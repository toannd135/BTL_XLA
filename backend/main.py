import io
import json
import os

# Bound CPU work before torch/BLAS initializes; the user can override this explicitly.
CPU_THREADS = max(1, min(8, int(os.getenv("CPU_THREADS", "2"))))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import SAM, YOLO

import collect
from quality import assess_detection
from detection import (
    crop_rotated,
    deduplicate_boxes,
    detect,
    detect_obb,
    detect_seg,
    fuse_teacher_boxes,
    refine_types,
    split_fruit_cluster_detections,
    compute_mask_metrics,
    configure_sam_refiner,
)

# Pillow tự cảnh báo/chặn ảnh lớn; giới hạn đã kiểm tra riêng bằng MAX_PIXELS
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent.parent
DET_PATH = os.getenv("MODEL_PATH", str(ROOT / "yolo11s-seg.pt" if (ROOT / "yolo11s-seg.pt").exists() else ROOT / "exports/det_obb_s.pt"))
OBB_PATH = os.getenv("OBB_MODEL_PATH", str(ROOT / "exports/det_obb_s.pt"))
OBB_ENSEMBLE_PATHS = [
    Path(value.strip()) for value in os.getenv(
        "OBB_ENSEMBLE_PATHS",
        ",".join(str(p) for p in (
            ROOT / "exports/det_obb_s.pt",
            ROOT / "exports/det_obb_map95_s1.pt",
            ROOT / "exports/det_obb_map95_s2.pt",
        )),
    ).split(",") if value.strip()
]
SAM_REFINE = os.getenv("SAM_REFINE", "1") == "1"
SAM_PATH = os.getenv("SAM_MODEL_PATH", str(next(
    (p for p in (ROOT / "work/weights/sam2.1_t.pt", ROOT / "work/weights/sam2.1_b.pt") if p.is_file()),
    ROOT / "work/weights/sam2.1_t.pt",
)))
HBB_FALLBACK = os.getenv("HBB_FALLBACK", "1") == "1"
HBB_PATH = os.getenv("HBB_MODEL_PATH", str(ROOT / "exports/det_B_clean_backup.pt"))
CLS_PATH = os.getenv("CLS_MODEL_PATH", str(ROOT / "exports/cls_finetune_best.pt"))
FEEDBACK_DIR = Path(os.getenv("FEEDBACK_DIR", str(ROOT / "work" / "feedback")))
TEACHER_CHECK = os.getenv("TEACHER_CHECK", "1") == "1"
TEACHER_PATH = os.getenv("TEACHER_MODEL_PATH", str(ROOT / "yolo11s.pt"))

# giống notebook: CONF_PRED, IMGSZ, CLS_IMGSZ, CROP_PAD
CONF = float(os.getenv("CONF", "0.4"))   # chốt 0.40 cho model C (F1 tổng 0.82, nền sai 1/71); hạ xuống 0.25-0.35 nếu hay bỏ sót chuối
IMGSZ = 640
QUALITY_IMGSZ = 640   # 960 đếm chuối thừa ~2.4x, nền sai 3/71 (docs/handoff/dem-chuoi.md)
OBB_IMGSZ = int(os.getenv("OBB_IMGSZ", "1280"))
SEG_IMGSZ = int(os.getenv("SEG_IMGSZ", "640"))
SEG_RETINA_MASKS = os.getenv("SEG_RETINA_MASKS", "1") == "1"
SEG_SIMPLIFY_EPS = float(os.getenv("SEG_SIMPLIFY_EPS", "0.75"))
ENHANCE_IMAGE = os.getenv("ENHANCE_IMAGE", "0") == "1"
INSTANCE_CROP_CHECKS = max(0, min(24, int(os.getenv("INSTANCE_CROP_CHECKS", "12"))))
NMS_IOU = float(os.getenv("NMS_IOU", "0.7"))   # giữ các quả chuối sát nhau; đặt 0.6 nếu tăng box trùng
QUANT = 16 if os.getenv("HALF", "1") == "1" else None   # FP16 trên GPU (ultralytics tự tắt khi chạy CPU); đặt HALF=0 để tắt
CLS_IMGSZ = 224
CROP_PAD = 0.10
# CLIP kiểm tra loại quả từng crop (zero-shot): bỏ vật không phải chuối/táo/cam (dâu, nho...), sửa nhầm táo/cam.
# Ngưỡng chọn trên val (work/fruitcls/eval_gate.py): test 373 ảnh đếm đúng 75.6% -> 78.8%, ảnh "vật khác" báo sai 318 -> 110/1054.
CLIP_CHECK = os.getenv("CLIP_CHECK", "1") == "1"
CLIP_T_OTHER = float(os.getenv("CLIP_T_OTHER", "0.8"))
CLIP_T_SWAP = float(os.getenv("CLIP_T_SWAP", "0.8"))   # val: 0.8 = 0.9 (cùng lỗi), 0.7 sai thêm; 0.8 sửa thêm táo bị gọi cam
# Có CLIP: chạy detector ở ngưỡng thấp hơn, box điểm < CONF chỉ giữ khi CLIP chắc >= CLIP_T_RESCUE là quả (bớt bỏ sót)
RESCUE_CONF = float(os.getenv("RESCUE_CONF", "0.25"))
CLIP_T_RESCUE = float(os.getenv("CLIP_T_RESCUE", "0.9"))

MAX_SIZE = 10 * 1024 * 1024
# file nén nhỏ vẫn có thể bung ra rất lớn sau khi giải mã, nên chặn theo số pixel
MAX_PIXELS = 40_000_000
ALLOWED = {"image/jpeg", "image/png", "image/webp"}
# mặc định nhận trang web mở bằng localhost hoặc 127.0.0.1; thêm địa chỉ khác qua CORS_ORIGINS (phân cách dấu phẩy)
ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

models = {}
# model ultralytics không an toàn khi nhiều luồng gọi cùng lúc
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(CPU_THREADS)
    cv2.setNumThreads(CPU_THREADS)
    models["det"] = YOLO(DET_PATH)
    models["cls"] = YOLO(CLS_PATH)
    if models["det"].task == "segment" and Path(OBB_PATH).exists():
        models["obb"] = YOLO(OBB_PATH)
        models["obb_ensemble"] = [
            YOLO(str(path)) for path in OBB_ENSEMBLE_PATHS
            if path.exists()
        ] or [models["obb"]]
    if models["det"].task == "segment" and SAM_REFINE and Path(SAM_PATH).is_file():
        models["sam"] = configure_sam_refiner(SAM(SAM_PATH))
    if HBB_FALLBACK and Path(DET_PATH).resolve() != Path(HBB_PATH).resolve():
        models["hbb"] = YOLO(HBB_PATH)
    if TEACHER_CHECK and models["det"].task == "obb":
        models["teacher"] = YOLO(TEACHER_PATH)
    if CLIP_CHECK:
        try:
            from clip_check import ClipChecker
            models["clip"] = ClipChecker()
        except Exception as e:   # thiếu open_clip/model: chạy tiếp không có bước kiểm tra
            print("CLIP_CHECK tắt:", e)
    # chạy thử 1 lượt để request đầu tiên không bị chậm
    blank = np.zeros((IMGSZ, IMGSZ, 3), np.uint8)
    models["det"].predict(blank, imgsz=IMGSZ, quantize=QUANT, verbose=False)
    if "obb" in models:
        models["obb"].predict(blank, imgsz=IMGSZ, quantize=QUANT, verbose=False)
    for model in models.get("obb_ensemble", [])[1:]:
        model.predict(blank, imgsz=IMGSZ, quantize=QUANT, verbose=False)
    if "hbb" in models:
        models["hbb"].predict(blank, imgsz=IMGSZ, quantize=QUANT, verbose=False)
    models["cls"].predict(blank, imgsz=CLS_IMGSZ, quantize=QUANT, verbose=False)
    if "teacher" in models:
        models["teacher"].predict(blank, imgsz=IMGSZ, quantize=QUANT, classes=[46, 47, 49], verbose=False)
    # Some Ultralytics device setup paths reset torch's thread count.
    torch.set_num_threads(CPU_THREADS)
    yield
    models.clear()


app = FastAPI(title="Nhận dạng trái cây", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ORIGINS if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def on_error(request, exc):
    return JSONResponse(status_code=500, content={"detail": "Lỗi khi xử lý ảnh trên server"})


def crop_box(img, xyxy, pad=CROP_PAD):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, min(w - 1, x1 - px))), int(max(0, min(h - 1, y1 - py)))
    x2, y2 = int(min(w, max(x1 + 1, x2 + px))), int(min(h, max(y1 + 1, y2 + py)))
    return img[y1:y2, x1:x2]


def predict(img, mode="careful"):
    det, cls = models["det"], models["cls"]
    with lock:
        torch.set_num_threads(CPU_THREADS)
        polys = None
        mask_metrics = None
        diagnostics = {"fallback_used": False, "primary_count": None, "fallback_count": None,
                       "banana_split_pending": 0, "banana_split_groups": 0, "primary_task": det.task}
        conf = min(CONF, RESCUE_CONF) if "clip" in models else CONF

        if det.task == "segment":
            # Instance Segmentation: đếm và phân tách quả bằng mask đa giác thực
            polys, boxes, scores, classes, mask_metrics = detect_seg(
                det, img, conf=conf, imgsz=SEG_IMGSZ, iou=NMS_IOU, quantize=QUANT,
                retina_masks=SEG_RETINA_MASKS, simplify_eps=SEG_SIMPLIFY_EPS,
                compute_metrics=False,
            )
            hbb_detections = None
            if "hbb" in models:
                hbb_detections = detect(
                    models["hbb"], img, conf=conf, imgsz=QUALITY_IMGSZ,
                    iou=NMS_IOU, quantize=QUANT, rotated=False,
                )
                diagnostics["fallback_count"] = len(hbb_detections[0])
            # All fruit classes can be grouped or missed, including apples.
            polys, boxes, scores, classes = split_fruit_cluster_detections(
                polys, boxes, scores, classes,
                models.get("obb_ensemble", models.get("obb")), img,
                conf=max(0.30, conf), det_names=det.names, quantize=QUANT,
                mask_model=models.get("sam"), diagnostics=diagnostics,
                hbb_detections=hbb_detections,
                hbb_names=models["hbb"].names if "hbb" in models else None,
                seg_model=det,
                max_crop_checks=INSTANCE_CROP_CHECKS,
            )
            diagnostics["primary_count"] = len(boxes)
        elif det.task == "obb":   # khung xoay: một lượt, không cần xoay ảnh (model đã học góc bất kỳ)
            polys, boxes, scores, classes = detect_obb(
                det, img, conf=conf, imgsz=OBB_IMGSZ, iou=NMS_IOU, quantize=QUANT,
            )
            diagnostics["primary_count"] = len(boxes)
            # OBB is better for crowded/diagonal fruit, while HBB B is stronger
            # on ordinary scenes. Use HBB only when it clearly recovers more
            # candidates and the OBB result is not already crowded.
            if "hbb" in models and len(boxes) <= 6:
                hbb_boxes, hbb_scores, hbb_classes = detect(
                    models["hbb"], img, conf=conf, imgsz=QUALITY_IMGSZ,
                    iou=NMS_IOU, quantize=QUANT, rotated=False,
                )
                diagnostics["fallback_count"] = len(hbb_boxes)
                if len(hbb_boxes) >= len(boxes) + 2:
                    boxes, scores, classes = hbb_boxes, hbb_scores, hbb_classes
                    polys = None
                    diagnostics["fallback_used"] = True
            if "teacher" in models and polys is not None:
                teacher_result = models["teacher"].predict(
                    img, conf=0.20, imgsz=IMGSZ, quantize=QUANT,
                    classes=[46, 47, 49], verbose=False,
                )[0]
                detector_ids = {name: idx for idx, name in det.names.items()}
                teacher_map = {46: detector_ids["banana"], 47: detector_ids["apple"], 49: detector_ids["orange"]}
                teacher_thresholds = {
                    teacher_map[46]: 0.60,
                    teacher_map[47]: 0.38,
                    teacher_map[49]: 0.80,
                }
                polys, boxes, scores, classes = fuse_teacher_boxes(
                    polys, boxes, scores, classes,
                    teacher_result.boxes.xyxy.cpu().numpy(),
                    np.asarray(teacher_result.boxes.conf.tolist(), dtype=np.float32),
                    np.asarray(teacher_result.boxes.cls.tolist(), dtype=np.int64),
                    teacher_map, teacher_thresholds,
                )
        else:
            boxes, scores, classes = detect(det, img, conf=conf,
                imgsz=QUALITY_IMGSZ if mode == "careful" else IMGSZ,
                iou=NMS_IOU, quantize=QUANT, rotated=mode == "careful")
            diagnostics["primary_count"] = len(boxes)
        if len(boxes) and "clip" in models:
            clip = models["clip"]
            keep, classes = refine_types(det.names, classes, clip.names,
                                         clip.probs_boxes(img, boxes, CROP_PAD, t_other=CLIP_T_OTHER), CLIP_T_OTHER, CLIP_T_SWAP,
                                         scores=scores, conf=CONF, t_rescue=CLIP_T_RESCUE,
                                         protected_min_conf=0.30)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
            if polys is not None:
                polys = [p for i, p in enumerate(polys) if keep[i]]
        # Recompute after splitting AND CLIP filtering so overlap indices and ratios
        # cannot refer to an instance that has already been removed.
        if det.task == "segment":
            mask_metrics = compute_mask_metrics(polys, boxes, img.shape[1], img.shape[0])
            diagnostics["primary_count"] = len(boxes)
        if polys is not None and det.task == "obb" and len(boxes):
            polys, boxes, scores, classes = deduplicate_boxes(polys, boxes, scores, classes)
        if not len(boxes):
            return [], diagnostics
        crops = [crop_rotated(img, q, CROP_PAD) for q in polys] if (polys is not None and det.task == "obb") else [crop_box(img, b) for b in boxes]
        # Explicit chunks: predict(list) does not honor batch in every Ultralytics version.
        rs = []
        for start in range(0, len(crops), 8):
            rs.extend(cls.predict(crops[start:start + 8], imgsz=CLS_IMGSZ, quantize=QUANT, verbose=False))
    out = []
    for k, (b, c, conf, p) in enumerate(zip(boxes, classes, scores, rs)):
        lbl = det.names[int(c)] if int(c) in det.names else f"fruit_{int(c)}"
        item = {
            "label": lbl,
            "confidence": round(float(conf), 4),
            "state": cls.names[int(p.probs.top1)],
            "state_confidence": round(float(p.probs.top1conf), 4),
            "bbox": [round(float(v), 1) for v in b],
            "rescued": bool("clip" in models and float(conf) < CONF),
        }
        if polys is not None and k < len(polys) and len(polys[k]) > 0:
            item["polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in polys[k]]
        item["geometry_type"] = ("mask" if det.task == "segment" else "obb") if "polygon" in item else "box"
        if mask_metrics is not None and k < len(mask_metrics):
            item["mask_area"] = mask_metrics[k]["mask_area"]
            item["solidity"] = mask_metrics[k]["solidity"]
            item["occlusion_ratio"] = mask_metrics[k]["occlusion_ratio"]
            item["is_occluded"] = mask_metrics[k]["is_occluded"]
            item["overlap_ratio"] = mask_metrics[k]["overlap_ratio"]
            item["has_mask_overlap"] = mask_metrics[k]["has_mask_overlap"]
            item["mask_valid"] = mask_metrics[k]["mask_valid"]
            item["mask_overlaps"] = mask_metrics[k]["overlaps"]
            if not item["mask_valid"]:
                item["geometry_type"] = "box"
        out.append(item)
    return out, diagnostics


@app.get("/api/health")
def health():
    return {
        "status": "ok" if models else "loading",
        # predictor mới giữ thiết bị thật (model.device vẫn báo cpu)
        "device": str(models["det"].predictor.device) if models else None,
        "det_model": Path(DET_PATH).name,
        "obb_helper": "obb" in models,
        "obb_model": Path(OBB_PATH).name if "obb" in models else None,
        "obb_ensemble": [Path(path).name for path in OBB_ENSEMBLE_PATHS
                         if path.exists()] if "obb" in models else [],
        "hbb_fallback": HBB_FALLBACK and "hbb" in models,
        "hbb_model": Path(HBB_PATH).name if "hbb" in models else None,
        "cls_model": Path(CLS_PATH).name,
        "cpu_threads": torch.get_num_threads(),
        "conf": CONF,
        "nms_iou": NMS_IOU,
        "default_mode": "careful",
        "careful_imgsz": QUALITY_IMGSZ,
        "obb_imgsz": OBB_IMGSZ,
        "seg_imgsz": SEG_IMGSZ,
        "seg_retina_masks": SEG_RETINA_MASKS,
        "seg_simplify_eps": SEG_SIMPLIFY_EPS,
        "sam_refine": "sam" in models,
        "sam_model": Path(SAM_PATH).name if "sam" in models else None,
        "enhance_image": ENHANCE_IMAGE,
        "instance_refine_classes": ["banana", "apple", "orange"],
        "instance_crop_checks": INSTANCE_CROP_CHECKS,
        "clip_check": "clip" in models,
        "teacher_check": "teacher" in models,
        "teacher_model": Path(TEACHER_PATH).name if "teacher" in models else None,
    }


async def read_image(file):
    """Kiểm tra và giải mã ảnh tải lên. Trả về (bytes, ảnh BGR)."""
    if file.content_type not in ALLOWED:
        raise HTTPException(400, "Chỉ nhận ảnh jpg, png hoặc webp")
    # đọc tối đa MAX_SIZE + 1 byte, không nạp cả file quá lớn vào bộ nhớ
    data = await file.read(MAX_SIZE + 1)
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "Ảnh vượt quá 10MB")
    if not data:
        raise HTTPException(400, "File ảnh rỗng")

    # đọc kích thước từ phần đầu file trước khi giải mã cả ảnh
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Exception:
        raise HTTPException(400, "Không đọc được file ảnh")
    if w * h > MAX_PIXELS:
        raise HTTPException(413, f"Ảnh quá lớn ({w}x{h}), tối đa khoảng 40 triệu điểm ảnh")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không đọc được file ảnh")
    if ENHANCE_IMAGE:
        img = enhance_image(img)
    return data, img


def enhance_image(img: np.ndarray) -> np.ndarray:
    """Tiền xử lý ảnh để tăng độ nét và chi tiết trước khi chạy detection.

    Pipeline:
    1. Upscale ảnh rất nhỏ (<480px) → YOLO có thêm pixel để lấy contour chính xác hơn
    2. Unsharp mask → làm nét cạnh (trái cây có màu đơn, nét rõ giúp mask đúng hơn)
    3. CLAHE trên kênh Luminance → tăng tương phản cục bộ, phân tách màu gần nhau
    """
    h, w = img.shape[:2]
    min_side = min(h, w)

    # 1. Upscale chỉ khi ảnh thực sự nhỏ (<480px chiều ngắn) để tránh chậm không cần thiết
    #    YOLO tự resize về imgsz nên chỉ cần upscale để tăng thông tin contour ban đầu
    if min_side < 480:
        scale = 480 / min_side
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 2. Unsharp mask: amount=0.8 — đủ để nét cạnh, không mất quả confidence thấp
    #    sharp = alpha * original + (1-alpha) * blur  →  original + (alpha-1)*(original-blur)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    img = cv2.addWeighted(img, 1.4, blur, -0.4, 0)

    # 3. CLAHE trên kênh L của LAB (tăng tương phản cục bộ, không làm bão hòa màu)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    lab = cv2.merge([l_chan, a_chan, b_chan])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img


def segment_prompt(img, prompts):
    """Return one SAM polygon for normalized positive/negative point prompts."""
    if "sam" not in models:
        raise RuntimeError("SAM chưa được bật hoặc chưa có trọng số")
    height, width = img.shape[:2]
    points = [[int(round(float(p["x"]) * (width - 1))),
               int(round(float(p["y"]) * (height - 1)))] for p in prompts]
    labels = [int(p["label"]) for p in prompts]
    with lock:
        result = models["sam"].predict(
            img, points=[points], labels=[labels], conf=0.0, verbose=False,
        )[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None
    masks = result.masks.data.cpu().numpy()
    scores = (result.boxes.conf.cpu().numpy()
              if result.boxes is not None else np.ones(len(masks), dtype=np.float32))
    best = int(np.argmax(scores))
    mask = (masks[best] > 0.5).astype(np.uint8)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 3 or cv2.contourArea(contour) < 32:
        return None
    return {
        "polygon": [[round(float(x) / width, 6), round(float(y) / height, 6)]
                    for x, y in contour],
        "confidence": round(float(scores[best]), 6),
    }


@app.post("/api/predict")
async def predict_api(file: UploadFile = File(...), mode: Literal["fast", "careful"] = Form("careful")):
    _, img = await read_image(file)
    t = time.perf_counter()
    objects, diagnostics = await run_in_threadpool(predict, img, mode)
    ms = (time.perf_counter() - t) * 1000

    h, w = img.shape[:2]
    quality = assess_detection(objects, w, h, diagnostics, conf_threshold=CONF)
    return {
        "image_width": w,
        "image_height": h,
        "inference_ms": round(ms, 1),
        "mode": mode,
        "objects": objects,
        "quality": quality | {
            "fallback_used": diagnostics["fallback_used"],
            "primary_count": diagnostics["primary_count"],
            "fallback_count": diagnostics["fallback_count"],
            "banana_split_pending": diagnostics["banana_split_pending"],
            "banana_split_groups": diagnostics["banana_split_groups"],
            "fruit_split_groups": diagnostics.get("fruit_split_groups", 0),
            "fruit_split_pending": diagnostics.get("fruit_split_pending", 0),
            "fruit_recovered": diagnostics.get("fruit_recovered", 0),
            "instance_refinement": diagnostics.get("instance_refinement", {}),
        },
    }


@app.post("/api/segment_prompt")
async def segment_prompt_api(file: UploadFile = File(...), prompts: str = Form(...)):
    """Create a review polygon from SAM point prompts; never changes detections."""
    if "sam" not in models:
        raise HTTPException(503, "SAM chưa sẵn sàng")
    try:
        payload = json.loads(prompts)
    except (TypeError, json.JSONDecodeError):
        raise HTTPException(400, "Danh sách điểm SAM không hợp lệ")
    if (not isinstance(payload, list) or not 1 <= len(payload) <= 16 or
            any(not isinstance(point, dict) or point.get("label") not in (0, 1)
                or not all(isinstance(point.get(key), (int, float))
                           and np.isfinite(point[key]) and 0 <= point[key] <= 1
                           for key in ("x", "y")) for point in payload)):
        raise HTTPException(400, "Điểm SAM phải có x, y chuẩn hóa và label 0/1")
    _, img = await read_image(file)
    try:
        result = await run_in_threadpool(segment_prompt, img, payload)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))
    if result is None:
        raise HTTPException(422, "SAM không tạo được mask từ các điểm đã chọn")
    return {"image_width": img.shape[1], "image_height": img.shape[0], **result}


@app.post("/api/feedback")
async def feedback_api(
    file: UploadFile = File(...),
    verdict: Literal["correct", "needs_review"] = Form(...),
    result: str = Form(...),
):
    """Store a reviewed prediction as a candidate for later annotation/training."""
    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        raise HTTPException(400, "Kết quả phản hồi không hợp lệ")
    if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
        raise HTTPException(400, "Kết quả phản hồi phải chứa danh sách objects")
    valid_labels = {"banana", "apple", "orange"}
    valid_states = {"fresh", "rotten"}
    for obj in payload["objects"]:
        if not isinstance(obj, dict):
            raise HTTPException(400, "Mỗi object phản hồi phải là một object JSON")
        if obj.get("label") not in valid_labels or obj.get("state") not in valid_states:
            raise HTTPException(400, "Nhãn quả hoặc trạng thái phản hồi không hợp lệ")
        bbox = obj.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise HTTPException(400, "Bounding box phản hồi không hợp lệ")
        try:
            coords = [float(value) for value in bbox]
        except (TypeError, ValueError):
            raise HTTPException(400, "Bounding box phản hồi phải là số")
        if not all(np.isfinite(coords)) or coords[2] <= coords[0] or coords[3] <= coords[1]:
            raise HTTPException(400, "Bounding box phản hồi phải có kích thước dương")

    _, img = await read_image(file)
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    feedback_id = uuid.uuid4().hex
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise HTTPException(500, "Không thể lưu ảnh phản hồi")
    (FEEDBACK_DIR / f"{feedback_id}.jpg").write_bytes(encoded.tobytes())
    (FEEDBACK_DIR / f"{feedback_id}.json").write_text(json.dumps({
        "id": feedback_id,
        "verdict": verdict,
        "source_name": file.filename or "upload",
        "result": payload,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, ensure_ascii=False, indent=2) + "\n")
    return {"id": feedback_id, "verdict": verdict, "status": "saved"}


@app.get("/api/feedback/stats")
def feedback_stats():
    files = list(FEEDBACK_DIR.glob("*.json")) if FEEDBACK_DIR.exists() else []
    counts = {"correct": 0, "needs_review": 0}
    for path in files:
        try:
            verdict = json.loads(path.read_text()).get("verdict")
        except (OSError, json.JSONDecodeError):
            continue
        if verdict in counts:
            counts[verdict] += 1
    return {"total": sum(counts.values()), **counts}


@app.post("/api/collect")
async def collect_api(file: UploadFile = File(...), session: str = Form(...)):
    if not collect.SESSION_RE.match(session):
        raise HTTPException(400, "Mã phiên thu thập không hợp lệ")
    data, img = await read_image(file)
    return await run_in_threadpool(collect.collect, data, img, session, models, lock, crop_box, CONF, IMGSZ, CLS_IMGSZ)


@app.get("/api/collect/stats")
def collect_stats():
    return collect.stats()
