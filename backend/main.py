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
from ultralytics import YOLO

import collect
from detection import (
    crop_rotated,
    deduplicate_boxes,
    detect,
    detect_obb,
    detect_seg,
    fuse_teacher_boxes,
    refine_types,
    split_banana_cluster_detections,
    compute_mask_metrics,
)

# Pillow tự cảnh báo/chặn ảnh lớn; giới hạn đã kiểm tra riêng bằng MAX_PIXELS
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent.parent
DET_PATH = os.getenv("MODEL_PATH", str(ROOT / "yolo11s-seg.pt" if (ROOT / "yolo11s-seg.pt").exists() else ROOT / "exports/det_obb_s.pt"))
OBB_PATH = os.getenv("OBB_MODEL_PATH", str(ROOT / "exports/det_obb_s.pt"))
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
    if Path(OBB_PATH).exists() and Path(DET_PATH).resolve() != Path(OBB_PATH).resolve():
        models["obb"] = YOLO(OBB_PATH)
    if HBB_FALLBACK and Path(DET_PATH).resolve() != Path(HBB_PATH).resolve():
        models["hbb"] = YOLO(HBB_PATH)
    if TEACHER_CHECK:
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


def assess_detection(objects, width, height, diagnostics=None):
    """Mark predictions that are unsafe to confirm without human review."""
    reasons = []
    image_area = max(width * height, 1)

    # 1. Per-object checks
    for index, obj in enumerate(objects):
        lbl = obj["label"]
        x1, y1, x2, y2 = obj["bbox"]
        area = max(x2 - x1, 0) * max(y2 - y1, 0)
        conf = obj.get("confidence", 1.0)

        # Low confidence
        if conf < CONF and not obj.get("rescued"):
            reasons.append(f"quả #{index + 1} ({lbl}) có độ tin cậy thấp ({conf:.2f} < {CONF})")

        # Oversized object: covering large portion of image, likely a cluster/group
        if area / image_area > 0.60:
            reasons.append(f"khung #{index + 1} quá lớn ({area / image_area:.0%}), có nguy cơ gộp nhiều quả")

        # Solidity check for segmentation masks (irregular/dumbbell shaped masks)
        solidity = obj.get("solidity", 1.0)
        if obj.get("polygon") and len(obj["polygon"]) >= 5:
            if lbl in ("apple", "orange") and solidity < 0.65:
                reasons.append(f"quả #{index + 1} ({lbl}) có hình dạng bất thường (độ lồi {solidity:.2f}), nghi gộp nhiều quả")
            elif lbl == "banana" and solidity < 0.40:
                reasons.append(f"quả #{index + 1} ({lbl}) có hình dạng bất thường (độ lồi {solidity:.2f}), nghi gộp cả nải")

        # High occlusion warning
        occlusion = obj.get("occlusion_ratio", 0.0)
        if occlusion >= 0.55:
            reasons.append(f"quả #{index + 1} ({lbl}) bị che khuất nhiều ({occlusion:.0%}), ranh giới không hoàn chỉnh")

    # 2. Pairwise instance overlap check
    for i in range(len(objects)):
        oi = objects[i]
        for j in range(i + 1, len(objects)):
            oj = objects[j]
            xi1, yi1, xi2, yi2 = oi["bbox"]
            xj1, yj1, xj2, yj2 = oj["bbox"]
            inter = max(0, min(xi2, xj2) - max(xi1, xj1)) * max(0, min(yi2, yj2) - max(yi1, yj1))
            area_i = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
            area_j = max(xj2 - xj1, 0) * max(yj2 - yj1, 0)
            union = area_i + area_j - inter
            iou = inter / max(union, 1e-9)

            if oi["label"] == oj["label"]:
                if iou >= 0.40:
                    reasons.append(f"hai quả cùng loại #{i + 1} và #{j + 1} ({oi['label']}) bị chồng lấn lớn (IoU {iou:.0%})")
            else:
                if iou >= 0.50:
                    reasons.append(f"hai quả khác loại #{i + 1} ({oi['label']}) và #{j + 1} ({oj['label']}) đè lên nhau quá nhiều (IoU {iou:.0%})")

    # 3. Model discrepancy check
    if diagnostics:
        if diagnostics.get("fallback_used"):
            reasons.append("mô hình kiểm chứng (fallback) cho kết quả khác biệt, cần đối chiếu số lượng")
        p_cnt = diagnostics.get("primary_count")
        f_cnt = diagnostics.get("fallback_count")
        if p_cnt is not None and f_cnt is not None and abs(p_cnt - f_cnt) >= 2:
            reasons.append(f"số lượng quả giữa các mô hình lệch nhau (Detector chính: {p_cnt}, Kiểm chứng: {f_cnt})")

    unique_reasons = list(dict.fromkeys(reasons))
    is_uncertain = len(unique_reasons) > 0
    return {
        "status": "needs_review" if is_uncertain else "ready",
        "is_uncertain": is_uncertain,
        "reasons": unique_reasons,
    }


def predict(img, mode="careful"):
    det, cls = models["det"], models["cls"]
    with lock:
        torch.set_num_threads(CPU_THREADS)
        polys = None
        mask_metrics = None
        diagnostics = {"fallback_used": False, "primary_count": None, "fallback_count": None}
        conf = min(CONF, RESCUE_CONF) if "clip" in models else CONF

        if det.task == "segment":
            # Instance Segmentation: đếm và phân tách quả bằng mask đa giác thực
            polys, boxes, scores, classes, mask_metrics = detect_seg(
                det, img, conf=conf, imgsz=IMGSZ, iou=NMS_IOU, quantize=QUANT,
            )
            # Tách nải chuối (banana cluster) thành từng quả riêng biệt bằng OBB spine partitioning
            if "obb" in models and len(boxes):
                polys, boxes, scores, classes = split_banana_cluster_detections(
                    polys, boxes, scores, classes, models["obb"], img,
                    conf=max(0.30, conf), det_names=det.names, quantize=QUANT,
                )
                mask_metrics = compute_mask_metrics(polys, boxes, img.shape[1], img.shape[0])
            diagnostics["primary_count"] = len(boxes)
            if "hbb" in models:
                hbb_boxes, _, _ = detect(
                    models["hbb"], img, conf=conf, imgsz=QUALITY_IMGSZ,
                    iou=NMS_IOU, quantize=QUANT, rotated=False,
                )
                diagnostics["fallback_count"] = len(hbb_boxes)
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
            if mask_metrics is not None:
                mask_metrics = [m for i, m in enumerate(mask_metrics) if keep[i]]
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
            "rescued": bool(float(conf) < CONF),   # điểm detector < CONF nhưng CLIP xác nhận là quả
        }
        if polys is not None and k < len(polys) and len(polys[k]) > 0:
            item["polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in polys[k]]
        if mask_metrics is not None and k < len(mask_metrics):
            item["mask_area"] = mask_metrics[k]["mask_area"]
            item["solidity"] = mask_metrics[k]["solidity"]
            item["occlusion_ratio"] = mask_metrics[k]["occlusion_ratio"]
            item["is_occluded"] = mask_metrics[k]["is_occluded"]
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
        "hbb_fallback": HBB_FALLBACK and "hbb" in models,
        "hbb_model": Path(HBB_PATH).name if "hbb" in models else None,
        "cls_model": Path(CLS_PATH).name,
        "cpu_threads": torch.get_num_threads(),
        "conf": CONF,
        "nms_iou": NMS_IOU,
        "default_mode": "careful",
        "careful_imgsz": QUALITY_IMGSZ,
        "obb_imgsz": OBB_IMGSZ,
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


@app.post("/api/predict")
async def predict_api(file: UploadFile = File(...), mode: Literal["fast", "careful"] = Form("careful")):
    _, img = await read_image(file)
    t = time.perf_counter()
    objects, diagnostics = await run_in_threadpool(predict, img, mode)
    ms = (time.perf_counter() - t) * 1000

    h, w = img.shape[:2]
    quality = assess_detection(objects, w, h, diagnostics)
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
        },
    }


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
