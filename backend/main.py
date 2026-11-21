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
from coordinate_transform import ImageTransform
from runtime_memory import memory_snapshot, reset_peak_memory, resolve_device
from inference.providers.yolo_provider import YoloSegProvider
from inference.router import InferenceRouter
from postprocessing.counting import count_instances
from postprocessing.quality_gate import assess_instances
from detection import (
    refine_types,
    compute_mask_metrics,
)

# Pillow tự cảnh báo/chặn ảnh lớn; giới hạn đã kiểm tra riêng bằng MAX_PIXELS
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent.parent
# Best saved held-out count result: work/audit/seg_candidate_count_test.json.
DEFAULT_DET_PATH = str(ROOT / "runs/seg/seg_boundary_candidate_gpu/weights/best.pt")
DET_PATH = os.getenv("MODEL_PATH", DEFAULT_DET_PATH)
CLS_PATH = os.getenv("CLS_MODEL_PATH", str(ROOT / "exports/cls_finetune_best.pt"))
FEEDBACK_DIR = Path(os.getenv("FEEDBACK_DIR", str(ROOT / "work" / "feedback")))
DEVICE_REQUESTED = os.getenv("DEVICE", "auto")
INFERENCE_PROVIDER = os.getenv("INFERENCE_PROVIDER", "local_yolo").strip().lower()
CLASSIFIER_CHECK = os.getenv("CLASSIFIER_CHECK", "1") == "1"
CLASSIFIER_DEVICE_REQUESTED = os.getenv("CLASSIFIER_DEVICE", "cpu")
CLIP_DEVICE_REQUESTED = os.getenv("CLIP_DEVICE", "cpu")

# giống notebook: CONF_PRED, IMGSZ, CLS_IMGSZ, CROP_PAD
CONF = float(os.getenv("CONF", "0.4"))
IMGSZ = int(os.getenv("MODEL_IMGSZ", "640"))
NMS_IOU = float(os.getenv("NMS_IOU", "0.6"))
QUANT = 16 if os.getenv("HALF", "1") == "1" else None   # FP16 trên GPU (ultralytics tự tắt khi chạy CPU); đặt HALF=0 để tắt
CLS_IMGSZ = 224
CROP_PAD = 0.10
# CLIP chỉ kiểm tra ngữ nghĩa; các instance chuối luôn được giữ nguyên.
CLIP_CHECK = os.getenv("CLIP_CHECK", "0") == "1"
CLIP_T_OTHER = float(os.getenv("CLIP_T_OTHER", "0.8"))
CLIP_T_SWAP = float(os.getenv("CLIP_T_SWAP", "0.8"))
CLIP_T_RESCUE = float(os.getenv("CLIP_T_RESCUE", "0.9"))

MAX_SIZE = 10 * 1024 * 1024
# file nén nhỏ vẫn có thể bung ra rất lớn sau khi giải mã, nên chặn theo số pixel
MAX_PIXELS = 40_000_000
ALLOWED = {"image/jpeg", "image/png", "image/webp"}
# mặc định nhận trang web mở bằng localhost hoặc 127.0.0.1; thêm địa chỉ khác qua CORS_ORIGINS (phân cách dấu phẩy)
ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

models = {}
runtime = {"device": None, "device_reason": None, "classifier_device": None,
           "clip_device": None, "clip_error": None, "last_gpu_memory": None}
# model ultralytics không an toàn khi nhiều luồng gọi cùng lúc
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(CPU_THREADS)
    cv2.setNumThreads(CPU_THREADS)
    runtime["device"], runtime["device_reason"] = resolve_device(DEVICE_REQUESTED)
    print(f"Detection device: {runtime['device']} ({runtime['device_reason']})")
    requested_gpu_models = int(runtime["device"].startswith("cuda"))
    if CLASSIFIER_CHECK and CLASSIFIER_DEVICE_REQUESTED != "cpu":
        runtime["classifier_device"], _ = resolve_device(CLASSIFIER_DEVICE_REQUESTED, "classifier")
        requested_gpu_models += int(runtime["classifier_device"].startswith("cuda"))
    if CLIP_CHECK and CLIP_DEVICE_REQUESTED != "cpu":
        runtime["clip_device"], _ = resolve_device(CLIP_DEVICE_REQUESTED, "CLIP")
        requested_gpu_models += int(runtime["clip_device"].startswith("cuda"))
    if requested_gpu_models > 1:
        raise RuntimeError("Only one inference model may be assigned to CUDA at a time; keep optional models on CPU or disable them")
    print(f"Detection checkpoint: {Path(DET_PATH).expanduser().resolve()}")
    if INFERENCE_PROVIDER != "local_yolo":
        raise RuntimeError(f"Unsupported or unconfigured INFERENCE_PROVIDER={INFERENCE_PROVIDER!r}")
    models["det"] = YOLO(DET_PATH)
    if models["det"].task != "segment":
        raise RuntimeError("Banana Instance Separation requires a YOLO segmentation model (task=segment).")
    class_names = {str(name).lower() for name in models["det"].names.values()}
    if not {"banana", "apple", "orange"}.issubset(class_names):
        raise RuntimeError(f"Segmentation checkpoint must support banana, apple and orange; got {models['det'].names}")
    print(f"Detection model loaded: task={models['det'].task}, classes={models['det'].names}")
    models["inference_router"] = InferenceRouter({
        "local_yolo": YoloSegProvider(
            models["det"], conf=CONF, imgsz=IMGSZ, iou=NMS_IOU,
            quantize=QUANT, device=runtime["device"],
            half=bool(QUANT and runtime["device"].startswith("cuda")),
        ),
    }, default_provider=INFERENCE_PROVIDER)
    if CLASSIFIER_CHECK:
        if runtime["classifier_device"] is None:
            runtime["classifier_device"], _ = resolve_device(CLASSIFIER_DEVICE_REQUESTED, "classifier")
        models["cls"] = YOLO(CLS_PATH)
        if models["cls"].task != "classify":
            raise RuntimeError(f"Classifier checkpoint must have task=classify; got {models['cls'].task}")
    if CLIP_CHECK:
        try:
            from clip_check import ClipChecker
            if runtime["clip_device"] is None:
                runtime["clip_device"], _ = resolve_device(CLIP_DEVICE_REQUESTED, "CLIP")
            models["clip"] = ClipChecker(device=runtime["clip_device"])
        except Exception as e:   # thiếu open_clip/model: chạy tiếp không có bước kiểm tra
            runtime["clip_error"] = str(e)
            print("CLIP_CHECK tắt:", e)
    # chạy thử 1 lượt để request đầu tiên không bị chậm
    blank = np.zeros((IMGSZ, IMGSZ, 3), np.uint8)
    models["det"].predict(blank, imgsz=IMGSZ, quantize=QUANT, device=runtime["device"],
                          half=bool(QUANT and runtime["device"].startswith("cuda")), verbose=False)
    if "cls" in models:
        models["cls"].predict(blank, imgsz=CLS_IMGSZ, quantize=QUANT,
                              device=runtime["classifier_device"],
                              half=bool(QUANT and runtime["classifier_device"].startswith("cuda")), verbose=False)
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


def assess_detection(objects, width, height):
    """Mark predictions that are unsafe to confirm without human review."""
    reasons = []
    image_area = max(width * height, 1)

    # 1. Per-object checks
    for index, obj in enumerate(objects):
        lbl = obj["label"]
        conf = obj.get("confidence", 1.0)

        # Low confidence
        if conf < CONF and not obj.get("rescued"):
            reasons.append(f"quả #{index + 1} ({lbl}) có độ tin cậy thấp ({conf:.2f} < {CONF})")

        mask_area = obj.get("mask_area", 0)
        if mask_area / image_area > 0.60:
            reasons.append(f"mask #{index + 1} quá lớn ({mask_area / image_area:.0%}), có nguy cơ gộp nhiều quả")

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

    # 2. Pairwise mask overlap check
    masks = []
    for obj in objects:
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = obj.get("polygon") or []
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 1)
        masks.append(mask)
    for i in range(len(objects)):
        oi = objects[i]
        for j in range(i + 1, len(objects)):
            oj = objects[j]
            inter = int(np.count_nonzero(masks[i] & masks[j]))
            area_i = int(np.count_nonzero(masks[i]))
            area_j = int(np.count_nonzero(masks[j]))
            union = area_i + area_j - inter
            iou = inter / max(union, 1e-9)

            if oi["label"] == oj["label"]:
                if iou >= 0.40:
                    reasons.append(f"hai mask cùng loại #{i + 1} và #{j + 1} ({oi['label']}) chồng lấn lớn (IoU {iou:.0%})")
            else:
                if iou >= 0.50:
                    reasons.append(f"hai mask khác loại #{i + 1} ({oi['label']}) và #{j + 1} ({oj['label']}) chồng lấn lớn (IoU {iou:.0%})")

    unique_reasons = list(dict.fromkeys(reasons))
    is_uncertain = len(unique_reasons) > 0
    return {
        "status": "needs_review" if is_uncertain else "ready",
        "is_uncertain": is_uncertain,
        "reasons": unique_reasons,
    }


def predict(img, mode="careful"):
    det, cls = models["det"], models.get("cls")
    with lock:
        torch.set_num_threads(CPU_THREADS)
        polys = None
        mask_metrics = None
        diagnostics = {"instance_count": 0}
        # Keep the detector threshold fixed: CLIP cannot rescue or discard
        # banana instances and therefore cannot change the banana count.
        conf = CONF

        # Keep model-produced instances intact. Distance-transform splitting is
        # benchmark-only until it improves held-out count results.
        reset_peak_memory(runtime["device"])
        try:
            provider_result = models["inference_router"].predict(img)
            instances = provider_result.instances
            polys = [instance.polygon for instance in instances]
            boxes = np.asarray([instance.bbox_xyxy for instance in instances], dtype=np.float32).reshape(-1, 4)
            scores = np.asarray([instance.confidence for instance in instances], dtype=np.float32)
            classes = np.asarray([instance.class_id for instance in instances], dtype=np.int64)
            diagnostics["provider"] = provider_result.provider
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError) or (runtime["device"].startswith("cuda") and "out of memory" in str(exc).lower()):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise HTTPException(503, "CUDA ran out of memory during segmentation; inference was not retried on CPU") from exc
            raise
        # Mask geometry is computed once after mapping polygons back to the
        # uploaded image, in restore_objects_to_original().
        mask_metrics = None
        if len(boxes) and "clip" in models:
            clip = models["clip"]
            try:
                type_probs = clip.probs_boxes(img, boxes, CROP_PAD, t_other=CLIP_T_OTHER)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError) or (runtime["clip_device"] and runtime["clip_device"].startswith("cuda") and "out of memory" in str(exc).lower()):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise HTTPException(503, "CUDA ran out of memory during CLIP; no CPU fallback was attempted") from exc
                raise
            keep, classes = refine_types(det.names, classes, clip.names, type_probs,
                                         CLIP_T_OTHER, CLIP_T_SWAP,
                                         scores=scores, conf=CONF, t_rescue=CLIP_T_RESCUE,
                                         protected_min_conf=0.30)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
            if polys is not None:
                polys = [p for i, p in enumerate(polys) if keep[i]]
            if mask_metrics is not None:
                mask_metrics = [m for i, m in enumerate(mask_metrics) if keep[i]]
        diagnostics["instance_count"] = len(boxes)
        runtime["last_gpu_memory"] = memory_snapshot(runtime["device"])
        diagnostics["gpu_memory"] = runtime["last_gpu_memory"]
        if not len(boxes):
            return [], diagnostics
        rs = []
        if cls is not None:
            crops = [crop_box(img, b) for b in boxes]
            # Explicit chunks: predict(list) does not honor batch in every Ultralytics version.
            try:
                for start in range(0, len(crops), 8):
                    rs.extend(cls.predict(crops[start:start + 8], imgsz=CLS_IMGSZ, quantize=QUANT,
                                          device=runtime["classifier_device"],
                                          half=bool(QUANT and runtime["classifier_device"].startswith("cuda")),
                                          verbose=False))
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError) or (runtime["classifier_device"].startswith("cuda") and "out of memory" in str(exc).lower()):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise HTTPException(503, "CUDA ran out of memory during classifier inference; no CPU fallback was attempted") from exc
                raise
        runtime["last_gpu_memory"] = memory_snapshot(runtime["device"])
        diagnostics["gpu_memory"] = runtime["last_gpu_memory"]
    out = []
    for k, (b, c, conf) in enumerate(zip(boxes, classes, scores)):
        lbl = det.names[int(c)] if int(c) in det.names else f"fruit_{int(c)}"
        item = {
            "label": lbl,
            "confidence": round(float(conf), 4),
        }
        if cls is not None:
            p = rs[k]
            item["state"] = cls.names[int(p.probs.top1)]
            item["state_confidence"] = round(float(p.probs.top1conf), 4)
        else:
            item["state"] = "unknown"
            item["state_confidence"] = 0.0
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
        "device": runtime["device"],
        "det_model": Path(DET_PATH).name,
        "cls_model": Path(CLS_PATH).name if "cls" in models else None,
        "classifier_check": "cls" in models,
        "classifier_device": runtime["classifier_device"],
        "clip_device": runtime["clip_device"],
        "clip_error": runtime["clip_error"],
        "device_reason": runtime["device_reason"],
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory": memory_snapshot(runtime["device"] or "cpu"),
        "last_inference_gpu_memory": runtime["last_gpu_memory"],
        "cpu_threads": torch.get_num_threads(),
        "det_task": models["det"].task if models else None,
        "fruit_class_names": models["det"].names if models else None,
        "conf": CONF,
        "nms_iou": NMS_IOU,
        "default_mode": "careful",
        "clip_check": "clip" in models,
        "inference_provider": INFERENCE_PROVIDER,
        "model_imgsz": IMGSZ,
    }


async def read_image(file, with_transform=False):
    """Validate/decode and preprocess an upload; optionally return its geometry transform."""
    if file.content_type not in ALLOWED:
        raise HTTPException(400, "Chỉ nhận ảnh jpg, png hoặc webp")
    # đọc tối đa MAX_SIZE + 1 byte, không nạp cả file quá lớn vào bộ nhớ
    data = await file.read(MAX_SIZE + 1)
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "Ảnh vượt quá 10MB")
    if not data:
        raise HTTPException(400, "File ảnh rỗng")

    img, transform = await run_in_threadpool(decode_and_preprocess, data)
    if with_transform:
        return data, img, transform
    return data, img


def decode_and_preprocess(data: bytes):
    """Decode, validate dimensions, and preprocess without blocking ASGI's event loop."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
    except Exception as exc:
        raise HTTPException(400, "Không đọc được file ảnh") from exc
    if width * height > MAX_PIXELS:
        raise HTTPException(413, f"Ảnh quá lớn ({width}x{height}), tối đa khoảng 40 triệu điểm ảnh")
    original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if original is None:
        raise HTTPException(400, "Không đọc được file ảnh")
    return preprocess_image(original)


def preprocess_image(img: np.ndarray):
    """Preprocess an image and retain exact original/processed geometry."""
    processed = enhance_image(img)
    original_h, original_w = img.shape[:2]
    processed_h, processed_w = processed.shape[:2]
    transform = ImageTransform.from_sizes((original_w, original_h),
                                          (processed_w, processed_h))
    return processed, transform


def restore_objects_to_original(objects, transform):
    """Map output polygons and mask metrics from processed to uploaded-image pixels."""
    polygons = []
    boxes = []
    for obj in objects:
        polygon = obj.get("polygon")
        if not polygon or len(polygon) < 3:
            polygons.append(np.zeros((0, 2), dtype=np.float32))
            boxes.append(np.zeros(4, dtype=np.float32))
            continue
        mapped = transform.map_points_to_original(polygon)
        obj["polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in mapped]
        polygons.append(mapped)
        boxes.append(np.asarray([mapped[:, 0].min(), mapped[:, 1].min(),
                                 mapped[:, 0].max(), mapped[:, 1].max()], dtype=np.float32))
    metrics = compute_mask_metrics(polygons, boxes, transform.original_width, transform.original_height)
    for obj, item in zip(objects, metrics):
        obj["mask_area"] = item["mask_area"]
        obj["solidity"] = item["solidity"]
        obj["occlusion_ratio"] = item["occlusion_ratio"]
        obj["is_occluded"] = item["is_occluded"]
    return objects


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
    request_started = time.perf_counter()
    _, img, transform = await read_image(file, with_transform=True)
    preprocessing_ms = (time.perf_counter() - request_started) * 1000
    t = time.perf_counter()
    objects, diagnostics = await run_in_threadpool(predict, img, mode)
    ms = (time.perf_counter() - t) * 1000

    objects = await run_in_threadpool(restore_objects_to_original, objects, transform)
    w, h = transform.original_width, transform.original_height
    quality = await run_in_threadpool(assess_detection, objects, w, h)
    total_ms = (time.perf_counter() - request_started) * 1000
    return {
        "image_width": w,
        "image_height": h,
        "preprocessing": transform.to_dict(),
        "inference_ms": round(ms, 1),
        "preprocessing_ms": round(preprocessing_ms, 1),
        "total_ms": round(total_ms, 1),
        "mode": mode,
        "objects": objects,
        "runtime": {
            "device": runtime["device"],
            "device_reason": runtime["device_reason"],
            "classifier_enabled": "cls" in models,
            "classifier_device": runtime["classifier_device"],
            "clip_enabled": "clip" in models,
            "gpu_memory": diagnostics.get("gpu_memory"),
        },
        "quality": quality | {
            "instance_count": diagnostics["instance_count"],
            "count": count_instances(objects),
            "instance_signals": assess_instances(objects, w, h, min_confidence=CONF),
            "provider": diagnostics.get("provider", INFERENCE_PROVIDER),
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
    valid_states = {"fresh", "rotten", "unknown"}
    for obj in payload["objects"]:
        if not isinstance(obj, dict):
            raise HTTPException(400, "Mỗi object phản hồi phải là một object JSON")
        if obj.get("label") not in valid_labels or obj.get("state") not in valid_states:
            raise HTTPException(400, "Nhãn quả hoặc trạng thái phản hồi không hợp lệ")
        polygon = obj.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise HTTPException(400, "Mask phản hồi không hợp lệ")

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
    if "cls" not in models:
        raise HTTPException(503, "Thu thập nhãn cần bật CLASSIFIER_CHECK")
    data, img = await read_image(file)
    return await run_in_threadpool(collect.collect, data, img, session, models, lock, crop_box, CONF, IMGSZ, CLS_IMGSZ)


@app.get("/api/collect/stats")
def collect_stats():
    return collect.stats()
