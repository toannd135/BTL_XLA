import io
import os

# Bound CPU work before torch/BLAS initializes; the user can override this explicitly.
CPU_THREADS = max(1, min(8, int(os.getenv("CPU_THREADS", "2"))))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
import threading
import time
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
from detection import crop_rotated, detect, detect_obb, refine_types

# Pillow tự cảnh báo/chặn ảnh lớn; giới hạn đã kiểm tra riêng bằng MAX_PIXELS
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent.parent
DET_PATH = os.getenv("MODEL_PATH", str(ROOT / "exports/det_obb_s.pt"))   # OBB Colab 60 epoch (2026-11-12); bản cũ: exports/det_C_candidate.pt
CLS_PATH = os.getenv("CLS_MODEL_PATH", str(ROOT / "exports/cls_finetune_best.pt"))

# giống notebook: CONF_PRED, IMGSZ, CLS_IMGSZ, CROP_PAD
CONF = float(os.getenv("CONF", "0.4"))   # chốt 0.40 cho model C (F1 tổng 0.82, nền sai 1/71); hạ xuống 0.25-0.35 nếu hay bỏ sót chuối
IMGSZ = 640
QUALITY_IMGSZ = 640   # 960 đếm chuối thừa ~2.4x, nền sai 3/71 (docs/handoff/dem-chuoi.md)
NMS_IOU = float(os.getenv("NMS_IOU", "0.6"))   # NMS chặt hơn mặc định 0.7: bớt box trùng trên cùng một quả (xem work/audit)
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
    if CLIP_CHECK:
        try:
            from clip_check import ClipChecker
            models["clip"] = ClipChecker()
        except Exception as e:   # thiếu open_clip/model: chạy tiếp không có bước kiểm tra
            print("CLIP_CHECK tắt:", e)
    # chạy thử 1 lượt để request đầu tiên không bị chậm
    blank = np.zeros((IMGSZ, IMGSZ, 3), np.uint8)
    models["det"].predict(blank, imgsz=IMGSZ, quantize=QUANT, verbose=False)
    models["cls"].predict(blank, imgsz=CLS_IMGSZ, quantize=QUANT, verbose=False)
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
        conf = min(CONF, RESCUE_CONF) if "clip" in models else CONF
        if det.task == "obb":   # khung xoay: một lượt, không cần xoay ảnh (model đã học góc bất kỳ)
            polys, boxes, scores, classes = detect_obb(det, img, conf=conf, imgsz=IMGSZ, iou=NMS_IOU, quantize=QUANT)
        else:
            boxes, scores, classes = detect(det, img, conf=conf,
                imgsz=QUALITY_IMGSZ if mode == "careful" else IMGSZ,
                iou=NMS_IOU, quantize=QUANT, rotated=mode == "careful")
        if len(boxes) and "clip" in models:
            clip = models["clip"]
            keep, classes = refine_types(det.names, classes, clip.names,
                                         clip.probs_boxes(img, boxes, CROP_PAD, t_other=CLIP_T_OTHER), CLIP_T_OTHER, CLIP_T_SWAP,
                                         scores=scores, conf=CONF, t_rescue=CLIP_T_RESCUE)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
            if polys is not None:
                polys = polys[keep]
        if not len(boxes):
            return []
        crops = [crop_rotated(img, q, CROP_PAD) for q in polys] if polys is not None else [crop_box(img, b) for b in boxes]
        # Explicit chunks: predict(list) does not honor batch in every Ultralytics version.
        rs = []
        for start in range(0, len(crops), 8):
            rs.extend(cls.predict(crops[start:start + 8], imgsz=CLS_IMGSZ, quantize=QUANT, verbose=False))
    out = []
    for k, (b, c, conf, p) in enumerate(zip(boxes, classes, scores, rs)):
        item = {
            "label": det.names[int(c)],
            "confidence": round(float(conf), 4),
            "state": cls.names[int(p.probs.top1)],
            "state_confidence": round(float(p.probs.top1conf), 4),
            "bbox": [round(float(v), 1) for v in b],
            "rescued": bool(float(conf) < CONF),   # điểm detector < CONF nhưng CLIP xác nhận là quả
        }
        if polys is not None:
            item["polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in polys[k]]
        out.append(item)
    return out


@app.get("/api/health")
def health():
    return {
        "status": "ok" if models else "loading",
        # predictor mới giữ thiết bị thật (model.device vẫn báo cpu)
        "device": str(models["det"].predictor.device) if models else None,
        "det_model": Path(DET_PATH).name,
        "cls_model": Path(CLS_PATH).name,
        "cpu_threads": torch.get_num_threads(),
        "conf": CONF,
        "default_mode": "careful",
        "careful_imgsz": QUALITY_IMGSZ,
        "clip_check": "clip" in models,
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
    return data, img


@app.post("/api/predict")
async def predict_api(file: UploadFile = File(...), mode: Literal["fast", "careful"] = Form("careful")):
    _, img = await read_image(file)
    t = time.perf_counter()
    objects = await run_in_threadpool(predict, img, mode)
    ms = (time.perf_counter() - t) * 1000

    h, w = img.shape[:2]
    return {
        "image_width": w,
        "image_height": h,
        "inference_ms": round(ms, 1),
        "mode": mode,
        "objects": objects,
    }


@app.post("/api/collect")
async def collect_api(file: UploadFile = File(...), session: str = Form(...)):
    if not collect.SESSION_RE.match(session):
        raise HTTPException(400, "Mã phiên thu thập không hợp lệ")
    data, img = await read_image(file)
    return await run_in_threadpool(collect.collect, data, img, session, models, lock, crop_box, CONF, IMGSZ, CLS_IMGSZ)


@app.get("/api/collect/stats")
def collect_stats():
    return collect.stats()
