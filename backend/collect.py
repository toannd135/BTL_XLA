"""Tự thu thập ảnh train từ camera: lưu ảnh gốc + nhãn YOLO 6 class vào own_cam/ (notebook ô 1.1c chép sang own/).

Không ai sửa nhãn bằng tay nên chỉ giữ khung mà nhãn tự sinh đáng tin:
- box: detector đang chạy và YOLO11l COCO (huấn luyện sẵn, độc lập với dữ liệu của mình) phải khớp nhau;
  quả chỉ một bên thấy thì chỉ nhận khi bên đó rất chắc, còn lại bỏ cả khung
- tình trạng: classifier phải chắc (>= STATE_CONF)
- bỏ khung mờ và khung gần giống khung vừa lưu
"""
import os
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.getenv("COLLECT_DIR", str(ROOT / "own_cam")))
TEACHER_PATH = ROOT / "yolo11l.pt"

FRUITS = ["banana", "apple", "orange"]          # đúng thứ tự class của detector
STATES = ["fresh", "rotten"]
CLASS_NAMES = [f"{f}_{s}" for f in FRUITS for s in STATES]   # giống notebook
COCO_ID = {46: "banana", 47: "apple", 49: "orange"}

IOU_MATCH = 0.5
DET_SURE = 0.7       # box chỉ detector thấy: nhận khi conf >= ngưỡng này
TEACHER_SURE = 0.6   # box chỉ YOLO11l thấy (detector bỏ sót): nhận khi conf >= ngưỡng này
TEACHER_MIN = 0.3    # box YOLO11l dưới ngưỡng này coi như không có
STATE_CONF = 0.8
BLUR_THRESHOLD = 100.0   # giống BLUR_THRESHOLD của notebook
DUP_HASH_DIST = 6        # giống DUP_HASH_DIST của notebook
MIN_BOX_FRAC = 0.002     # bỏ khung có box quá nhỏ (nhãn kém tin cậy)

SESSION_RE = re.compile(r"^[0-9A-Za-z-]{1,32}$")
REASONS = {
    "blurry": "ảnh mờ",
    "duplicate": "giống khung vừa lưu",
    "empty": "không thấy quả",
    "disagree": "hai model không khớp nhau",
    "unsure_state": "chưa chắc tươi hay hỏng",
    "tiny": "quả quá nhỏ",
    "group": "có khung bao cả chùm quả",
}

_teacher = None
_recent = {}   # session -> dHash các khung đã lưu gần đây
_io = threading.Lock()


def teacher():
    global _teacher
    if _teacher is None:
        _teacher = YOLO(str(TEACHER_PATH))
    return _teacher


def dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def covers(big, small):
    """Phần diện tích của `small` nằm trong `big`."""
    x1, y1, x2, y2 = max(big[0], small[0]), max(big[1], small[1]), min(big[2], small[2]), min(big[3], small[3])
    area = (small[2] - small[0]) * (small[3] - small[1])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / area if area > 0 else 0.0


def blur_score(img):
    h, w = img.shape[:2]
    s = 640 / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img
    return cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def agree(dets, teach):
    """Ghép box detector với box YOLO11l. Trả về [(fruit, xyxy)] hoặc None nếu không đủ tin cậy."""
    out, used = [], set()
    for fruit, conf, box in dets:
        j = max(((k, iou(box, t[2])) for k, t in enumerate(teach) if k not in used),
                key=lambda x: x[1], default=(None, 0))
        if j[0] is not None and j[1] >= IOU_MATCH:
            used.add(j[0])
            if teach[j[0]][0] != fruit:
                return None
            out.append((fruit, box))
        elif conf >= DET_SURE:
            out.append((fruit, box))
        else:
            return None
    for k, (fruit, conf, box) in enumerate(teach):
        if k in used:
            continue
        if conf < TEACHER_SURE:
            return None
        out.append((fruit, box))
    return out


def label_frame(img, models, lock, crop_box, conf, imgsz, cls_imgsz):
    """Trả về (danh sách (class_id, xyxy), None) hoặc (None, lý do)."""
    det, cls = models["det"], models["cls"]
    with lock:
        r = det.predict(img, conf=conf, imgsz=imgsz, agnostic_nms=True, verbose=False)[0]
        t = teacher().predict(img, conf=TEACHER_MIN, imgsz=imgsz, classes=list(COCO_ID),
                              agnostic_nms=True, verbose=False)[0]
    dets = [(det.names[int(c)], float(p), b) for b, c, p in
            zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
    teach = [(COCO_ID[int(c)], float(p), b) for b, c, p in
             zip(t.boxes.xyxy.tolist(), t.boxes.cls.tolist(), t.boxes.conf.tolist())]
    if not dets and not teach:
        return None, "empty"
    boxes = agree(dets, teach)
    if boxes is None:
        return None, "disagree"
    # box bao trọn >= 2 box khác (cả nải chuối, cả đĩa cam): dữ liệu public đánh từng quả nên bỏ khung
    if any(sum(covers(a, b) >= 0.8 for _, b in boxes if b is not a) >= 2 for _, a in boxes):
        return None, "group"
    h, w = img.shape[:2]
    if any((b[2] - b[0]) * (b[3] - b[1]) < MIN_BOX_FRAC * w * h for _, b in boxes):
        return None, "tiny"
    with lock:
        rs = cls.predict([crop_box(img, b) for _, b in boxes], imgsz=cls_imgsz, verbose=False)
    out = []
    for (fruit, b), p in zip(boxes, rs):
        if float(p.probs.top1conf) < STATE_CONF:
            return None, "unsure_state"
        state = cls.names[int(p.probs.top1)]
        out.append((CLASS_NAMES.index(f"{fruit}_{state}"), b))
    return out, None


def save(data, img, session, labels):
    h, w = img.shape[:2]
    lines = []
    for c, (x1, y1, x2, y2) in labels:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    with _io:
        (OUT / "images").mkdir(parents=True, exist_ok=True)
        (OUT / "labels").mkdir(parents=True, exist_ok=True)
        if not (OUT / "data.yaml").exists():
            (OUT / "data.yaml").write_text(yaml.safe_dump(
                {"path": str(OUT), "names": dict(enumerate(CLASS_NAMES))}, allow_unicode=True, sort_keys=False))
        # số cuối tên tệp tăng dần theo thời gian: notebook gom theo "cam_<phiên>" và chia tập theo khối khung liên tiếp
        stem = f"cam_{session}_{time.time_ns() // 1_000_000}"
        # giữ nguyên bytes JPEG gốc (không nén lại); png/webp thì đổi sang jpg
        jpg = data if data[:2] == b"\xff\xd8" else cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        (OUT / "images" / f"{stem}.jpg").write_bytes(jpg)
        (OUT / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    return stem


def collect(data, img, session, models, lock, crop_box, conf, imgsz, cls_imgsz):
    if blur_score(img) < BLUR_THRESHOLD:
        return {"saved": False, "reason": REASONS["blurry"]}
    hsh = dhash(img)
    recent = _recent.setdefault(session, [])
    if any(bin(hsh ^ x).count("1") <= DUP_HASH_DIST for x in recent):
        return {"saved": False, "reason": REASONS["duplicate"]}
    labels, why = label_frame(img, models, lock, crop_box, conf, imgsz, cls_imgsz)
    if labels is None:
        return {"saved": False, "reason": REASONS[why]}
    stem = save(data, img, session, labels)
    recent.append(hsh)
    del recent[:-50]
    return {"saved": True, "name": stem, "labels": [CLASS_NAMES[c] for c, _ in labels]}


def stats():
    counts = dict.fromkeys(CLASS_NAMES, 0)
    images = 0
    for p in (OUT / "labels").glob("*.txt") if (OUT / "labels").exists() else []:
        images += 1
        for line in p.read_text().split("\n"):
            if line.strip():
                counts[CLASS_NAMES[int(line.split()[0])]] += 1
    return {"images": images, "boxes": counts, "folder": str(OUT)}
