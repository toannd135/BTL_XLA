"""Dựng tập crop cho bộ phân loại loại quả 4 lớp (banana/apple/orange/other) -> work/fruitcls/data/{train,val,test}/<lớp>/*.jpg
- banana/apple/orange: box nhãn gốc local_dataset/det/{train,val,test} (giữ đúng split), nới 10% như app (CROP_PAD).
- other: (1) work/exp/neg/{train,test} (ảnh nền, test = 71 ảnh giữ riêng); (2) own_other/raw (quả khác/vật tròn, Wikimedia),
  bỏ ảnh mà YOLO11l thấy chuối/táo/cam (conf>=0.3), chia train/val/test 80/10/10 theo hash tên file.
  Mỗi ảnh "other": box mà các detector (C, OBB, D1) báo ở conf>=0.1 (ca khó) + cả ảnh; tối đa 8 crop/ảnh.
Dùng: python work/fruitcls/build_crops.py"""
import csv, hashlib, shutil, sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from detection import detect, detect_obb  # noqa: E402

OUT = ROOT / "work/fruitcls/data"
NAMES = ["banana", "apple", "orange"]
PAD, MIN_SIDE, MAX_PER_IMG = 0.10, 24, 8
if OUT.exists():
    shutil.rmtree(OUT)
for s in ("train", "val", "test"):
    for c in NAMES + ["other"]:
        (OUT / s / c).mkdir(parents=True, exist_ok=True)


def crop(img, b, pad=PAD):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = b
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1, x2, y2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))
    if min(x2 - x1, y2 - y1) < MIN_SIDE:
        return None
    return img[y1:y2, x1:x2]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


# 1. Quả thật từ nhãn gốc
n_fruit = {}
for split in ("train", "val", "test"):
    d = ROOT / "local_dataset/det" / split
    for lp in sorted((d / "labels").glob("*.txt")):
        ip = next((p for p in (d / "images").glob(lp.stem + ".*")), None)
        if ip is None:
            continue
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        for k, ln in enumerate(lp.read_text().split("\n")):
            p = ln.split()
            if len(p) != 5:
                continue
            c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
            cr = crop(img, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
            if cr is not None:
                cv2.imwrite(str(OUT / split / NAMES[c] / f"{lp.stem}_{k}.jpg"), cr)
                n_fruit[(split, NAMES[c])] = n_fruit.get((split, NAMES[c]), 0) + 1
print("quả:", n_fruit)

# 2. Other: ca khó do detector báo + cả ảnh
dets = [YOLO(str(ROOT / w)) for w in ("exports/det_C_candidate.pt", "exports/det_obb_s.pt", "runs/exp/det_D_synth/weights/best.pt")]
teacher = YOLO(str(ROOT / "yolo11l.pt"))
COCO_FRUIT = {46, 47, 49}


def proposals(img):
    boxes = []
    for m in dets:
        b = detect_obb(m, img, conf=0.1, quantize=16)[1] if m.task == "obb" else detect(m, img, conf=0.1, quantize=16)[0]
        for bb in b:
            if all(iou(bb, o) < 0.7 for o in boxes):
                boxes.append(bb)
    return boxes[:MAX_PER_IMG - 1]


def split_of(name):
    v = int(hashlib.md5(name.encode()).hexdigest(), 16) % 10
    return "test" if v == 0 else "val" if v == 1 else "train"


srcs = [(p, "test" if "/test/" in str(p) else ("val" if split_of(p.name) == "val" else "train"))
        for p in sorted((ROOT / "work/exp/neg").glob("*/images/*")) if "/cam/" not in str(p)]
dropped, n_other = [], {}
for p in sorted((ROOT / "own_other/raw").glob("*.jpg")):
    img = cv2.imread(str(p))
    if img is None:
        continue
    r = teacher.predict(img, conf=0.3, verbose=False)[0]
    if any(int(c) in COCO_FRUIT for c in r.boxes.cls.tolist()):
        dropped.append(p.name); continue
    srcs.append((p, split_of(p.name)))
for p, split in srcs:
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    crops = [img] + [c for c in (crop(img, b) for b in proposals(img)) if c is not None]
    for k, cr in enumerate(crops):
        cv2.imwrite(str(OUT / split / "other" / f"{p.stem[:60]}_{k}.jpg"), cr)
    n_other[split] = n_other.get(split, 0) + len(crops)
print("other:", n_other, "| ảnh own_other bỏ vì YOLO11l thấy chuối/táo/cam:", len(dropped))
(OUT / "dropped_own_other.txt").write_text("\n".join(dropped) + "\n")
