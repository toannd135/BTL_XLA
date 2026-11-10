"""Train detector thêm ảnh NỀN/vật khác (không có quả) để giảm nhận nhầm. Không ghi đè exports/.
Cách dùng: python train_detection.py [model=yolo11s.pt] [name=det_neg_s] [epochs=60] [batch=16] [neg_rep=3]
Đặt ảnh (jpg/png) hoặc video (mp4/mov/avi/mkv) chứa vật KHÔNG phải chuối/táo/cam vào own_neg/ (tạo thư mục con tuỳ ý)."""
import hashlib
import random
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DET = ROOT / "local_dataset" / "det"
NEG_SRC = ROOT / "own_neg"
OUT = ROOT / "work" / "exp" / "neg"
a = sys.argv[1:] + [None] * 5
model, name = a[0] or str(ROOT / "exports" / "det_finetune_best.pt"), a[1] or "det_C_neg"   # mặc định fine-tune từ B
epochs, batch, neg_rep = int(a[2] or 25), int(a[3] or 16), min(int(a[4] or 2), 2)
VIDEO_STEP, BLOCK = 10, 20          # lấy mỗi 10 khung; 20 ảnh liên tiếp cùng một tập (tránh rò rỉ khung gần giống)
IMG, VID = {".jpg", ".jpeg", ".png", ".webp"}, {".mp4", ".mov", ".avi", ".mkv"}


def save(img, name_, bucket):
    d = OUT / bucket
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    s = 1280 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    cv2.imwrite(str(d / "images" / f"{name_}.jpg"), img)
    (d / "labels" / f"{name_}.txt").write_text("")      # nhãn rỗng = ảnh nền


import shutil
shutil.rmtree(OUT, ignore_errors=True)
files = sorted(p for p in NEG_SRC.rglob("*") if p.suffix.lower() in IMG | VID)
if not files:
    print("own_neg/ trống: train không có ảnh nền (chỉ thêm own_cam)")
n = 0
for f in files:
    if f.suffix.lower() in IMG:
        frames = [cv2.imread(str(f))]
    else:
        cap, frames, i = cv2.VideoCapture(str(f)), [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % VIDEO_STEP == 0:
                frames.append(fr)
            i += 1
    for k, fr in enumerate(frames):
        if fr is None:
            continue
        if len(frames) >= BLOCK * 2:
            held = (k // BLOCK) % 5 == 0                             # video: ~20% khối liên tiếp giữ lại
        else:                                                         # ảnh lẻ: ~20% theo hash tên (cố định, lặp lại được)
            held = f.suffix.lower() in IMG and int(hashlib.md5(f.relative_to(NEG_SRC).as_posix().encode()).hexdigest(), 16) % 5 == 0
        save(fr, f"neg-{f.stem}-{k:04d}", "test" if held else "train")
        n += 1
ntr = len(list((OUT / "train" / "images").glob("*"))) if (OUT / "train").exists() else 0
nte = len(list((OUT / "test" / "images").glob("*"))) if (OUT / "test").exists() else 0
print(f"negatives: {ntr} train, {nte} giữ lại để đo (từ {len(files)} tệp)")

quar = set(Path(ROOT / "work" / "audit" / "quarantine_train.txt").read_text().split())   # đúng cấu hình B
pub = [p for p in sorted((DET / "train" / "images").iterdir()) if p.name not in quar]
own = sorted(p for p in (DET / "train_ft" / "images").iterdir() if p.name.startswith("own-"))
neg = sorted((OUT / "train" / "images").iterdir()) if ntr else []
# ảnh thu thập từ webcam (own_cam, nhãn 6 class) -> nhãn detection 3 class (id // 2)
cam = []
for im in sorted((ROOT / "own_cam" / "images").glob("*.jpg")):
    lb = ROOT / "own_cam" / "labels" / f"{im.stem}.txt"
    if not lb.exists():
        continue
    d = OUT / "cam"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(im, d / "images" / im.name)
    (d / "labels" / lb.name).write_text("".join(f"{int(l.split()[0]) // 2} {' '.join(l.split()[1:])}\n" for l in lb.read_text().splitlines() if l.strip()))
    cam.append(d / "images" / im.name)
lst = ROOT / "work" / "exp" / f"{name}_train.txt"
lst.write_text("\n".join(str(p) for p in pub + own * 4 + neg * neg_rep) + "\n")
yaml = ROOT / "work" / "exp" / f"{name}.yaml"
yaml.write_text(f"path: {DET}\ntrain: {lst}\nval: val/images\ntest: test/images\nnc: 3\nnames:\n  0: banana\n  1: apple\n  2: orange\n")
print(f"train: {len(pub)} public + {len(own)} own x4 + (bỏ own_cam) {len(neg)} neg x{neg_rep}")

YOLO(model).train(data=str(yaml), epochs=epochs, imgsz=640, batch=batch, optimizer="SGD", lr0=0.002, warmup_epochs=1, seed=42,
                  deterministic=True, patience=10, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, workers=4,
                  project=str(ROOT / "runs" / "exp"), name=name, exist_ok=True, plots=False)

# --- So sánh model đang dùng và model mới: nhận nhầm trên ảnh nền giữ lại + mAP trên test ---
new = ROOT / "runs" / "exp" / name / "weights" / "best.pt"
held = sorted((OUT / "test" / "images").glob("*")) if nte else []
cam_eval = cam
for tag, w in (("đang dùng", ROOT / "exports" / "det_finetune_best.pt"), ("mới", new)):
    m = YOLO(str(w))
    fp = 0
    for i in range(0, len(held), 16):
        fp += sum(len(r.boxes) > 0 for r in m.predict([str(p) for p in held[i:i + 16]], conf=0.25, imgsz=640, verbose=False))
    mp = m.val(data=str(yaml), split="test", imgsz=640, batch=8, verbose=False, plots=False)
    print(f"[{tag}] ảnh nền bị nhận nhầm: {fp}/{len(held)} | test mAP50={mp.box.map50:.3f} mAP50-95={mp.box.map:.3f}")
