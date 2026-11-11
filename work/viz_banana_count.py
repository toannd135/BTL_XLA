"""Vẽ nhãn (xanh lá) và dự đoán chuối (đỏ) cho các ảnh sai nhiều nhất; ghép lưới.
python work/viz_banana_count.py <tag> <weights> [imgsz] [rotated 0/1] [k]  -> work/diag/banana/<tag>_grid*.jpg"""
import csv, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(ROOT / "work"))
from detection import detect
from eval_banana_count import load_gt
tag, w = sys.argv[1], sys.argv[2]; imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 640
rot = len(sys.argv) > 4 and sys.argv[4] == "1"; k = int(sys.argv[5]) if len(sys.argv) > 5 else 16
rows = sorted(csv.DictReader(open(ROOT / f"work/audit/banana_count_{tag}.csv")), key=lambda r: -abs(int(r["err"])))[:k]
m = YOLO(w); out = ROOT / "work/diag/banana"; out.mkdir(parents=True, exist_ok=True); tiles = []
for r in rows:
    p = ROOT / "local_dataset/det/test/images" / r["image"]; img = cv2.imread(str(p)); h, ww = img.shape[:2]
    for b in load_gt(p.parent.parent / "labels" / (p.stem + ".txt"), ww, h).astype(int):
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), max(2, ww // 200))
    b, s, c = detect(m, img, conf=0.4, imgsz=imgsz, iou=0.6, quantize=16, rotated=rot)
    for bb, ss, cc in zip(np.asarray(b).astype(int), s, c):
        col = (0, 0, 255) if cc == 0 else (255, 0, 0)
        cv2.rectangle(img, tuple(bb[:2]), tuple(bb[2:]), col, max(1, ww // 300))
    t = cv2.resize(img, (400, int(400 * h / ww))); t = cv2.copyMakeBorder(t, 0, max(0, 400 - t.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:400]
    cv2.putText(t, f"{r['image'][:28]} gt{r['gt']} pr{r['pred']}", (4, 18), 0, 0.5, (0, 255, 255), 2); tiles.append(t)
for g in range(0, len(tiles), 8):
    ch = tiles[g:g + 8]; ch += [np.zeros_like(tiles[0])] * (8 - len(ch))
    cv2.imwrite(str(out / f"{tag}_grid{g // 8}.jpg"), np.vstack([np.hstack(ch[:4]), np.hstack(ch[4:])]))
print(out)
