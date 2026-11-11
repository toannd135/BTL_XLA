"""Gán nhãn nháp cho own_mixed/raw bằng YOLO11l (COCO: banana 46, apple 47, orange 49), conf>=0.35. Giữ ảnh có >=2 loại quả.
Ra: own_mixed/images, own_mixed/labels (nhãn 3 class: 0 banana, 1 apple, 2 orange), own_mixed/preview (ảnh vẽ box để người duyệt)."""
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "own_mixed"
for k in ("images", "labels", "preview"):
    shutil.rmtree(D / k, ignore_errors=True); (D / k).mkdir()
T = YOLO(str(ROOT / "yolo11l.pt")); M = {46: 0, 47: 1, 49: 2}; NM = ["banana", "apple", "orange"]
keep = 0
for p in sorted((D / "raw").glob("*.jpg")):
    if cv2.imread(str(p)) is None:
        print("bỏ ảnh hỏng", p.name); continue
    r = T.predict(str(p), conf=0.35, imgsz=1280, classes=list(M), verbose=False)[0]
    cls = [M[int(c)] for c in r.boxes.cls]
    if len(set(cls)) < 2:
        continue
    shutil.copy(p, D / "images" / p.name)
    (D / "labels" / f"{p.stem}.txt").write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, (x, y, w, h) in zip(cls, r.boxes.xywhn.tolist())))
    im = r.orig_img.copy()
    for c, b in zip(cls, r.boxes.xyxy.tolist()):
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), [(0, 220, 255), (0, 0, 255), (0, 140, 255)][c], 3)
        cv2.putText(im, NM[c], (int(b[0]), int(b[1]) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(D / "preview" / p.name), cv2.resize(im, None, fx=640 / max(im.shape[:2]), fy=640 / max(im.shape[:2])))
    keep += 1
print("giữ", keep, "/", len(list((D / "raw").glob("*.jpg"))))
