"""Chạy detection + phân loại tươi/hỏng trên ảnh hoặc thư mục ảnh.
Dùng: python run_detection.py <ảnh|thư mục> [--conf 0.4]
Kết quả: ảnh vẽ box trong thư mục out/, in danh sách quả ra màn hình."""
import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--iou", type=float, default=0.6)
a = ap.parse_args()

det = YOLO(str(ROOT / "exports/det_finetune_best.pt"))
cls = YOLO(str(ROOT / "exports/cls_finetune_best.pt"))
src = Path(a.src)
files = sorted(p for p in ([src] if src.is_file() else src.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
if not files:
    sys.exit(f"Không thấy ảnh trong {src}")
out = ROOT / "out"
out.mkdir(exist_ok=True)


def crop(img, b, pad=0.10):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = b
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, min(w - 1, x1 - px))), int(max(0, min(h - 1, y1 - py)))
    x2, y2 = int(min(w, max(x1 + 1, x2 + px))), int(min(h, max(y1 + 1, y2 + py)))
    return img[y1:y2, x1:x2]


for f in files:
    img = cv2.imread(str(f))
    if img is None:
        print(f"{f.name}: không đọc được"); continue
    r = det.predict(img, conf=a.conf, iou=a.iou, imgsz=640, agnostic_nms=True, verbose=False)[0]
    boxes = r.boxes.xyxy.tolist()
    print(f"{f.name}: {len(boxes)} quả")
    if boxes:
        rs = cls.predict([crop(img, b) for b in boxes], imgsz=224, verbose=False)
        for b, c, s, p in zip(boxes, r.boxes.cls.tolist(), r.boxes.conf.tolist(), rs):
            x1, y1, x2, y2 = map(int, b)
            lab = f"{det.names[int(c)]} {cls.names[int(p.probs.top1)]} {s:.2f}"
            print("  ", lab)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, lab, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    cv2.imwrite(str(out / f.name), img)
print(f"Đã lưu ảnh kết quả vào {out}/")
