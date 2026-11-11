"""Chẩn đoán một ảnh: so inference trực tiếp với cấu hình backend, lưu crop đưa vào classifier và ảnh vẽ box.
Dùng: python work/diag_image.py <ảnh> [weights=exports/det_finetune_best.pt]
Kết quả: work/diag/<tên ảnh>/{raw_*.png, crops/, report.txt}. Không ghi đè exports/."""
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
img_path = Path(sys.argv[1])
w = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "exports" / "det_finetune_best.pt")
img = cv2.imread(str(img_path))
out = ROOT / "work" / "diag" / img_path.stem
(out / "crops").mkdir(parents=True, exist_ok=True)
det, cls = YOLO(w), YOLO(str(ROOT / "exports" / "cls_finetune_best.pt"))
print("names:", det.names, "| ảnh:", img.shape[1], "x", img.shape[0])


def run(tag, **kw):
    r = det.predict(img, imgsz=640, verbose=False, **kw)[0]
    rows = [(det.names[int(c)], float(s), [round(v) for v in b.tolist()]) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    cv2.imwrite(str(out / f"raw_{tag}.jpg"), r.plot())
    by = {}
    for n, s, _ in rows:
        by[n] = by.get(n, 0) + 1
    print(f"[{tag}] {len(rows)} box {by}")
    return r, rows


# 1) cấu hình backend (conf 0.4, NMS 0.6, agnostic) và các biến thể để tách nguyên nhân
_, backend = run("backend_conf0.40_agnostic", conf=0.4, iou=0.6, agnostic_nms=True, quantize=16)
run("conf0.40_perclass_nms", conf=0.4, iou=0.6, agnostic_nms=False)
run("conf0.25_agnostic", conf=0.25, iou=0.6, agnostic_nms=True)
run("conf0.25_perclass_nms", conf=0.25, iou=0.6, agnostic_nms=False)
r, rows = run("conf0.10_perclass_nms_maxdet300", conf=0.10, iou=0.6, agnostic_nms=False, max_det=300)

# 2) crop mà backend thực sự đưa vào classifier (đủ bước pad 10% như main.crop_box)
from main import crop_box  # noqa: E402
lines = []
r0 = det.predict(img, conf=0.4, imgsz=640, iou=0.6, agnostic_nms=True, verbose=False)[0]
for k, (b, c, s) in enumerate(zip(r0.boxes.xyxy.tolist(), r0.boxes.cls.tolist(), r0.boxes.conf.tolist())):
    cr = crop_box(img, b)
    cv2.imwrite(str(out / "crops" / f"{k}_{det.names[int(c)]}_{s:.2f}.png"), cr)
    p = cls.predict(cr, imgsz=224, verbose=False)[0]
    area = (b[2] - b[0]) * (b[3] - b[1]) / (img.shape[0] * img.shape[1])
    lines.append(f"{k}: {det.names[int(c)]} det={s:.2f} | {cls.names[int(p.probs.top1)]} cls={float(p.probs.top1conf):.2f} | "
                 f"box={[round(v) for v in b]} chiếm {area:.1%} ảnh")
(out / "report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("→", out)
