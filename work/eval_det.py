"""Đánh giá detector theo bộ chỉ số cố định để so sánh các thí nghiệm.
Cách dùng: python work/eval_det.py <weights.pt> <tag> [split=test]
Ra: work/audit/eval_<tag>.json, work/audit/fp_<tag>.csv (mỗi box thừa kèm error_type)."""
import csv, json, statistics, sys, time
from pathlib import Path
import numpy as np, torch, ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
W, TAG = sys.argv[1], sys.argv[2]
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "test"
DATA, N, CONF, IMGSZ = ROOT / "work" / "exp" / "det_joint_s.yaml", ["banana", "apple", "orange"], 0.25, 640
OUT = ROOT / "work" / "audit"; OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = ROOT / "local_dataset" / "det" / SPLIT
m = YOLO(W)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
def inside(s, b):
    cx, cy = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
def area(b): return (b[2] - b[0]) * (b[3] - b[1])

# 1) mAP (conf thấp) và P/R + ma trận nhầm lẫn (conf 0.25)
r1 = m.val(data=str(DATA), split=SPLIT, imgsz=IMGSZ, batch=8, verbose=False, plots=False, project=str(OUT / "val"), name=TAG, exist_ok=True)
res = {"tag": TAG, "weights": W, "split": SPLIT, "ultralytics": ultralytics.__version__, "imgsz": IMGSZ, "conf_pr": CONF,
       "mAP50": round(float(r1.box.map50), 4), "mAP50-95": round(float(r1.box.map), 4),
       "AP50": {n: round(float(x), 4) for n, x in zip(N, r1.box.ap50)}}
r2 = m.val(data=str(DATA), split=SPLIT, imgsz=IMGSZ, batch=8, conf=CONF, verbose=False, plots=True, project=str(OUT / "val"), name=TAG + "_c25", exist_ok=True)
cm = np.array(r2.confusion_matrix.matrix).astype(int)       # hàng = dự đoán, cột = thực tế, hàng/cột cuối = nền
res["confusion_c25"] = cm.tolist(); res["P_all"], res["R_all"] = round(float(r2.box.mp), 4), round(float(r2.box.mr), 4)
res["per_class_PR_c25"] = {n: {"P": round(cm[k, k] / max(cm[k].sum(), 1), 4), "R": round(cm[k, k] / max(cm[:, k].sum(), 1), 4),
                               "TP": int(cm[k, k]), "FP": int(cm[k].sum() - cm[k, k]), "FN": int(cm[3, k])} for k, n in enumerate(N)}

# 2) phân loại từng box thừa
rows, imgs = [], sorted((IMG_DIR / "images").glob("*"))
for i in range(0, len(imgs), 16):
    for p, r in zip(imgs[i:i + 16], m.predict([str(x) for x in imgs[i:i + 16]], conf=CONF, imgsz=IMGSZ, verbose=False)):
        h, w = r.orig_shape; gt = []
        lf = IMG_DIR / "labels" / f"{p.stem}.txt"
        if lf.exists():
            for l in lf.read_text().splitlines():
                if l.strip():
                    c, x, y, bw, bh = [float(v) for v in l.split()[:5]]; gt.append((int(c), [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
        used = set(); preds = sorted(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()), key=lambda t: -t[1])
        for c, s, b in preds:
            c = int(c); best = max(((k, iou(g[1], b)) for k, g in enumerate(gt) if g[0] == c and k not in used), key=lambda t: t[1], default=(None, 0))
            if best[1] >= 0.5:
                used.add(best[0]); continue                                   # TP
            oth = max(((k, iou(g[1], b)) for k, g in enumerate(gt) if g[0] != c), key=lambda t: t[1], default=(None, 0))
            same = max(((k, iou(g[1], b)) for k, g in enumerate(gt) if g[0] == c), key=lambda t: t[1], default=(None, 0))
            if oth[1] >= 0.5: et, mg, v = "class_fp", gt[oth[0]][0], oth[1]
            elif same[0] is not None and same[1] >= 0.5: et, mg, v = "duplicate", c, same[1]
            elif any(g[0] == c and inside(b, g[1]) and area(b) <= 0.6 * area(g[1]) for g in gt): et, mg, v = "split_of_larger_gt", c, same[1]
            elif same[1] >= 0.1: et, mg, v = "localization_fp", c, same[1]
            else: et, mg, v = "background_fp" , "", max(same[1], oth[1])
            rows.append({"image_path": str(p), "predicted_class": N[c], "confidence": round(s, 3), "bbox_x1": round(b[0]), "bbox_y1": round(b[1]),
                         "bbox_x2": round(b[2]), "bbox_y2": round(b[3]), "matched_ground_truth": N[mg] if mg != "" else "", "iou": round(v, 3), "error_type": et})
with open(OUT / f"fp_{TAG}.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)
res["fp_types"] = {n: {} for n in N}
for r in rows: res["fp_types"][r["predicted_class"]][r["error_type"]] = res["fp_types"][r["predicted_class"]].get(r["error_type"], 0) + 1

# 3) latency FP16 (batch 1, đã gồm tiền/hậu xử lý) và VRAM đỉnh
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
sample = [str(x) for x in imgs[:60]]; m.predict(sample[0], imgsz=IMGSZ, quantize=16, verbose=False); ts = []
for s in sample:
    t = time.perf_counter(); m.predict(s, imgsz=IMGSZ, quantize=16, verbose=False); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
res["latency_ms_fp16_median"] = round(statistics.median(ts), 1); res["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
json.dump(res, open(OUT / f"eval_{TAG}.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k not in ("confusion_c25",)}, ensure_ascii=False, indent=1))
