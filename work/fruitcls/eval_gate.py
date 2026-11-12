"""Đánh giá detector + bộ phân loại loại quả (refine_types) so với chỉ detector, cùng conf 0.4 / IoU 0.6 như app.
Tập: test (373 ảnh, đếm theo loại như work/obb/eval_count.py), neg (71 ảnh nền giữ riêng), own_mixed (nhãn nháp YOLO11l, ma trận nhầm).
Dùng: python work/fruitcls/eval_gate.py <det.pt> <typecls.pt> [split=test|val] [t_other,t_swap ...]
  split=val: dùng local_dataset/det/val + ảnh nền val để chọn ngưỡng (không đụng test)."""
import json, sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(ROOT / "work/obb"))
from detection import detect, detect_obb, refine_types  # noqa: E402

N = ["banana", "apple", "orange"]
import os
DET_CONF = float(os.getenv("DET_CONF", "0.4"))
det = YOLO(sys.argv[1])
if sys.argv[2] == "clip":
    from clip_check import ClipChecker
    typ = ClipChecker(); type_names = typ.names
else:
    typ = YOLO(sys.argv[2]); type_names = typ.names
split = sys.argv[3] if len(sys.argv) > 3 else "test"
grid = [tuple(map(float, a.split(","))) for a in sys.argv[4:]] or [(0.6, 0.7)]


def crop(img, b, pad=0.10):
    h, w = img.shape[:2]; x1, y1, x2, y2 = b; px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, min(w - 1, x1 - px))), int(max(0, min(h - 1, y1 - py)))
    return img[y1:int(min(h, max(y1 + 1, y2 + py))), x1:int(min(w, max(x1 + 1, x2 + px)))]


def run(img):
    if det.task == "obb":
        _, b, s, c = detect_obb(det, img, conf=DET_CONF, quantize=16)
    else:
        b, s, c = detect(det, img, conf=DET_CONF, quantize=16)
    crops = [crop(img, bb) for bb in b]
    if isinstance(typ, YOLO):
        probs = []
        for i in range(0, len(crops), 16):
            probs += [r.probs.data.cpu().numpy() for r in typ.predict(crops[i:i + 16], imgsz=224, quantize=16, verbose=False)]
    else:
        probs = list(typ.probs_boxes(img, b)) if os.getenv('TWO_VIEW', '1') == '1' else list(typ.probs(crops))
    return np.asarray(b), np.asarray(c, dtype=np.int64), probs, np.asarray(s, dtype=np.float32)


def gt_yolo(lp, w, h):
    out = []
    for ln in lp.read_text().split("\n"):
        p = ln.split()
        if len(p) == 5:
            c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
            out.append((c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    return out


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def load(d_img, d_lbl):
    items = []
    for ip in sorted(Path(d_img).glob("*")):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        lp = Path(d_lbl) / (ip.stem + ".txt") if d_lbl else None
        gt = gt_yolo(lp, img.shape[1], img.shape[0]) if lp and lp.exists() else []
        items.append((ip.name, gt, run(img)))
    return items


D = ROOT / "local_dataset/det" / split
sets = {"det": load(D / "images", D / "labels")}
if split == "test":
    sets["neg"] = load(ROOT / "work/exp/neg/test/images", None)
    sets["mixed"] = load(ROOT / "own_mixed/images", ROOT / "own_mixed/labels")
    sets["other"] = load(ROOT / "own_other/raw", None)
else:
    sets["neg"] = load(ROOT / "work/exp/neg/train/images", None)   # ảnh nền không thuộc 71 ảnh giữ riêng

res = {"det": sys.argv[1], "typecls": sys.argv[2], "split": split, "rows": []}
for t in [None] + grid:
    row = {"t_other,t_swap": t}
    for name, items in sets.items():
        cnt_g, cnt_p, cm, fp_img = [], [], Counter(), 0
        for _, gt, (b, c, probs, sc) in items:
            if t is not None and len(c):
                keep, c2 = refine_types(det.names, c, type_names, probs, *t[:2], **({'scores': sc, 'conf': 0.4, 't_rescue': t[2]} if len(t) > 2 else {}))
                b, c = b[keep], c2[keep]
            if name in ("neg", "other"):
                fp_img += int(len(c) > 0); continue
            pc = Counter(int(x) for x in c)
            cnt_g.append([sum(1 for g, _ in gt if g == k) for k in range(3)]); cnt_p.append([pc[k] for k in range(3)])
            for gc, gb in gt:
                best = max(((iou(gb, pb), int(k)) for pb, k in zip(b, c)), default=(0, -1))
                cm[(N[gc], N[best[1]] if best[0] >= 0.4 else "miss")] += 1
        if name in ("neg", "other"):
            row[f"{name}_fp_images"] = f"{fp_img}/{len(items)}"; continue
        g, p = np.array(cnt_g), np.array(cnt_p)
        row[name] = {"mae": {N[k]: round(float(np.abs(p[g[:, k] > 0, k] - g[g[:, k] > 0, k]).mean()), 3) for k in range(3)},
                     "all_exact": round(float((p == g).all(1).mean()), 3),
                     "pred_total": [int(x) for x in p.sum(0)], "gt_total": [int(x) for x in g.sum(0)],
                     "confusion": {gn: {pn: cm[(gn, pn)] for pn in N + ["miss"]} for gn in N}}
    res["rows"].append(row)
    print(json.dumps(row, ensure_ascii=False))
out = ROOT / "work/fruitcls" / f"eval_gate_{Path(sys.argv[2]).stem}_{split}_c{DET_CONF}.json"
out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
