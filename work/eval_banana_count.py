"""Đo sai số ĐẾM chuối của đúng pipeline backend (backend/detection.py::detect).

Cách dùng:
  python work/eval_banana_count.py --tag base                       # model app mặc định, chế độ nhanh
  python work/eval_banana_count.py --tag careful --mode careful     # chế độ "careful" (imgsz 960 + xoay)
  python work/eval_banana_count.py --tag x --opt banana_iou=0.75    # tham số thêm cho detect()

Ra: work/audit/banana_count_<tag>.csv (mỗi ảnh) và .json (tổng hợp).
Tập "clean" = ảnh test có chuối, không bị cờ bunch/nested/missing:banana trong
work/audit/det_label_audit.csv. Nhãn test còn lẫn box nải, nên số trên tập "all" chỉ để tham khảo.
"""
import argparse, csv, json, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from detection import detect  # noqa: E402

BANANA = 0


def parse_opt(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                v = {"true": True, "false": False}.get(v.lower(), v)
        out[k] = v
    return out


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1); ab = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def n_group(boxes):
    """Số box chứa >= 2 box khác (>= 80% diện tích box con nằm trong)."""
    cnt = 0
    for i, p in enumerate(boxes):
        lt = np.maximum(boxes[:, :2], p[:2]); rb = np.minimum(boxes[:, 2:], p[2:])
        inside = np.clip(rb - lt, 0, None).prod(1) / np.maximum((boxes[:, 2:] - boxes[:, :2]).prod(1), 1e-9)
        inside[i] = 0
        cnt += int((inside >= 0.8).sum() >= 2)
    return cnt


def greedy_tp(pred, gt, thr=0.5):
    m = iou_matrix(pred, gt); tp = 0
    while m.size and m.max() >= thr:
        i, j = np.unravel_index(m.argmax(), m.shape); tp += 1; m[i, :] = 0; m[:, j] = 0
    return tp


def load_gt(lbl, w, h):
    rows = [l.split() for l in open(lbl)] if lbl.exists() else []
    b = [[(float(x) - float(bw) / 2) * w, (float(y) - float(bh) / 2) * h, (float(x) + float(bw) / 2) * w,
          (float(y) + float(bh) / 2) * h] for c, x, y, bw, bh in (r[:5] for r in rows if r) if int(c) == BANANA]
    return np.asarray(b, dtype=np.float32).reshape(-1, 4)


def flagged():
    out = set()
    for r in csv.DictReader(open(ROOT / "work/audit/det_label_audit.csv")):
        if r["split"] == "test" and any(f.split(":")[-1].startswith("banana") for f in r["flags"].split(";") if ":" in f):
            out.add(r["image"])
    return out


def summarize(rows):
    if not rows:
        return {}
    err = np.array([r["pred"] - r["gt"] for r in rows]); tp = sum(r["tp"] for r in rows)
    P = tp / max(sum(r["pred"] for r in rows), 1); R = tp / max(sum(r["gt"] for r in rows), 1)
    return {"n_img": len(rows), "gt_total": int(sum(r["gt"] for r in rows)), "pred_total": int(sum(r["pred"] for r in rows)),
            "MAE": round(float(np.abs(err).mean()), 3), "bias": round(float(err.mean()), 3),
            "exact": round(float((err == 0).mean()), 3), "within1": round(float((np.abs(err) <= 1).mean()), 3),
            "over": int((err > 0).sum()), "under": int((err < 0).sum()),
            "group_boxes": int(sum(r["group"] for r in rows)), "dup_pairs": int(sum(r["dup"] for r in rows)),
            "P50": round(P, 3), "R50": round(R, 3), "F1_50": round(2 * P * R / max(P + R, 1e-9), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "exports/det_finetune_best.pt"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["fast", "careful"], default="fast")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--opt", nargs="*")
    a = ap.parse_args()
    imgsz = a.imgsz or (960 if a.mode == "careful" else 640)
    extra = parse_opt(a.opt)
    m = YOLO(a.weights)
    kw = dict(conf=a.conf, imgsz=imgsz, iou=a.iou, quantize=16 if a.half else None, rotated=a.mode == "careful", **extra)

    test = ROOT / "local_dataset/det/test"; bad = flagged(); rows = []
    for p in sorted((test / "images").glob("*")):
        img = cv2.imread(str(p)); h, w = img.shape[:2]
        gt = load_gt(test / "labels" / (p.stem + ".txt"), w, h)
        if not len(gt):
            continue
        b, s, c = detect(m, img, **kw)
        pb = np.asarray(b, dtype=np.float32).reshape(-1, 4)[np.asarray(c) == BANANA]
        iu = iou_matrix(pb, pb); np.fill_diagonal(iu, 0)
        rows.append({"image": p.name, "subset": "flagged" if p.name in bad else "clean", "gt": len(gt), "pred": len(pb),
                     "err": len(pb) - len(gt), "tp": greedy_tp(pb, gt), "group": n_group(pb),
                     "dup": int((np.triu(iu, 1) > 0.5).sum()), "other_cls": int((np.asarray(c) != BANANA).sum())})

    # Ảnh nền giữ riêng (71): ảnh nào có >= 1 box bất kỳ là báo sai
    neg = sorted((ROOT / "work/exp/neg/test/images").glob("*"))
    neg_fp = sum(len(detect(m, cv2.imread(str(p)), **kw)[0]) > 0 for p in neg)

    out = ROOT / "work/audit"; out.mkdir(parents=True, exist_ok=True)
    with open(out / f"banana_count_{a.tag}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    res = {"tag": a.tag, "weights": a.weights, "kw": {k: v for k, v in kw.items()},
           "all": summarize(rows), "clean": summarize([r for r in rows if r["subset"] == "clean"]),
           "clean_multi(gt>=2)": summarize([r for r in rows if r["subset"] == "clean" and r["gt"] >= 2]),
           "neg_fp": f"{neg_fp}/{len(neg)}"}
    json.dump(res, open(out / f"banana_count_{a.tag}.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
