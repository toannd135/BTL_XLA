"""Đo đếm quả theo từng loại cho model HBB (detect) hoặc OBB, cùng một cách tính.
Tập: synth_holdout (ảnh ghép seed khác, lạc quan vì dùng chung ảnh cắt), test (nhãn gốc, có nhãn nải nên chỉ để so tương đối),
naichuoi (~20 chuối, đếm tay), count_gt (nếu có work/obb/count_gt.csv do người điền), neg (71 ảnh nền giữ riêng).
Dùng: python work/obb/eval_count.py <weights> <tag> [conf=0.4] [iou=0.6] [sets=all]"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
NAMES = ["banana", "apple", "orange"]


def gt_from_labels(lp):
    c = Counter()
    if lp.exists():
        for r in lp.read_text().splitlines():
            if r.strip():
                c[int(r.split()[0])] += 1
    return c


def sets():
    s = {}
    sh = ROOT / "work/obb/data/synth_holdout"
    if sh.exists():
        s["synth_holdout"] = [(str(p), gt_from_labels(sh / "labels" / f"{p.stem}.txt")) for p in sorted((sh / "images").glob("*.jpg"))]
    td = ROOT / "local_dataset/det/test"
    s["test"] = [(str(p), gt_from_labels(td / "labels" / f"{p.stem}.txt")) for p in sorted((td / "images").iterdir())]
    s["naichuoi"] = [(str(ROOT / "naichuoi.jpeg"), Counter({0: 20}))]
    cg = ROOT / "work/obb/count_gt.csv"
    if cg.exists():
        rows = [r for r in csv.DictReader(open(cg)) if r.get("confidence", "high") != "skip"]
        s["count_gt"] = [(r["path"], Counter({i: int(r[n] or 0) for i, n in enumerate(NAMES)})) for r in rows]
    s["neg"] = [(str(p), Counter()) for p in sorted((ROOT / "work/exp/neg/test/images").glob("*"))]
    return s


def main():
    w, tag = sys.argv[1], sys.argv[2]
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    iou = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
    want = sys.argv[5].split(",") if len(sys.argv) > 5 else None
    m = YOLO(w)
    res = {"weights": w, "conf": conf, "iou": iou, "task": m.task}
    for name, items in sets().items():
        if want and name not in want:
            continue
        per = []
        for i in range(0, len(items), 16):
            batch = items[i:i + 16]
            rs = m.predict([p for p, _ in batch], conf=conf, iou=iou, agnostic_nms=True, imgsz=640, quantize=16, verbose=False, max_det=300)
            for (p, gt), r in zip(batch, rs):
                b = r.obb if m.task == "obb" else r.boxes
                pc = Counter(int(c) for c in b.cls.tolist())
                per.append({"image": Path(p).name, "gt": [gt[k] for k in range(3)], "pred": [pc[k] for k in range(3)]})
        g = np.array([x["gt"] for x in per])
        pr = np.array([x["pred"] for x in per])
        out = {"n": len(per)}
        if name == "neg":
            out["fp_images"] = int((pr.sum(1) > 0).sum())
        else:
            for k, n in enumerate(NAMES):
                sel = g[:, k] > 0
                if sel.sum() == 0:
                    continue
                out[n] = {"imgs": int(sel.sum()), "gt": int(g[sel, k].sum()), "pred": int(pr[sel, k].sum()),
                          "mae": round(float(np.abs(pr[sel, k] - g[sel, k]).mean()), 3),
                          "exact": round(float((pr[sel, k] == g[sel, k]).mean()), 3),
                          "rel_err": round(float((np.abs(pr[sel, k] - g[sel, k]) / g[sel, k]).mean()), 3)}
            out["all_mae"] = round(float(np.abs(pr - g).sum(1).mean()), 3)
            out["all_exact"] = round(float((pr == g).all(1).mean()), 3)
        out["per_image"] = per if name in ("naichuoi", "count_gt") else None
        res[name] = out
        print(name, json.dumps({k: v for k, v in out.items() if k != "per_image"}, ensure_ascii=False))
        if name == "naichuoi":
            print("  naichuoi pred:", per[0]["pred"])
    od = ROOT / "work/obb/eval"
    od.mkdir(exist_ok=True)
    (od / f"{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
