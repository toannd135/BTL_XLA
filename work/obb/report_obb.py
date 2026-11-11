"""Bảng + biểu đồ cho model khung xoay (OBB), cùng kiểu README mục 5 (work/make_figures.py), để so với A/B/C.
Dùng: python work/obb/report_obb.py <best.pt> <run_dir có results.csv> <out_dir> [hbb_ref=exports/det_C_candidate.pt]
Cần chạy trước: eval_count.py cho model OBB (tag = tên thư mục run) và cho C (tag C_hbb) -> work/obb/eval/*.json.
Ra: <out_dir>/tables.md (bảng markdown, dán được vào README) và <out_dir>/figures/*.png.

Lưu ý so sánh: số A/B/C lấy từ README (test box thẳng gốc). Test của OBB là cùng 373 ảnh nhưng nhãn chuối đã đổi
sang khung xoay ở chỗ mặt nạ SAM rõ một quả, còn lại giữ box thẳng, nên mAP chỉ so tương đối."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/obb/data"
NAMES = ["banana", "apple", "orange"]
VI = {"banana": "chuối", "apple": "táo", "orange": "cam"}

# Số liệu README mục 5.1 / 5.7 (test 373 ảnh, conf 0.25, NMS 0.7)
REF = {
    "A": {"map50": 0.903, "map": 0.758, "ap50": [0.802, 0.927, 0.979], "neg25": None},
    "B": {"map50": 0.910, "map": 0.759, "ap50": [0.836, 0.935, 0.960], "neg25": 34},
    "C": {"map50": 0.889, "map": 0.732, "ap50": [0.805, 0.905, 0.958], "neg25": 3},
}
REF_C_F1_40 = {"P": 0.80, "R": 0.84, "F1": 0.82, "banana": 0.73, "apple": 0.89, "orange": 0.90, "neg40": 1}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f3(v):
    return "-" if v is None else f"{v:.3f}"


def test_items():
    items = []
    for line in (DATA / "test.txt").read_text().split():
        p = DATA / line[2:] if line.startswith("./") else Path(line)
        lp = p.parent.parent / "labels" / f"{p.stem}.txt"
        rows = [r.split() for r in lp.read_text().splitlines() if r.strip()] if lp.exists() else []
        items.append((p, np.array([[float(v) for v in r[1:9]] for r in rows]).reshape(-1, 8), np.array([int(r[0]) for r in rows])))
    return items


def neg_images():
    return sorted(str(p) for p in (ROOT / "work/exp/neg/test/images").glob("*"))


def prf_at_conf(m, items, conf=0.4, iou_nms=0.6, iou_match=0.5):
    """P/R/F1 theo lớp ở một ngưỡng conf, ghép tham lam theo probIoU (cách Ultralytics đo OBB)."""
    tp, fp, fn = Counter(), Counter(), Counter()
    lat = []
    for i in range(0, len(items), 16):
        batch = items[i:i + 16]
        rs = m.predict([str(p) for p, _, _ in batch], conf=conf, iou=iou_nms, agnostic_nms=True, imgsz=640,
                       max_det=300, verbose=False)
        for (p, g, gc), r in zip(batch, rs):
            lat.append(r.speed["inference"])
            h, w = r.orig_shape
            pb = r.obb.xywhr.cpu().numpy() if len(r.obb) else np.zeros((0, 5))
            pc = r.obb.cls.cpu().numpy().astype(int)
            ps = r.obb.conf.cpu().numpy()
            gb = xyxyxyxy2xywhr(g * np.array([w, h] * 4)) if len(g) else np.zeros((0, 5))
            for k in range(3):
                pi, gi = np.where(pc == k)[0], np.where(gc == k)[0]
                pi = pi[np.argsort(-ps[pi])]
                if len(pi) and len(gi):
                    ious = batch_probiou(torch.tensor(pb[pi], dtype=torch.float32),
                                         torch.tensor(np.asarray(gb)[gi], dtype=torch.float32)).numpy()
                else:
                    ious = np.zeros((len(pi), len(gi)))
                used = set()
                for a in range(len(pi)):
                    cand = [(ious[a, b], b) for b in range(len(gi)) if b not in used and ious[a, b] >= iou_match]
                    if cand:
                        used.add(max(cand)[1])
                        tp[k] += 1
                    else:
                        fp[k] += 1
                fn[k] += len(gi) - len(used)
    res = {}
    for k in list(range(3)) + ["all"]:
        t = sum(tp.values()) if k == "all" else tp[k]
        f = sum(fp.values()) if k == "all" else fp[k]
        n = sum(fn.values()) if k == "all" else fn[k]
        P = t / (t + f) if t + f else 0.0
        R = t / (t + n) if t + n else 0.0
        res[k] = {"P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0, "tp": t, "fp": f, "fn": n}
    return res, float(np.median(lat)) if lat else None


def neg_fp(m, conf):
    imgs, n = neg_images(), 0
    for i in range(0, len(imgs), 16):
        for r in m.predict(imgs[i:i + 16], conf=conf, iou=0.6, agnostic_nms=True, imgsz=640, max_det=300, verbose=False):
            n += int(len(r.obb) > 0)
    return n, len(imgs)


def load_eval(tag):
    p = ROOT / "work/obb/eval" / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, out, name):
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  hình:", out / name)


def main():
    w, run_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    hbb_ref = sys.argv[4] if len(sys.argv) > 4 else str(ROOT / "exports/det_C_candidate.pt")
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    tag = run_dir.name
    m = YOLO(w)
    md = [f"# Kết quả model khung xoay `{tag}`\n", f"Trọng số: `{w}`\n"]

    # 1. mAP trên test (conf 0.25, NMS 0.7 như README 5.1) + confusion matrix của Ultralytics
    print("== val trên test (conf 0.25, iou 0.7)")
    mt = m.val(data=str(DATA / "obb.yaml"), split="test", conf=0.25, iou=0.7, imgsz=640, batch=16, plots=True,
               project=str(out), name="val_test", exist_ok=True, verbose=False)
    ap50 = {NAMES[int(c)]: float(v) for c, v in zip(mt.box.ap_class_index, mt.box.ap50)}
    obb = {"map50": float(mt.box.map50), "map": float(mt.box.map), "ap50": [ap50.get(n, 0.0) for n in NAMES]}
    n25, nneg = neg_fp(m, 0.25)
    n40, _ = neg_fp(m, 0.40)
    obb["neg25"] = n25

    def negs(v):
        return "-" if v is None else f"{v} ({v / nneg:.0%})"

    cols = ["A", "B", "C (đang dùng)", f"OBB `{tag}`"]
    allm = [REF["A"], REF["B"], REF["C"], obb]
    rows = [["mAP50"] + [f3(x["map50"]) for x in allm], ["mAP50-95"] + [f3(x["map"]) for x in allm]]
    rows += [[f"AP50 {VI[n]}"] + [f3(x["ap50"][k]) for x in allm] for k, n in enumerate(NAMES)]
    rows += [[f"Ảnh nền báo sai (trên {nneg} ảnh giữ riêng)"] + [negs(x["neg25"]) for x in allm]]
    md += ["## 5.1 Detector (test 373 ảnh, conf 0.25, NMS IoU 0.7)\n", md_table([""] + cols, rows),
           "\nA/B/C: số trong README (test box thẳng). OBB: cùng 373 ảnh, nhãn chuối một phần đã đổi sang khung xoay -> so tương đối.\n"]

    # 2. P/R/F1 ở conf 0.40 (README 5.2) + độ trễ
    print("== P/R/F1 ở conf 0.40")
    prf, lat = prf_at_conf(m, test_items(), conf=0.4)
    rows = [["Precision / Recall / F1 tổng", f"{REF_C_F1_40['P']:.2f} / {REF_C_F1_40['R']:.2f} / {REF_C_F1_40['F1']:.2f}",
             f"{prf['all']['P']:.2f} / {prf['all']['R']:.2f} / {prf['all']['F1']:.2f}"]]
    rows += [[f"F1 {VI[n]} (P / R)", f"{REF_C_F1_40[n]:.2f}",
              f"{prf[k]['F1']:.2f} ({prf[k]['P']:.2f} / {prf[k]['R']:.2f})"] for k, n in enumerate(NAMES)]
    rows += [[f"Ảnh nền báo sai (conf 0.40)", f"{REF_C_F1_40['neg40']}/71", f"{n40}/{nneg}"],
             ["Độ trễ detector (trung vị, ms/ảnh, GPU Colab)", "16.5 (bản B, máy local)", f"{lat:.1f}" if lat else "-"]]
    md += ["## 5.2 Ngưỡng conf 0.40 (NMS 0.6, như backend)\n", md_table(["Chỉ số", "C (README)", f"OBB `{tag}`"], rows),
           "\nGhép box theo probIoU >= 0.5 (cách Ultralytics đo OBB). TP/FP/FN tổng: "
           f"{prf['all']['tp']}/{prf['all']['fp']}/{prf['all']['fn']}.\n"]

    # 3. Đếm quả (từ eval_count.py)
    ev_o, ev_c = load_eval(tag), load_eval("C_hbb")
    if ev_o:
        def cnt(ev, s, n, key):
            v = (ev or {}).get(s, {}).get(n)
            return "-" if not v else v[key]
        rows = []
        if "naichuoi" in ev_o:
            nc = lambda ev: (ev or {}).get("naichuoi", {}).get("per_image") or [{"pred": ["-"]}]
            rows.append(["naichuoi.jpeg: số chuối (thật ~20)", nc(ev_c)[0]["pred"][0], nc(ev_o)[0]["pred"][0]])
        for s in ("test", "synth_holdout", "count_gt"):
            if s not in ev_o:
                continue
            for n in NAMES:
                rows.append([f"{s}: MAE {VI[n]} / đếm đúng", f"{cnt(ev_c, s, n, 'mae')} / {cnt(ev_c, s, n, 'exact')}",
                             f"{cnt(ev_o, s, n, 'mae')} / {cnt(ev_o, s, n, 'exact')}"])
        if "neg" in ev_o:
            rows.append([f"neg: ảnh báo sai (conf {ev_o['conf']})", (ev_c or {}).get("neg", {}).get("fp_images", "-"),
                         ev_o["neg"]["fp_images"]])
        md += ["## Đếm quả (conf 0.40, NMS 0.6)\n", md_table(["Tập / chỉ số", "C (box thẳng)", f"OBB `{tag}`"], rows),
               "\ntest: nhãn gốc còn box nải nên chỉ so tương đối; synth_holdout: ảnh ghép, lạc quan.\n"]
    else:
        print("!! chưa có work/obb/eval/%s.json: chạy eval_count.py trước để có bảng đếm" % tag)

    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    # ---- Hình 1: AP50 theo lớp, A/B/C/OBB
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    cats = ["mAP50"] + [f"AP50 {VI[n]}" for n in NAMES]
    x = np.arange(len(cats))
    wd = 0.2
    for k, (name, d) in enumerate(zip(["A", "B", "C", "OBB"], allm)):
        v = [d["map50"]] + d["ap50"]
        bars = ax.bar(x + (k - 1.5) * (wd + 0.01), v, wd, color=S[k], label=name, zorder=3)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.008, f"{val:.3f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=4, loc="lower right")
    ax.set_title("So sánh detector A, B, C và OBB trên test (conf 0.25, NMS 0.7)")
    save(fig, figs, "01_detector_A_B_C_OBB.png")

    # ---- Hình 2: đánh đổi mAP50 và ảnh nền báo sai (B, C, OBB)
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.4))
    names3, ds = ["B", "C", "OBB"], [REF["B"], REF["C"], obb]
    for a, key, title, fmt in ((axs[0], "map50", "mAP50 trên test", "{:.3f}"), (axs[1], "neg", "Ảnh nền báo sai (%)", "{:.0f}%")):
        v = [d["map50"] for d in ds] if key == "map50" else [100 * d["neg25"] / nneg for d in ds]
        bars = a.bar(names3, v, 0.5, color=S[1:4], zorder=3)
        for b, val in zip(bars, v):
            a.text(b.get_x() + b.get_width() / 2, val * 1.01 + 0.005, fmt.format(val), ha="center", fontsize=8.5, color=INK2)
        a.set_title(title)
        a.set_ylim(0, 1.05 if key == "map50" else max(v + [5]) * 1.25)
    save(fig, figs, "02_tradeoff_B_C_OBB.png")

    # ---- Hình 3: đường huấn luyện OBB
    rc = run_dir / "results.csv"
    if rc.exists():
        rs = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(rc))]
        col = lambda k: [float(r[k]) for r in rs]
        ep = col("epoch")
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
        axs[0].plot(ep, col("metrics/mAP50(B)"), color=S[0], lw=2, label="mAP50 (val)")
        axs[0].plot(ep, col("metrics/mAP50-95(B)"), color=S[1], lw=2, label="mAP50-95 (val)")
        axs[0].plot(ep, col("metrics/precision(B)"), color=S[2], lw=1.2, ls="--", label="precision (val)")
        axs[0].plot(ep, col("metrics/recall(B)"), color=S[3], lw=1.2, ls="--", label="recall (val)")
        axs[0].set_title(f"OBB {tag}: chỉ số val theo epoch")
        axs[0].set_xlabel("epoch")
        axs[0].legend(frameon=False, fontsize=8)
        for k, (kk, lab) in enumerate((("box_loss", "box"), ("cls_loss", "cls"), ("angle_loss", "angle"))):
            axs[1].plot(ep, col(f"train/{kk}"), color=S[k], lw=2, label=f"train {lab}")
            axs[1].plot(ep, col(f"val/{kk}"), color=S[k], lw=1.2, ls="--", label=f"val {lab}")
        axs[1].set_title("Loss train (liền) và val (đứt)")
        axs[1].set_xlabel("epoch")
        axs[1].legend(frameon=False, fontsize=8, ncol=3)
        save(fig, figs, "03_obb_training_curves.png")

    # ---- Hình 4: đếm quả C vs OBB
    if ev_o:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
        nc_c = ((ev_c or {}).get("naichuoi", {}).get("per_image") or [{"pred": [0]}])[0]["pred"][0]
        nc_o = (ev_o.get("naichuoi", {}).get("per_image") or [{"pred": [0]}])[0]["pred"][0]
        bars = axs[0].bar(["Thật", "C", "OBB"], [20, nc_c, nc_o], 0.5, color=[GRID, S[1], S[3]], zorder=3)
        for b, val in zip(bars, [20, nc_c, nc_o]):
            axs[0].text(b.get_x() + b.get_width() / 2, val + 0.4, str(val), ha="center", fontsize=9, color=INK2)
        axs[0].set_title("Số chuối trên naichuoi.jpeg")
        s = "test" if "test" in ev_o else next(iter(k for k in ev_o if isinstance(ev_o[k], dict) and "banana" in ev_o[k]), None)
        if s:
            x = np.arange(3)
            for k, (name, ev, c) in enumerate((("C", ev_c, S[1]), ("OBB", ev_o, S[3]))):
                v = [((ev or {}).get(s, {}).get(n) or {}).get("mae", 0) for n in NAMES]
                bars = axs[1].bar(x + (k - 0.5) * 0.36, v, 0.34, color=c, label=name, zorder=3)
                for b, val in zip(bars, v):
                    axs[1].text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=8, color=INK2)
            axs[1].set_xticks(x, [VI[n] for n in NAMES])
            axs[1].set_title(f"Sai số đếm (MAE, quả/ảnh) trên {s}")
            axs[1].legend(frameon=False)
        save(fig, figs, "04_count_C_vs_OBB.png")

    # ---- Hình 5: naichuoi.jpeg, C (box thẳng) và OBB
    img = ROOT / "naichuoi.jpeg"
    if img.exists():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
        for a, (lab, mm) in zip(axs, (("C (box thẳng)", YOLO(hbb_ref)), (f"OBB {tag}", m))):
            r = mm.predict(str(img), conf=0.4, iou=0.6, agnostic_nms=True, imgsz=640, max_det=300, verbose=False)[0]
            b = r.obb if r.obb is not None else r.boxes
            nb = int((b.cls == 0).sum())
            a.imshow(cv2.cvtColor(r.plot(font_size=8, line_width=1), cv2.COLOR_BGR2RGB))
            a.set_title(f"{lab}: {nb} chuối (thật ~20)")
            a.axis("off")
        save(fig, figs, "05_naichuoi_C_vs_OBB.png")

    print("\nConfusion matrix / PR curve của Ultralytics:", mt.save_dir)


if __name__ == "__main__":
    main()
