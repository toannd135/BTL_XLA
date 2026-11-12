"""Vẽ biểu đồ cho báo cáo. Chạy: python work/make_figures.py  ->  docs/figures/*.png
Số liệu lấy từ runs/*/results.csv, work/audit/eval_*.json, work/fruitcls/eval_gate_clip_test*.json, work/obb/eval/*.json và các bảng trong README (mục 5)."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # thứ tự cố định: xanh, cam, aqua, vàng, hồng
BLUE_SEQ = LinearSegmentedColormap.from_list("b", ["#f3f8fe", "#86b6ef", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("ok", name)


def rows(path):
    r = list(csv.DictReader(open(ROOT / path)))
    return [{k.strip(): v for k, v in x.items()} for x in r]


def col(rs, key):
    return np.array([float(x[key]) for x in rs])


def heat(ax, m, labels_x, labels_y, title, fmt="{:.0f}", norm_rows=False):
    m = np.array(m, float)
    shade = m / m.sum(1, keepdims=True) if norm_rows else m / m.max()
    ax.imshow(shade, cmap=BLUE_SEQ, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if shade[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(labels_x)), labels_x)
    ax.set_yticks(range(len(labels_y)), labels_y)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------- 1. So sánh detector A / B / C ----------
metrics = ["mAP50", "mAP50-95", "AP50 chuối", "AP50 táo", "AP50 cam"]
vals = {"A": [0.903, 0.758, 0.802, 0.927, 0.979],
        "B": [0.910, 0.759, 0.836, 0.935, 0.960],
        "C": [0.889, 0.732, 0.805, 0.905, 0.958]}
for tag, name in (("candC", "C'"), ("D1best", "D1 (không nhận)")):   # work/audit/eval_<tag>.json
    e = json.load(open(ROOT / f"work/audit/eval_{tag}.json"))
    vals[name] = [e["mAP50"], e["mAP50-95"], e["AP50"]["banana"], e["AP50"]["apple"], e["AP50"]["orange"]]
fig, ax = plt.subplots(figsize=(11, 4.4))
n = len(vals)
w, x = 0.8 / n, np.arange(len(metrics))
for k, ((name, v), c) in enumerate(zip(vals.items(), S)):
    bars = ax.bar(x + (k - (n - 1) / 2) * w, v, w * 0.92, color=c, label=name, zorder=3)
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.008, f"{val:.2f}", ha="center", fontsize=6.5, color=INK2)
ax.set_xticks(x, metrics)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Giá trị")
ax.set_title("Detector trên test 373 ảnh (conf 0.25, NMS IoU 0.7)")
ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.grid(axis="x", visible=False)
save(fig, "01_detector_A_B_C.png")

# ---------- 2. Đánh đổi B -> C: hai biểu đồ riêng, không dùng hai trục ----------
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.6))
for ax, title, v, fmt, ylim in (
    (axs[0], "mAP50 trên test (có quả)", [0.910, 0.889], "{:.3f}", (0, 1.05)),
    (axs[1], "Ảnh nền báo có quả (71 ảnh giữ riêng)", [48, 4], "{:.0f}%", (0, 60)),
):
    bars = ax.bar(["B", "C"], v, 0.5, color=[S[0], S[1]], zorder=3)
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + (ylim[1] - ylim[0]) * 0.02, fmt.format(val), ha="center", fontweight="bold")
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", visible=False)
fig.suptitle("Đánh đổi khi thêm ảnh nền: mAP50 giảm 0.021, báo sai trên nền giảm từ 48% xuống 4%", x=0.01, ha="left", fontsize=10, color=INK2)
fig.tight_layout()
save(fig, "02_tradeoff_B_vs_C.png")

# ---------- 3. Đường huấn luyện detector (B: 80 epoch, C: 11 epoch) ----------
fig, axs = plt.subplots(2, 2, figsize=(10, 6.2))
for j, (run, title) in enumerate((("runs/exp/det_B_clean", "B (YOLO11s, 80 epoch)"), ("runs/exp/det_C_neg", "C (fine-tune, 11 epoch)"))):
    rs = rows(f"{run}/results.csv")
    ep = col(rs, "epoch")
    a = axs[0, j]
    a.plot(ep, col(rs, "metrics/mAP50(B)"), color=S[0], lw=2, label="mAP50 (val)")
    a.plot(ep, col(rs, "metrics/mAP50-95(B)"), color=S[1], lw=2, label="mAP50-95 (val)")
    a.set_title(f"{title}: độ chính xác trên val", fontsize=10)
    a.set_ylabel("Giá trị")
    a.legend(frameon=False, loc="center right")
    a = axs[1, j]
    a.plot(ep, col(rs, "train/box_loss"), color=S[0], lw=2, label="train box loss")
    a.plot(ep, col(rs, "val/box_loss"), color=S[1], lw=2, label="val box loss")
    a.plot(ep, col(rs, "train/cls_loss"), color=S[2], lw=2, label="train cls loss")
    a.plot(ep, col(rs, "val/cls_loss"), color=S[3], lw=2, label="val cls loss")
    a.set_title(f"{title}: loss", fontsize=10)
    a.set_xlabel("Epoch")
    a.set_ylabel("Loss")
    a.legend(frameon=False, ncol=2, fontsize=8)
fig.tight_layout()
save(fig, "03_detector_training_curves.png")

# ---------- 4. Đường huấn luyện classifier ----------
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, run, title in ((axs[0], "runs/cls_baseline", "Baseline (30 epoch)"), (axs[1], "runs/cls_finetune", "Fine-tune (15 epoch)")):
    rs = rows(f"{run}/results.csv")
    ep = col(rs, "epoch")
    ax.plot(ep, col(rs, "train/loss"), color=S[0], lw=2, label="train loss")
    ax.plot(ep, col(rs, "val/loss"), color=S[1], lw=2, label="val loss")
    ax.set_title(f"Classifier {title}", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
fig.tight_layout()
save(fig, "04_classifier_training_curves.png")

# ---------- 5. Confusion matrix classifier ----------
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.8))
heat(axs[0], [[959, 2], [0, 1309]], ["fresh", "rotten"], ["fresh", "rotten"], "Test public (2270 crop), accuracy 0.9991")
heat(axs[1], [[8, 0], [1, 19]], ["fresh", "rotten"], ["fresh", "rotten"], "Test tự chụp (28 crop), accuracy 0.964")
fig.tight_layout()
save(fig, "05_classifier_confusion.png")

# ---------- 6. Confusion matrix detector B (conf 0.25, test) ----------
eb = json.load(open(ROOT / "work/audit/eval_B_clean.json"))
m = np.array(eb["confusion_c25"], float).T   # Ultralytics lưu [dự đoán][thật]; chuyển vị để hàng = thật (cột cuối = bỏ sót, hàng cuối = báo nhầm nền)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
labs = ["chuối", "táo", "cam", "nền"]
labs_x = ["chuối", "táo", "cam", "bỏ sót"]
labs_y = ["chuối", "táo", "cam", "nền (báo nhầm)"]
heat(axs[0], m, labs_x, labs_y, "Detector B: số box (hàng = thật, cột = dự đoán)")
nm = m / m.sum(1, keepdims=True)
heat(axs[1], nm, labs_x, labs_y, "Chuẩn hóa theo hàng", fmt="{:.2f}", norm_rows=True)
fig.tight_layout()
save(fig, "06_detector_B_confusion.png")

# ---------- 7. Loại box thừa của detector B ----------
types = ["duplicate", "split_of_larger_gt", "localization_fp", "background_fp", "class_fp"]
names = ["trùng lặp", "từng quả trong nhãn nải", "lệch vị trí", "nền thật", "nhầm loại"]
fruits = [("banana", "Chuối"), ("apple", "Táo"), ("orange", "Cam")]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
left = np.zeros(3)
for t, n, c in zip(types, names, S):
    v = np.array([eb["fp_types"][f].get(t, 0) for f, _ in fruits], float)
    bars = ax.barh([f for _, f in fruits], v, left=left, color=c, label=n, height=0.55, edgecolor=SURFACE, linewidth=2, zorder=3)
    for b, val, l in zip(bars, v, left):
        if val >= 6:
            ax.text(l + val / 2, b.get_y() + b.get_height() / 2, f"{val:.0f}", ha="center", va="center", fontsize=9, color="white" if c in (S[0], S[1], S[2]) else INK)
    left += v
for i, tot in enumerate(left):
    ax.text(tot + 1.2, i, f"{tot:.0f}", va="center", fontweight="bold")
ax.invert_yaxis()
ax.set_xlabel("Số box thừa (conf 0.25)")
ax.set_title("Detector B: nguồn gốc box thừa trên test (tổng theo loại lỗi đã phân tích)")
ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.grid(axis="y", visible=False)
save(fig, "07_detector_B_fp_types.png")

# ---------- 8. End-to-end (28 ảnh tự chụp) ----------
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axs[0]
mt = ["Accuracy e2e", "Recall detector", "F1 macro"]
b, f = [0.464, 0.500, 0.380], [0.607, 0.607, 0.604]
x = np.arange(3)
for k, (v, name, c) in enumerate(((b, "Baseline", S[1]), (f, "Fine-tune", S[0]))):
    bars = ax.bar(x + (k - 0.5) * 0.36, v, 0.34, color=c, label=name, zorder=3)
    for bb, val in zip(bars, v):
        ax.text(bb.get_x() + bb.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(x, mt)
ax.set_ylim(0, 1)
ax.set_title("End-to-end trên 28 ảnh tự chụp")
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="x", visible=False)
ax = axs[1]
cats = ["Đúng cả loại và\ntươi/hỏng", "Bỏ sót quả", "Báo nhầm nền", "Nhầm loại quả", "Nhầm tươi/hỏng"]
v = [17, 11, 12, 0, 0]
bars = ax.barh(cats, v, color=[S[0], S[1], S[3], S[2], S[4]], height=0.6, zorder=3)
for bb, val in zip(bars, v):
    ax.text(val + 0.4, bb.get_y() + bb.get_height() / 2, str(val), va="center", fontweight="bold")
ax.invert_yaxis()
ax.set_xlim(0, 21)
ax.set_title("Bản fine-tune: kết quả từng loại")
ax.set_xlabel("Số quả / lần báo")
ax.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "08_end_to_end.png")

# ---------- 9. Dữ liệu ----------
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axs[0]
sp = ["Train", "Val", "Test"]
v = [2365, 301, 373]
bars = ax.bar(sp, v, 0.55, color=S[0], zorder=3)
for bb, val in zip(bars, v):
    ax.text(bb.get_x() + bb.get_width() / 2, val + 30, str(val), ha="center", fontweight="bold")
ax.set_title("Detector: số ảnh mỗi tập")
ax.set_ylabel("Số ảnh")
ax.grid(axis="x", visible=False)
ax = axs[1]
fr = ["Chuối", "Táo", "Cam"]
fresh, rot = np.array([246, 248, 467]), np.array([602, 279, 428])
x = np.arange(3)
for k, (v, name, c) in enumerate(((fresh, "fresh", S[2]), (rot, "rotten", S[1]))):
    bars = ax.bar(x + (k - 0.5) * 0.38, v, 0.36, color=c, label=name, zorder=3)
    for bb, val in zip(bars, v):
        ax.text(bb.get_x() + bb.get_width() / 2, val + 8, str(val), ha="center", fontsize=9)
ax.set_xticks(x, fr)
ax.set_title("Classifier: test public (2270 crop)")
ax.set_ylabel("Số crop")
ax.legend(frameon=False)
ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "09_dataset.png")


# ---------- 10. CLIP kiểm tra loại quả sau detector OBB (test 373 ảnh) ----------
# work/fruitcls/eval_gate_clip_test.json (B-32), _c0.4.json (L-14), _c0.25.json (L-14 + cứu box + hai lần nhìn)
fc = ROOT / "work/fruitcls"
src = [("Chỉ OBB", "eval_gate_clip_test_c0.4.json", 0), ("+ CLIP B-32", "eval_gate_clip_test.json", -1),
       ("+ CLIP L-14", "eval_gate_clip_test_c0.4.json", -1), ("+ cứu box,\nhai lần nhìn\n(mặc định)", "eval_gate_clip_test_c0.25.json", -1)]
names, exact, other, miss = [], [], [], []
for name, f, i in src:
    r = json.load(open(fc / f))["rows"][i]
    a, b = map(int, r["other_fp_images"].split("/"))
    names.append(name); exact.append(100 * r["det"]["all_exact"]); other.append(100 * a / b)
    miss.append(r["det"]["confusion"]["banana"]["miss"])
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
cols = [S[4], S[3], S[0], S[2]]
for ax, title, v, fmt, ylim in (
    (axs[0], "Ảnh đếm đúng hoàn toàn (%)", exact, "{:.1f}", (0, 100)),
    (axs[1], "Ảnh vật khác báo có quả (% trên 1054)", other, "{:.1f}", (0, 35)),
    (axs[2], "Chuối bị bỏ sót (trên 156 quả)", miss, "{:.0f}", (0, 40)),
):
    bars = ax.bar(range(len(v)), v, 0.6, color=cols, zorder=3)
    for bb, val in zip(bars, v):
        ax.text(bb.get_x() + bb.get_width() / 2, val + (ylim[1] - ylim[0]) * 0.02, fmt.format(val), ha="center", fontweight="bold", fontsize=9)
    ax.set_xticks(range(len(v)), names, fontsize=8)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", visible=False)
fig.suptitle("CLIP sau detector OBB, test 373 ảnh: lọc vật khác 30% -> 7.5%, chuối sót 32 -> 17", x=0.01, ha="left", fontsize=10, color=INK2)
fig.tight_layout()
save(fig, "10_clip_gate.png")

# ---------- 11. Đếm quả: C' (box thẳng) và OBB (box xoay), conf 0.40 ----------
cnt = {"C'": json.load(open(ROOT / "work/obb/eval/C_hbb.json")), "OBB": json.load(open(ROOT / "work/obb/eval/obb_colab.json"))}
fruits = [("banana", "Chuối"), ("apple", "Táo"), ("orange", "Cam")]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 2]})
x = np.arange(len(fruits))
for k, (name, d) in enumerate(cnt.items()):
    v = [d["test"][f]["mae"] for f, _ in fruits]
    bars = axs[0].bar(x + (k - 0.5) * 0.36, v, 0.34, color=S[k], label=name, zorder=3)
    for bb, val in zip(bars, v):
        axs[0].text(bb.get_x() + bb.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", fontsize=8.5)
axs[0].set_xticks(x, [t for _, t in fruits])
axs[0].set_title("MAE đếm số quả mỗi ảnh, test 373 ảnh (thấp hơn là tốt)", fontsize=10)
axs[0].legend(frameon=False)
axs[0].grid(axis="x", visible=False)
v = [100 * cnt[n]["test"]["all_exact"] for n in cnt]
bars = axs[1].bar(list(cnt), v, 0.5, color=S[:2], zorder=3)
for bb, val in zip(bars, v):
    axs[1].text(bb.get_x() + bb.get_width() / 2, val + 2, f"{val:.1f}%", ha="center", fontweight="bold")
axs[1].set_ylim(0, 100)
axs[1].set_title("Ảnh đếm đúng hoàn toàn", fontsize=10)
axs[1].grid(axis="x", visible=False)
fig.suptitle("Nhãn chuối của test là box cả nải nên số chuối chỉ so tương đối", x=0.01, ha="left", fontsize=10, color=INK2)
fig.tight_layout()
save(fig, "11_count_C_vs_OBB.png")
