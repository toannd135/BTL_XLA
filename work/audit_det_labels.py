"""Rà nhãn detection: (1) nhãn ôm cả nải, (2) thiếu nhãn, (3) rò rỉ giữa train/val/test. Chỉ ĐỌC dữ liệu, không sửa nhãn.
Ra: work/audit/det_label_audit.csv + work/audit/{bunch,missing}.jpg (ảnh minh hoạ: xanh = nhãn, đỏ = model)."""
import csv, re
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DET, OUT = ROOT / "local_dataset" / "det", ROOT / "work" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
N = ["banana", "apple", "orange"]
SPLITS = ["train", "val", "test"]
CONF_SURE = 0.5          # box model đủ chắc để coi là "quả thật"
RF = re.compile(r"(_(jpg|jpeg|png|bmp|webp))?\.rf\.[0-9A-Za-z]+$", re.I)
model = YOLO(str(ROOT / "exports" / "det_finetune_best.pt"))


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def inside(small, big):
    cx, cy = (small[0] + small[2]) / 2, (small[1] + small[3]) / 2
    return big[0] <= cx <= big[2] and big[1] <= cy <= big[3]


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


rows, groups, hashes = [], defaultdict(set), {}
for sp in SPLITS:
    imgs = sorted((DET / sp / "images").glob("*"))
    for i in range(0, len(imgs), 16):
        batch = imgs[i:i + 16]
        for p, r in zip(batch, model.predict([str(x) for x in batch], conf=0.25, imgsz=640, verbose=False)):
            h, w = r.orig_shape
            gt = []
            lf = DET / sp / "labels" / f"{p.stem}.txt"
            if lf.exists():
                for l in lf.read_text().splitlines():
                    if l.strip():
                        c, x, y, bw, bh = l.split()[:5]
                        x, y, bw, bh = float(x), float(y), float(bw), float(bh)
                        gt.append((int(c), [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
            pr = [(int(c), b, s) for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            sure = [q for q in pr if q[2] >= CONF_SURE]
            flags = []
            # 1) nhãn ôm cả nải: một box nhãn chứa >= 2 box model chắc cùng class, mỗi box nhỏ hơn hẳn nhãn
            for c, g in gt:
                inner = [q for q in sure if q[0] == c and inside(q[1], g) and area(q[1]) <= 0.6 * area(g)]
                if len(inner) >= 2:
                    flags.append(f"bunch:{N[c]}x{len(inner)}")
            # box nhãn bao trọn >= 2 box nhãn khác
            for k, (c, g) in enumerate(gt):
                if sum(inside(g2, g) and area(g2) <= 0.6 * area(g) for j, (_, g2) in enumerate(gt) if j != k) >= 2:
                    flags.append(f"nested:{N[c]}")
            # 2) thiếu nhãn: box model chắc không trùng nhãn nào (IoU < 0.3) và không nằm trong/bao nhãn nào
            miss = [q for q in sure if all(iou(q[1], g) < 0.3 and not inside(q[1], g) and not inside(g, q[1]) for _, g in gt)]
            if miss:
                flags.append("missing:" + "/".join(sorted({N[q[0]] for q in miss})) + f"x{len(miss)}")
            if not gt and any(True for _ in pr):
                flags.append("empty-label")
            groups[sp].add(RF.sub("", p.stem))
            hashes[(sp, p.name)] = dhash(cv2.imread(str(p)))
            rows.append({"split": sp, "image": p.name, "n_gt": len(gt), "n_pred": len(pr), "flags": ";".join(flags)})
            if flags:
                rows[-1]["_draw"] = (p, gt, sure)

with open(OUT / "det_label_audit.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, ["split", "image", "n_gt", "n_pred", "flags"])
    wr.writeheader()
    wr.writerows({k: r[k] for k in wr.fieldnames} for r in rows if r["flags"])

print("== Cờ theo tập (số ảnh) ==")
for sp in SPLITS:
    rs = [r for r in rows if r["split"] == sp]
    cnt = defaultdict(int)
    for r in rs:
        for kind in {fl.split(":")[0] for fl in r["flags"].split(";") if fl}:
            cnt[kind] += 1
    print(f"{sp:6s} ảnh={len(rs):5d}", dict(cnt))

print("== Rò rỉ giữa các tập ==")
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    print(f"cùng nguồn Roboflow {a}-{b}: {len(groups[a] & groups[b])} nhóm")
tr = [(k, v) for k, v in hashes.items() if k[0] == "train"]
trv = np.array([v for _, v in tr], dtype=object)
for sp in ("val", "test"):
    near = sum(any(bin(v ^ t).count("1") <= 6 for t in trv) for (s, _), v in hashes.items() if s == sp)
    print(f"{sp}: {near}/{sum(1 for k in hashes if k[0] == sp)} ảnh gần giống ảnh train (dHash <= 6 bit)")


def montage(kind, name):
    sel = [r for r in rows if r["flags"] and any(fl.startswith(kind) for fl in r["flags"].split(";"))][:12]
    tiles = []
    for r in sel:
        p, gt, sure = r["_draw"]
        im = cv2.imread(str(p))
        for _, g in gt:
            cv2.rectangle(im, tuple(int(v) for v in g[:2]), tuple(int(v) for v in g[2:]), (0, 200, 0), 3)
        for c, b, s in sure:
            cv2.rectangle(im, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:]), (0, 0, 255), 2)
        t = cv2.resize(im, (480, 360))
        cv2.putText(t, f"{r['split']} {r['flags'][:40]}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        tiles.append(t)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros((360, 480, 3), np.uint8))
        cv2.imwrite(str(OUT / name), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


montage("bunch", "bunch.jpg")
montage("missing", "missing.jpg")
