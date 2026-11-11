"""Tạo ảnh ghép NHIỀU LOẠI QUẢ: cắt từng quả (GrabCut trong box) từ ảnh 1 quả của local_dataset/det/{train|test}, dán lên nền own_neg.
Nhãn = bbox của mặt nạ (từng quả). Train chỉ dùng quả từ split train + nền neg train; bộ đo dùng quả split test + nền neg giữ riêng.
Ra: work/synth/{train,test}/{images,labels}, work/synth/pool_preview.jpg. Dùng: python work/gen_synth.py [n_train=700] [n_test=120]"""
import random, shutil, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]; DET = ROOT / "local_dataset" / "det"; OUT = ROOT / "work" / "synth"
NTR, NTE = (int(x) for x in (sys.argv[1:3] + [None, None])[:2]) if len(sys.argv) > 2 else (700, 120)
rng = random.Random(42); S = 640
quar = set((ROOT / "work/audit/quarantine_train.txt").read_text().split())


def is_green(c):
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3); m = hsv[hsv[:, 1] > 60]
    return len(m) > 20 and 50 <= np.median(m[:, 0]) * 2 <= 100


def pool(split, per_class=120):
    """Danh sách (cls, ảnh BGRA đã cắt, xanh?) từ ảnh có đúng 1 box lớn."""
    files = [p for p in sorted((DET / split / "labels").glob("*.txt"))]
    rng.shuffle(files); got = {0: [], 1: [], 2: []}; ng = 0
    for lb in files:
        L = [l.split() for l in lb.read_text().splitlines() if l.strip()]
        if len(L) != 1 or lb.stem + ".jpg" in quar or min(len(v) for v in got.values()) >= per_class:
            continue
        c = int(L[0][0]); 
        if len(got[c]) >= per_class: continue
        im = cv2.imread(str(DET / split / "images" / (lb.stem + ".jpg")))
        if im is None: continue
        h, w = im.shape[:2]; x, y, bw, bh = (float(v) for v in L[0][1:5])
        if not 0.25 <= bw * bh <= 0.85: continue
        x1, y1, x2, y2 = int((x - bw / 2) * w), int((y - bh / 2) * h), int((x + bw / 2) * w), int((y + bh / 2) * h)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        s = 320 / max(h, w); im = cv2.resize(im, None, fx=s, fy=s) if s < 1 else im; k = im.shape[0] / h
        r = (int(x1 * k), int(y1 * k), int((x2 - x1) * k), int((y2 - y1) * k))
        m = np.zeros(im.shape[:2], np.uint8); bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
        try: cv2.grabCut(im, m, r, bg, fg, 4, cv2.GC_INIT_WITH_RECT)
        except cv2.error: continue
        mk = ((m == 1) | (m == 3)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mk)
        if n < 2: continue
        mk = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
        frac = mk.sum() / max(r[2] * r[3], 1)
        if not 0.4 <= frac <= 0.95: continue            # mặt nạ lọt ra ngoài/quá nhỏ -> loại
        ys, xs = np.where(mk); b = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        crop, mc = im[b[1]:b[3], b[0]:b[2]], mk[b[1]:b[3], b[0]:b[2]]
        got[c].append((c, np.dstack([crop, mc * 255]), c == 1 and is_green(crop)))
    return got


def compose(P, bgs, name, d, green_boost=True):
    bg = cv2.imread(str(rng.choice(bgs))); h, w = bg.shape[:2]; s = S / min(h, w)
    bg = cv2.resize(bg, None, fx=s, fy=s); y0, x0 = rng.randint(0, bg.shape[0] - S), rng.randint(0, bg.shape[1] - S)
    img = bg[y0:y0 + S, x0:x0 + S].copy(); boxes = []
    classes = rng.sample([0, 1, 2], rng.choice([2, 3, 3]))
    for _ in range(rng.randint(4, 10)):
        c = rng.choice(classes) if len(boxes) >= len(classes) else classes[len(boxes)]
        pl = P[c]
        if c == 1 and green_boost and rng.random() < 0.45 and any(g for _, _, g in pl):
            pl = [t for t in pl if t[2]]
        _, rgba, _ = rng.choice(pl); rgba = rgba.copy()
        if rng.random() < 0.5: rgba = rgba[:, ::-1]
        sc = rng.uniform(90, 230) / max(rgba.shape[:2]); rgba = cv2.resize(np.ascontiguousarray(rgba), None, fx=sc, fy=sc)
        hh, ww = rgba.shape[:2]
        if hh >= S or ww >= S: continue
        for _try in range(15):
            px, py = rng.randint(0, S - ww), rng.randint(0, S - hh); nb = (px, py, px + ww, py + hh)
            ok = True
            for o in boxes:
                iw, ih = min(nb[2], o[1][2]) - max(nb[0], o[1][0]), min(nb[3], o[1][3]) - max(nb[1], o[1][1])
                if iw > 0 and ih > 0 and iw * ih > 0.2 * min(ww * hh, (o[1][2] - o[1][0]) * (o[1][3] - o[1][1])): ok = False; break
            if ok: break
        else: continue
        a = cv2.GaussianBlur(rgba[:, :, 3].astype(np.float32) / 255, (3, 3), 0)[..., None]
        roi = img[py:py + hh, px:px + ww].astype(np.float32); sh = np.roll(a, (4, 4), (0, 1)) * 0.25     # bóng đổ nhẹ
        roi = roi * (1 - sh); img[py:py + hh, px:px + ww] = (rgba[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)
        boxes.append((c, nb))
    if len({c for c, _ in boxes}) < 2: return False
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.85, 1.15), beta=rng.uniform(-15, 15))
    (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "images" / f"{name}.jpg"), img)
    (d / "labels" / f"{name}.txt").write_text("".join(f"{c} {(b[0]+b[2])/2/S:.6f} {(b[1]+b[3])/2/S:.6f} {(b[2]-b[0])/S:.6f} {(b[3]-b[1])/S:.6f}\n" for c, b in boxes))
    return True


shutil.rmtree(OUT, ignore_errors=True)
NEG = ROOT / "work" / "exp" / "neg"
for split, n, bgdir in (("train", NTR, NEG / "train" / "images"), ("test", NTE, NEG / "test" / "images")):
    P = pool(split, 120 if split == "train" else 40); bgs = sorted(bgdir.glob("*.jpg"))
    print(split, "pool:", {k: len(v) for k, v in P.items()}, "táo xanh:", sum(g for _, _, g in P[1]), "| nền:", len(bgs), flush=True)
    i = 0
    while i < n:
        i += compose(P, bgs, f"synth_{split}_{i:04d}", OUT / split)
print("xong ->", OUT)
