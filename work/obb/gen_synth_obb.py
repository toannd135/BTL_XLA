"""Sinh ảnh ghép nhiều quả có nhãn khung xoay (YOLO-OBB) từ ảnh cắt SAM2 (work/obb/cutouts, chỉ nguồn train).
Kiểu cảnh: nải chuối xòe (có cuống làm mẫu âm), chuối xếp song song, đống táo/cam chồng nhau, cảnh trộn.
Nhãn = minAreaRect phần NHÌN THẤY của mỗi quả (chuối) / box thẳng phần nhìn thấy (táo, cam); quả bị che > 65% thì bỏ nhãn
nhưng vẫn vẽ (hiếm). Dùng: python work/obb/gen_synth_obb.py [n=2000] [out=work/obb/data/synth] [seed=0]"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OBB = ROOT / "work" / "obb"
CLS = {"banana": 0, "apple": 1, "orange": 2}


def load_cutouts():
    pools = {"banana": [], "apple": [], "orange": []}
    for line in open(OBB / "instances.jsonl"):
        r = json.loads(line)
        if "cutout" not in r:
            continue
        name = ["banana", "apple", "orange"][r["cls"]]
        if name == "banana":
            ok = r["aspect"] >= 2.6 and r["solidity"] >= 0.55 and r["fill"] >= 0.12
        else:
            ok = r["aspect"] <= 1.35 and r["solidity"] >= 0.92 and r["fill"] >= 0.6
        if ok:
            pools[name].append(str(OBB / "cutouts" / name / r["cutout"]))
    return pools


def read_rgba(p):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    return im if im is not None and im.shape[2] == 4 else None


def jitter(rgba, rng):
    bgr = rgba[..., :3].astype(np.float32)
    hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
    hsv[..., 1] *= rng.uniform(0.8, 1.15)
    hsv[..., 2] *= rng.uniform(0.75, 1.15)
    bgr = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.dstack([bgr, rgba[..., 3]])


def banana_canonical(rgba):
    """Xoay ảnh cắt chuối để trục dài nằm ngang (trái -> phải), cắt sát."""
    m = (rgba[..., 3] > 127).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cx, cy), (w, h), ang = cv2.minAreaRect(max(cs, key=cv2.contourArea))
    if w < h:
        ang += 90
    return rotate(rgba, ang)


def rotate(rgba, ang, scale=1.0):
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    ys, xs = np.where(out[..., 3] > 127)
    if len(xs) == 0:
        return out
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def background(rng, bgs, W, H):
    t = rng.random()
    if t < 0.55 and bgs:
        im = cv2.imread(rng.choice(bgs))
        if im is not None:
            return cv2.resize(im, (W, H))
    if t < 0.8:
        c = np.array([rng.randint(150, 255)] * 3) if rng.random() < 0.6 else np.array([rng.randint(30, 255) for _ in range(3)])
        im = np.ones((H, W, 3), np.float32) * c
        g = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(0.9, 1.1), H)[:, None, None]
        return np.clip(im * g + np.random.normal(0, 3, im.shape), 0, 255).astype(np.uint8)
    base = np.array([rng.randint(60, 200) for _ in range(3)], np.float32)   # vân gỗ / vải
    noise = cv2.resize(np.random.rand(H // 8 + 1, 4).astype(np.float32), (W, H)) if rng.random() < 0.5 else \
        cv2.GaussianBlur(np.random.rand(H, W).astype(np.float32), (0, 0), 6)
    return np.clip(base * (0.75 + 0.5 * noise[..., None]), 0, 255).astype(np.uint8)


class Canvas:
    def __init__(self, img):
        self.img = img
        self.H, self.W = img.shape[:2]
        self.owner = np.full((self.H, self.W), -1, np.int32)
        self.items = []          # (cls, full_area)

    def paste(self, rgba, x, y, cls, shadow=True):
        """Dán rgba với góc trên-trái (x, y); cls=None nghĩa là vật không phải quả (không gán nhãn)."""
        h, w = rgba.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, self.W), min(y + h, self.H)
        if x1 <= x0 or y1 <= y0:
            return
        sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
        a = sub[..., 3:4].astype(np.float32) / 255
        if shadow:
            sh = cv2.GaussianBlur(a[..., 0], (0, 0), max(2, w / 40))
            dy = int(h * 0.04) + 1
            reg = self.img[y0:y1, x0:x1].astype(np.float32)
            shm = np.zeros_like(sh)
            shm[dy:] = sh[:-dy] if dy < sh.shape[0] else 0
            reg *= (1 - 0.35 * shm[..., None])
            self.img[y0:y1, x0:x1] = reg.astype(np.uint8)
        reg = self.img[y0:y1, x0:x1].astype(np.float32)
        self.img[y0:y1, x0:x1] = (reg * (1 - a) + sub[..., :3].astype(np.float32) * a).astype(np.uint8)
        mask = sub[..., 3] > 127
        idx = len(self.items)
        self.owner[y0:y1, x0:x1][mask] = idx if cls is not None else -2
        self.items.append((cls, int(mask.sum())))

    def labels(self):
        rows = []
        for i, (cls, full) in enumerate(self.items):
            if cls is None or full < 50:
                continue
            m = (self.owner == i).astype(np.uint8)
            vis = int(m.sum())
            if vis < 0.35 * full or vis < 60:
                continue
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            m = (lab == j).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cs, key=cv2.contourArea)
            if cls == 0:
                pts = cv2.boxPoints(cv2.minAreaRect(cnt))
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
            pts[:, 0] = np.clip(pts[:, 0] / self.W, 0, 1)
            pts[:, 1] = np.clip(pts[:, 1] / self.H, 0, 1)
            rows.append(f"{cls} " + " ".join(f"{v:.5f}" for v in pts.flatten()))
        return rows


def crown(rng, size):
    """Cuống nải: hình elip xanh-nâu (mẫu âm, không gán nhãn)."""
    w, h = int(size * rng.uniform(0.25, 0.4)), int(size * rng.uniform(0.25, 0.45))
    img = np.zeros((h, w, 4), np.uint8)
    col = (rng.randint(40, 90), rng.randint(110, 170), rng.randint(90, 140)) if rng.random() < 0.6 else \
        (rng.randint(60, 110), rng.randint(110, 150), rng.randint(150, 200))
    cv2.ellipse(img, (w // 2, h // 2), (w // 2 - 1, h // 2 - 1), 0, 0, 360, (*col, 255), -1)
    if rng.random() < 0.5:   # mặt cắt nâu
        cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, (60, 100, 150, 255), -1)
    return img


def banana_hand(cv, rng, pool, cx, cy, size):
    """Nải xòe: N quả toả ra từ cuống (cx, cy); size = chiều dài một quả (px)."""
    src = [banana_canonical(jitter(read_rgba(p), rng)) for p in rng.sample(pool, min(len(pool), rng.randint(1, 2)))]
    src = [s for s in src if s is not None and s.shape[1] > 10]
    if not src:
        return
    n = rng.randint(3, 20)
    rows = 2 if n >= 8 and rng.random() < 0.7 else 1
    up = rng.uniform(0, 360)                      # hướng chính của nải
    spread = rng.uniform(60, 200) if n >= 5 else rng.uniform(25, 70)
    places = []
    for r in range(rows):
        k = n // rows + (1 if r < n % rows else 0)
        for j in range(k):
            a = up - spread / 2 + spread * (j + 0.5 * (r % 2) + 0.25) / max(k, 1) + rng.uniform(-4, 4)
            places.append((r, a))
    rng.shuffle(places)
    places.sort(key=lambda t: -t[0])              # hàng sau vẽ trước
    for r, a in places:
        s = rng.choice(src)
        L = size * rng.uniform(0.85, 1.1) * (0.95 if r else 1.0)
        sc = L / s.shape[1]
        b = cv2.resize(s, (max(int(s.shape[1] * sc), 4), max(int(s.shape[0] * sc), 4)))
        if rng.random() < 0.5:
            b = b[:, ::-1]
        if rng.random() < 0.5:
            b = b[::-1]
        b = rotate(b, a)
        rad = L * (0.5 + 0.12 * r) + size * 0.08
        t = np.deg2rad(a)
        px, py = cx + rad * np.cos(t), cy - rad * np.sin(t)
        cv.paste(b, int(px - b.shape[1] / 2), int(py - b.shape[0] / 2), 0)
    if rng.random() < 0.6:
        c = crown(rng, size)
        cv.paste(c, int(cx - c.shape[1] / 2), int(cy - c.shape[0] / 2), None, shadow=False)


def banana_row(cv, rng, pool, cx, cy, size):
    """Chuối xếp gần song song (nhìn ngang nải / bày thành hàng)."""
    s0 = banana_canonical(jitter(read_rgba(rng.choice(pool)), rng))
    n = rng.randint(2, 9)
    base = rng.uniform(0, 180)
    step = size * rng.uniform(0.12, 0.35)
    for j in range(n):
        sc = size * rng.uniform(0.9, 1.1) / s0.shape[1]
        b = cv2.resize(s0, (max(int(s0.shape[1] * sc), 4), max(int(s0.shape[0] * sc), 4)))
        b = rotate(b, base + rng.uniform(-8, 8))
        t = np.deg2rad(base + 90)
        off = (j - n / 2) * step
        cv.paste(b, int(cx + off * np.cos(t) - b.shape[1] / 2), int(cy - off * np.sin(t) - b.shape[0] / 2), 0)


def round_pile(cv, rng, pools, cx, cy, size, kinds):
    n = rng.randint(3, 25)
    r_area = size * np.sqrt(n) * 0.5
    pts = []
    for _ in range(n):
        for _ in range(30):
            x, y = cx + rng.gauss(0, r_area * 0.6), cy + rng.gauss(0, r_area * 0.45)
            if all((x - a) ** 2 + (y - b) ** 2 > (size * 0.55) ** 2 for a, b in pts):
                break
        pts.append((x, y))
    pts.sort(key=lambda t: t[1])                  # dưới (y lớn) che trên
    cache = {}
    for x, y in pts:
        k = rng.choice(kinds)
        p = rng.choice(pools[k])
        if p not in cache or rng.random() < 0.3:
            cache[p] = read_rgba(p)
        f = cache[p]
        if f is None:
            continue
        f = jitter(f, rng)
        d = size * rng.uniform(0.85, 1.15)
        sc = d / max(f.shape[:2])
        f = rotate(cv2.resize(f, (max(int(f.shape[1] * sc), 4), max(int(f.shape[0] * sc), 4))), rng.uniform(-30, 30))
        cv.paste(f, int(x - f.shape[1] / 2), int(y - f.shape[0] / 2), CLS[k])


def make(rng, pools, bgs):
    W = rng.choice([640, 800, 960, 1024])
    H = int(W * rng.choice([0.75, 1.0, 0.66, 1.33]))
    cv = Canvas(background(rng, bgs, W, H))
    t = rng.random()
    m = min(W, H)
    if t < 0.4:                                   # nải chuối xòe
        banana_hand(cv, rng, pools["banana"], W * rng.uniform(0.35, 0.65), H * rng.uniform(0.35, 0.75), m * rng.uniform(0.25, 0.45))
    elif t < 0.55:                                # chuối song song
        banana_row(cv, rng, pools["banana"], W / 2, H / 2, m * rng.uniform(0.35, 0.6))
    elif t < 0.8:                                 # đống táo/cam
        kinds = rng.choice([["apple"], ["orange"], ["apple", "orange"]])
        round_pile(cv, rng, pools, W / 2, H / 2, m * rng.uniform(0.09, 0.2), kinds)
    else:                                         # trộn
        banana_hand(cv, rng, pools["banana"], W * rng.uniform(0.25, 0.4), H * rng.uniform(0.4, 0.7), m * rng.uniform(0.2, 0.35))
        round_pile(cv, rng, pools, W * rng.uniform(0.6, 0.75), H * rng.uniform(0.4, 0.7), m * rng.uniform(0.09, 0.16),
                   rng.choice([["apple"], ["orange"], ["apple", "orange"]]))
    img = cv.img
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.5:
        img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.2) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    return img, cv.labels()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OBB / "data" / "synth"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rng = random.Random(seed)
    np.random.seed(seed)
    pools = load_cutouts()
    print({k: len(v) for k, v in pools.items()})
    bgs = [str(p) for p in (ROOT / "work/exp/neg/train/images").glob("*.jpg")]
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    tot = 0
    for i in range(n):
        img, rows = make(rng, pools, bgs)
        cv2.imwrite(str(out / "images" / f"syn{seed}_{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
        (out / "labels" / f"syn{seed}_{i:05d}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        tot += len(rows)
    print(f"{n} ảnh, {tot} nhãn -> {out}")


if __name__ == "__main__":
    main()
