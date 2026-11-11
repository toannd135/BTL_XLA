"""Chạy SAM2 (prompt = box nhãn gốc) cho mọi box của train (danh sách C), val, test.
Ghi work/obb/instances.jsonl: mỗi box -> polygon mặt nạ, khung xoay (minAreaRect), chỉ số hình dạng.
Lưu ảnh cắt RGBA của quả vào work/obb/cutouts/<class>/ (chỉ từ train, để ghép ảnh tổng hợp).
Không sửa nhãn gốc. Dùng: python work/obb/sam_instances.py"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

ROOT = Path(__file__).resolve().parents[2]
DET = ROOT / "local_dataset" / "det"
OUT = ROOT / "work" / "obb"
NAMES = {0: "banana", 1: "apple", 2: "orange"}


def label_path(img):
    return img.parent.parent / "labels" / f"{img.stem}.txt"


def read_boxes(lp, w, h):
    rows = []
    if not lp.exists():
        return rows
    for r in lp.read_text().splitlines():
        v = r.split()
        if not v:
            continue
        c, p = int(v[0]), list(map(float, v[1:]))
        if len(p) == 4:
            cx, cy, bw, bh = p
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        else:                                   # polygon cũ -> box bao
            xs, ys = p[::2], p[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        rows.append((c, [x1 * w, y1 * h, x2 * w, y2 * h]))
    return rows


def main():
    sam = SAM(str(ROOT / "work" / "weights" / "sam2.1_b.pt"))
    train = [Path(p) for p in (ROOT / "work/exp/det_C_neg_train.txt").read_text().split()]
    seen, jobs = set(), []
    for p in train:
        if p not in seen and "/neg/" not in str(p):
            seen.add(p)
            jobs.append(("train", p))
    for sp in ("val", "test"):
        jobs += [(sp, p) for p in sorted((DET / sp / "images").iterdir())]
    for c in NAMES.values():
        (OUT / "cutouts" / c).mkdir(parents=True, exist_ok=True)
    f = open(OUT / "instances.jsonl", "w")
    for k, (sp, p) in enumerate(jobs):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = read_boxes(label_path(p), w, h)
        if not boxes:
            continue
        r = sam(img, bboxes=[b for _, b in boxes], verbose=False)[0]
        masks = r.masks.data.cpu().numpy() if r.masks is not None else []
        for i, (c, b) in enumerate(boxes):
            rec = {"split": sp, "image": str(p), "i": i, "cls": c, "box": [round(v, 1) for v in b], "w": w, "h": h}
            if i < len(masks):
                m = cv2.resize(masks[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                n, lab, st, _ = cv2.connectedComponentsWithStats(m)
                if n > 1:
                    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                    m = (lab == j).astype(np.uint8)
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cs:
                    cnt = max(cs, key=cv2.contourArea)
                    area = float(cv2.contourArea(cnt))
                    hull = float(cv2.contourArea(cv2.convexHull(cnt))) or 1.0
                    (cx, cy), (rw, rh), ang = cv2.minAreaRect(cnt)
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    rec.update(obb=cv2.boxPoints(((cx, cy), (rw, rh), ang)).round(1).tolist(),
                               fill=area / max(bw * bh, 1), solidity=area / hull,
                               aspect=max(rw, rh) / max(min(rw, rh), 1), rect_fill=area / max(rw * rh, 1),
                               poly=cv2.approxPolyDP(cnt, 1.5, True).reshape(-1, 2).tolist())
                    if sp == "train" and min(bw, bh) >= 40:
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        rgba = np.dstack([img, m * 255])[y:y + ch, x:x + cw]
                        name = f"{p.stem}_{i}.png"
                        cv2.imwrite(str(OUT / "cutouts" / NAMES[c] / name), rgba)
                        rec["cutout"] = name
            f.write(json.dumps(rec) + "\n")
        if k % 200 == 0:
            print(k, "/", len(jobs), flush=True)
    f.close()
    print("xong")


if __name__ == "__main__":
    main()
