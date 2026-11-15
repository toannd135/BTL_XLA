"""Build a segmentation dataset with isolated per-fruit crowded banana examples.

The original SAM polygons are retained as source data. Synthetic images use the
alpha mask of each cutout as an independent instance, so no enclosing group
polygon can be introduced by augmentation.
"""
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OBB = ROOT / "work" / "obb"
SOURCE = OBB / "instances.jsonl"
OUT = OBB / "seg_data_clean"
CLS = {0: "banana", 1: "apple", 2: "orange"}


def copy_source(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("poly") and Path(row["image"]).exists():
            grouped[(row["split"], Path(row["image"]))].append(row)
    for (split, image), items in grouped.items():
        target_image = OUT / split / "images" / image.name
        target_label = OUT / split / "labels" / f"{image.stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target_image)
        lines = []
        for item in items:
            pts = item["poly"]
            if len(pts) < 3:
                continue
            coords = []
            for x, y in pts:
                coords.extend((np.clip(x / item["w"], 0, 1), np.clip(y / item["h"], 0, 1)))
            lines.append(" ".join([str(item["cls"]), *[f"{v:.6f}" for v in coords]]))
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""))


def load_bananas(rows):
    paths = []
    for row in rows:
        cutout = row.get("cutout")
        if row.get("cls") != 0 or not cutout:
            continue
        path = OBB / "cutouts" / "banana" / cutout
        if path.exists() and row.get("solidity", 0) >= 0.55 and row.get("fill", 0) >= 0.12:
            paths.append(path)
    return sorted(set(paths))


def paste(canvas, owner, rgba, x, y, item_id):
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return 0
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = src[..., 3] > 127
    region = canvas[y0:y1, x0:x1]
    a = (src[..., 3:4].astype(np.float32) / 255)
    canvas[y0:y1, x0:x1] = (region * (1 - a) + src[..., :3] * a).astype(np.uint8)
    owner[y0:y1, x0:x1][alpha] = item_id
    return int(alpha.sum())


def make_synthetic(paths, count=600, seed=42):
    rng = random.Random(seed)
    bg_files = [p for p in (ROOT / "own_neg").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    out_split = OUT / "train"
    (out_split / "images").mkdir(parents=True, exist_ok=True)
    (out_split / "labels").mkdir(parents=True, exist_ok=True)
    for index in range(count):
        width = height = 640
        if bg_files and rng.random() < 0.6:
            bg = cv2.imread(str(rng.choice(bg_files)))
            canvas = cv2.resize(bg, (width, height)) if bg is not None else np.full((height, width, 3), 235, np.uint8)
        else:
            canvas = np.full((height, width, 3), rng.randint(180, 250), np.uint8)
        owner = np.full((height, width), -1, np.int32)
        masks = []
        for item_id in range(rng.randint(3, 5)):
            rgba = cv2.imread(str(rng.choice(paths)), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                continue
            scale = rng.uniform(0.35, 0.65) * width / max(rgba.shape[:2])
            rgba = cv2.resize(rgba, (max(4, int(rgba.shape[1] * scale)), max(4, int(rgba.shape[0] * scale))))
            angle = rng.uniform(-35, 35)
            h, w = rgba.shape[:2]
            mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            rgba = cv2.warpAffine(rgba, mat, (w, h), borderValue=(0, 0, 0, 0))
            x = rng.randint(-40, width - max(4, rgba.shape[1]) + 40)
            y = rng.randint(-40, height - max(4, rgba.shape[0]) + 40)
            full = paste(canvas, owner, rgba, x, y, item_id)
            if full:
                masks.append((item_id, full))
        labels = []
        for item_id, full in masks:
            mask = (owner == item_id).astype(np.uint8)
            if mask.sum() < max(80, full * 0.35):
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
            if len(contour) < 3:
                continue
            points = contour.astype(np.float32)
            points[:, 0] /= width
            points[:, 1] /= height
            labels.append("0 " + " ".join(f"{v:.6f}" for v in points.ravel()))
        if labels:
            stem = f"synthetic_crowded_banana_{index:04d}"
            cv2.imwrite(str(out_split / "images" / f"{stem}.jpg"), canvas)
            (out_split / "labels" / f"{stem}.txt").write_text("\n".join(labels) + "\n")


def main():
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    if OUT.exists():
        shutil.rmtree(OUT)
    copy_source(rows)
    bananas = load_bananas(rows)
    if not bananas:
        raise RuntimeError("No usable banana cutouts found")
    make_synthetic(bananas)
    (OUT / "seg.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        "names:\n  0: banana\n  1: apple\n  2: orange\n"
        "source_note: original polygons plus independently masked synthetic bananas\n"
    )
    print(f"wrote {OUT} using {len(bananas)} banana cutouts")


if __name__ == "__main__":
    main()
