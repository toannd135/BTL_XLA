#!/usr/bin/env python3
"""Validate an extracted YOLO instance-segmentation dataset without modifying it."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from phase1_dataset_audit import polygon_self_intersects

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Extracted YOLO-seg dataset root")
    parser.add_argument("--yaml", type=Path, help="Dataset yaml; auto-discovered if omitted")
    parser.add_argument("--check", choices=("all", "structural", "topology"), default="all",
                        help="Run both checks, YOLO structure checks only, or polygon self-intersection detection only")
    args = parser.parse_args()
    root = args.dataset.resolve()
    config_path = args.yaml.resolve() if args.yaml else next(
        (p for p in (root / "data.yaml", root / "seg.yaml") if p.exists()), None)
    structural_problems, topology_problems = [], []
    counts = defaultdict(lambda: {"images": 0, "instances": 0, "negative_images": 0})
    def add_problem(message, category="structural"):
        target = topology_problems if category == "topology" else structural_problems
        if args.check == "all" or args.check == category:
            target.append(message)
    if config_path is None:
        raise SystemExit("No data.yaml or seg.yaml found")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    names = config.get("names", {})
    if isinstance(names, list):
        names = {i: x for i, x in enumerate(names)}
    else:
        names = {int(k): v for k, v in names.items()}
    if list(names.values()) != ["banana", "apple", "orange"]:
        add_problem(f"class order must be banana, apple, orange; found {names}")
    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if split_value is None:
            continue
        data_base = Path(config.get("path", "."))
        if not data_base.is_absolute():
            data_base = root / data_base
        split_path = Path(split_value)
        if not split_path.is_absolute():
            split_path = data_base / split_path
        split_path = split_path.resolve()
        image_labels = []
        if split_path.is_file() and split_path.suffix.lower() == ".txt":
            for line in split_path.read_text(encoding="utf-8").splitlines():
                value = Path(line.strip())
                if not line.strip():
                    continue
                image_path = value if value.is_absolute() else (root / value).resolve()
                image_labels.append((image_path, None))
        elif split_path.is_dir():
            images = sorted(p for p in split_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            for image_path in images:
                parts = list(image_path.parts)
                if "images" in parts:
                    idx = len(parts) - 1 - parts[::-1].index("images")
                    parts[idx] = "labels"
                    label_path = Path(*parts).with_suffix(".txt")
                else:
                    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
                image_labels.append((image_path, label_path))
        else:
            add_problem(f"{split}: image directory/list missing: {split_path}")
            continue
        if not image_labels:
            add_problem(f"{split}: no images found")
        image_stems = {image.stem for image, _ in image_labels}
        expected_label_paths = {label for _, label in image_labels if label is not None}
        discovered_labels = set()
        for candidate_root in (root / "labels" / split, root / split / "labels", split_path.parent / "labels"):
            if candidate_root.exists():
                discovered_labels.update(candidate_root.glob("*.txt"))
        for label in discovered_labels - expected_label_paths:
            if label.stem not in image_stems:
                add_problem(f"{split}: label without image: {label}")
        for image, label in image_labels:
            counts[split]["images"] += 1
            if label is None:
                # The images/<split> export layout is paired with labels/<split>.
                label = root / "labels" / split / f"{image.stem}.txt"
            if not label.exists():
                add_problem(f"{split}: image without label file: {image}")
                continue
            frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if frame is None:
                add_problem(f"{split}/{image.name}: unreadable image")
                continue
            h, w = frame.shape[:2]
            masks, seen = [], set()
            lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                counts[split]["negative_images"] += 1
            for line_no, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) < 7 or (len(parts) - 1) % 2:
                    add_problem(f"{split}/{label.name}:{line_no}: malformed segmentation row")
                    continue
                try:
                    cls = int(parts[0]); pts = np.asarray([float(x) for x in parts[1:]], dtype=np.float64).reshape(-1, 2)
                except Exception:
                    add_problem(f"{split}/{label.name}:{line_no}: non-numeric row")
                    continue
                if cls not in names:
                    add_problem(f"{split}/{label.name}:{line_no}: unknown class id {cls}")
                if len(pts) < 3 or not np.isfinite(pts).all() or (pts < 0).any() or (pts > 1).any():
                    add_problem(f"{split}/{label.name}:{line_no}: invalid normalized polygon")
                    continue
                if polygon_self_intersects(pts):
                    add_problem(f"{split}/{label.name}:{line_no}: self-intersecting polygon", "topology")
                pixel = np.rint(pts * np.array([w - 1, h - 1])).astype(np.int32)
                mask = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mask, [pixel], 1)
                if mask.sum() == 0 or cv2.contourArea(pixel) <= 0:
                    add_problem(f"{split}/{label.name}:{line_no}: empty/zero-area mask")
                token = (cls, tuple(np.round(pts, 7).reshape(-1)))
                if token in seen:
                    add_problem(f"{split}/{label.name}:{line_no}: exact duplicate polygon")
                seen.add(token)
                for prev_cls, prev_mask, prev_idx in masks:
                    if cls != prev_cls:
                        continue
                    inter = int(np.logical_and(mask, prev_mask).sum())
                    union = int(np.logical_or(mask, prev_mask).sum())
                    if union and inter / union >= .98:
                        add_problem(f"{split}/{label.name}:{line_no}: duplicate mask IoU>=0.98 with row {prev_idx}")
                masks.append((cls, mask, line_no))
                counts[split]["instances"] += 1
    problems = structural_problems + topology_problems
    print(json.dumps({"dataset": str(root), "yaml": str(config_path), "check": args.check, "names": names,
                      "counts": dict(counts), "structural_problem_count": len(structural_problems),
                      "topology_problem_count": len(topology_problems), "problem_count": len(problems),
                      "problems": problems[:500]}, indent=2))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
