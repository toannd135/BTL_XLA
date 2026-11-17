"""Segmentation Dataset Builder & Quality Auditor.

Audits and builds YOLO instance segmentation datasets adhering to production requirements:
1. Multi-fruit crowded scenes (300-500 images)
2. Occluded / overlapping instances (100-200 images)
3. Banana clusters vs individual fruit (100 images)
4. Negative background images with 0 fruit (100 images)
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


LABELS = {0: "banana", 1: "apple", 2: "orange"}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit and prepare segmentation datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("work/obb/seg_data_clean"))
    parser.add_argument("--output-report", type=Path, default=Path("work/audit/seg_dataset_audit.json"))
    return parser.parse_args()


def validate_polygon(points):
    """Check if polygon coordinates form a valid geometric area."""
    if len(points) < 3:
        return False, "polygon has fewer than 3 points"
    pts = np.asarray(points, dtype=np.float32)
    if np.any(pts < 0.0) or np.any(pts > 1.0):
        return False, "coordinates outside normalized [0, 1] range"
    area = cv2.contourArea(pts.reshape(-1, 1, 2))
    if area <= 1e-7:
        return False, "zero or negative polygon area"
    return True, "valid"


def parse_label_file(label_path):
    instances = []
    if not label_path.exists():
        return instances
    for line_idx, line in enumerate(label_path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:  # class + at least 3 points (x, y) = 7 tokens
            continue
        try:
            cls_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            if len(coords) % 2 != 0:
                continue
            pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            valid, msg = validate_polygon(pts)
            instances.append({
                "class_id": cls_id,
                "label": LABELS.get(cls_id, f"class_{cls_id}"),
                "points": pts,
                "valid": valid,
                "validation_msg": msg,
            })
        except ValueError:
            continue
    return instances


def audit_dataset(data_dir):
    stats = {
        "splits": {},
        "total_images": 0,
        "total_instances": 0,
        "class_counts": Counter(),
        "categories": {
            "mixed_multi_fruit": 0,
            "crowded_instances": 0,
            "banana_clusters": 0,
            "negatives_background": 0,
        },
        "target_progress": {
            "mixed_multi_fruit": {"target": "300-500", "current": 0, "status": "pending"},
            "crowded_instances": {"target": "100-200", "current": 0, "status": "pending"},
            "banana_clusters": {"target": "100", "current": 0, "status": "pending"},
            "negatives_background": {"target": "100", "current": 0, "status": "pending"},
        },
        "invalid_polygons": 0,
    }

    for split in ("train", "val", "test"):
        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"
        if not img_dir.exists():
            continue
        images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        split_stats = {
            "images": len(images),
            "instances": 0,
            "classes": Counter(),
            "empty_images": 0,
        }

        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            instances = parse_label_file(lbl_path)
            cnt = len(instances)
            split_stats["instances"] += cnt
            stats["total_instances"] += cnt
            classes_in_img = set(inst["class_id"] for inst in instances)

            for inst in instances:
                split_stats["classes"][inst["label"]] += 1
                stats["class_counts"][inst["label"]] += 1
                if not inst["valid"]:
                    stats["invalid_polygons"] += 1

            if cnt == 0:
                split_stats["empty_images"] += 1
                stats["categories"]["negatives_background"] += 1
            if len(classes_in_img) >= 2:
                stats["categories"]["mixed_multi_fruit"] += 1
            if cnt >= 4:
                stats["categories"]["crowded_instances"] += 1
            if sum(1 for inst in instances if inst["class_id"] == 0) >= 3:
                stats["categories"]["banana_clusters"] += 1

        stats["splits"][split] = {
            "images": split_stats["images"],
            "instances": split_stats["instances"],
            "classes": dict(split_stats["classes"]),
            "empty_images": split_stats["empty_images"],
        }
        stats["total_images"] += len(images)

    # Progress assessment towards production dataset targets
    stats["target_progress"]["mixed_multi_fruit"]["current"] = stats["categories"]["mixed_multi_fruit"]
    stats["target_progress"]["crowded_instances"]["current"] = stats["categories"]["crowded_instances"]
    stats["target_progress"]["banana_clusters"]["current"] = stats["categories"]["banana_clusters"]
    stats["target_progress"]["negatives_background"]["current"] = stats["categories"]["negatives_background"]

    for k in stats["target_progress"]:
        cur = stats["target_progress"][k]["current"]
        tgt = stats["target_progress"][k]["target"]
        min_val = int(tgt.split("-")[0])
        stats["target_progress"][k]["status"] = "achieved" if cur >= min_val else f"needs {min_val - cur} more"

    return stats


def main():
    args = parse_args()
    if not args.data_dir.exists():
        print(f"Data directory {args.data_dir} does not exist.")
        return
    print(f"Auditing dataset at: {args.data_dir}...")
    stats = audit_dataset(args.data_dir)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Audit completed. Report saved to: {args.output_report}")
    print("\n=== DATASET AUDIT SUMMARY ===")
    print(f"Total images: {stats['total_images']}, Total instances: {stats['total_instances']}")
    print(f"Class distribution: {dict(stats['class_counts'])}")
    print("\n--- Production Target Progress ---")
    for k, v in stats["target_progress"].items():
        print(f"  * {k:22s}: current={v['current']:4d} / target={v['target']:8s} -> {v['status']}")


if __name__ == "__main__":
    main()
