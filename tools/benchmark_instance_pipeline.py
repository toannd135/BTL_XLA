"""Benchmark the production-style segmentation refinement pipeline.

This evaluates count behavior after OBB proposals, SAM refinement, crop
verification, and final mask deduplication. It intentionally does not run the
classifier or CLIP gate because those stages do not change instance geometry.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM, YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from detection import (compute_mask_metrics, configure_sam_refiner, detect_seg,
                       split_fruit_cluster_detections)
from quality import assess_detection


LABELS = {0: "banana", 1: "apple", 2: "orange"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "yolo11s-seg.pt")
    parser.add_argument("--obb-model", type=Path)
    parser.add_argument("--sam-model", type=Path)
    parser.add_argument("--data-split", type=Path, default=ROOT / "local_dataset/det/val")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "work/audit/instance_pipeline_results.json")
    return parser.parse_args()


def load_ground_truth(path):
    counts = Counter()
    if not path.exists():
        raise FileNotFoundError(f"Missing ground truth: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            counts[LABELS.get(int(parts[0]), f"class_{parts[0]}")] += 1
    return sum(counts.values()), counts


def main():
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if args.obb_model and not args.obb_model.is_file():
        raise FileNotFoundError(f"OBB model not found: {args.obb_model}")
    if args.sam_model and not args.sam_model.is_file():
        raise FileNotFoundError(f"SAM model not found: {args.sam_model}")

    model = YOLO(str(args.model))
    obb = YOLO(str(args.obb_model)) if args.obb_model else None
    sam = configure_sam_refiner(SAM(str(args.sam_model))) if args.sam_model else None
    image_dir = args.data_split / "images"
    label_dir = args.data_split / "labels"
    images = sorted(p for p in image_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if args.max_images:
        images = sorted(random.Random(args.seed).sample(images, min(args.max_images, len(images))))

    rows = []
    exact = over = under = absolute_error = caught = miscounted = 0
    for image_path in images:
        true_total, true_counts = load_ground_truth(label_dir / f"{image_path.stem}.txt")
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode image: {image_path}")
        height, width = image.shape[:2]
        polygons, boxes, scores, classes, _ = detect_seg(
            model, image, conf=args.conf, imgsz=args.imgsz, compute_metrics=False,
        )
        diagnostics = {}
        polygons, boxes, scores, classes = split_fruit_cluster_detections(
            polygons, boxes, scores, classes, obb, image,
            conf=args.conf, det_names=model.names, mask_model=sam,
            diagnostics=diagnostics, seg_model=model,
        )
        metrics = compute_mask_metrics(polygons, boxes, width, height)
        objects = [{
            "label": model.names[int(cls)],
            "confidence": float(score),
            "bbox": box.tolist(),
            **metric,
            "mask_overlaps": metric["overlaps"],
        } for box, score, cls, metric in zip(boxes, scores, classes, metrics)]
        predicted_counts = Counter(obj["label"] for obj in objects)
        predicted_total = len(objects)
        error = abs(predicted_total - true_total)
        quality = assess_detection(objects, width, height, diagnostics, conf_threshold=args.conf)
        exact += predicted_total == true_total
        over += predicted_total > true_total
        under += predicted_total < true_total
        absolute_error += error
        if predicted_total != true_total:
            miscounted += 1
            caught += quality["status"] == "needs_review"
        rows.append({
            "image": image_path.name,
            "true_total": true_total,
            "predicted_total": predicted_total,
            "absolute_count_error": error,
            "true_counts": dict(true_counts),
            "predicted_counts": dict(predicted_counts),
            "flagged_needs_review": quality["status"] == "needs_review",
            "diagnostics": diagnostics,
        })

    total = len(rows)
    summary = {
        "model": str(args.model),
        "obb_model": str(args.obb_model) if args.obb_model else None,
        "sam_model": str(args.sam_model) if args.sam_model else None,
        "pipeline": "segmentation_refinement",
        "images": total,
        "exact_count_accuracy": exact / total if total else 0.0,
        "count_mae": absolute_error / total if total else 0.0,
        "over_count_rate": over / total if total else 0.0,
        "under_count_rate": under / total if total else 0.0,
        "quality_gate_safety_rate": caught / miscounted if miscounted else 1.0,
        "miscounted": miscounted,
        "miscounted_caught": caught,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "details": rows},
                                      indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
