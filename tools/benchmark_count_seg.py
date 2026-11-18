"""Specialized Count-oriented Benchmark for Instance Segmentation & Detection.

Measures count accuracy metrics instead of relying solely on mAP:
1. Exact Count Accuracy (ECA)
2. Mean Absolute Error (MAE) & RMSE
3. Over-count Rate (OCR) & Under-count Rate (UCR)
4. Per-class Count MAE
5. Quality Gate Safety Rate: % of miscounted images flagged as 'needs_review'
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from detection import detect, detect_obb, detect_seg
from quality import assess_detection


LABELS = {0: "banana", 1: "apple", 2: "orange"}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark fruit counting performance.")
    parser.add_argument("--model", type=Path, default=ROOT / "yolo11s-seg.pt",
                        help="Path to YOLO model (seg, obb, or hbb)")
    parser.add_argument("--data-split", type=Path, default=ROOT / "work/obb/seg_data_clean/val",
                        help="Path to split directory containing images/ and labels/")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=ROOT / "work/audit/benchmark_count_results.json")
    parser.add_argument("--max-images", type=int, default=0, help="0 evaluates the entire split")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_ground_truth(label_path):
    counts = Counter()
    total = 0
    if not label_path.exists():
        raise FileNotFoundError(f"Missing ground truth (not a verified negative image): {label_path}")
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            lbl = LABELS.get(cls_id, f"class_{cls_id}")
            counts[lbl] += 1
            total += 1
    return total, counts


def main():
    args = parse_args()
    model_path = args.model
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}...")
    model = YOLO(str(model_path))
    task = model.task
    print(f"Model task: {task}")

    img_dir = args.data_split / "images"
    lbl_dir = args.data_split / "labels"
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if args.max_images:
        images = sorted(random.Random(args.seed).sample(images, min(args.max_images, len(images))))

    print(f"Running count benchmark on {len(images)} images (conf={args.conf})...")

    results = []
    exact_count_hits = 0
    total_abs_error = 0
    over_count_hits = 0
    under_count_hits = 0
    miscounted_caught_by_gate = 0
    total_miscounted = 0
    class_errors = defaultdict(int)

    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        true_total, true_counts = load_ground_truth(lbl_path)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        objects = []
        if task == "segment":
            polys, boxes, scores, classes, mask_metrics = detect_seg(
                model, img, conf=args.conf, imgsz=args.imgsz,
            )
            for k in range(len(boxes)):
                c = int(classes[k])
                objects.append({
                    "label": model.names[c],
                    "confidence": float(scores[k]),
                    "bbox": boxes[k].tolist(),
                    "polygon": polys[k].tolist() if k < len(polys) else None,
                    **mask_metrics[k],
                    "mask_overlaps": mask_metrics[k]["overlaps"],
                })
        elif task == "obb":
            polys, boxes, scores, classes = detect_obb(model, img, conf=args.conf, imgsz=args.imgsz)
            for k in range(len(boxes)):
                c = int(classes[k])
                objects.append({
                    "label": model.names[c],
                    "confidence": float(scores[k]),
                    "bbox": boxes[k].tolist(),
                    "polygon": polys[k].tolist() if k < len(polys) else None,
                })
        else:
            boxes, scores, classes = detect(model, img, conf=args.conf, imgsz=args.imgsz)
            for k in range(len(boxes)):
                c = int(classes[k])
                objects.append({
                    "label": model.names[c],
                    "confidence": float(scores[k]),
                    "bbox": boxes[k].tolist(),
                })

        pred_total = len(objects)
        pred_counts = Counter(o["label"] for o in objects)
        abs_err = abs(pred_total - true_total)
        total_abs_error += abs_err

        is_exact = (pred_total == true_total)
        if is_exact:
            exact_count_hits += 1
        elif pred_total > true_total:
            over_count_hits += 1
        else:
            under_count_hits += 1

        quality = assess_detection(objects, w, h, conf_threshold=args.conf)
        is_flagged = (quality["status"] == "needs_review")

        if not is_exact:
            total_miscounted += 1
            if is_flagged:
                miscounted_caught_by_gate += 1

        for c_lbl in ("banana", "apple", "orange"):
            class_errors[c_lbl] += abs(pred_counts[c_lbl] - true_counts[c_lbl])

        results.append({
            "image": img_path.name,
            "true_total": true_total,
            "pred_total": pred_total,
            "abs_error": abs_err,
            "is_exact": is_exact,
            "flagged_needs_review": is_flagged,
            "reasons": quality["reasons"],
        })

    n = len(results)
    benchmark_summary = {
        "model": str(model_path.name),
        "task": task,
        "pipeline": "detector_only",
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "sample_seed": args.seed,
        "retina_masks": task == "segment",
        "total_images": n,
        "exact_count_accuracy": round(exact_count_hits / n, 4) if n else 0.0,
        "count_mae": round(total_abs_error / n, 4) if n else 0.0,
        "over_count_rate": round(over_count_hits / n, 4) if n else 0.0,
        "under_count_rate": round(under_count_hits / n, 4) if n else 0.0,
        "per_class_mae": {c: round(class_errors[c] / n, 4) for c in class_errors},
        "quality_gate_safety_rate": round(miscounted_caught_by_gate / total_miscounted, 4) if total_miscounted else 1.0,
        "total_miscounted": total_miscounted,
        "miscounted_caught": miscounted_caught_by_gate,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": benchmark_summary, "details": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== COUNT BENCHMARK RESULTS ({model_path.name}) ===")
    print(f"Exact Count Accuracy (ECA) : {benchmark_summary['exact_count_accuracy'] * 100:.2f}%")
    print(f"Count MAE                  : {benchmark_summary['count_mae']:.3f} fruits/image")
    print(f"Over-count Rate (OCR)      : {benchmark_summary['over_count_rate'] * 100:.2f}%")
    print(f"Under-count Rate (UCR)     : {benchmark_summary['under_count_rate'] * 100:.2f}%")
    print(f"Per-Class MAE              : {benchmark_summary['per_class_mae']}")
    print(f"Quality Gate Safety Rate   : {benchmark_summary['quality_gate_safety_rate'] * 100:.2f}% "
          f"({miscounted_caught_by_gate}/{total_miscounted} errors caught)")
    print(f"Saved full report to: {args.output}")


if __name__ == "__main__":
    main()
