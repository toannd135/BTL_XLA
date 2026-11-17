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
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from detection import detect, detect_obb, detect_seg


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
    parser.add_argument("--max-images", type=int, default=100)
    return parser.parse_args()


def load_ground_truth(label_path):
    counts = Counter()
    total = 0
    if not label_path.exists():
        return total, counts
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


def assess_uncertainty(objects, width, height, conf_threshold):
    reasons = []
    image_area = max(width * height, 1)
    for idx, obj in enumerate(objects):
        lbl = obj["label"]
        x1, y1, x2, y2 = obj["bbox"]
        area = max(x2 - x1, 0) * max(y2 - y1, 0)
        conf = obj.get("confidence", 1.0)
        if conf < conf_threshold:
            reasons.append(f"khung #{idx+1} ({lbl}) điểm thấp ({conf:.2f})")
        if area / image_area > 0.60:
            reasons.append(f"khung #{idx+1} quá lớn ({area/image_area:.0%})")
        solidity = obj.get("solidity", 1.0)
        if obj.get("polygon") and len(obj["polygon"]) >= 5:
            if lbl in ("apple", "orange") and solidity < 0.65:
                reasons.append(f"khung #{idx+1} độ lồi bất thường ({solidity:.2f})")
            elif lbl == "banana" and solidity < 0.40:
                reasons.append(f"khung #{idx+1} độ lồi bất thường ({solidity:.2f})")
        if obj.get("occlusion_ratio", 0.0) >= 0.55:
            reasons.append(f"khung #{idx+1} che khuất nhiều ({obj['occlusion_ratio']:.0%})")

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            oi, oj = objects[i], objects[j]
            xi1, yi1, xi2, yi2 = oi["bbox"]
            xj1, yj1, xj2, yj2 = oj["bbox"]
            inter = max(0, min(xi2, xj2) - max(xi1, xj1)) * max(0, min(yi2, yj2) - max(yi1, yj1))
            area_i = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
            area_j = max(xj2 - xj1, 0) * max(yi2 - yi1, 0)
            union = area_i + area_j - inter
            iou = inter / max(union, 1e-9)
            if oi["label"] == oj["label"] and iou >= 0.40:
                reasons.append(f"trùng lặp khung #{i+1} và #{j+1}")

    unique = list(dict.fromkeys(reasons))
    return {"status": "needs_review" if unique else "ready", "reasons": unique}


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
        images = images[:args.max_images]

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
                    "label": LABELS.get(c, f"class_{c}"),
                    "confidence": float(scores[k]),
                    "bbox": boxes[k].tolist(),
                    "polygon": polys[k].tolist() if k < len(polys) else None,
                    "solidity": mask_metrics[k]["solidity"] if k < len(mask_metrics) else 1.0,
                    "occlusion_ratio": mask_metrics[k]["occlusion_ratio"] if k < len(mask_metrics) else 0.0,
                })
        elif task == "obb":
            polys, boxes, scores, classes = detect_obb(model, img, conf=args.conf, imgsz=args.imgsz)
            for k in range(len(boxes)):
                c = int(classes[k])
                objects.append({
                    "label": LABELS.get(c, f"class_{c}"),
                    "confidence": float(scores[k]),
                    "bbox": boxes[k].tolist(),
                    "polygon": polys[k].tolist() if k < len(polys) else None,
                })
        else:
            boxes, scores, classes = detect(model, img, conf=args.conf, imgsz=args.imgsz)
            for k in range(len(boxes)):
                c = int(classes[k])
                objects.append({
                    "label": LABELS.get(c, f"class_{c}"),
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

        quality = assess_uncertainty(objects, w, h, args.conf)
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
    args.output.write_text(json.dumps({"summary": benchmark_summary, "details": results}, indent=2, ensure_ascii=False))
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
