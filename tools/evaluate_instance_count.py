"""Measure per-fruit counting and localization against individually reviewed labels.

Matching uses maximum-cardinality, class-aware IoU matching, not AP. Equal counts
alone do not pass the perfect-instance check when boxes are duplicated or missing.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

from instance_review import sha256, validate_boxes


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return intersection / max(area_a+area_b-intersection, 1e-12)


def match_instances(truth, predictions, threshold):
    """Each input is (class, x1, y1, x2, y2); match each instance at most once."""
    edges = []
    for pred in predictions:
        candidates = [(overlap(pred[1:], gt[1:]), i) for i, gt in enumerate(truth) if pred[0] == gt[0]]
        edges.append([i for value, i in sorted(candidates, reverse=True) if value >= threshold])
    owner = {}

    def assign(prediction, visited):
        for target in edges[prediction]:
            if target in visited:
                continue
            visited.add(target)
            if target not in owner or assign(owner[target], visited):
                owner[target] = prediction
                return True
        return False

    for index in range(len(predictions)):
        assign(index, set())
    tp = len(owner)
    return {"tp": tp, "fp": len(predictions)-tp, "fn": len(truth)-tp,
            "perfect_instances": tp == len(truth) == len(predictions)}


def score_image(truth, predictions):
    actual, predicted = Counter(x[0] for x in truth), Counter(x[0] for x in predictions)
    return {"true_count": len(truth), "predicted_count": len(predictions),
            "absolute_count_error": abs(len(truth)-len(predictions)),
            "exact_count": len(truth) == len(predictions),
            "per_class_absolute_error": {c: abs(actual[c]-predicted[c]) for c in (0, 1, 2)},
            "iou_50": match_instances(truth, predictions, 0.5),
            "iou_75": match_instances(truth, predictions, 0.75)}


def summarize(rows):
    if not rows:
        raise ValueError("No approved evaluation images; review labels before evaluating")
    total = len(rows)
    result = {"images": total, "count_mae": sum(r["absolute_count_error"] for r in rows)/total,
              "exact_count_rate": sum(r["exact_count"] for r in rows)/total,
              "per_class_mae": {c: sum(r["per_class_absolute_error"][c] for r in rows)/total for c in (0, 1, 2)}}
    for key in ("iou_50", "iou_75"):
        sums = {metric: sum(r[key][metric] for r in rows) for metric in ("tp", "fp", "fn")}
        tp, fp, fn = sums["tp"], sums["fp"], sums["fn"]
        result[key] = {**sums, "precision": tp/(tp+fp) if tp+fp else None,
                       "recall": tp/(tp+fn) if tp+fn else None,
                       "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                       "perfect_instances_rate": sum(r[key]["perfect_instances"] for r in rows)/total}
    return result


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=repo / "out/instance_review_v1")
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=repo / "exports/det_C_candidate.pt")
    parser.add_argument("--roles", nargs="+", choices=("diagnostic", "val", "test"), default=["diagnostic", "val"])
    parser.add_argument("--output", type=Path, default=repo / "out/instance_count_baseline.json")
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        raise ValueError("Confidence must be greater than zero and at most one")
    if args.output.exists():
        raise ValueError("Report exists; choose another output name")
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8-sig"))
    if reviews.get("schema") != 1 or reviews.get("dataset_id") != manifest["dataset_id"]:
        raise ValueError("Reviews belong to another dataset")
    catalog = {x["id"]: x for x in manifest["items"]}
    selected, seen = [], set()
    for review in reviews["reviews"]:
        identifier = review.get("id")
        if identifier not in catalog or identifier in seen:
            raise ValueError("Unknown or duplicate review id")
        seen.add(identifier)
        item = catalog[identifier]
        if review.get("status") != "approved" or item["role"] not in args.roles:
            continue
        boxes = validate_boxes(review.get("boxes"))
        count = review.get("expected_count")
        if isinstance(count, bool) or not isinstance(count, int) or count != len(boxes):
            raise ValueError("Manual count differs from approved boxes")
        image = (args.bundle / item["image"]).resolve()
        if not image.is_relative_to(args.bundle.resolve()) or sha256(image) != item["display_image_sha256"]:
            raise ValueError("Review image missing or changed")
        selected.append((item, image, boxes))
    if not selected:
        raise ValueError("No approved images in requested roles; review them first")
    if not args.model.is_file():
        raise ValueError("Supply an existing local model")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    sys.path.insert(0, str(repo / "backend"))
    from detection import detect
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    model = YOLO(str(args.model))
    if {int(k): v for k, v in model.names.items()} != {0: "banana", 1: "apple", 2: "orange"}:
        raise ValueError("Unexpected model classes")
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), device="cpu", verbose=False)
    torch.set_num_threads(2)
    rows = []
    for item, image_path, boxes in selected:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        predicted, scores, classes = detect(model, image, conf=args.conf, imgsz=960, rotated=True)
        truth = [[c, x-w/2, y-h/2, x+w/2, y+h/2] for c, x, y, w, h in boxes]
        predictions = [[int(c), x1/width, y1/height, x2/width, y2/height] for (x1, y1, x2, y2), c in zip(predicted, classes)]
        row = {"image": item["source_image"], "role": item["role"], **score_image(truth, predictions),
               "predictions": predictions, "scores": scores.tolist()}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("image", "true_count", "predicted_count", "absolute_count_error")}), flush=True)
    report = {"model": str(args.model), "model_sha256": sha256(args.model), "dataset_id": manifest["dataset_id"],
              "reviews_sha256": sha256(args.reviews), "postprocess_sha256": sha256(repo / "backend/detection.py"),
              "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
              "conf": args.conf, "imgsz": 960, "iou_nms": 0.6, "mode": "careful", "device": "cpu", "cpu_threads": 2,
              "summary": summarize(rows), "by_role": {role: summarize([r for r in rows if r["role"] == role])
                                                       for role in args.roles if any(r["role"] == role for r in rows)},
              "rows": rows, "limitation": "Diagnostic images already used during development are not an independent final test."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
