"""Reproducible same-split comparison for fruit YOLO-Seg checkpoints."""

import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from detection import detect_seg, split_banana_masks_by_distance
from runtime_memory import memory_snapshot, reset_peak_memory, resolve_device

FRUITS = ("banana", "apple", "orange")
COCO_TO_FRUIT = {46: "banana", 47: "apple", 49: "orange"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ground_truth(path, width, height):
    instances = []
    if not path.exists():
        return instances
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        try:
            class_id = int(values[0])
            coords = np.asarray([float(x) for x in values[1:]], dtype=np.float32)
        except ValueError:
            continue
        if coords.size % 2:
            continue
        points = coords.reshape(-1, 2) * np.asarray([width, height], dtype=np.float32)
        if len(points) >= 3 and np.isfinite(points).all():
            instances.append({"label": FRUITS[class_id] if 0 <= class_id < len(FRUITS) else f"class_{class_id}",
                              "polygon": points})
    return instances


def polygon_mask(polygon, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask


def mask_iou(a, b):
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def match_instances(gt, predictions, width, height, threshold=0.50):
    gt_masks = [polygon_mask(x["polygon"], width, height) for x in gt]
    pred_masks = [polygon_mask(x["polygon"], width, height) for x in predictions]
    matches, gt_used, pred_used = [], set(), set()
    pairs = []
    for i, truth in enumerate(gt):
        for j, pred in enumerate(predictions):
            if truth["label"] == pred["label"]:
                pairs.append((mask_iou(gt_masks[i], pred_masks[j]), i, j))
    for overlap, i, j in sorted(pairs, reverse=True):
        if overlap < threshold:
            break
        if i not in gt_used and j not in pred_used:
            matches.append((i, j, overlap))
            gt_used.add(i)
            pred_used.add(j)

    # A merged prediction covers at least 25% of two distinct same-class GT masks.
    merged = set()
    for j, pred in enumerate(predictions):
        supported = 0
        pred_area = max(int(pred_masks[j].sum()), 1)
        for i, truth in enumerate(gt):
            if truth["label"] != pred["label"]:
                continue
            gt_area = max(int(gt_masks[i].sum()), 1)
            inter = int(np.count_nonzero(pred_masks[j] & gt_masks[i]))
            if inter / gt_area >= 0.25 and inter / pred_area >= 0.10:
                supported += 1
        if supported >= 2:
            merged.add(j)

    # Duplicates are additional same-class prediction masks with IoU >= 0.70.
    duplicate_ids = set()
    for j in range(len(predictions)):
        for k in range(j + 1, len(predictions)):
            if predictions[j]["label"] == predictions[k]["label"] and mask_iou(pred_masks[j], pred_masks[k]) >= 0.70:
                duplicate_ids.add(k)
    return matches, gt_used, pred_used, merged, duplicate_ids


def split_manifest(data_split):
    image_dir = data_split / "images"
    label_dir = data_split / "labels"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    rows, by_image = [], {}
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        row = {"image": image.relative_to(data_split).as_posix(),
               "image_sha256": sha256_file(image),
               "label": label.relative_to(data_split).as_posix(),
               "label_sha256": sha256_file(label) if label.exists() else None}
        rows.append(row)
        by_image[image] = label
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return images, by_image, rows, hashlib.sha256(encoded).hexdigest()


def get_model_fruits(model):
    names = {int(k): str(v).lower() for k, v in model.names.items()}
    if set(FRUITS).issubset(names.values()):
        return names
    if len(names) >= 80 and all(i in names for i in COCO_TO_FRUIT):
        return names
    raise ValueError(f"Checkpoint must contain all three fruit classes; got names={names}")


def evaluate_model(model_path, split_paths, args, device, device_reason):
    start_load = time.perf_counter()
    model_path = model_path.resolve()
    model = YOLO(str(model_path))
    names = get_model_fruits(model)
    if model.task != "segment":
        raise ValueError(f"{model_path} task is {model.task}, expected segment")
    load_seconds = time.perf_counter() - start_load
    model_result = {
        "checkpoint": str(model_path),
        "checkpoint_sha256": sha256_file(model_path),
        "task": model.task,
        "class_names": names,
        "input_resolution": args.imgsz,
        "load_seconds": round(load_seconds, 3),
        "splits": {},
    }
    for split_name, data_split in split_paths.items():
        images, labels, manifest_rows, manifest_hash = split_manifest(data_split)
        manifest_dir = args.output.parent / f"{args.output.stem}_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{split_name}-{manifest_hash}.json"
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps({"split": split_name, "entries": manifest_rows},
                                                indent=2, ensure_ascii=False) + "\n")
        if args.max_images:
            images = images[:args.max_images]
        if not images:
            raise ValueError(f"No images found in {data_split}")
        summary = {
            "total_images": 0, "gt_instances": 0, "pred_instances": 0,
            "exact_count_images": 0, "count_absolute_error": 0,
            "gt_by_class": Counter(), "matched_by_class": Counter(),
            "banana_cluster_images": 0, "banana_cluster_exact": 0,
            "banana_cluster_absolute_error": 0,
            "ordinary_single_fruit_images": 0, "ordinary_single_fruit_exact": 0,
            "ordinary_single_fruit_absolute_error": 0,
            "merge_predictions": 0, "duplicate_predictions": 0,
            "negative_images": 0, "negative_false_positive_images": 0,
            "inference_seconds": [],
        }
        detail_rows = []
        reset_peak_memory(device)
        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode {image_path}")
            height, width = image.shape[:2]
            gt = load_ground_truth(labels[image_path], width, height)
            started = time.perf_counter()
            polygons, boxes, scores, classes, _ = detect_seg(
                model, image, conf=args.conf, imgsz=args.imgsz, iou=args.iou,
                device=device, half=args.precision == "fp16",
            )
            if args.distance_refinement:
                polygons, boxes, scores, classes = split_banana_masks_by_distance(
                    polygons, boxes, scores, classes, model.names, image,
                )
            elapsed = time.perf_counter() - started
            pred = []
            for polygon, class_id, score in zip(polygons, classes, scores):
                name = names.get(int(class_id), f"class_{int(class_id)}")
                if name in COCO_TO_FRUIT:
                    name = COCO_TO_FRUIT[name]
                if name in FRUITS:
                    pred.append({"label": name, "polygon": polygon, "confidence": float(score)})
            matches, gt_used, pred_used, merged, duplicates = match_instances(gt, pred, width, height)
            gt_count = len(gt)
            pred_count = len(pred)
            counts_exact = gt_count == pred_count
            summary["total_images"] += 1
            summary["gt_instances"] += gt_count
            summary["pred_instances"] += pred_count
            summary["exact_count_images"] += int(counts_exact)
            summary["count_absolute_error"] += abs(gt_count - pred_count)
            summary["merge_predictions"] += len(merged)
            summary["duplicate_predictions"] += len(duplicates)
            for fruit in FRUITS:
                true_count = sum(x["label"] == fruit for x in gt)
                summary["gt_by_class"][fruit] += true_count
            for i in gt_used:
                summary["matched_by_class"][gt[i]["label"]] += 1
            banana_gt = sum(x["label"] == "banana" for x in gt)
            banana_pred = sum(x["label"] == "banana" for x in pred)
            if banana_gt >= 2:
                summary["banana_cluster_images"] += 1
                summary["banana_cluster_exact"] += int(banana_gt == banana_pred)
                summary["banana_cluster_absolute_error"] += abs(banana_gt - banana_pred)
            if gt_count == 1:
                summary["ordinary_single_fruit_images"] += 1
                summary["ordinary_single_fruit_exact"] += int(counts_exact)
                summary["ordinary_single_fruit_absolute_error"] += abs(gt_count - pred_count)
            if gt_count == 0:
                summary["negative_images"] += 1
                summary["negative_false_positive_images"] += int(pred_count > 0)
            summary["inference_seconds"].append(elapsed)
            detail_rows.append({"image": image_path.name, "gt_count": gt_count,
                                "pred_count": pred_count, "exact": counts_exact,
                                "gt_bananas": banana_gt, "pred_bananas": banana_pred,
                                "mask_matches_iou50": len(matches), "merge_predictions": len(merged),
                                "duplicate_predictions": len(duplicates), "inference_seconds": round(elapsed, 5)})

        n = summary["total_images"]
        durations = summary.pop("inference_seconds")
        summary["exact_count_accuracy"] = summary.pop("exact_count_images") / n
        summary["count_mae"] = summary.pop("count_absolute_error") / n
        summary["per_class_recall_iou50"] = {
            fruit: (summary["matched_by_class"][fruit] / summary["gt_by_class"][fruit]
                    if summary["gt_by_class"][fruit] else None) for fruit in FRUITS
        }
        summary["banana_cluster_exact_count_accuracy"] = (
            summary["banana_cluster_exact"] / summary["banana_cluster_images"]
            if summary["banana_cluster_images"] else None)
        summary["banana_cluster_count_mae"] = (
            summary["banana_cluster_absolute_error"] / summary["banana_cluster_images"]
            if summary["banana_cluster_images"] else None)
        summary.pop("banana_cluster_exact")
        summary["ordinary_single_fruit_exact_count_accuracy"] = (
            summary["ordinary_single_fruit_exact"] / summary["ordinary_single_fruit_images"]
            if summary["ordinary_single_fruit_images"] else None)
        summary["ordinary_single_fruit_count_mae"] = (
            summary["ordinary_single_fruit_absolute_error"] / summary["ordinary_single_fruit_images"]
            if summary["ordinary_single_fruit_images"] else None)
        summary["merge_rate"] = summary["merge_predictions"] / max(summary["pred_instances"], 1)
        summary["duplicate_rate"] = summary["duplicate_predictions"] / max(summary["pred_instances"], 1)
        summary["negative_false_positive_images_rate"] = (
            summary["negative_false_positive_images"] / summary["negative_images"]
            if summary["negative_images"] else None)
        summary["mean_inference_seconds_per_image"] = statistics.mean(durations)
        summary["p95_inference_seconds_per_image"] = float(np.percentile(durations, 95))
        summary["execution_seconds"] = sum(durations)
        summary["gt_by_class"] = dict(summary["gt_by_class"])
        summary["matched_by_class_iou50"] = dict(summary.pop("matched_by_class"))
        summary["dataset_manifest"] = str(manifest_path.resolve())
        summary["dataset_manifest_sha256"] = manifest_hash
        summary["manifest_images"] = len(manifest_rows)
        model_result["splits"][split_name] = {"summary": summary, "images": detail_rows}
        if device.startswith("cuda"):
            model_result["splits"][split_name]["gpu_memory"] = memory_snapshot(device)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "yolo11s-seg.pt")
    parser.add_argument("--custom", type=Path,
                        default=ROOT / "runs/seg/seg_boundary_candidate_gpu/weights/best.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "work/obb/seg_data_candidate")
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=("val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--distance-refinement", action="store_true",
                        help="Evaluate the experimental banana watershed helper (off by default)")
    parser.add_argument("--max-images", type=int, default=0, help="0 evaluates every image")
    parser.add_argument("--output", type=Path, default=ROOT / "work/audit/phase05_benchmark.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device, reason = resolve_device(args.device, "benchmark")
    if args.precision == "fp16" and not device.startswith("cuda"):
        raise SystemExit("fp16 was requested but the selected device is not CUDA; refusing an implicit precision change")
    split_paths = {split: args.dataset / split for split in args.splits}
    for path in (args.baseline, args.custom, *split_paths.values()):
        if not path.exists():
            raise FileNotFoundError(path)
    records = []
    for checkpoint in (args.baseline, args.custom):
        print(f"Evaluating {checkpoint} on {', '.join(args.splits)} using {device} ({reason})", flush=True)
        records.append(evaluate_model(checkpoint, split_paths, args, device, reason))
    # Require exactly identical manifests across the two checkpoint records.
    for split in args.splits:
        hashes = [x["splits"][split]["summary"]["dataset_manifest_sha256"] for x in records]
        if len(set(hashes)) != 1:
            raise RuntimeError(f"Model comparisons used different {split} image/label manifests: {hashes}")
    report = {
        "benchmark": "fruit-seg-same-split-v1",
        "config": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
                   "device": device, "device_reason": reason, "precision": args.precision,
                   "distance_refinement": args.distance_refinement,
                   "max_images": args.max_images or None},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                     "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
                     "numpy": np.__version__},
        "gpu_memory_note": "Unavailable" if not torch.cuda.is_available() else "Peak allocated/reserved per split recorded",
        "models": records,
        "metric_definitions": {
            "per_class_recall_iou50": "Greedy one-to-one matching of same-class raster masks at mask IoU >= 0.50",
            "merge_rate": "Predicted masks with >=25% coverage of at least two same-class GT masks / predictions",
            "duplicate_rate": "Extra member of same-class predicted-mask pairs with IoU >= 0.70 / predictions",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
