#!/usr/bin/env python3
"""Fresh, resumable CLIP swap-only benchmark. Inference requires --run-inference."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import jsonschema
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "backend"))

FRUITS = ("banana", "apple", "orange")
CLIP_NAMES = ("banana", "apple", "orange", "other")
IOU_THRESHOLD = 0.50
HASH_SCHEMA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
PREDICTION_RECORD_SCHEMA = {
    "type": "object",
    "required": ["image_id", "image_path", "image_sha256", "label_path", "label_sha256",
                 "width", "height", "inference_status", "detector_checkpoint_sha256",
                 "config_sha256", "prediction_count_before_clip", "prediction_count_after_clip", "instances"],
    "properties": {
        "image_id": {"type": "string", "minLength": 1}, "image_path": {"type": "string"},
        "image_sha256": HASH_SCHEMA, "label_path": {"type": "string"}, "label_sha256": HASH_SCHEMA,
        "width": {"type": "integer", "minimum": 1}, "height": {"type": "integer", "minimum": 1},
        "inference_status": {"const": "success"}, "detector_checkpoint_sha256": HASH_SCHEMA,
        "config_sha256": HASH_SCHEMA, "prediction_count_before_clip": {"type": "integer", "minimum": 0},
        "prediction_count_after_clip": {"type": "integer", "minimum": 0},
        "instances": {"type": "array", "items": {"type": "object",
            "required": ["instance_id", "class_id", "class_name", "confidence", "bbox", "polygon", "mask_area", "mask_sha256"],
            "properties": {
                "instance_id": {"type": "string"}, "class_id": {"type": "integer", "minimum": 0, "maximum": 2},
                "class_name": {"enum": list(FRUITS)}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "bbox": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}},
                "polygon": {"type": "array", "minItems": 3, "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}},
                "mask_area": {"type": "integer", "minimum": 0}, "mask_sha256": HASH_SCHEMA,
                "candidate": {"type": "object", "required": ["grouped_probabilities", "first_view_grouped_probabilities",
                    "selected_view", "original_class_id", "result_class_id", "swapped", "reason"],
                    "properties": {"grouped_probabilities": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}},
                        "first_view_grouped_probabilities": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}},
                        "inner_view_grouped_probabilities": {"oneOf": [{"type": "null"}, {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}}]},
                        "selected_view": {"enum": ["expanded", "inner"]}, "original_class_id": {"type": "integer"},
                        "result_class_id": {"type": "integer"}, "swapped": {"type": "boolean"}, "reason": {"type": "string"}}}
            }}}
    }
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict[str, Any]) -> str:
    payload = {k: v for k, v in value.items() if k != "config_sha256"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def prompt_payload_hash(source_path: Path) -> str:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"NAMES", "FRUIT_PROMPTS", "OTHER_PROMPTS", "TEMPLATE"}
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != wanted:
        raise ValueError(f"Could not extract pinned prompt payload: {sorted(values)}")
    raw = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
        dir_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def finite_vector(scores: Any, expected: int = 4) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.shape != (expected,) or not np.isfinite(arr).all():
        raise ValueError(f"Expected {expected} finite grouped scores; got shape={arr.shape}, values={arr!r}")
    if (arr < 0).any() or not np.isclose(float(arr.sum()), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"Grouped CLIP probabilities must be nonnegative and sum to 1; got {arr.tolist()}")
    return arr


def group_prompt_probabilities(prompt_probabilities: Any, prompt_groups: Any) -> np.ndarray:
    """Sum a softmax normalized across all prompts into the four ordered groups."""
    p = np.asarray(prompt_probabilities, dtype=np.float64)
    groups = np.asarray(prompt_groups, dtype=np.int64)
    if p.ndim != 1 or groups.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Prompt probabilities and group IDs must be finite aligned vectors")
    if (p < 0).any() or not np.isclose(float(p.sum()), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Prompt probabilities must already be softmax-normalized across all prompts")
    if (groups < 0).any() or (groups > 3).any():
        raise ValueError("Prompt group IDs must be in 0..3")
    grouped = np.zeros(4, dtype=np.float64)
    np.add.at(grouped, groups, p)
    return finite_vector(grouped)


def swap_decision(original_class_id: int, grouped_probs: Any, threshold: float = 0.80) -> dict[str, Any]:
    """Pure swap-only policy. np.argmax gives first-index tie priority."""
    p = finite_vector(grouped_probs)
    if original_class_id not in range(3):
        raise ValueError(f"Unknown detector fruit class id: {original_class_id}")
    top = int(np.argmax(p))
    changed = top < 3 and top != original_class_id and float(p[top]) >= threshold
    output = top if changed else original_class_id
    if top == 3:
        reason = "top_group_other_no_swap"
    elif top == original_class_id:
        reason = "top_group_matches_detector"
    elif float(p[top]) < threshold:
        reason = "different_top_group_below_threshold"
    else:
        reason = "different_top_fruit_at_or_above_threshold"
    return {
        "original_class_id": original_class_id,
        "original_class_name": FRUITS[original_class_id],
        "result_class_id": output,
        "result_class_name": FRUITS[output],
        "swapped": bool(changed),
        "top_group_id": top,
        "top_group_name": CLIP_NAMES[top],
        "top_group_probability": float(p[top]),
        "threshold": float(threshold),
        "reason": reason,
    }


def clip_box_crop(image: np.ndarray, box: Any, fraction: float) -> np.ndarray:
    """Exact current ClipChecker.probs_boxes integer crop calculation."""
    height, width = image.shape[:2]
    # Keep input scalar dtypes and Python arithmetic identical to probs_boxes.
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * fraction, (y2 - y1) * fraction
    left = int(max(0, min(width - 1, x1 - px)))
    top = int(max(0, min(height - 1, y1 - py)))
    right = int(min(width, max(left + 1, x2 + px)))
    bottom = int(min(height, max(top + 1, y2 + py)))
    return image[top:bottom, left:right]


def score_boxes(checker: Any, image: np.ndarray, boxes: np.ndarray,
                pad: float = 0.10, inner: float = -0.10,
                t_other: float = 0.80, t_inner: float = 0.90) -> list[dict[str, Any]]:
    """Return first and selected view scores and reject non-finite model outputs."""
    if len(boxes) == 0:
        return []
    first_crops = [clip_box_crop(image, b, pad) for b in boxes]
    first = np.asarray(checker.probs(first_crops), dtype=np.float64)
    if first.shape != (len(boxes), 4) or not np.isfinite(first).all():
        raise ValueError(f"Invalid first-view CLIP output: shape={first.shape}")
    final = first.copy()
    results = [{"first_view_grouped_probabilities": finite_vector(row).tolist(),
                "inner_view_grouped_probabilities": None,
                "selected_view": "expanded"} for row in first]
    redo = [i for i, row in enumerate(first) if row[3] >= t_other]
    if redo:
        inner_crops = [clip_box_crop(image, boxes[i], inner) for i in redo]
        second = np.asarray(checker.probs(inner_crops), dtype=np.float64)
        if second.shape != (len(redo), 4) or not np.isfinite(second).all():
            raise ValueError(f"Invalid inner-view CLIP output: shape={second.shape}")
        for row_index, scores in zip(redo, second):
            scores = finite_vector(scores)
            results[row_index]["inner_view_grouped_probabilities"] = scores.tolist()
            if float(scores[:3].max()) >= t_inner:
                final[row_index] = scores
                results[row_index]["selected_view"] = "inner"
    for item, row in zip(results, final):
        item["grouped_probabilities"] = finite_vector(row).tolist()
    return results


def polygon_to_mask(polygon: Any, width: int, height: int, normalized: bool = False) -> np.ndarray:
    """Rasterize a single xy polygon; invalid/empty input maps to an all-zero mask.

    Coordinates are continuous pixel-edge coordinates; normalized GT is scaled by
    width/height. Vertices are rounded with NumPy rint (ties-to-even), then clipped
    to valid pixel-center indices before OpenCV fillPoly (OpenCV 4-connectivity).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    try:
        points = np.asarray(polygon, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            return mask
        if normalized:
            points = points * np.asarray([width, height], dtype=np.float64)
        points = np.rint(points)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        points = points.astype(np.int32)
        if len(np.unique(points, axis=0)) < 3:
            return mask
        cv2.fillPoly(mask, [points], 1, lineType=cv2.LINE_8)
    except (TypeError, ValueError, OverflowError):
        return mask
    return mask


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Mask shape mismatch: {left.shape} != {right.shape}")
    a, b = left.astype(bool), right.astype(bool)
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def _max_weight_assignment(weights: np.ndarray) -> float:
    """Maximum positive assignment allowing every row to remain unmatched."""
    rows, cols = weights.shape
    if rows == 0 or cols == 0:
        return 0.0
    # One dummy column per row represents an unmatched GT; forbidden edges are 0.
    padded = np.concatenate([weights, np.zeros((rows, rows), dtype=np.float64)], axis=1)
    r, c = linear_sum_assignment(-padded)
    return float(padded[r, c].sum())


def class_agnostic_match(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray],
                         threshold: float = IOU_THRESHOLD) -> dict[str, Any]:
    """Maximum-sum eligible-IoU assignment, lexicographically tie-broken by IDs.

    GT and prediction list order define stable IDs. Exact objective ties are
    resolved GT-by-GT, choosing the lowest available prediction ID before leaving
    that GT unmatched, provided the global optimum remains attainable.
    """
    if gt_masks and pred_masks and gt_masks[0].shape != pred_masks[0].shape:
        raise ValueError("Ground-truth and prediction masks must share dimensions")
    ious = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float64)
    for i, gm in enumerate(gt_masks):
        for j, pm in enumerate(pred_masks):
            ious[i, j] = mask_iou(gm, pm)
    eligible_weights = np.where(ious >= threshold, ious, 0.0)
    remaining_preds = list(range(len(pred_masks)))
    target = _max_weight_assignment(eligible_weights)
    selected: list[tuple[int, int]] = []
    fixed_total = 0.0
    # Tolerance only compensates floating summation differences, not IoU eligibility.
    tolerance = 1e-12 * max(1.0, abs(target))
    for gt_id in range(len(gt_masks)):
        options = [p for p in remaining_preds if eligible_weights[gt_id, p] > 0] + [None]
        chosen = None
        for pred_id in options:
            edge = 0.0 if pred_id is None else float(eligible_weights[gt_id, pred_id])
            next_preds = remaining_preds if pred_id is None else [p for p in remaining_preds if p != pred_id]
            residual = eligible_weights[gt_id + 1:, next_preds]
            possible = fixed_total + edge + _max_weight_assignment(residual)
            if abs(possible - target) <= tolerance:
                chosen = pred_id
                fixed_total += edge
                if pred_id is not None:
                    selected.append((gt_id, pred_id))
                    remaining_preds.remove(pred_id)
                break
        if chosen is None and abs(fixed_total + _max_weight_assignment(eligible_weights[gt_id + 1:, remaining_preds]) - target) > tolerance:
            raise RuntimeError("Deterministic tie refinement could not preserve optimum")
    selected.sort()
    used_gt = {i for i, _ in selected}
    used_pred = {j for _, j in selected}
    return {
        "matches": [{"gt_id": i, "prediction_id": j, "mask_iou": float(ious[i, j])} for i, j in selected],
        "unmatched_gt_ids": [i for i in range(len(gt_masks)) if i not in used_gt],
        "unmatched_prediction_ids": [j for j in range(len(pred_masks)) if j not in used_pred],
        "iou_matrix": ious.tolist(),
    }


def validate_swap_invariants(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    if baseline["image_id"] != candidate["image_id"]:
        raise AssertionError("image_id changed")
    if (baseline["width"], baseline["height"]) != (candidate["width"], candidate["height"]):
        raise AssertionError("image dimensions changed")
    before, after = baseline["instances"], candidate["instances"]
    if len(before) != len(after):
        raise AssertionError("instance count changed")
    if [x["instance_id"] for x in before] != [x["instance_id"] for x in after]:
        raise AssertionError("instance IDs/order changed")
    for a, b in zip(before, after):
        for key in ("confidence", "bbox", "polygon", "mask_area", "mask_sha256"):
            if a[key] != b[key]:
                raise AssertionError(f"non-class field changed for {a['instance_id']}: {key}")
        decision = b.get("candidate")
        if not decision:
            raise AssertionError(f"candidate decision missing for {b['instance_id']}")
        if decision["original_class_id"] != a["class_id"]:
            raise AssertionError(f"candidate original class does not match baseline for {b['instance_id']}")
        if decision["result_class_id"] != b["class_id"] or decision["result_class_name"] != b["class_name"]:
            raise AssertionError(f"candidate decision does not match output class for {b['instance_id']}")
        if decision["swapped"] != (a["class_id"] != b["class_id"]):
            raise AssertionError(f"candidate swap flag disagrees with class transition for {b['instance_id']}")
        replay = swap_decision(a["class_id"], decision["grouped_probabilities"], decision["threshold"])
        for key in ("result_class_id", "result_class_name", "swapped", "reason"):
            if replay[key] != decision[key]:
                raise AssertionError(f"candidate decision replay mismatch for {b['instance_id']}: {key}")


def validate_prediction_record(record: dict[str, Any]) -> None:
    jsonschema.validate(record, PREDICTION_RECORD_SCHEMA)
    if record["prediction_count_before_clip"] != record["prediction_count_after_clip"]:
        raise ValueError("Swap-only record instance counts differ")
    if record["prediction_count_after_clip"] != len(record["instances"]):
        raise ValueError("Prediction count does not match persisted instance records")
    for instance in record["instances"]:
        if instance["class_name"] != FRUITS[instance["class_id"]]:
            raise ValueError(f"class ID/name mismatch for {instance['instance_id']}")
        numeric_geometry = np.asarray([instance["confidence"], *instance["bbox"],
                                       *[v for point in instance["polygon"] for v in point]], dtype=np.float64)
        if not np.isfinite(numeric_geometry).all():
            raise ValueError(f"Non-finite geometry/score for {instance['instance_id']}")
        if "candidate" in instance:
            candidate = instance["candidate"]
            finite_vector(candidate["grouped_probabilities"])
            finite_vector(candidate["first_view_grouped_probabilities"])
            if candidate["inner_view_grouped_probabilities"] is not None:
                finite_vector(candidate["inner_view_grouped_probabilities"])


def reusable_completed_pair(baseline_path: Path, candidate_path: Path, config_hash: str,
                            image_id: str | None = None, image_sha256: str | None = None,
                            label_sha256: str | None = None) -> bool:
    """Resume gate: both durable records must match config and satisfy invariants."""
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        if baseline.get("inference_status") != "success" or candidate.get("inference_status") != "success":
            return False
        if baseline.get("config_sha256") != config_hash or candidate.get("config_sha256") != config_hash:
            return False
        if image_id is not None and baseline.get("image_id") != image_id:
            return False
        for record in (baseline, candidate):
            if image_sha256 is not None and record.get("image_sha256") != image_sha256:
                return False
            if label_sha256 is not None and record.get("label_sha256") != label_sha256:
                return False
        validate_swap_invariants(baseline, candidate)
        validate_prediction_record(baseline)
        validate_prediction_record(candidate)
        return True
    except (OSError, ValueError, KeyError, TypeError, AssertionError, json.JSONDecodeError):
        return False


def evaluate_image(gt_instances: list[dict[str, Any]], prediction_record: dict[str, Any],
                   width: int, height: int) -> dict[str, Any]:
    """Class-agnostic geometry assignment, then classification accounting."""
    gt_masks = [obj.get("mask") if "mask" in obj else
                polygon_to_mask(obj["polygon_normalized"], width, height, normalized=True)
                for obj in gt_instances]
    pred = prediction_record["instances"]
    pred_masks = [polygon_to_mask(obj["polygon"], width, height) for obj in pred]
    match = class_agnostic_match(gt_masks, pred_masks, IOU_THRESHOLD)
    confusion = {truth: {predicted: 0 for predicted in FRUITS} | {"miss": 0} for truth in FRUITS}
    for gt_id, gt_obj in enumerate(gt_instances):
        found = next((m for m in match["matches"] if m["gt_id"] == gt_id), None)
        if found is None:
            confusion[gt_obj["class_name"]]["miss"] += 1
        else:
            confusion[gt_obj["class_name"]][pred[found["prediction_id"]]["class_name"]] += 1
    false_positives = [{"prediction_id": i, "class_name": pred[i]["class_name"]}
                       for i in match["unmatched_prediction_ids"]]
    return {"matches": match["matches"], "confusion": confusion,
            "false_positives": false_positives,
            "false_negatives": match["unmatched_gt_ids"],
            "matched_count": len(match["matches"]),
            "gt_count": len(gt_instances), "prediction_count": len(pred)}


def load_and_verify_config(config_path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    declared = config.get("config_sha256")
    actual = canonical_hash(config)
    if declared != actual:
        raise ValueError(f"Config hash mismatch: declared={declared}, computed={actual}")
    for artifact in config["artifacts"].values():
        path = ROOT / artifact["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != artifact["sha256"]:
            raise ValueError(f"Artifact hash mismatch for {path}: {actual_hash}")
    prompt_source = ROOT / config["artifacts"]["clip_source"]["path"]
    prompt_hash = prompt_payload_hash(prompt_source)
    if prompt_hash != config["clip"]["prompt_payload_sha256"]:
        raise ValueError(f"Prompt payload hash mismatch: {prompt_hash}")
    for package, expected in config["runtime"]["packages"].items():
        actual_version = importlib.metadata.version(package)
        if actual_version != expected:
            raise ValueError(f"Version mismatch for {package}: {actual_version} != {expected}")
    if platform.python_version() != config["runtime"]["python"]:
        raise ValueError(f"Python version mismatch: {platform.python_version()} != {config['runtime']['python']}")
    if cv2.__version__ != config["runtime"]["opencv_runtime_version"]:
        raise ValueError(f"OpenCV runtime mismatch: {cv2.__version__} != {config['runtime']['opencv_runtime_version']}")
    return config, actual


def dry_run(config_path: Path) -> dict[str, Any]:
    config, config_hash = load_and_verify_config(config_path)
    manifest_path = ROOT / config["dataset"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    issues = []
    seen = set()
    valid = 0
    split_root = ROOT / config["dataset"]["split_root"]
    if len(entries) != config["dataset"]["expected_images"]:
        issues.append({"error": f"manifest has {len(entries)} entries; expected {config['dataset']['expected_images']}"})
    for row in entries:
        image_rel, label_rel = row["image"], row["label"]
        if image_rel in seen:
            issues.append({"image": image_rel, "error": "duplicate manifest image"})
            continue
        seen.add(image_rel)
        image_path, label_path = split_root / image_rel, split_root / label_rel
        try:
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            image_hash, label_hash = sha256_file(image_path), sha256_file(label_path)
            if image_hash != row["image_sha256"]:
                raise ValueError(f"image hash mismatch: {image_hash}")
            if label_hash != row["label_sha256"]:
                raise ValueError(f"label hash mismatch: {label_hash}")
            valid += 1
        except Exception as exc:
            issues.append({"image": image_rel, "label": label_rel, "error": f"{type(exc).__name__}: {exc}"})
    return {"status": "ok" if valid == len(entries) and not issues else "failed",
            "config_sha256": config_hash, "manifest_entries": len(entries),
            "unique_images": len(seen), "valid_entries": valid,
            "invalid_entries": len(issues), "issues": issues}


def load_ground_truth(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    result = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"Invalid polygon row {label_path}:{line_no}")
        class_id = int(fields[0])
        if class_id not in range(3):
            raise ValueError(f"Unknown GT class at {label_path}:{line_no}: {class_id}")
        coords = np.asarray([float(v) for v in fields[1:]], dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(coords).all():
            raise ValueError(f"Non-finite GT polygon at {label_path}:{line_no}")
        result.append({"instance_id": f"gt_{len(result):04d}", "class_id": class_id,
                       "class_name": FRUITS[class_id], "polygon_normalized": coords.tolist(),
                       "mask": polygon_to_mask(coords, width, height, normalized=True)})
    return result


def inference(config_path: Path) -> None:
    config, config_hash = load_and_verify_config(config_path)
    # Verification above occurs before either model is instantiated.
    from ultralytics import YOLO
    from detection import detect_seg
    from clip_check import ClipChecker

    detector_path = ROOT / config["artifacts"]["detector"]["path"]
    clip_checker = ClipChecker(arch=config["clip"]["arch"], pretrained=config["clip"]["pretrained"], device="cpu")
    detector = YOLO(str(detector_path))
    if detector.task != "segment" or tuple(detector.names.values()) != FRUITS:
        raise ValueError(f"Unexpected detector task/classes: {detector.task}/{detector.names}")
    manifest = json.loads((ROOT / config["dataset"]["manifest_path"]).read_text(encoding="utf-8"))
    out_dir = HERE / "records"
    progress_path = HERE / "progress.json"
    event_path = HERE / "inference_events.jsonl"
    state = {row["image"]: "pending" for row in manifest["entries"]}
    if progress_path.exists():
        existing = json.loads(progress_path.read_text())
        if existing.get("config_sha256") != config_hash:
            raise ValueError("Existing progress belongs to another config")
        state.update(existing["images"])
    atomic_json(progress_path, {"config_sha256": config_hash, "manifest_id": config["dataset"]["manifest_id"], "images": state})
    split_root = ROOT / config["dataset"]["split_root"]
    for row in manifest["entries"]:
        rel = row["image"]
        key = hashlib.sha256(rel.encode()).hexdigest()
        base_path, cand_path = out_dir / "baseline" / f"{key}.json", out_dir / "candidate" / f"{key}.json"
        if state.get(rel) == "success" and reusable_completed_pair(
                base_path, cand_path, config_hash, rel, row["image_sha256"], row["label_sha256"]):
            continue
        image_path, label_path = split_root / rel, split_root / row["label"]
        event = {"image": rel, "started_unix": time.time(), "status": "running"}
        state[rel] = "running"
        atomic_json(progress_path, {"config_sha256": config_hash, "manifest_id": config["dataset"]["manifest_id"], "images": state})
        try:
            if sha256_file(image_path) != row["image_sha256"] or sha256_file(label_path) != row["label_sha256"]:
                raise ValueError("Manifest source hash changed")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode image: {image_path}")
            h, w = image.shape[:2]
            polygons, boxes, scores, classes, _ = detect_seg(
                detector, image, conf=config["detector"]["conf"], imgsz=config["detector"]["imgsz"],
                iou=config["detector"]["iou"], device="cpu", half=False)
            if any(int(c) not in range(3) for c in classes):
                raise ValueError(f"Unexpected predicted class IDs: {classes}")
            boxes = np.asarray(boxes).reshape(-1, 4)
            scores = np.asarray(scores, dtype=np.float64)
            classes = np.asarray(classes, dtype=np.int64)
            if not (len(polygons) == len(boxes) == len(scores) == len(classes)):
                raise ValueError("Detector instance arrays have different lengths")
            clip_rows = score_boxes(clip_checker, image, boxes,
                                    t_other=config["clip"]["t_other"], t_inner=config["clip"]["t_inner"])
            baseline_instances, candidate_instances = [], []
            for i, (poly, box, score, cls, clip_row) in enumerate(zip(polygons, boxes, scores, classes, clip_rows)):
                polygon = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
                if (len(polygon) < 3 or not np.isfinite(polygon).all() or not math.isfinite(float(score))
                        or not np.isfinite(box).all()):
                    raise ValueError(f"Invalid detector geometry/score for instance {i}")
                instance_mask = polygon_to_mask(polygon, w, h)
                area = int(instance_mask.sum())
                common = {"instance_id": f"det_{i:04d}", "confidence": float(score),
                          "bbox": box.tolist(), "polygon": polygon.tolist(), "mask_area": area,
                          "mask_sha256": hashlib.sha256(instance_mask.tobytes(order="C")).hexdigest()}
                b = {**common, "class_id": int(cls), "class_name": FRUITS[int(cls)]}
                decision = swap_decision(int(cls), clip_row["grouped_probabilities"], config["clip"]["swap_threshold"])
                c = {**common, "class_id": decision["result_class_id"], "class_name": decision["result_class_name"],
                     "candidate": {**clip_row, **decision}}
                baseline_instances.append(b)
                candidate_instances.append(c)
            label = load_ground_truth(label_path, w, h)
            detector_hash = config["artifacts"]["detector"]["sha256"]
            base_record = _image_record(row, rel, image_path, label_path, w, h, config_hash, detector_hash, baseline_instances, len(classes))
            cand_record = _image_record(row, rel, image_path, label_path, w, h, config_hash, detector_hash, candidate_instances, len(classes))
            cand_record["ground_truth"] = [{k: v for k, v in obj.items() if k != "mask"} for obj in label]
            # Persist GT masks once in a separate reference record.
            gt_record = {"image_id": rel, "image_path": str(image_path.relative_to(ROOT)),
                         "image_sha256": row["image_sha256"], "label_path": str(label_path.relative_to(ROOT)),
                         "label_sha256": row["label_sha256"], "width": w, "height": h,
                         "config_sha256": config_hash,
                         "instances": [{**{k: v for k, v in obj.items() if k != "mask"},
                                        "mask_rle": _encode_mask(obj["mask"])} for obj in label]}
            gt_path = out_dir / "ground_truth" / f"{key}.json"
            atomic_json(gt_path, gt_record)
            validate_swap_invariants(base_record, cand_record)
            validate_prediction_record(base_record)
            validate_prediction_record(cand_record)
            atomic_json(base_path, base_record)
            atomic_json(cand_path, cand_record)
            # Read-back validation means success follows durable complete records.
            validate_swap_invariants(json.loads(base_path.read_text()), json.loads(cand_path.read_text()))
            validate_prediction_record(json.loads(base_path.read_text()))
            validate_prediction_record(json.loads(cand_path.read_text()))
            state[rel] = "success"
            event.update(status="success", finished_unix=time.time(), prediction_count=len(classes))
        except Exception as exc:
            state[rel] = "error"
            event.update(status="error", finished_unix=time.time(), exception_type=type(exc).__name__,
                         exception=str(exc), traceback=traceback.format_exc())
            atomic_json(out_dir / "errors" / f"{key}.json", {
                "image_id": rel, "image_path": str(image_path.relative_to(ROOT)),
                "image_sha256": row.get("image_sha256"), "label_path": str(label_path.relative_to(ROOT)),
                "label_sha256": row.get("label_sha256"), "config_sha256": config_hash,
                "inference_status": "error", "exception_type": type(exc).__name__,
                "exception": str(exc), "traceback": traceback.format_exc()})
        with event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        atomic_json(progress_path, {"config_sha256": config_hash, "manifest_id": config["dataset"]["manifest_id"], "images": state})
    _export_jsonl(out_dir / "baseline", HERE / "predictions_baseline.jsonl")
    _export_jsonl(out_dir / "candidate", HERE / "predictions_candidate.jsonl")


def _image_record(row, rel, image_path, label_path, width, height, config_hash, detector_hash, instances, count):
    return {"image_id": rel, "image_path": str(image_path.relative_to(ROOT)),
            "image_sha256": row["image_sha256"], "label_path": str(label_path.relative_to(ROOT)),
            "label_sha256": row["label_sha256"], "width": width, "height": height,
            "inference_status": "success", "detector_checkpoint_sha256": detector_hash,
            "config_sha256": config_hash, "prediction_count_before_clip": count,
            "prediction_count_after_clip": len(instances), "instances": instances}


def _encode_mask(mask: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    runs, last, count = [], 0, 0
    for value in flat:
        value = int(value)
        if value == last:
            count += 1
        else:
            runs.append(count); count = 1; last = value
    runs.append(count)
    return {"size": list(mask.shape), "order": "C", "counts": runs}


def _export_jsonl(record_dir: Path, output: Path) -> None:
    records = []
    for path in sorted(record_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp_name, output)
    except BaseException:
        try: os.unlink(temp_name)
        except FileNotFoundError: pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate config, artifacts, versions, manifest files and hashes; never load models")
    parser.add_argument("--run-inference", action="store_true", help="Explicitly run the fresh test evaluation")
    args = parser.parse_args()
    if args.dry_run == args.run_inference:
        parser.error("choose exactly one of --dry-run or --run-inference")
    if args.dry_run:
        result = dry_run(args.config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "ok" else 1
    inference(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
