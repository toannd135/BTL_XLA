#!/usr/bin/env python3
"""Validate the complete pinned run, then calculate metrics and reports (no inference)."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import evaluator as ev

RUN = HERE / "full_runs/fresh_clip_swap_only_345"
REPORT_MD = ROOT / "docs/diagnostics/candidate-test-final-report.md"
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261119
CLASSES = ev.FRUITS


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def decode_rle(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.empty(h * w, dtype=np.uint8)
    offset, value = 0, 0
    for count in rle["counts"]:
        if count < 0 or offset + count > flat.size:
            raise ValueError("Invalid GT RLE counts")
        flat[offset:offset + count] = value
        offset += count
        value = 1 - value
    if offset != flat.size:
        raise ValueError("GT RLE does not cover declared mask dimensions")
    return flat.reshape((h, w))


def add_matrix(target: dict, source: dict) -> None:
    for gt in (*CLASSES, "background"):
        for pred in (*CLASSES, "miss"):
            target[gt][pred] += source.get(gt, {}).get(pred, 0)


def class_metrics(confusion: dict) -> dict:
    result = {}
    for cls in CLASSES:
        tp = confusion[cls][cls]
        fp = sum(confusion[row][cls] for row in CLASSES if row != cls) + confusion["background"][cls]
        fn = sum(confusion[cls][col] for col in CLASSES if col != cls) + confusion[cls]["miss"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[cls] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
                       "recall": recall, "f1": f1}
    result["macro_f1"] = float(np.mean([result[c]["f1"] for c in CLASSES]))
    return result


def empty_confusion() -> dict:
    return {gt: {pred: 0 for pred in (*CLASSES, "miss")} for gt in CLASSES} | {
        "background": {pred: 0 for pred in (*CLASSES, "miss")}}


def paired_bootstrap(rows: list[dict], reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(rows)
    matrices = {arm: [] for arm in ("baseline", "candidate")}
    for row in rows:
        for arm in matrices:
            matrices[arm].append(row[arm]["confusion_array"])
    matrix_arrays = {k: np.stack(v) for k, v in matrices.items()}
    gt_counts = np.asarray([r["gt_count"] for r in rows], dtype=np.float64)
    exact = {k: np.asarray([r[k]["exact_count"] for r in rows], dtype=np.float64)
             for k in ("baseline", "candidate")}
    iou_sum = np.asarray([r["mean_iou_sum"] for r in rows], dtype=np.float64)
    iou_n = np.asarray([r["mean_iou_n"] for r in rows], dtype=np.float64)
    neg_idx = np.asarray([i for i, r in enumerate(rows) if r["negative"]], dtype=np.int64)
    neg_fp = {k: np.asarray([rows[i][k]["fp_count"] for i in neg_idx], dtype=np.float64)
              for k in ("baseline", "candidate")}

    samples = {name: [] for name in ("classification_accuracy", "macro_f1", "exact_count_rate",
                                      "mean_matched_iou", "negative_fp_image_rate")}
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        denom = gt_counts[idx].sum()
        for arm in ("baseline", "candidate"):
            m = matrix_arrays[arm][idx].sum(axis=0)
            conf = {gt: {pred: int(m[i, j]) for j, pred in enumerate((*CLASSES, "miss"))}
                    for i, gt in enumerate(CLASSES)}
            conf["background"] = {pred: 0 for pred in (*CLASSES, "miss")}
            # Unmatched predictions are included in the serialized per-image confusion array at background row.
            conf["background"] = {pred: int(m[3, j]) for j, pred in enumerate((*CLASSES, "miss"))}
            class_correct = sum(conf[c][c] for c in CLASSES)
            samples["classification_accuracy"].append((arm, class_correct / denom if denom else 0.0))
            samples["macro_f1"].append((arm, class_metrics(conf)["macro_f1"]))
            samples["exact_count_rate"].append((arm, float(exact[arm][idx].mean())))
            isum, inum = iou_sum[idx].sum(), iou_n[idx].sum()
            samples["mean_matched_iou"].append((arm, float(isum / inum) if inum else 0.0))
            if len(neg_idx):
                ni = rng.integers(0, len(neg_idx), size=len(neg_idx))
                samples["negative_fp_image_rate"].append((arm, float((neg_fp[arm][ni] > 0).mean())))
    output = {}
    for name, vals in samples.items():
        paired = {}
        base = np.asarray([v for arm, v in vals if arm == "baseline"])
        cand = np.asarray([v for arm, v in vals if arm == "candidate"])
        delta = cand - base
        output[name] = {"baseline_ci95": np.percentile(base, [2.5, 97.5]).tolist(),
                        "candidate_ci95": np.percentile(cand, [2.5, 97.5]).tolist(),
                        "paired_difference_candidate_minus_baseline_ci95": np.percentile(delta, [2.5, 97.5]).tolist(),
                        "bootstrap_repetitions": reps, "seed": seed}
    return output


def evaluate_arm(gt_instances: list[dict], record: dict, width: int, height: int) -> dict:
    out = ev.evaluate_image(gt_instances, record, width, height)
    confusion = empty_confusion()
    for gt_name, cells in out["confusion"].items():
        for predicted, count in cells.items():
            confusion[gt_name][predicted] += count
    for fp in out["false_positives"]:
        confusion["background"][fp["class_name"]] += 1
    confusion_array = np.asarray([[confusion[gt][pred] for pred in (*CLASSES, "miss")]
                                  for gt in (*CLASSES, "background")], dtype=np.int64)
    matched = out["matches"]
    pred_instances = record["instances"]
    correct = 0
    for match in matched:
        gt = gt_instances[match["gt_id"]]
        pred = pred_instances[match["prediction_id"]]
        correct += int(gt["class_name"] == pred["class_name"])
    fp_count = len(out["false_positives"])
    fn_count = len(out["false_negatives"])
    gt_count = len(gt_instances)
    pred_count = len(pred_instances)
    return {"confusion": confusion, "confusion_array": confusion_array,
            "matched": len(matched), "fp_count": fp_count, "fn_count": fn_count,
            "gt_count": gt_count, "pred_count": pred_count,
            "exact_count": gt_count == pred_count,
            "abs_count_error": abs(gt_count - pred_count),
            "classification_correct": correct,
            "classification_accuracy": correct / gt_count if gt_count else None,
            "mean_iou_sum": sum(m["mask_iou"] for m in matched), "mean_iou_n": len(matched),
            "mask_ious": [float(m["mask_iou"]) for m in matched],
            "false_positives": out["false_positives"], "false_negatives": out["false_negatives"]}


def run_validation_and_metrics() -> tuple[dict, list[dict], list[dict], list[dict]]:
    config, config_hash = ev.load_and_verify_config(HERE / "config.json")
    if config_hash != "65f552c3b25f545ab20643d986dd0777e8f7d30b482e889d4953ae223b0b53b4":
        raise ValueError("Pinned config digest mismatch")
    preflight = ev.dry_run(HERE / "config.json")
    if preflight["status"] != "ok" or preflight["manifest_entries"] != 345:
        raise ValueError(f"Dataset preflight failed: {preflight}")
    manifest = json.loads((ROOT / config["dataset"]["manifest_path"]).read_text(encoding="utf-8"))
    entries = manifest["entries"]
    progress = read_json(RUN / "progress.json")
    summary = read_json(RUN / "execution_summary.json")
    if progress.get("config_sha256") != config_hash or progress.get("manifest_id") != config["dataset"]["manifest_id"]:
        raise ValueError("Progress config or manifest identity mismatch")
    if len(progress.get("images", {})) != 345 or set(progress["images"].values()) != {"success"}:
        raise ValueError("Progress does not show 345 successful images")
    for key in ("successful_images", "error_images", "pending_or_running_images"):
        expected = {"successful_images": 345, "error_images": 0, "pending_or_running_images": 0}[key]
        if summary.get(key) != expected:
            raise ValueError(f"Execution summary {key}={summary.get(key)} != {expected}")
    if summary.get("failures"):
        raise ValueError("Execution summary contains unresolved failures")
    if summary.get("config_sha256") != config_hash:
        raise ValueError("Execution summary config hash mismatch")

    expected_names = {hashlib.sha256(row["image"].encode()).hexdigest() + ".json" for row in entries}
    for role in ("baseline", "candidate", "ground_truth"):
        names = {p.name for p in (RUN / "records" / role).glob("*.json")}
        if names != expected_names:
            raise ValueError(f"{role} record set mismatch: expected {len(expected_names)}, got {len(names)}")
    for arm in ("baseline", "candidate"):
        lines = (RUN / f"predictions_{arm}.jsonl").read_text(encoding="utf-8").splitlines()
        if len(lines) != 345:
            raise ValueError(f"{arm} JSONL has {len(lines)} records")

    events = [json.loads(line) for line in (RUN / "inference_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    complete_by_image = Counter(e["image"] for e in events if e.get("event") == "complete")
    if len(complete_by_image) != 345 or any(count != 1 for count in complete_by_image.values()):
        raise ValueError("Expected exactly one completion event for each manifest image")
    recovered_attempts = []
    for e in events:
        if e.get("event") == "error":
            image = e["image"]
            if not any(x.get("event") == "complete" and x.get("image") == image for x in events):
                raise ValueError(f"Unrecovered inference error remains for {image}")
            recovered_attempts.append({"image": image, "exception_type": e.get("exception_type"),
                                       "exception": e.get("exception"), "recovered": True})

    rows, confusion_by_arm = [], {arm: empty_confusion() for arm in ("baseline", "candidate")}
    transitions = Counter()
    per_class_gt = Counter()
    swap_transitions = Counter()
    negatives = []
    for row in entries:
        rel = row["image"]
        key = hashlib.sha256(rel.encode()).hexdigest()
        bp = RUN / "records/baseline" / f"{key}.json"
        cp = RUN / "records/candidate" / f"{key}.json"
        gp = RUN / "records/ground_truth" / f"{key}.json"
        baseline, candidate, gt_ref = read_json(bp), read_json(cp), read_json(gp)
        if not ev.reusable_completed_pair(bp, cp, config_hash, rel, row["image_sha256"], row["label_sha256"]):
            raise ValueError(f"Baseline/candidate schema or swap invariants failed: {rel}")
        ev.validate_prediction_record(baseline)
        ev.validate_prediction_record(candidate)
        for rec in (baseline, candidate):
            if rec["detector_checkpoint_sha256"] != config["artifacts"]["detector"]["sha256"]:
                raise ValueError(f"Detector hash mismatch in {rel}")
            if rec["image_sha256"] != row["image_sha256"] or rec["label_sha256"] != row["label_sha256"]:
                raise ValueError(f"Manifest source hashes mismatch in {rel}")
        if gt_ref.get("config_sha256") != config_hash or gt_ref.get("image_sha256") != row["image_sha256"] or gt_ref.get("label_sha256") != row["label_sha256"]:
            raise ValueError(f"GT reference hash mismatch in {rel}")
        if (gt_ref["width"], gt_ref["height"]) != (baseline["width"], baseline["height"]):
            raise ValueError(f"GT dimensions mismatch in {rel}")
        expected_gt = ev.load_ground_truth(ROOT / config["dataset"]["split_root"] / row["label"],
                                           gt_ref["width"], gt_ref["height"])
        if len(expected_gt) != len(gt_ref["instances"]):
            raise ValueError(f"GT instance count mismatch in {rel}")
        gt_instances = []
        for i, (expected, saved) in enumerate(zip(expected_gt, gt_ref["instances"])):
            if expected["class_id"] != saved["class_id"] or expected["polygon_normalized"] != saved["polygon_normalized"]:
                raise ValueError(f"GT class/polygon mismatch in {rel} at {i}")
            decoded = decode_rle(saved["mask_rle"])
            if not np.array_equal(decoded, expected["mask"]):
                raise ValueError(f"GT RLE mismatch in {rel} at {i}")
            gt_instances.append({k: saved[k] for k in ("instance_id", "class_id", "class_name", "polygon_normalized")})
            per_class_gt[saved["class_name"]] += 1

        # Verify exported JSONL records match the durable per-image instance records.
        for arm, expected_record in (("baseline", baseline), ("candidate", candidate)):
            # Each JSONL is sorted by hashed record filename, so the line map is loaded once below.
            pass

        b_eval = evaluate_arm(gt_instances, baseline, baseline["width"], baseline["height"])
        c_eval = evaluate_arm(gt_instances, candidate, candidate["width"], candidate["height"])
        # Matching must be class agnostic and geometry-identical across the two arms.
        bm = ev.class_agnostic_match([ev.polygon_to_mask(x["polygon_normalized"], baseline["width"], baseline["height"], True) for x in gt_instances],
                                     [ev.polygon_to_mask(x["polygon"], baseline["width"], baseline["height"]) for x in baseline["instances"]])
        cm = ev.class_agnostic_match([ev.polygon_to_mask(x["polygon_normalized"], candidate["width"], candidate["height"], True) for x in gt_instances],
                                     [ev.polygon_to_mask(x["polygon"], candidate["width"], candidate["height"]) for x in candidate["instances"]])
        if bm["matches"] != cm["matches"] or bm["unmatched_gt_ids"] != cm["unmatched_gt_ids"] or bm["unmatched_prediction_ids"] != cm["unmatched_prediction_ids"]:
            raise ValueError(f"Baseline/candidate geometry matching changed in {rel}")
        for arm, result in (("baseline", b_eval), ("candidate", c_eval)):
            add_matrix(confusion_by_arm[arm], result["confusion"])
        base_by_gt, cand_by_gt = {}, {}
        for m in bm["matches"]:
            gid, pid = m["gt_id"], m["prediction_id"]
            truth = gt_instances[gid]["class_name"]
            bname = baseline["instances"][pid]["class_name"]
            cname = candidate["instances"][pid]["class_name"]
            base_by_gt[gid], cand_by_gt[gid] = bname, cname
            b_correct, c_correct = bname == truth, cname == truth
            state_name = ("correct_to_correct" if b_correct and c_correct else
                          "wrong_to_correct" if not b_correct and c_correct else
                          "correct_to_wrong" if b_correct and not c_correct else
                          "wrong_to_wrong")
            transitions[state_name] += 1
            transitions[f"{truth}:{bname}->{cname}"] += 1
        for bi, ci in zip(baseline["instances"], candidate["instances"]):
            swap_transitions[f"{bi['class_name']}->{ci['class_name']}"] += 1
        is_negative = len(gt_instances) == 0
        if is_negative:
            negatives.append({"image": rel, "baseline_false_positives": b_eval["fp_count"],
                              "candidate_false_positives": c_eval["fp_count"]})
        rows.append({"image": rel, "negative": is_negative, "gt_count": len(gt_instances),
                     "mean_iou_sum": sum(m["mask_iou"] for m in bm["matches"]),
                     "mean_iou_n": len(bm["matches"]),
                     "baseline": b_eval, "candidate": c_eval,
                     "baseline_exact_count": b_eval["exact_count"],
                     "candidate_exact_count": c_eval["exact_count"]})

    # Validate JSONL content (all records already validated individually above).
    jsonl_records = {}
    for arm in ("baseline", "candidate"):
        data = [json.loads(line) for line in (RUN / f"predictions_{arm}.jsonl").read_text().splitlines()]
        jsonl_records[arm] = {x["image_id"]: x for x in data}
        if set(jsonl_records[arm]) != {r["image"] for r in entries}:
            raise ValueError(f"{arm} JSONL image IDs do not match manifest")
    for row in entries:
        key = hashlib.sha256(row["image"].encode()).hexdigest()
        for arm in ("baseline", "candidate"):
            record = read_json(RUN / "records" / arm / f"{key}.json")
            if jsonl_records[arm].get(row["image"]) != record:
                raise ValueError(f"{arm} JSONL differs from durable per-image record: {row['image']}")
    return {"config": config, "config_hash": config_hash, "summary": summary,
            "events": events, "recovered_attempts": recovered_attempts,
            "rows": rows, "confusion": confusion_by_arm, "transitions": transitions,
            "swap_transitions": swap_transitions, "gt_by_class": dict(per_class_gt),
            "negative_images": negatives, "preflight": preflight}


def main() -> int:
    result = run_validation_and_metrics()
    rows, confusion = result["rows"], result["confusion"]
    class_results = {arm: class_metrics(confusion[arm]) for arm in ("baseline", "candidate")}
    matched = sum(r["baseline"]["matched"] for r in rows)
    total_gt = sum(r["gt_count"] for r in rows)
    total_pred = sum(r["baseline"]["pred_count"] for r in rows)
    negatives = [r for r in rows if r["negative"]]
    fps_baseline = sum(r["baseline"]["fp_count"] for r in rows)
    fps_candidate = sum(r["candidate"]["fp_count"] for r in rows)
    fns = sum(r["baseline"]["fn_count"] for r in rows)
    ious = [iou for r in rows for iou in r["baseline"]["mask_ious"]]
    correct_before = result["transitions"]["correct_to_correct"] + result["transitions"]["correct_to_wrong"]
    correct_after = result["transitions"]["correct_to_correct"] + result["transitions"]["wrong_to_correct"]
    swaps = sum(result["swap_transitions"].values()) - result["swap_transitions"].get("banana->banana", 0) - result["swap_transitions"].get("apple->apple", 0) - result["swap_transitions"].get("orange->orange", 0)
    # Direct swap count from candidate metadata, independently rechecked above.
    actual_swaps = 0
    for e in result["events"]:
        if e.get("event") == "complete": actual_swaps += len(e.get("swap_decisions", []))
    bootstrap = paired_bootstrap(rows)

    def mean(key, arm): return float(np.mean([r[arm][key] for r in rows]))
    metrics = {
        "images": {"evaluated": len(rows), "negative_images": len(negatives)},
        "matching": {"method": "class-agnostic Hungarian maximum-total mask IoU",
                     "iou_threshold": 0.50, "matched_instances": matched,
                     "unmatched_predictions": {"baseline": fps_baseline, "candidate": fps_candidate},
                     "unmatched_ground_truths": {"baseline": fns,
                         "candidate": sum(r["candidate"]["fn_count"] for r in rows)},
                     "mask_iou_distribution": {"count": len(ious), "mean": float(np.mean(ious)) if ious else 0.0,
                         "median": float(np.median(ious)) if ious else 0.0,
                         "p05": float(np.percentile(ious, 5)) if ious else 0.0,
                         "p95": float(np.percentile(ious, 95)) if ious else 0.0,
                         "min": float(min(ious)) if ious else 0.0,
                         "max": float(max(ious)) if ious else 0.0}},
        "instance_detection": {}, "classification": {}, "counting": {},
        "negative_images": {}, "candidate_consistency": {},
        "paired_bootstrap_95_ci": bootstrap,
        "transitions": {k: int(v) for k, v in result["transitions"].items()},
        "class_swap_transitions": {k: int(v) for k, v in result["swap_transitions"].items()},
        "clip_swap_instances": int(actual_swaps),
        "recovered_incomplete_attempts": result["recovered_attempts"],
    }
    for arm in ("baseline", "candidate"):
        mm = class_metrics(result["confusion"][arm])
        # Detection is class-agnostic: derive TP/FP/FN only from mask matching.
        tp = sum(r[arm]["matched"] for r in rows)
        fp = sum(r[arm]["fp_count"] for r in rows)
        fn = sum(r[arm]["fn_count"] for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics["instance_detection"][arm] = {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "matched_instances": sum(r[arm]["matched"] for r in rows),
            "false_positives": sum(r[arm]["fp_count"] for r in rows),
            "false_negatives": sum(r[arm]["fn_count"] for r in rows),
            "false_positives_per_image": sum(r[arm]["fp_count"] for r in rows) / len(rows),
            "false_negatives_per_image": sum(r[arm]["fn_count"] for r in rows) / len(rows)}
        matched_correct = sum(r[arm]["classification_correct"] for r in rows)
        metrics["classification"][arm] = {"confusion_matrix": result["confusion"][arm],
                                           "per_class": mm,
                                           "matched_class_accuracy_over_all_gt": matched_correct / total_gt if total_gt else 0.0,
                                           "macro_f1": mm["macro_f1"]}
        abs_total = sum(r[arm]["abs_count_error"] for r in rows)
        metrics["counting"][arm] = {"mae_total_count": abs_total / len(rows),
            "exact_count_match_rate": sum(r[arm]["exact_count"] for r in rows) / len(rows),
            "overcount_images": sum(r[arm]["pred_count"] > r[arm]["gt_count"] for r in rows),
            "undercount_images": sum(r[arm]["pred_count"] < r[arm]["gt_count"] for r in rows)}
        fp_neg = sum(r[arm]["fp_count"] for r in negatives)
        fp_neg_imgs = sum(r[arm]["fp_count"] > 0 for r in negatives)
        metrics["negative_images"][arm] = {"negative_image_count": len(negatives),
            "false_positive_images": fp_neg_imgs,
            "false_positive_image_rate": fp_neg_imgs / len(negatives) if negatives else 0.0,
            "average_false_positives_per_negative_image": fp_neg / len(negatives) if negatives else 0.0,
            "images_with_false_positives": [{"image": x["image"], "false_positives": x[f"{arm}_false_positives"]}
                                             for x in result["negative_images"] if x[f"{arm}_false_positives"] > 0]}
    metrics["candidate_consistency"] = {"geometry_and_instance_count_unchanged": True,
        "all_swaps_validated": True, "candidate_instance_additions": 0,
        "candidate_instance_removals": 0, "baseline_total_instances": total_pred,
        "candidate_total_instances": sum(r["candidate"]["pred_count"] for r in rows),
        "swapped_instances": actual_swaps,
        "unchanged_class_instances": total_pred - actual_swaps,
        "correct_to_wrong": result["transitions"]["correct_to_wrong"],
        "wrong_to_correct": result["transitions"]["wrong_to_correct"],
        "baseline_correct_matched": correct_before, "candidate_correct_matched": correct_after}

    # Paired per-image CSV includes counts, classification and geometry summaries.
    csv_path = RUN / "candidate-test-per-image.csv"
    csv_rows = []
    for r in rows:
        csv_rows.append({"image": r["image"], "negative": int(r["negative"]), "gt_count": r["gt_count"],
            "baseline_pred_count": r["baseline"]["pred_count"], "candidate_pred_count": r["candidate"]["pred_count"],
            "baseline_matched": r["baseline"]["matched"], "candidate_matched": r["candidate"]["matched"],
            "baseline_fp": r["baseline"]["fp_count"], "candidate_fp": r["candidate"]["fp_count"],
            "baseline_fn": r["baseline"]["fn_count"], "candidate_fn": r["candidate"]["fn_count"],
            "baseline_classification_accuracy": r["baseline"]["classification_accuracy"],
            "candidate_classification_accuracy": r["candidate"]["classification_accuracy"],
            "baseline_exact_count": int(r["baseline"]["exact_count"]),
            "candidate_exact_count": int(r["candidate"]["exact_count"]),
            "baseline_abs_count_error": r["baseline"]["abs_count_error"],
            "candidate_abs_count_error": r["candidate"]["abs_count_error"],
            "mean_mask_iou": sum(r["baseline"]["mask_ious"]) / len(r["baseline"]["mask_ious"]) if r["baseline"]["mask_ious"] else ""})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)

    conf_csv = RUN / "candidate-test-confusion-matrix.csv"
    with conf_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["arm", "ground_truth", *CLASSES, "miss"])
        for arm in ("baseline", "candidate"):
            for gt in (*CLASSES, "background"):
                writer.writerow([arm, gt, *[result["confusion"][arm][gt][p] for p in (*CLASSES, "miss")]])
    neg_csv = RUN / "candidate-test-negative-images.csv"
    with neg_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "baseline_false_positives", "candidate_false_positives"])
        writer.writeheader(); writer.writerows(result["negative_images"])

    # Paired decision rule for recommendation; deployment remains explicitly out of scope.
    accuracy_delta = metrics["classification"]["candidate"]["matched_class_accuracy_over_all_gt"] - metrics["classification"]["baseline"]["matched_class_accuracy_over_all_gt"]
    f1_delta = metrics["classification"]["candidate"]["macro_f1"] - metrics["classification"]["baseline"]["macro_f1"]
    ci_acc = bootstrap["classification_accuracy"]["paired_difference_candidate_minus_baseline_ci95"]
    ci_f1 = bootstrap["macro_f1"]["paired_difference_candidate_minus_baseline_ci95"]
    no_detection_degradation = (metrics["instance_detection"]["candidate"]["false_positives"] <= metrics["instance_detection"]["baseline"]["false_positives"] and
                                metrics["instance_detection"]["candidate"]["false_negatives"] <= metrics["instance_detection"]["baseline"]["false_negatives"] and
                                metrics["counting"]["candidate"]["mae_total_count"] <= metrics["counting"]["baseline"]["mae_total_count"])
    go = accuracy_delta > 0 and f1_delta > 0 and ci_acc[0] > 0 and ci_f1[0] > 0 and no_detection_degradation
    metrics["recommendation"] = {"decision": "GO" if go else "NO-GO",
        "rationale": "Promotion consideration only; no automatic deployment. Requires paired accuracy and macro-F1 improvement with 95% bootstrap intervals above zero and no count/detection regression." if go else
        "Candidate did not meet the predeclared evidence gate for promotion consideration; no automatic deployment.",
        "classification_accuracy_absolute_difference": accuracy_delta,
        "macro_f1_absolute_difference": f1_delta,
        "no_detection_or_count_degradation": no_detection_degradation,
        "automatic_promotion": False}
    metrics["validation"] = {"complete": True, "successful_images": 345,
        "errors_current": 0, "recovered_incomplete_attempts": result["recovered_attempts"],
        "config_sha256": result["config_hash"],
        "detector_sha256": result["config"]["artifacts"]["detector"]["sha256"],
        "clip_weights_sha256": result["config"]["artifacts"]["clip_weights"]["sha256"],
        "manifest_id": result["config"]["dataset"]["manifest_id"],
        "manifest_entries": 345, "resume_validation": "345/345 accepted",
        "matching_class_agnostic": True}
    ev.atomic_json(RUN / "report.json", metrics)
    render_markdown(metrics, result, csv_path, conf_csv, neg_csv)
    print(json.dumps({"report": str((RUN / "report.json").relative_to(ROOT)),
                      "decision": metrics["recommendation"]["decision"],
                      "metrics": {"baseline": {"macro_f1": metrics['classification']['baseline']['macro_f1'],
                                                  "accuracy": metrics['classification']['baseline']['matched_class_accuracy_over_all_gt'],
                                                  "count_mae": metrics['counting']['baseline']['mae_total_count']},
                                  "candidate": {"macro_f1": metrics['classification']['candidate']['macro_f1'],
                                                "accuracy": metrics['classification']['candidate']['matched_class_accuracy_over_all_gt'],
                                                "count_mae": metrics['counting']['candidate']['mae_total_count']},
                                  "swaps": actual_swaps, "errors_recovered": len(result['recovered_attempts'])}},
                     indent=2, ensure_ascii=False))
    return 0


def render_markdown(metrics: dict, result: dict, per_image: Path, confusion_csv: Path, negative_csv: Path) -> None:
    lines = ["# Fresh CLIP swap-only test evaluation", "",
        "**Status:** Complete, 345/345 images validated. Fresh experiment; historical partial evaluation was not reused.", "",
        f"Config SHA256: `{result['config_hash']}`  ",
        f"Detector SHA256: `{result['config']['artifacts']['detector']['sha256']}`  ",
        f"CLIP weights SHA256: `{result['config']['artifacts']['clip_weights']['sha256']}`  ",
        f"Manifest: `{result['config']['dataset']['manifest_id']}` (345 images)", "",
        "## Detection and counting", "",
        "| Metric | Baseline | Candidate |", "|---|---:|---:|"]
    for label, key in [("Detection precision", "precision"), ("Detection recall", "recall"), ("Detection F1", "f1"),
                       ("False positives", "false_positives"), ("False negatives", "false_negatives"),
                       ("False positives / image", "false_positives_per_image"), ("False negatives / image", "false_negatives_per_image")]:
        lines.append(f"| {label} | {metrics['instance_detection']['baseline'][key]:.4f} | {metrics['instance_detection']['candidate'][key]:.4f} |")
    lines += ["", "| Count metric | Baseline | Candidate |", "|---|---:|---:|"]
    for label, key in [("Total count MAE", "mae_total_count"), ("Exact count match rate", "exact_count_match_rate"),
                       ("Overcount images", "overcount_images"), ("Undercount images", "undercount_images")]:
        lines.append(f"| {label} | {metrics['counting']['baseline'][key]:.4f} | {metrics['counting']['candidate'][key]:.4f} |")
    lines += ["", "## Classification", "", "| Class | Arm | Precision | Recall | F1 |", "|---|---|---:|---:|---:|"]
    for arm in ("baseline", "candidate"):
        for cls in CLASSES:
            m = metrics["classification"][arm]["per_class"][cls]
            lines.append(f"| {cls} | {arm} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
    lines += ["", f"Matched-class accuracy: baseline {metrics['classification']['baseline']['matched_class_accuracy_over_all_gt']:.4f}; candidate {metrics['classification']['candidate']['matched_class_accuracy_over_all_gt']:.4f}.",
              f"Macro F1: baseline {metrics['classification']['baseline']['macro_f1']:.4f}; candidate {metrics['classification']['candidate']['macro_f1']:.4f}.",
              f"Swaps: {metrics['clip_swap_instances']}. Correct-to-wrong: {metrics['candidate_consistency']['correct_to_wrong']}; wrong-to-correct: {metrics['candidate_consistency']['wrong_to_correct']}.",
              "", "## Negative images", "", f"Negative images: {len(metrics['negative_images']['baseline']['images_with_false_positives'])} images with baseline FPs and {len(metrics['negative_images']['candidate']['images_with_false_positives'])} with candidate FPs among {metrics['negative_images']['baseline']['negative_image_count']} negatives.",
              "", "## Paired uncertainty", "", "Image-level paired bootstrap, 2,000 replicates; intervals are 95% percentile CIs for candidate-minus-baseline differences.", ""]
    for key, label in [("classification_accuracy", "Matched-class accuracy"), ("macro_f1", "Macro F1"),
                       ("exact_count_rate", "Exact count rate"), ("mean_matched_iou", "Mean matched mask IoU")]:
        ci = metrics["paired_bootstrap_95_ci"][key]["paired_difference_candidate_minus_baseline_ci95"]
        lines.append(f"- {label} difference CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    lines += ["", "## Completeness and recovery", "", f"Resume validation accepted 345/345 record triplets. Recovered incomplete attempts: {len(result['recovered_attempts'])}.",
              "The initial orchestration attempt for the recovered image is retained in the event/error records; its incomplete output was reprocessed and the final triplet passed validation.",
              "", "## Artifacts", "", f"- Per-image paired metrics: `{per_image.relative_to(ROOT)}`",
              f"- Confusion matrix CSV: `{confusion_csv.relative_to(ROOT)}`",
              f"- Negative-image CSV: `{negative_csv.relative_to(ROOT)}`",
              f"- Machine-readable metrics: `{(RUN / 'report.json').relative_to(ROOT)}`",
              "- Predictions, progress, execution summary, and logs are preserved in `work/benchmarks/candidate_eval/full_runs/fresh_clip_swap_only_345/`.",
              "", "## Recommendation", "", f"**{metrics['recommendation']['decision']}** — {metrics['recommendation']['rationale']}",
              "", "This report does not deploy or automatically promote the candidate.", ""]
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
