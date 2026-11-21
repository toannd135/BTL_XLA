#!/usr/bin/env python3
"""Guarded, resumable full-run orchestrator for the pinned fresh experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ["HF_HOME"] = str(ROOT / "work/weights/hf")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "backend"))

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO
from detection import detect_seg

import evaluator as ev

EXPECTED_CONFIG_SHA256 = "65f552c3b25f545ab20643d986dd0777e8f7d30b482e889d4953ae223b0b53b4"
MIN_RAM_BYTES = 1 * 1024**3
MIN_DISK_BYTES = 1 * 1024**3
RUN_NAME = "fresh_clip_swap_only_345"


def resources() -> dict:
    proc = psutil.Process()
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(ROOT)
    cuda = torch.cuda.is_available()
    return {
        "ram_available_bytes": int(vm.available),
        "process_rss_bytes": int(proc.memory_info().rss),
        "process_uss_bytes": int(proc.memory_full_info().uss),
        "process_peak_rss_bytes": int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * 1024),
        "disk_total_bytes": int(disk.total), "disk_free_bytes": int(disk.free),
        "cuda_available": bool(cuda), "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()) if cuda else 0,
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()) if cuda else 0,
    }


def check_resources(phase: str) -> dict:
    snapshot = resources()
    if snapshot["ram_available_bytes"] < MIN_RAM_BYTES:
        raise RuntimeError(f"Stop condition: less than 1 GiB RAM available at {phase}: {snapshot}")
    if snapshot["disk_free_bytes"] < MIN_DISK_BYTES:
        raise RuntimeError(f"Stop condition: less than 1 GiB disk free at {phase}: {snapshot}")
    return snapshot


def safe_json(path: Path, value) -> None:
    check_resources(f"before atomic write {path.name}")
    ev.atomic_json(path, value)
    check_resources(f"after atomic write {path.name}")


def append_event(path: Path, event: dict) -> None:
    check_resources("before event append")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    check_resources("after event append")


def sha_key(image_rel: str) -> str:
    return hashlib.sha256(image_rel.encode()).hexdigest()


def valid_triplet(run_dir: Path, row: dict, config_hash: str) -> bool:
    key = sha_key(row["image"])
    b_path = run_dir / "records/baseline" / f"{key}.json"
    c_path = run_dir / "records/candidate" / f"{key}.json"
    g_path = run_dir / "records/ground_truth" / f"{key}.json"
    if not all(p.is_file() for p in (b_path, c_path, g_path)):
        return False
    try:
        if not ev.reusable_completed_pair(b_path, c_path, config_hash, row["image"],
                                          row["image_sha256"], row["label_sha256"]):
            return False
        b = json.loads(b_path.read_text(encoding="utf-8"))
        c = json.loads(c_path.read_text(encoding="utf-8"))
        g = json.loads(g_path.read_text(encoding="utf-8"))
        if g.get("config_sha256") != config_hash or g.get("image_id") != row["image"]:
            return False
        if g.get("image_sha256") != row["image_sha256"] or g.get("label_sha256") != row["label_sha256"]:
            return False
        if g.get("width") != b.get("width") or g.get("height") != b.get("height"):
            return False
        if b.get("prediction_count_after_clip") != len(b.get("instances", [])):
            return False
        if c.get("prediction_count_after_clip") != len(c.get("instances", [])):
            return False
        for rec in (b, c):
            for instance in rec["instances"]:
                mask = ev.polygon_to_mask(instance["polygon"], rec["width"], rec["height"])
                if int(mask.sum()) != instance["mask_area"]:
                    return False
                if hashlib.sha256(mask.tobytes(order="C")).hexdigest() != instance["mask_sha256"]:
                    return False
        return True
    except Exception:
        return False


def load_state(progress_path: Path, entries: list[dict], config_hash: str, manifest_id: str) -> dict[str, str]:
    expected = {row["image"] for row in entries}
    if not progress_path.exists():
        return {row["image"]: "pending" for row in entries}
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    if data.get("config_sha256") != config_hash or data.get("manifest_id") != manifest_id:
        raise RuntimeError("Existing full-run progress belongs to another config or manifest")
    states = data.get("images", {})
    if set(states) != expected:
        raise RuntimeError("Existing progress manifest does not match the exact 345-image set")
    return states


def make_records(config, config_hash, row, rel, image_path, label_path, width, height,
                 baseline_instances, candidate_instances, gt_instances, pred_count):
    detector_hash = config["artifacts"]["detector"]["sha256"]
    baseline = ev._image_record(row, rel, image_path, label_path, width, height, config_hash,
                                detector_hash, baseline_instances, pred_count)
    candidate = ev._image_record(row, rel, image_path, label_path, width, height, config_hash,
                                 detector_hash, candidate_instances, pred_count)
    candidate["ground_truth"] = [{k: v for k, v in x.items() if k != "mask"} for x in gt_instances]
    gt = {"image_id": rel, "image_path": str(image_path.relative_to(ROOT)),
          "image_sha256": row["image_sha256"], "label_path": str(label_path.relative_to(ROOT)),
          "label_sha256": row["label_sha256"], "width": width, "height": height,
          "config_sha256": config_hash,
          "instances": [{**{k: v for k, v in x.items() if k != "mask"},
                         "mask_rle": ev._encode_mask(x["mask"])} for x in gt_instances]}
    return baseline, candidate, gt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE / "full_runs" / RUN_NAME)
    args = parser.parse_args()
    config_path = HERE / "config.json"

    # Hashes, package/runtime versions, manifest file, every image and label are checked before model loading.
    config, config_hash = ev.load_and_verify_config(config_path)
    if config_hash != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(f"Config hash differs from authorization: {config_hash}")
    dry = ev.dry_run(config_path)
    if dry["status"] != "ok" or dry["manifest_entries"] != 345 or dry["unique_images"] != 345:
        raise RuntimeError(f"Preflight manifest validation failed: {dry}")
    if torch.cuda.is_available() or config["detector"]["device"] != "cpu" or config["clip"]["device"] != "cpu":
        raise RuntimeError("Runtime device does not match pinned CPU configuration")

    manifest_path = ROOT / config["dataset"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    if len(entries) != 345 or len({r["image"] for r in entries}) != 345:
        raise RuntimeError("Manifest must contain exactly 345 unique images")

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("records/baseline", "records/candidate", "records/ground_truth", "records/errors"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    events_path = run_dir / "inference_events.jsonl"
    state = load_state(progress_path, entries, config_hash, config["dataset"]["manifest_id"])

    # Refuse conflicting untracked files; only records keyed by the locked manifest are permitted.
    expected_keys = {sha_key(row["image"]) + ".json" for row in entries}
    for role in ("baseline", "candidate", "ground_truth"):
        existing = {p.name for p in (run_dir / "records" / role).glob("*.json")}
        if existing - expected_keys:
            raise RuntimeError(f"Unexpected/conflicting {role} records in run dir: {sorted(existing - expected_keys)[:5]}")
    check_resources("pre-model-load")
    cache_ref = ROOT / "work/weights/hf/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/refs/main"
    if cache_ref.read_text().strip() != config["clip"]["repository_revision"]:
        raise RuntimeError("Cached CLIP revision differs from pinned revision")
    preflight = {"config_sha256": config_hash, "config_file_sha256": ev.sha256_file(config_path),
                 "manifest_id": config["dataset"]["manifest_id"], "manifest_entries": len(entries),
                 "dry_run": dry, "run_dir": str(run_dir.relative_to(ROOT)),
                 "smoke_dir": "work/benchmarks/candidate_eval/smoke_runs/20261118T153115.170985Z",
                 "minimum_ram_bytes": MIN_RAM_BYTES, "minimum_disk_bytes": MIN_DISK_BYTES,
                 "resource_snapshot": resources(), "full_runner_sha256": ev.sha256_file(Path(__file__))}
    safe_json(run_dir / "preflight.json", preflight)

    # Never load or use a different artifact: both hashes were checked before these loads.
    clip_start = time.perf_counter()
    from clip_check import ClipChecker
    checker = ClipChecker(arch=config["clip"]["arch"], pretrained=config["clip"]["pretrained"], device="cpu")
    clip_load_seconds = time.perf_counter() - clip_start
    after_clip = check_resources("after-CLIP-load")
    detector_start = time.perf_counter()
    detector = YOLO(str(ROOT / config["artifacts"]["detector"]["path"]))
    detector_load_seconds = time.perf_counter() - detector_start
    after_models = check_resources("after-detector-load")
    if detector.task != "segment" or tuple(detector.names.values()) != ev.FRUITS:
        raise RuntimeError(f"Detector task/classes mismatch: {detector.task}/{detector.names}")
    model_load = {"clip_load_seconds": clip_load_seconds, "detector_load_seconds": detector_load_seconds,
                  "detector_task": detector.task, "detector_names": {str(k): v for k, v in detector.names.items()},
                  "clip_arch": config["clip"]["arch"], "clip_pretrained": config["clip"]["pretrained"],
                  "clip_revision": cache_ref.read_text().strip(),
                  "clip_weights_sha256": ev.sha256_file(ROOT / config["artifacts"]["clip_weights"]["path"]),
                  "after_clip_resources": after_clip, "after_models_resources": after_models}
    safe_json(run_dir / "model_load.json", model_load)
    safe_json(progress_path, {"config_sha256": config_hash, "manifest_id": config["dataset"]["manifest_id"],
                              "expected_images": 345, "images": state})

    split_root = ROOT / config["dataset"]["split_root"]
    attempted = 0
    failures = []
    latencies = []
    for ordinal, row in enumerate(entries):
        rel = row["image"]
        key = sha_key(rel)
        base_path = run_dir / "records/baseline" / f"{key}.json"
        cand_path = run_dir / "records/candidate" / f"{key}.json"
        gt_path = run_dir / "records/ground_truth" / f"{key}.json"
        if state.get(rel) == "success" and valid_triplet(run_dir, row, config_hash):
            append_event(events_path, {"event": "resume_validated_skip", "ordinal": ordinal,
                                       "image": rel, "utc": datetime.now(timezone.utc).isoformat(),
                                       "reason": "previous successful triplet passed schema, hash and invariant validation"})
            continue
        # A triplet that validates can be recovered even if interruption preceded the progress update.
        if all(p.is_file() for p in (base_path, cand_path, gt_path)) and valid_triplet(run_dir, row, config_hash):
            state[rel] = "success"
            safe_json(progress_path, {"config_sha256": config_hash, "manifest_id": config["dataset"]["manifest_id"],
                                      "expected_images": 345, "images": state})
            append_event(events_path, {"event": "recovered_valid_triplet", "ordinal": ordinal,
                                       "image": rel, "utc": datetime.now(timezone.utc).isoformat(),
                                       "inference_repeated": False})
            continue

        image_path, label_path = split_root / rel, split_root / row["label"]
        start_event = {"event": "start", "ordinal": ordinal, "image": rel,
                       "utc": datetime.now(timezone.utc).isoformat(),
                       "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"]}
        append_event(events_path, start_event)
        state[rel] = "running"
        safe_json(progress_path, {"config_sha256": config_hash, "manifest_id": config["dataset"]["manifest_id"],
                                  "expected_images": 345, "images": state})
        attempted += 1
        started = time.perf_counter()
        try:
            check_resources(f"before image {ordinal}")
            if ev.sha256_file(image_path) != row["image_sha256"] or ev.sha256_file(label_path) != row["label_sha256"]:
                raise RuntimeError("Source image or label hash changed")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode image: {image_path}")
            height, width = image.shape[:2]
            detector_started = time.perf_counter()
            polygons, boxes, scores, classes, _ = detect_seg(
                detector, image, conf=config["detector"]["conf"], imgsz=config["detector"]["imgsz"],
                iou=config["detector"]["iou"], device="cpu", half=False)
            detector_seconds = time.perf_counter() - detector_started
            boxes = np.asarray(boxes).reshape(-1, 4)
            scores = np.asarray(scores, dtype=np.float64)
            classes = np.asarray(classes, dtype=np.int64)
            if not (len(polygons) == len(boxes) == len(scores) == len(classes)):
                raise ValueError("Detector output arrays have inconsistent lengths")

            # Baseline instances are frozen and persisted from this one detector invocation.
            baseline_instances = []
            for i, (poly_raw, box, score, class_id) in enumerate(zip(polygons, boxes, scores, classes)):
                class_id = int(class_id)
                if class_id not in range(3):
                    raise ValueError(f"Invalid detector class id {class_id}")
                polygon = np.asarray(poly_raw, dtype=np.float64).reshape(-1, 2)
                if len(polygon) < 3 or not np.isfinite(polygon).all() or not np.isfinite(box).all() or not np.isfinite(score):
                    raise ValueError(f"Invalid detector geometry/confidence on instance {i}")
                mask = ev.polygon_to_mask(polygon, width, height)
                baseline_instances.append({"instance_id": f"det_{i:04d}", "class_id": class_id,
                    "class_name": ev.FRUITS[class_id], "confidence": float(score), "bbox": box.tolist(),
                    "polygon": polygon.tolist(), "mask_area": int(mask.sum()),
                    "mask_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest()})
            detector_hash = config["artifacts"]["detector"]["sha256"]
            baseline_record = ev._image_record(row, rel, image_path, label_path, width, height,
                                               config_hash, detector_hash, baseline_instances, len(classes))
            ev.validate_prediction_record(baseline_record)
            safe_json(base_path, baseline_record)

            # Candidate is derived from the same boxes/classes/scores/polygons; no second detector pass.
            clip_started = time.perf_counter()
            clip_rows = ev.score_boxes(checker, image, boxes,
                                      t_other=config["clip"]["t_other_for_second_look"],
                                      t_inner=config["clip"]["t_inner_for_selecting_inner_scores"])
            clip_seconds = time.perf_counter() - clip_started
            candidate_instances = []
            for original, clip_row in zip(baseline_instances, clip_rows):
                decision = ev.swap_decision(original["class_id"], clip_row["grouped_probabilities"],
                                            config["clip"]["swap_threshold"])
                candidate_instances.append({**{k: original[k] for k in
                    ("instance_id", "confidence", "bbox", "polygon", "mask_area", "mask_sha256")},
                    "class_id": decision["result_class_id"], "class_name": decision["result_class_name"],
                    "candidate": {**clip_row, **decision}})
            gt_instances = ev.load_ground_truth(label_path, width, height)
            candidate_record = ev._image_record(row, rel, image_path, label_path, width, height,
                                                config_hash, detector_hash, candidate_instances, len(classes))
            candidate_record["ground_truth"] = [{k: v for k, v in x.items() if k != "mask"} for x in gt_instances]
            gt_record = {"image_id": rel, "image_path": str(image_path.relative_to(ROOT)),
                "image_sha256": row["image_sha256"], "label_path": str(label_path.relative_to(ROOT)),
                "label_sha256": row["label_sha256"], "width": width, "height": height,
                "config_sha256": config_hash,
                "instances": [{**{k: v for k, v in x.items() if k != "mask"},
                               "mask_rle": ev._encode_mask(x["mask"])} for x in gt_instances]}
            ev.validate_swap_invariants(baseline_record, candidate_record)
            ev.validate_prediction_record(candidate_record)
            safe_json(cand_path, candidate_record)
            safe_json(gt_path, gt_record)
            # Read-back all artifacts and validate before any success state is written.
            if not valid_triplet(run_dir, row, config_hash):
                raise RuntimeError("Persisted triplet failed schema, hash, mask or swap invariant validation")
            completion = {"event": "complete", "ordinal": ordinal, "image": rel,
                "utc": datetime.now(timezone.utc).isoformat(), "inference_calls": {"detector": 1},
                "prediction_count": len(candidate_instances), "detector_seconds": detector_seconds,
                "clip_seconds": clip_seconds, "total_seconds": time.perf_counter() - started,
                "swap_decisions": [{"instance_id": x["instance_id"], "from": x["candidate"]["original_class_name"],
                    "to": x["class_name"], "scores": x["candidate"]["grouped_probabilities"],
                    "reason": x["candidate"]["reason"]} for x in candidate_instances if x["candidate"]["swapped"]],
                "invariants": "passed", "resume_validation": "accepted"}
            append_event(events_path, completion)
            state[rel] = "success"
            safe_json(progress_path, {"config_sha256": config_hash, "manifest_id": config["dataset"]["manifest_id"],
                                      "expected_images": 345, "images": state})
            latencies.append({"image": rel, "detector_seconds": detector_seconds,
                              "clip_seconds": clip_seconds, "total_seconds": completion["total_seconds"]})
        except Exception as exc:
            state[rel] = "error"
            error = {"image_id": rel, "image_path": str(image_path.relative_to(ROOT)),
                "image_sha256": row["image_sha256"], "label_path": str(label_path.relative_to(ROOT)),
                "label_sha256": row["label_sha256"], "config_sha256": config_hash,
                "inference_status": "error", "exception_type": type(exc).__name__,
                "exception": str(exc), "traceback": traceback.format_exc()}
            safe_json(run_dir / "records/errors" / f"{key}.json", error)
            append_event(events_path, {"event": "error", "ordinal": ordinal, "image": rel,
                "utc": datetime.now(timezone.utc).isoformat(), "exception_type": type(exc).__name__,
                "exception": str(exc), "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started})
            safe_json(progress_path, {"config_sha256": config_hash, "manifest_id": config["dataset"]["manifest_id"],
                                      "expected_images": 345, "images": state})
            failures.append({"image": rel, "exception_type": type(exc).__name__, "exception": str(exc)})
            # Preserve all preceding work and stop at the first failure for an explicit resume/recovery decision.
            break

    summary = {"config_sha256": config_hash, "run_dir": str(run_dir.relative_to(ROOT)),
        "full_runner_sha256": ev.sha256_file(Path(__file__)), "expected_images": 345,
        "successful_images": sum(value == "success" for value in state.values()),
        "error_images": sum(value == "error" for value in state.values()),
        "pending_or_running_images": sum(value in ("pending", "running") for value in state.values()),
        "attempted_this_process": attempted, "failures": failures,
        "model_load": model_load, "latencies_this_process": latencies,
        "resources_final": resources(), "completed_utc": datetime.now(timezone.utc).isoformat()}
    safe_json(run_dir / "execution_summary.json", summary)
    if summary["successful_images"] == 345 and not failures:
        # JSONL exports are produced only for a complete record set.
        ev._export_jsonl(run_dir / "records/baseline", run_dir / "predictions_baseline.jsonl")
        ev._export_jsonl(run_dir / "records/candidate", run_dir / "predictions_candidate.jsonl")
        summary["jsonl_exports"] = "complete"
        safe_json(run_dir / "execution_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["successful_images"] == 345 and not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "aborted", "exception_type": type(exc).__name__, "exception": str(exc)}, indent=2))
        raise
