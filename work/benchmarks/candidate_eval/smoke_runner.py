#!/usr/bin/env python3
"""Isolated, hard-limited five-image smoke run for the approved fresh config."""
from __future__ import annotations

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

LIMIT = 5
MIN_RAM_BYTES = 1 * 1024**3
MIN_DISK_BYTES = 1 * 1024**3


def resources() -> dict:
    process = psutil.Process()
    mem = psutil.virtual_memory()
    root_disk = shutil.disk_usage(ROOT)
    cuda_available = torch.cuda.is_available()
    return {
        "mem_available_bytes": int(mem.available),
        "process_rss_bytes": int(process.memory_info().rss),
        "process_uss_bytes": int(process.memory_full_info().uss),
        "process_peak_rss_bytes": int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * 1024),
        "disk_total_bytes": int(root_disk.total),
        "disk_free_bytes": int(root_disk.free),
        "cuda_available": bool(cuda_available),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()) if cuda_available else 0,
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()) if cuda_available else 0,
    }


def require_safe_resources(snapshot: dict, phase: str) -> None:
    if snapshot["mem_available_bytes"] < MIN_RAM_BYTES:
        raise RuntimeError(f"Unsafe available RAM during {phase}: {snapshot['mem_available_bytes']} bytes")
    if snapshot["disk_free_bytes"] < MIN_DISK_BYTES:
        raise RuntimeError(f"Unsafe free disk during {phase}: {snapshot['disk_free_bytes']} bytes")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    config_path = HERE / "config.json"
    config, config_hash = ev.load_and_verify_config(config_path)
    expected_config_hash = "65f552c3b25f545ab20643d986dd0777e8f7d30b482e889d4953ae223b0b53b4"
    if config_hash != expected_config_hash:
        raise RuntimeError(f"Config digest differs from authorized value: {config_hash}")
    if torch.cuda.is_available() or config["detector"]["device"] != "cpu" or config["clip"]["device"] != "cpu":
        raise RuntimeError("Runtime device differs from pinned CPU configuration")

    manifest = json.loads((ROOT / config["dataset"]["manifest_path"]).read_text(encoding="utf-8"))
    entries = manifest["entries"][:LIMIT]
    if len(entries) != LIMIT:
        raise RuntimeError(f"Expected at least {LIMIT} manifest entries, found {len(entries)}")
    output_root = HERE / "smoke_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_root.mkdir(parents=True, exist_ok=False)
    for sub in ("records/baseline", "records/candidate", "records/ground_truth", "records/errors"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    events_path = output_root / "inference_events.jsonl"
    state = {row["image"]: "pending" for row in entries}
    split_root = ROOT / config["dataset"]["split_root"]
    for row in entries:
        image_path, label_path = split_root / row["image"], split_root / row["label"]
        if ev.sha256_file(image_path) != row["image_sha256"]:
            raise RuntimeError(f"First-five image hash mismatch: {row['image']}")
        if ev.sha256_file(label_path) != row["label_sha256"]:
            raise RuntimeError(f"First-five label hash mismatch: {row['label']}")

    before_model = resources()
    require_safe_resources(before_model, "pre-model-load")
    clip_ref = ROOT / "work/weights/hf/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/refs/main"
    if clip_ref.read_text().strip() != config["clip"]["repository_revision"]:
        raise RuntimeError(f"CLIP cache revision mismatch: {clip_ref.read_text().strip()}")
    (output_root / "preflight.json").write_text(json.dumps({
        "config_sha256": config_hash, "config_file_sha256": ev.sha256_file(config_path),
        "smoke_runner_sha256": ev.sha256_file(Path(__file__)), "scope_limit": LIMIT,
        "manifest_order_image_ids": [r["image"] for r in entries],
        "pre_model_resources": before_model, "minimum_available_ram_bytes": MIN_RAM_BYTES,
        "minimum_disk_free_bytes": MIN_DISK_BYTES,
        "offline_hub_mode": True,
    }, indent=2) + "\n", encoding="utf-8")

    clip_load_started = time.perf_counter()
    from clip_check import ClipChecker
    clip_checker = ClipChecker(arch=config["clip"]["arch"], pretrained=config["clip"]["pretrained"], device="cpu")
    clip_load_seconds = time.perf_counter() - clip_load_started
    after_clip_load = resources()
    require_safe_resources(after_clip_load, "post-CLIP-load")
    detector_load_started = time.perf_counter()
    detector_path = ROOT / config["artifacts"]["detector"]["path"]
    detector = YOLO(str(detector_path))
    detector_load_seconds = time.perf_counter() - detector_load_started
    after_model = resources()
    require_safe_resources(after_model, "post-model-load")
    if detector.task != "segment" or tuple(detector.names.values()) != ev.FRUITS:
        raise RuntimeError(f"Unexpected detector task/classes: {detector.task}/{detector.names}")
    model_meta = {
        "detector_load_seconds": detector_load_seconds,
        "clip_load_seconds": clip_load_seconds,
        "detector_task": detector.task,
        "detector_names": {str(k): v for k, v in detector.names.items()},
        "clip_arch": config["clip"]["arch"],
        "clip_pretrained": config["clip"]["pretrained"],
        "clip_cache_ref": (ROOT / "work/weights/hf/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/refs/main").read_text().strip(),
        "clip_weight_sha256": ev.sha256_file(ROOT / config["artifacts"]["clip_weights"]["path"]),
        "post_model_resources": after_model,
        "after_clip_load_resources": after_clip_load,
    }
    ev.atomic_json(output_root / "model_load.json", model_meta)
    ev.atomic_json(progress_path, {"config_sha256": config_hash, "manifest_id": config["dataset"]["manifest_id"],
                                  "scope_limit": LIMIT, "images": state})

    outcomes, stop = [], False
    for ordinal, row in enumerate(entries):
        rel = row["image"]
        image_path, label_path = split_root / rel, split_root / row["label"]
        key = sha256_bytes(rel.encode())
        event = {"ordinal": ordinal, "image": rel, "started_utc": datetime.now(timezone.utc).isoformat(),
                 "status": "running"}
        state[rel] = "running"
        ev.atomic_json(progress_path, {"config_sha256": config_hash, "manifest_id": config["dataset"]["manifest_id"],
                                      "scope_limit": LIMIT, "images": state})
        image_started = time.perf_counter()
        try:
            require_safe_resources(resources(), f"before-image-{ordinal}")
            if ev.sha256_file(image_path) != row["image_sha256"] or ev.sha256_file(label_path) != row["label_sha256"]:
                raise RuntimeError("Image/label hash changed since manifest validation")
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
            clip_started = time.perf_counter()
            clip_rows = ev.score_boxes(clip_checker, image, boxes,
                                      t_other=config["clip"]["t_other_for_second_look"],
                                      t_inner=config["clip"]["t_inner_for_selecting_inner_scores"])
            clip_seconds = time.perf_counter() - clip_started
            baseline_instances, candidate_instances = [], []
            for i, (polygon_raw, box, score, class_id, clip_row) in enumerate(zip(polygons, boxes, scores, classes, clip_rows)):
                class_id = int(class_id)
                if class_id not in range(3):
                    raise ValueError(f"Invalid detector class id {class_id} on instance {i}")
                polygon = np.asarray(polygon_raw, dtype=np.float64).reshape(-1, 2)
                if len(polygon) < 3 or not np.isfinite(polygon).all() or not np.isfinite(box).all() or not np.isfinite(score):
                    raise ValueError(f"Invalid geometry/confidence on instance {i}")
                mask = ev.polygon_to_mask(polygon, width, height)
                common = {"instance_id": f"det_{i:04d}", "confidence": float(score), "bbox": box.tolist(),
                          "polygon": polygon.tolist(), "mask_area": int(mask.sum()),
                          "mask_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest()}
                baseline_instances.append({**common, "class_id": class_id, "class_name": ev.FRUITS[class_id]})
                decision = ev.swap_decision(class_id, clip_row["grouped_probabilities"], config["clip"]["swap_threshold"])
                candidate_instances.append({**common, "class_id": decision["result_class_id"],
                                            "class_name": decision["result_class_name"],
                                            "candidate": {**clip_row, **decision}})
            gt_instances = ev.load_ground_truth(label_path, width, height)
            detector_hash = config["artifacts"]["detector"]["sha256"]
            baseline_record = ev._image_record(row, rel, image_path, label_path, width, height, config_hash,
                                               detector_hash, baseline_instances, len(classes))
            candidate_record = ev._image_record(row, rel, image_path, label_path, width, height, config_hash,
                                                detector_hash, candidate_instances, len(classes))
            candidate_record["ground_truth"] = [{k: v for k, v in x.items() if k != "mask"} for x in gt_instances]
            gt_record = {"image_id": rel, "image_path": str(image_path.relative_to(ROOT)),
                         "image_sha256": row["image_sha256"], "label_path": str(label_path.relative_to(ROOT)),
                         "label_sha256": row["label_sha256"], "width": width, "height": height,
                         "config_sha256": config_hash,
                         "instances": [{**{k: v for k, v in x.items() if k != "mask"},
                                        "mask_rle": ev._encode_mask(x["mask"])} for x in gt_instances]}
            ev.validate_swap_invariants(baseline_record, candidate_record)
            ev.validate_prediction_record(baseline_record)
            ev.validate_prediction_record(candidate_record)
            b_path = output_root / "records/baseline" / f"{key}.json"
            c_path = output_root / "records/candidate" / f"{key}.json"
            g_path = output_root / "records/ground_truth" / f"{key}.json"
            ev.atomic_json(g_path, gt_record)
            ev.atomic_json(b_path, baseline_record)
            ev.atomic_json(c_path, candidate_record)
            if not ev.reusable_completed_pair(b_path, c_path, config_hash, rel,
                                               row["image_sha256"], row["label_sha256"]):
                raise RuntimeError("Persisted record pair failed resume/invariant validation")
            if json.loads(g_path.read_text())["config_sha256"] != config_hash:
                raise RuntimeError("Persisted ground-truth reference failed config hash validation")
            latency = {"detector_seconds": detector_seconds, "clip_seconds": clip_seconds,
                       "total_seconds": time.perf_counter() - image_started}
            state[rel] = "success"
            event.update(status="success", latencies=latency, prediction_count=len(candidate_instances),
                         swapped_instances=[{"instance_id": x["instance_id"],
                                             "from": x["candidate"]["original_class_name"],
                                             "to": x["class_name"],
                                             "probabilities": x["candidate"]["grouped_probabilities"],
                                             "reason": x["candidate"]["reason"]}
                                            for x in candidate_instances if x["candidate"]["swapped"]],
                         invariant_status="passed", resume_validation="accepted")
            outcomes.append({"image": rel, "status": "success", "latencies": latency,
                             "instances": len(candidate_instances),
                             "swap_decisions": event["swapped_instances"],
                             "invariant_status": "passed", "resume_validation": "accepted"})
        except Exception as exc:
            state[rel] = "error"
            message = {"image_id": rel, "image_path": str(image_path.relative_to(ROOT)),
                       "image_sha256": row["image_sha256"], "label_path": str(label_path.relative_to(ROOT)),
                       "label_sha256": row["label_sha256"], "config_sha256": config_hash,
                       "inference_status": "error", "exception_type": type(exc).__name__,
                       "exception": str(exc), "traceback": traceback.format_exc()}
            ev.atomic_json(output_root / "records/errors" / f"{key}.json", message)
            event.update(status="error", exception_type=type(exc).__name__, exception=str(exc),
                         traceback=traceback.format_exc(), latency_seconds=time.perf_counter() - image_started)
            outcomes.append({"image": rel, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            stop = True
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        ev.atomic_json(progress_path, {"config_sha256": config_hash, "manifest_id": config["dataset"]["manifest_id"],
                                      "scope_limit": LIMIT, "images": state})
        if stop:
            break

    final_resources = resources()
    summary = {"config_sha256": config_hash, "smoke_runner_sha256": ev.sha256_file(Path(__file__)),
               "scope_limit": LIMIT, "attempted": len(outcomes),
               "completed": sum(x["status"] == "success" for x in outcomes),
               "failed": sum(x["status"] == "error" for x in outcomes),
               "model_load": model_meta, "resources_before_models": before_model,
               "resources_after_models": after_model, "resources_after_run": final_resources,
               "images": outcomes}
    ev.atomic_json(output_root / "smoke_summary.json", summary)
    print(json.dumps({"output_dir": str(output_root.relative_to(ROOT)), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["completed"] == LIMIT else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "aborted", "exception_type": type(exc).__name__, "exception": str(exc)}, indent=2))
        raise
