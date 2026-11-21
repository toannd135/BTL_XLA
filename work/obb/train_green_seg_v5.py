#!/usr/bin/env python3
"""Fine-tune an isolated green-fruit segmentation candidate with added apple masks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/obb/seg_green_v5_candidate/data"
SUMMARY = ROOT / "work/obb/seg_green_v5_candidate/candidate_summary.json"
INITIAL = ROOT / "runs/seg/seg_green_v4_candidate/weights/best.pt"
PROJECT = ROOT / "runs/seg"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="seg_green_v5_candidate")
    args = parser.parse_args()
    run_dir = PROJECT / args.name
    if not DATA.joinpath("seg.yaml").is_file() or not SUMMARY.is_file():
        raise SystemExit("Missing isolated v5 candidate dataset")
    if run_dir.exists():
        raise SystemExit(f"Refusing to overwrite run directory: {run_dir}")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary.get("status") != "CANDIDATE_READY_FOR_OFFLINE_TRAINING":
        raise SystemExit("Candidate status does not permit training")
    if not INITIAL.is_file():
        raise SystemExit(f"Initialization checkpoint missing: {INITIAL}")
    initial_sha = sha256(INITIAL)

    import torch
    from ultralytics import YOLO

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(INITIAL))
    if model.task != "segment" or model.names != {0: "banana", 1: "apple", 2: "orange"}:
        raise SystemExit(f"Unexpected initialization model task/classes: {model.task}, {model.names}")
    results = model.train(
        data=str(DATA / "seg.yaml"), epochs=args.epochs, imgsz=640, batch=1,
        device=device, workers=0, cache=False, amp=device != "cpu",
        seed=20261120, deterministic=True, optimizer="AdamW", lr0=0.0001,
        lrf=0.01, weight_decay=0.0005, patience=6, close_mosaic=5,
        mosaic=1.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        project=str(PROJECT), name=args.name, exist_ok=False, plots=True,
        val=True, split="val",
    )
    report = {
        "status": "TRAINING_COMPLETED",
        "run_dir": str(run_dir),
        "data_yaml": str(DATA / "seg.yaml"),
        "initial_checkpoint": str(INITIAL),
        "initial_checkpoint_sha256": initial_sha,
        "device": device,
        "requested_epochs": args.epochs,
        "best_checkpoint": str(run_dir / "weights/best.pt"),
        "results_save_dir": str(getattr(results, "save_dir", run_dir)),
        "test_used_during_training": False,
    }
    (run_dir / "green_v5_training_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
