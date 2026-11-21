#!/usr/bin/env python3
"""Train an offline Green Banana/Apple segmentation candidate.

The added BananaID masks are SAM pseudo-labels. This script never changes the
backend checkpoint and refuses to overwrite an existing Ultralytics run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "work/obb/seg_green_v4_candidate/data"
INITIAL = ROOT / "runs/seg/seg_boundary_candidate_gpu/weights/best.pt"
EXPECTED_INITIAL_SHA = "6da28310bb7627d00dfa574cc36147d215afb48e4d0c3d0705e62ecb05917982"
PROJECT = ROOT / "runs/seg"


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="seg_green_v4_candidate")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary_path = DATA.parent / "candidate_summary.json"
    if not DATA.joinpath("seg.yaml").is_file() or not summary_path.is_file():
        raise SystemExit(f"Missing prepared pseudo-label candidate: {DATA}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "CANDIDATE_READY_FOR_OFFLINE_TRAINING":
        raise SystemExit("Candidate status is not ready for offline training")
    if not INITIAL.is_file() or digest(INITIAL) != EXPECTED_INITIAL_SHA:
        raise SystemExit("The expected current backend initialization checkpoint is missing or has changed")
    run_dir = PROJECT / args.name
    if run_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")

    import torch
    from ultralytics import YOLO

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(INITIAL))
    if model.task != "segment" or set(model.names.values()) != {"banana", "apple", "orange"}:
        raise SystemExit(f"Unexpected initialization model: task={model.task}, names={model.names}")
    model.train(
        data=str(DATA / "seg.yaml"), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=device, workers=0, cache=False, amp=device != "cpu",
        seed=20261120, deterministic=True, optimizer="AdamW", lr0=0.0002,
        lrf=0.01, weight_decay=0.0005, patience=6, close_mosaic=5,
        mosaic=1.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        project=str(PROJECT), name=args.name, exist_ok=False, plots=True,
    )
    print(f"Candidate training completed at {run_dir}", flush=True)
    print("Production/backend weights were not changed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        raise
