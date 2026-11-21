#!/usr/bin/env python3
"""Validation-only, same-split comparison of V2 and V3 YOLO11s-Seg."""

import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
V2 = Path("/mnt/01DC7E1E309A75E0/runs/segment/runs/seg/seg_v2_finetune/weights/best.pt")
V3 = ROOT / "runs/seg/seg_v3_qg_controlled/weights/best.pt"
DATA = ROOT / "work/audit/seg_data_training_prep_v1_qg.yaml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metric_row(metrics, family: str) -> dict:
    values = getattr(metrics, family)
    ids = [int(x) for x in values.ap_class_index]
    names = metrics.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {
        str(names[c]): {
            "precision": float(values.class_result(i)[0]),
            "recall": float(values.class_result(i)[1]),
            "map50": float(values.class_result(i)[2]),
            "map50_95": float(values.class_result(i)[3]),
        }
        for i, c in enumerate(ids)
    }


def validate(tag: str, checkpoint: Path, run_name: str) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (ROOT / "runs/val" / run_name).exists():
        raise FileExistsError(ROOT / "runs/val" / run_name)
    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(DATA),
        split="val",
        imgsz=640,
        batch=1,
        device=0,
        workers=2,
        cache=False,
        amp=True,
        conf=0.001,
        iou=0.70,
        plots=True,
        project=str(ROOT / "runs/val"),
        name=run_name,
        exist_ok=False,
        verbose=True,
    )
    result = {
        "tag": tag,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "split": "val",
        "data_yaml": str(DATA),
        "settings": {
            "imgsz": 640,
            "batch": 1,
            "device": 0,
            "workers": 2,
            "cache": False,
            "amp": True,
            "conf": 0.001,
            "iou": 0.70,
        },
        "overall": {k: float(v) for k, v in metrics.results_dict.items()},
        "per_class_box": metric_row(metrics, "box"),
        "per_class_mask": metric_row(metrics, "seg"),
        "val_output_dir": str(ROOT / "runs/val" / run_name),
    }
    del metrics, model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3060 Laptop GPU":
        raise RuntimeError("Host CUDA preflight failed; refusing validation on a fallback device")
    results = [
        validate("v2", V2, "seg_v2_on_v3_validation"),
        validate("v3_best", V3, "seg_v3_best_on_v3_validation"),
    ]
    payload = {
        "schema_version": "1.0",
        "comparison": "V2 and V3 best, same 301-image validation split; test not loaded",
        "runtime": {
            "python_cuda": True,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "results": results,
    }
    out = ROOT / "work/audit/seg_v3_validation_comparison.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
