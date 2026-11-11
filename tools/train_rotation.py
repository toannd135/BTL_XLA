"""GPU-only Colab experiment with exact quarter turns; never replaces the base model."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_resume_checkpoint(checkpoint, expected_epochs):
    """Reject finalized/incompatible weights before YOLO can fall back to a new run."""
    epoch = checkpoint.get("epoch", -1)
    if epoch is None or epoch < 0 or checkpoint.get("optimizer") is None:
        raise ValueError("Checkpoint is finalized or lacks optimizer state; it cannot resume. "
                         "If this run completed, use evaluate with --split val instead.")
    epochs = checkpoint.get("train_args", {}).get("epochs")
    if epochs != expected_epochs:
        raise ValueError(f"Checkpoint plans {epochs} epochs, but this experiment requires {expected_epochs}. "
                         "Inspect the previous run configuration; use a new run name for a different recipe.")
    if epoch + 1 >= epochs:
        raise ValueError("Checkpoint has reached its final epoch; use evaluate with --split val.")
    return epoch + 1


def configure(root, use_synthetic):
    import yaml
    manifest = json.loads((root / "rotation_manifest.json").read_text())
    if use_synthetic and not manifest.get("synthetic_review_approved", False):
        raise ValueError("Draft synthetic data is not approved: use the rotation-only experiment. Supply a curated dataset before enabling synthesis.")
    paths = manifest["train_files"] + (manifest["draft_synthetic_files"] if use_synthetic else [])
    absolute = [(root / p).resolve() for p in paths]
    if not absolute or not all(p.is_file() and p.is_relative_to(root.resolve()) for p in absolute):
        raise ValueError("Missing or invalid packaged training paths")
    listing = root / "train_current.txt"
    listing.write_text("\n".join(str(p) for p in absolute) + "\n")
    config = {"path": str(root / "dataset"), "train": str(listing),
              "val": ["val/images", "val_orientation/images"], "test": "test/images",
              "names": {0:"banana",1:"apple",2:"orange"}}
    path = root / "data_current.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return manifest, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "evaluate", "export"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="rotation_v1")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Opt in only after reviewing the draft synthetic data")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best",
                        help="Select on validation first; test evaluates only baseline and this checkpoint")
    args = parser.parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for this script. Use a Colab GPU runtime; do not train on the laptop CPU.")
    if ultralytics.__version__ != "8.4.180":
        raise RuntimeError("Use ultralytics==8.4.180 to match the baseline")
    if args.batch <= 0 or Path(args.name).name != args.name:
        raise ValueError("Invalid batch size or run name")
    root = args.root.resolve()
    manifest, data = configure(root, args.use_synthetic)
    base = root / "exports/base.pt"
    assert sha256(base) == manifest["source_hashes"]["base_model"]
    suffix = "_smoke" if args.mode == "smoke" else ""
    run_dir = args.project / (args.name + suffix)
    identity = {"base_model_sha256": sha256(base), "manifest_sha256": sha256(root / "rotation_manifest.json"),
                "use_synthetic": args.use_synthetic, "ultralytics": ultralytics.__version__,
                "purpose": "quarter-turn augmented fine-tuning; draft synthesis is opt-in"}
    if args.mode in {"smoke", "train"}:
        args.project.mkdir(parents=True, exist_ok=True)
        identity_path = args.project / (args.name + suffix + "_identity.json")
        if args.resume:
            previous = json.loads(identity_path.read_text())
            if previous != identity:
                raise ValueError("Cannot resume with changed model, data, or synthetic setting")
            checkpoint = run_dir / "weights/last.pt"
            if not checkpoint.is_file():
                raise ValueError("No checkpoint to resume; use a new run name")
            model = YOLO(str(checkpoint))
            completed = validate_resume_checkpoint(model.ckpt, 1 if args.mode == "smoke" else 25)
            print(f"Verified resume: {completed} completed epochs; "
                  f"target {model.ckpt['train_args']['epochs']} total epochs", flush=True)
            model.train(resume=True, data=str(data), batch=args.batch, workers=2, device=0)
        else:
            if run_dir.exists():
                raise ValueError("Run exists; resume an interrupted run or choose a new name")
            identity_path.write_text(json.dumps(identity, indent=2))
            model = YOLO(str(base))
            model.train(data=str(data), epochs=1 if args.mode == "smoke" else 25,
                        imgsz=640, batch=args.batch, optimizer="SGD", lr0=0.001, warmup_epochs=1,
                        seed=42, deterministic=True, patience=8, degrees=15,
                        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, close_mosaic=5,
                        workers=2, device=0, amp=True, cache=False, plots=True,
                        project=str(args.project), name=args.name + suffix, exist_ok=False,
                        save=True, save_period=5)
        shutil.copy2(identity_path, run_dir / "input_identity.json")
        print("Finished:", run_dir)
        return

    if not (run_dir / "input_identity.json").is_file():
        raise ValueError("Training not completed or run identity missing")
    if json.loads((run_dir / "input_identity.json").read_text()) != identity:
        raise ValueError("Evaluation settings differ from the training identity")
    if args.mode == "export":
        destination = run_dir / "candidate"
        destination.mkdir(exist_ok=True)
        for name in ("best", "last"):
            shutil.copy2(run_dir / f"weights/{name}.pt", destination / f"det_rotation_{name}.pt")
        print("Candidates saved for review, base model unchanged:", destination)
        return

    # Validation before checkpoint selection; original legacy test only for a final regression check.
    held = sorted((root / "dataset/neg_test/images").glob("*.jpg"))
    rows = []
    candidates = [("baseline", base)]
    for checkpoint in (("best", "last") if args.split == "val" else (args.checkpoint,)):
        candidates.append((checkpoint, run_dir / f"weights/{checkpoint}.pt"))
    for tag, weights in candidates:
        model = YOLO(str(weights))
        metrics = model.val(data=str(data), split=args.split, device=0, imgsz=640,
                            batch=args.batch, workers=2, conf=0.001, iou=0.6,
                            agnostic_nms=True, quantize=16, plots=False,
                            project=str(run_dir / "evaluation"), name=tag+"_"+args.split, exist_ok=True)
        row = {"model":tag, "split":args.split, "mAP50":float(metrics.box.map50),
               "mAP50_95":float(metrics.box.map)}
        for cls, value in zip(metrics.box.ap_class_index, metrics.box.ap50):
            row["AP50_"+model.names[int(cls)]] = float(value)
        # The established 71 images remain a regression set, never used for training.
        bad = 0
        for start in range(0,len(held),8):
            results = model.predict([str(p) for p in held[start:start+8]], stream=True,
                                    imgsz=640, conf=0.4, iou=0.6, agnostic_nms=True,
                                    device=0, quantize=16, verbose=False)
            bad += sum(len(result.boxes)>0 for result in results)
        row.update(background_false_positive_images=bad,background_images=len(held),background_conf=0.4)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    (run_dir / ("comparison_"+args.split+".json")).write_text(json.dumps(rows,indent=2))
    print("Legacy labels still limit these metrics. Check new real multi-fruit validation scenes before accepting a model.")


if __name__ == "__main__":
    main()
