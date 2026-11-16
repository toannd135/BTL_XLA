"""Bounded three-class Colab experiment; never promote a model automatically."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("smoke", "train", "val", "test"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--name", default="all3_balanced_v2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--eval-imgsz", type=int, default=1280)
    p.add_argument("--checkpoint", choices=("baseline", "trial_v1", "trial_v2"), default="trial_v2")
    p.add_argument("--no-auto-resume", action="store_true")
    args = p.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        raise ValueError("NAME only allows letters, numbers, underscore, hyphen")
    if not 1 <= args.epochs <= 60 or not 1 <= args.batch <= 16 or args.imgsz not in (640, 1024, 1280) or args.eval_imgsz not in (640, 1024, 1280):
        raise ValueError("Invalid train/evaluation configuration")
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("Select T4 GPU in Colab. This runner will not train or evaluate the whole dataset on laptop CPU.")
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for info in manifest["models"].values():
        assert sha(root / info["path"]) == info["sha256"], "Input model changed"
    name = args.name + ("_smoke" if args.action == "smoke" else "")
    run = args.project / name
    last = run / "weights/last.pt"
    progress = args.project / "progress" / (name + ".json")
    progress.parent.mkdir(exist_ok=True, parents=True)
    contract = {"manifest_sha256": sha(root / "manifest.json"), "imgsz": args.imgsz,
                "epochs": 1 if args.action == "smoke" else args.epochs, "lr0": .0001}

    def callback(trainer):
        progress.write_text(json.dumps({"epoch": trainer.epoch + 1, "epochs": trainer.epochs,
                                        "metrics": {k: float(v) for k, v in trainer.metrics.items()},
                                        "time": time.time()}), encoding="utf-8")

    if args.action in ("smoke", "train"):
        receipt = args.project / "contracts" / (name + ".json")
        receipt.parent.mkdir(exist_ok=True, parents=True)
        if receipt.exists() and json.loads(receipt.read_text()) != contract:
            raise ValueError("Training data/settings differ from this run. Keep original settings to resume, or choose a new NAME.")
        if last.exists():
            ckpt = torch.load(last, map_location="cpu", weights_only=False)
            complete = ckpt.get("epoch", -1) < 0 and ckpt.get("optimizer") is None
            epoch = ckpt.get("epoch", -1)
            resumable = epoch >= 0 and ckpt.get("optimizer") is not None
            del ckpt
            if complete:
                print("This run has already completed. Do not train again. Set ACTION='train' after smoke, or ACTION='val' after full train.", flush=True)
                return
            if not resumable:
                raise ValueError("Checkpoint has no valid optimizer/epoch for resume. Preserve it and choose a NEW NAME.")
            if args.no_auto_resume:
                raise ValueError("Interrupted checkpoint exists. Enable AUTO_RESUME or choose another NAME.")
            print("Auto-resume saved checkpoint, epoch:", epoch, flush=True)
            model = YOLO(str(last))
            model.add_callback("on_fit_epoch_end", callback)
            model.train(resume=True, device=0, batch=args.batch)
        else:
            if run.exists():
                raise ValueError("Run folder exists without last.pt. No saved epoch can be resumed. Preserve this folder and choose a NEW NAME.")
            receipt.write_text(json.dumps(contract, indent=2), encoding="utf-8")
            model = YOLO(str(root / manifest["models"]["baseline"]["path"]))
            model.add_callback("on_fit_epoch_end", callback)
            model.train(data=str(root / "data.yaml"), epochs=contract["epochs"], imgsz=args.imgsz,
                        batch=args.batch, device=0, optimizer="AdamW", lr0=.0001, warmup_epochs=3,
                        seed=42, patience=8, degrees=10, flipud=.1, fliplr=.5, mosaic=.2,
                        close_mosaic=0 if args.action == "smoke" else 5,
                        workers=2, cache=False, amp=True, project=str(args.project), name=name,
                        exist_ok=False, plots=True, save=True)
        print("Completed:", run / "weights", flush=True)
        return

    from metrics_all3 import evaluate_counts, save_previews
    candidate = run / "weights/best.pt"
    paths = {name: root / info["path"] for name, info in manifest["models"].items()}
    paths["trial_v2"] = candidate
    if args.action == "test":
        if not (run / "comparison_val.json").exists():
            raise ValueError("Run validation and choose CHECKPOINT before touching the legacy test regression set.")
        paths = {name: paths[name] for name in dict.fromkeys(("baseline", args.checkpoint))}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing model: {path}; complete full train first")
    split = "val" if args.action == "val" else "test"
    report_path = run / f"comparison_{split}.json"
    protocol = {"split": split, "imgsz": args.eval_imgsz, "AP_conf": .001,
                "count_conf": .4, "iou": .7, "agnostic_nms": True, "probIoU_matching_threshold": .5,
                "models": {label: sha(path) for label, path in paths.items()}, "manifest_sha256": sha(root / "manifest.json")}
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous["protocol"] != protocol:
            raise ValueError("Existing evaluation uses other settings/models. Preserve it; choose a new experiment directory.")
        print("Report already completed:", report_path, flush=True)
        return
    output = run / "evaluation" / split
    output.mkdir(parents=True, exist_ok=True)
    selected = [r for r in manifest["files"] if r["split"] == split]
    completed = []
    for label, path in paths.items():
        cache = output / (label + ".json")
        if cache.exists():
            row = json.loads(cache.read_text(encoding="utf-8"))
            if row.get("protocol") != protocol:
                raise ValueError("Evaluation cache settings differ; preserve the folder")
            completed.append(row)
            print("Reusing completed evaluation:", label, flush=True)
            continue
        print("Evaluating", label, "on", split, "at", args.eval_imgsz, flush=True)
        model = YOLO(str(path))
        stats = model.val(data=str(root / "data.yaml"), split=split, imgsz=args.eval_imgsz,
                          conf=.001, iou=.7, agnostic_nms=True, batch=args.batch, device=0,
                          project=str(output), name=label + "_AP", exist_ok=True, plots=True)
        counts = evaluate_counts(model, root, selected, args.eval_imgsz, args.batch)
        row = {"model": label, "sha256": sha(path), "protocol": protocol,
               "mAP50": float(stats.box.map50), "mAP50_95": float(stats.box.map),
               "AP50": {model.names[int(c)]: float(v) for c, v in zip(stats.box.ap_class_index, stats.box.ap50)},
               "operating_point": counts["summary"]}
        (output / (label + "_per_image.json")).write_text(json.dumps(counts, indent=2), encoding="utf-8")
        save_previews(root, selected, counts["rows"], output / (label + "_previews"))
        if args.action == "test":
            negatives = [r for r in manifest["files"] if r["split"] == "neg_test"]
            neg = evaluate_counts(model, root, negatives, args.eval_imgsz, args.batch)
            row["negative_regression"] = {"images": len(negatives),
                                          "false_positive_images": sum(sum(r["pred_counts"]) > 0 for r in neg["rows"])}
        cache.write_text(json.dumps(row, indent=2), encoding="utf-8")
        completed.append(row)
        del model
        torch.cuda.empty_cache()
    report = {"protocol": protocol, "results": completed, "limitations": manifest["limitations"],
              "acceptance": "Review each class AP/F1/count errors and real images. Do not automatically replace the app checkpoint. Compare only rows measured under this same protocol."}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# Three-fruit {split} comparison (imgsz {args.eval_imgsz})", "",
             "| Model | mAP50 | mAP50-95 | Banana AP50 | Apple AP50 | Orange AP50 | Macro F1 @.4 | Count MAE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in completed:
        op = row["operating_point"]
        lines.append(f"| {row['model']} | {row['mAP50']:.4f} | {row['mAP50_95']:.4f} | " +
                     " | ".join(f"{row['AP50'].get(k, 0):.4f}" for k in ("banana", "apple", "orange")) +
                     f" | {op['macro_F1']:.4f} | {op['count_MAE_all_classes']:.3f} |")
    lines.extend(["", "## Each fruit at confidence 0.40", "",
                  "| Model | Fruit | Precision | Recall | F1 | Count MAE (all images) |",
                  "|---|---|---:|---:|---:|---:|"])
    for row in completed:
        for fruit, values in row["operating_point"]["per_class"].items():
            lines.append(f"| {row['model']} | {fruit} | {values['precision']:.3f} | {values['recall']:.3f} | {values['F1']:.3f} | {values['count_MAE_all_images']:.3f} |")
    lines.extend(["", "Legacy labels are imperfect; this is a regression/development comparison, not an independent real-world benchmark."])
    (run / f"comparison_{split}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("Report:", report_path, flush=True)


if __name__ == "__main__":
    main()
