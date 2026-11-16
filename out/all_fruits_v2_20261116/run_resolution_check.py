"""Validation-only comparison at 640/1024; reuse existing 1280 report. No training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

EXPECTED_V2 = "d0abaa79cc3784849dc3bf7c178033c56df2990b71a2f69397dfb94fc463796d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="all3_balanced_v2")
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        raise ValueError("Invalid NAME")
    if not 1 <= args.batch <= 8:
        raise ValueError("Invalid BATCH")
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("Select T4 GPU in Colab. No full validation on laptop CPU.")
    from metrics_all3 import evaluate_counts
    root = args.root.resolve()
    run = args.project / args.name
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    prior_path = run / "comparison_val.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    manifest_hash = sha(root / "manifest.json")
    assert prior["protocol"]["manifest_sha256"] == manifest_hash
    assert prior["protocol"]["imgsz"] == 1280 and prior["protocol"]["split"] == "val"
    assert prior["protocol"]["AP_conf"] == .001 and prior["protocol"]["count_conf"] == .4
    assert prior["protocol"]["iou"] == .7 and prior["protocol"]["agnostic_nms"] is True
    paths = {name: root / info["path"] for name, info in manifest["models"].items()}
    paths["trial_v2"] = run / "weights/best.pt"
    hashes = {name: sha(path) for name, path in paths.items()}
    assert hashes == prior["protocol"]["models"], "Model changed since the submitted report"
    assert hashes["trial_v2"] == EXPECTED_V2, "Not the candidate from the submitted comparison"
    output = run / f"resolution_check_{EXPECTED_V2[:8]}_batch{args.batch}"
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.json"
    selected = [r for r in manifest["files"] if r["split"] == "val"]
    assert len(selected) == 297
    rows = []
    for old in prior["results"]:
        rows.append({"imgsz": 1280, "batch": None, "source": "existing submitted validation; original batch was not recorded in protocol",
                     **{key: old[key] for key in ("model", "sha256", "mAP50", "mAP50_95", "AP50", "operating_point")}})
    for size in (640, 1024):
        for name, path in paths.items():
            cache_dir = output / f"imgsz_{size}"
            cache_dir.mkdir(exist_ok=True)
            cache = cache_dir / (name + ".json")
            receipt = {"imgsz": size, "batch": args.batch, "models": hashes,
                       "manifest_sha256": manifest_hash, "AP_conf": .001, "count_conf": .4,
                       "iou": .7, "agnostic_nms": True}
            if cache.exists():
                row = json.loads(cache.read_text(encoding="utf-8"))
                assert row["receipt"] == receipt, "Existing cache uses other settings"
                rows.append(row)
                continue
            def status(stage):
                progress.write_text(json.dumps({"stage": f"{name} imgsz{size}: {stage}", "time": time.time()}), encoding="utf-8")
            status("AP evaluation")
            model = YOLO(str(path))
            metrics = model.val(data=str(root / "data.yaml"), split="val", imgsz=size,
                                conf=.001, iou=.7, agnostic_nms=True, batch=args.batch, device=0,
                                project=str(cache_dir), name=name + "_AP", exist_ok=True, plots=False)
            status("per-class F1 and counting")
            counts = evaluate_counts(model, root, selected, size, args.batch)
            row = {"imgsz": size, "batch": args.batch, "source": "new validation; no training", "receipt": receipt,
                   "model": name, "sha256": hashes[name], "mAP50": float(metrics.box.map50),
                   "mAP50_95": float(metrics.box.map),
                   "AP50": {model.names[int(c)]: float(v) for c, v in zip(metrics.box.ap_class_index, metrics.box.ap50)},
                   "operating_point": counts["summary"]}
            (cache_dir / (name + "_per_image.json")).write_text(json.dumps(counts, indent=2), encoding="utf-8")
            cache.write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            del model
            torch.cuda.empty_cache()
    rows.sort(key=lambda r: (r["imgsz"], list(paths).index(r["model"])))
    result = {"split": "val", "manifest_sha256": manifest_hash, "models": hashes, "results": rows,
              "limitations": manifest["limitations"] + [
                  "1280 rows reused from the submitted run; batch was not recorded there. Only 640/1024 newly evaluated with the same explicit batch.",
                  "Resolution selection uses development validation, not legacy test or an independent final benchmark.",
                  "No model training, app changes, CLIP or teacher evaluation in this check."]}
    target = output / "resolution_comparison.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Validation by model and image size", "",
             "| Size | Model | mAP50 | mAP50-95 | AP banana | AP apple | AP orange | Macro F1 @.4 | Count MAE |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        op = row["operating_point"]
        lines.append(f"| {row['imgsz']} | {row['model']} | {row['mAP50']:.4f} | {row['mAP50_95']:.4f} | " +
                     " | ".join(f"{row['AP50'].get(k, 0):.4f}" for k in ("banana", "apple", "orange")) +
                     f" | {op['macro_F1']:.4f} | {op['count_MAE_all_classes']:.3f} |")
    lines.extend(["", "Legacy labels contain bunch boxes; inspect individual-fruit scenes before selecting a model. Not a full-app benchmark."])
    target.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    progress.write_text(json.dumps({"stage": "completed", "time": time.time()}), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("Report:", target, flush=True)


if __name__ == "__main__":
    main()
