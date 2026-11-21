#!/usr/bin/env python3
"""Train an exploratory Green Banana vs Green Apple baseline from this candidate.

This entrypoint is intentionally separate from the Phase 7 three-class trainer.
It verifies the approved local ImageNet checkpoint and dataset before epoch 1.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import random
import sys
import time
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[4]
EXP = ROOT / "work" / "fruit_type_cls_experiment"
DATA = pathlib.Path(os.environ.get("PHASE8E_DATA_DIR", str(pathlib.Path(__file__).resolve().parent))).expanduser().resolve()
RUN = pathlib.Path(os.environ.get(
    "PHASE8E_RUN_DIR", str(EXP / "phase8e_green_binary_baseline_v1")
)).expanduser().resolve()
MANIFEST = DATA / "manifest.csv"
CLASS_TO_IDX = {"GreenBanana": 0, "GreenApple": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
SEED = 20261120
EXPECTED_SHA256 = "5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997"
DEFAULT_WEIGHTS = pathlib.Path("/home/toannd/models/mobilenet/mobilenet_v3_large-5c1a4163.pth")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def preflight():
    problems = []
    if not MANIFEST.is_file():
        return [], [f"Missing dataset manifest: {MANIFEST}"]
    rows = read_manifest()
    counts = Counter((r.get("split", ""), r.get("class_label", "")) for r in rows)
    summary_path = DATA / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    expected = {}
    for key, value in summary.get("counts", {}).items():
        split, label = key.split("/", 1)
        expected[(split, label)] = int(value)
    if not expected or len(rows) != int(summary.get("total_images", -1)) or dict(counts) != expected:
        problems.append(f"Unexpected candidate counts: rows={len(rows)}, counts={dict(counts)}")
    ids = [r.get("asset_id", "") for r in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        problems.append("Missing or duplicate asset IDs")
    fam_splits, hash_splits = defaultdict(set), defaultdict(set)
    for r in rows:
        if r.get("class_label") not in CLASS_TO_IDX:
            problems.append(f"Unknown class label: {r.get('class_label')}")
        if r.get("split") not in {"train", "val", "test"}:
            problems.append(f"Invalid split for {r.get('asset_id')}: {r.get('split')}")
        if r.get("review_status") != "VERIFIED_CORRECT":
            problems.append(f"Non-verified sample: {r.get('asset_id')}")
        if not r.get("source_family_id") and not (r.get("split") == "train" and r.get("family_status") == "UNKNOWN_TRAIN_ONLY"):
            problems.append(f"Missing family ID: {r.get('asset_id')}")
        if r.get("source_family_id"):
            fam_splits[(r.get("class_label"), r.get("source_family_id"))].add(r.get("split"))
        hash_splits[r.get("crop_sha256")].add(r.get("split"))
        img = DATA / r.get("target_crop_path", "")
        if not img.is_file():
            problems.append(f"Missing image: {r.get('asset_id')} ({img})")
            continue
        if sha256(img) != r.get("crop_sha256"):
            problems.append(f"Image SHA-256 mismatch: {r.get('asset_id')}")
        try:
            from PIL import Image
            with Image.open(img) as im:
                im.verify()
        except Exception as e:
            problems.append(f"Unreadable image {r.get('asset_id')}: {e}")
    if any(len(s) != 1 for s in fam_splits.values()):
        problems.append("A source family crosses dataset splits")
    if any(len(s) != 1 for s in hash_splits.values()):
        problems.append("An exact duplicate crosses dataset splits")
    return rows, problems


def load_verified_model(num_classes=2):
    import torch
    from torch import nn
    from torchvision.models import mobilenet_v3_large

    path = pathlib.Path(os.environ.get("PHASE7_PRETRAINED_WEIGHTS", str(DEFAULT_WEIGHTS))).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Approved local checkpoint is missing: {path}")
    actual = sha256(path)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Checkpoint is not a plain state_dict")
    model = mobilenet_v3_large(weights=None)  # Never invokes download/cache resolution.
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    shape_bad = sorted(k for k in set(expected) & set(state)
                       if not hasattr(state[k], "shape") or tuple(state[k].shape) != tuple(expected[k].shape))
    if missing or unexpected or shape_bad:
        raise RuntimeError(f"Checkpoint structure mismatch: missing={missing}, unexpected={unexpected}, shape={shape_bad}")
    model.load_state_dict(state, strict=True)
    backbone_keys = [k for k in expected if k.startswith("features.")]
    if not all(torch.equal(model.state_dict()[k].cpu(), state[k].cpu()) for k in backbone_keys):
        raise RuntimeError("Loaded backbone differs from verified checkpoint tensors")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model, {"checkpoint_path": str(path), "checkpoint_sha256": actual,
                   "expected_sha256": EXPECTED_SHA256, "strict_load": True,
                   "backbone_tensors_compared": len(backbone_keys),
                   "all_backbone_tensors_match": True, "classifier_outputs": num_classes}


def main():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from torchvision.models import MobileNet_V3_Large_Weights
    from PIL import Image
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    rows, problems = preflight()
    if problems:
        raise RuntimeError("Dataset preflight failed: " + "; ".join(problems[:20]))
    model, weights_check = load_verified_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 32 if device.type == "cuda" else 16
    workers = min(int(os.environ.get("PHASE8E_NUM_WORKERS", "4")), os.cpu_count() or 1)
    torch.set_num_threads(min(12, os.cpu_count() or 1)) if device.type == "cpu" else None
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1),
                                     interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.08, 0.01), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    eval_tf = MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()

    class Crops(Dataset):
        def __init__(self, split, tf):
            self.items = [r for r in rows if r["split"] == split]; self.tf = tf
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            r = self.items[i]
            with Image.open(DATA / r["target_crop_path"]) as im: x = self.tf(im.convert("RGB"))
            return x, CLASS_TO_IDX[r["class_label"]]

    ds = {s: Crops(s, train_tf if s == "train" else eval_tf) for s in ("train", "val", "test")}
    loaders = {s: DataLoader(ds[s], batch_size=batch, shuffle=(s == "train"),
                             num_workers=workers, pin_memory=device.type == "cuda") for s in ds}
    model.to(device)
    class_counts = Counter(r["class_label"] for r in rows if r["split"] == "train")
    class_weights = torch.tensor([len(ds["train"]) / (2 * class_counts[IDX_TO_CLASS[i]]) for i in range(2)], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    model_dir = RUN
    (model_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics").mkdir(parents=True, exist_ok=True)
    scheduler = None
    for epoch in range(1, 41):
        if epoch == 1:
            for n, p in model.named_parameters(): p.requires_grad = n.startswith("classifier.")
            opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)
        elif epoch == 6:
            for p in model.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW([{"params": model.features.parameters(), "lr": 3e-5},
                                     {"params": model.classifier.parameters(), "lr": 3e-4}], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)
        model.train()
        if epoch <= 5: model.features.eval()
        tr_loss = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device); opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y); loss.backward(); opt.step(); tr_loss += loss.item() * y.size(0)
        model.eval(); val_loss = 0.0; vy=[]; vp=[]
        with torch.no_grad():
            for x, y in loaders["val"]:
                x, y = x.to(device), y.to(device); logits = model(x)
                val_loss += criterion(logits, y).item() * y.size(0); vy.extend(y.cpu().tolist()); vp.extend(logits.argmax(1).cpu().tolist())
        val_loss /= len(ds["val"])
        history.append({"epoch": epoch, "train_loss": tr_loss/len(ds["train"]), "val_loss": val_loss,
                        "val_accuracy": accuracy_score(vy, vp), "val_macro_f1": f1_score(vy, vp, average="macro")})
        # Checkpoint selection must track the actual minimum validation loss.
        # Early-stopping tolerance must not suppress a better checkpoint.
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_val_loss": val_loss,
                        "class_to_idx": CLASS_TO_IDX, "pretrained": weights_check}, RUN/"checkpoints"/"best_model.pt")
        else: stale += 1
        scheduler.step(val_loss)
        if epoch >= 6 and stale >= 8: break
    with (RUN/"metrics"/"training_history.json").open("w") as f: json.dump(history, f, indent=2)
    print(f"Training completed. Best epoch={best_epoch}; validation loss={best_loss:.5f}.")
    # Test is evaluated exactly once, after validation-loss checkpoint selection.
    state = torch.load(RUN/"checkpoints"/"best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"]); model.eval(); ty=[]; tp=[]
    with torch.no_grad():
        for x, y in loaders["test"]:
            pred = model(x.to(device)).argmax(1).cpu().tolist(); ty.extend(y.tolist()); tp.extend(pred)
    report = classification_report(ty, tp, labels=[0,1], target_names=[IDX_TO_CLASS[0],IDX_TO_CLASS[1]], output_dict=True, zero_division=0)
    result = {"test_evaluations": 1, "selection_metric": "minimum validation loss", "best_epoch": best_epoch,
              "accuracy": accuracy_score(ty,tp), "macro_f1": f1_score(ty,tp,average="macro"),
              "confusion_matrix": confusion_matrix(ty,tp,labels=[0,1]).tolist(), "per_class": report,
              "sample_count": len(ty), "interpretation": "Exploratory only; strong source/background and class-diversity limitations."}
    (RUN/"metrics"/"test_metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    original = (EXP / "phase7_baseline").resolve()
    if RUN == original or (RUN.exists() and any(RUN.iterdir())):
        print(f"Refusing unsafe output directory: {RUN}", file=sys.stderr); sys.exit(10)
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        sys.exit(main())
    except Exception as e:
        print(f"TRAINING_BLOCKED before/within run: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
