"""Check the handoff files without loading models, training, or using a GPU."""

import json
from pathlib import Path
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    required = ["backend/main.py", "backend/detection.py", "backend/requirements.txt",
                "frontend/package.json", "frontend/package-lock.json", "frontend/.env.example",
                "exports/det_finetune_best.pt", "exports/det_C_candidate.pt", "exports/cls_finetune_best.pt",
                "work/exp/det_neg.py", "work/eval_det.py", "work/audit_det_labels.py",
                "work/audit/quarantine_train.txt", "README.md", "docs/toi-uu-detection.md"]
    errors = ["Missing: " + name for name in required if not (repo / name).is_file()]
    models, splits = {}, {}
    for name in required:
        path = repo / name
        if path.suffix != ".pt" or not path.is_file():
            continue
        with path.open("rb") as stream:
            header = stream.read(100)
        if header.startswith(b"version https://git-lfs.github.com/spec/"):
            errors.append("Model is an LFS pointer; run git lfs pull: " + name)
        if path.stat().st_size < 1024:
            errors.append("Model appears empty or incomplete: " + name)
        models[name] = path.stat().st_size
    for split in ("train", "val", "test", "train_ft"):
        folder = repo / "local_dataset/det" / split
        images = sorted(p for p in (folder / "images").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
        missing = [p.name for p in images if not (folder / "labels" / (p.stem + ".txt")).is_file()]
        if not images:
            errors.append("Missing or empty detection split: " + split)
        if missing:
            errors.append(f"{split}: {len(missing)} images missing label files")
        splits[split] = {"images": len(images), "missing_labels": len(missing)}
    negatives = sum(p.is_file() for p in (repo / "own_neg").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not negatives:
        errors.append("Missing own_neg images")
    print(json.dumps({"ok": not errors, "models_bytes": models, "detection": splits,
                      "negative_images": negatives, "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
