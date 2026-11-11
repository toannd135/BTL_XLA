"""Bundle the imported C data and model for Colab without moving source folders."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(repo, data_root, output):
    pairs = []
    counts = {}
    for split in ("train", "val", "test", "train_ft"):
        folder = data_root / "local_dataset" / "det" / split
        images = sorted(p for p in (folder / "images").rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"Missing dataset images: {folder}")
        empty = polygons = 0
        for image in images:
            label = folder / "labels" / image.relative_to(folder / "images").with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"Missing label: {label}")
            rows = [row for row in label.read_text(encoding="utf-8-sig").splitlines() if row.strip()]
            empty += not rows
            for row in rows:
                values = list(map(float, row.split()))
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(f"Non-finite label: {label}")
                if len(values) < 5 or not values[0].is_integer() or int(values[0]) not in range(3):
                    raise ValueError(f"Invalid detection class or row: {label}")
                if len(values) != 5:
                    if len(values) < 7 or len(values) % 2 != 1:
                        raise ValueError(f"Invalid polygon row: {label}")
                    polygons += 1
                if not all(0 <= v <= 1 for v in values[1:]):
                    raise ValueError(f"Unnormalized label: {label}")
            pairs.extend([(image, image.relative_to(data_root).as_posix()),
                          (label, label.relative_to(data_root).as_posix())])
        counts[split] = {"images": len(images), "empty_labels": empty, "polygon_rows": polygons}

    negatives = sorted(p for p in (data_root / "own_neg").rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    held = [p for p in negatives if int(hashlib.md5(
        p.relative_to(data_root / "own_neg").as_posix().encode()).hexdigest(), 16) % 5 == 0]
    if len(negatives) != 358 or len(held) != 71:
        raise ValueError(f"Unexpected C negative split: {len(negatives)} images, {len(held)} held out")
    pairs.extend((p, p.relative_to(data_root).as_posix()) for p in negatives)

    # The source script prepares these crops but excludes them from C's training list.
    for folder in (data_root / "own_cam" / "images", data_root / "own_cam" / "labels"):
        pairs.extend((p, p.relative_to(data_root).as_posix()) for p in sorted(folder.rglob("*"))
                     if p.is_file() and (p.suffix.lower() in IMAGE_EXTENSIONS or p.suffix.lower() == ".txt"))

    for relative in ("work/exp/det_neg.py", "work/eval_det.py", "work/audit_det_labels.py",
                     "work/audit/quarantine_train.txt"):
        path = data_root / relative
        if not path.is_file():
            raise ValueError(f"Missing original C input: {path}")
        pairs.append((path, relative))
    for relative in ("train_detection.py", "run_detection.py", "README.md",
                     "exports/det_finetune_best.pt", "exports/cls_finetune_best.pt"):
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"Missing repository input: {path}")
        pairs.append((path, relative))

    quarantine = set((data_root / "work/audit/quarantine_train.txt").read_text().split())
    train_images = list((data_root / "local_dataset/det/train/images").glob("*"))
    public_count = sum(p.is_file() and p.name not in quarantine for p in train_images)
    own_count = sum(p.is_file() and p.name.startswith("own-")
                    for p in (data_root / "local_dataset/det/train_ft/images").glob("*"))
    manifest = {
        "recipe": "C-style further fine-tuning from exported C; not D1 and not a replay from model B",
        "names": {0: "banana", 1: "apple", 2: "orange"},
        "counts": counts,
        "quarantine_count": len(quarantine),
        "training_list": {"public_after_quarantine": public_count, "own_images": own_count,
                          "own_repeat": 4, "negative_images": len(negatives) - len(held),
                          "negative_repeat": 2,
                          "total_entries": public_count + own_count * 4 + (len(negatives) - len(held)) * 2},
        "heldout_negative_count": len(held),
        "model_sha256": sha256(repo / "exports/det_finetune_best.pt"),
        "det_neg_script_sha256": sha256(data_root / "work/exp/det_neg.py"),
        "data_policy": "Original labels retained, including empty and polygon labels; original test is unchanged.",
        "known_empty_banana_examples": [
            "local_dataset/det/train/images/rf_v7__b_f011_png.rf.2940965015f7ed033a77dd07894d7bae.jpg",
            "local_dataset/det/test/images/rf_v7__b_f077_png.rf.26b27227d9088256a3f0f67e33bf3cfc.jpg"],
        "files": [],
    }
    names = [arcname for _, arcname in pairs]
    if len(names) != len(set(names)):
        raise ValueError("Archive paths collide")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for source, arcname in pairs:
            archive.write(source, arcname)
            manifest["files"].append({"path": arcname, "size": source.stat().st_size,
                                      "sha256": sha256(source)})
        archive.writestr("colab_input_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(temporary) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Archive integrity check failed: {bad_member}")
    os.replace(temporary, output)
    report = dict(manifest)
    report.pop("files")
    report.update(archive_file=output.name, archive_bytes=output.stat().st_size, archive_sha256=sha256(output))
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return report


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=repo.parent / "detection_C_colab_input.zip")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root or (repo if (repo / "local_dataset/det").exists() else repo.parent)
    if args.output.exists() and not args.force:
        parser.error(f"Output exists: {args.output}; use --force to rebuild this generated archive")
    prepare(repo, data_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
