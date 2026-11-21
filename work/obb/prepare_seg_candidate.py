"""Create an isolated segmentation candidate dataset with clean background samples."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/obb/seg_data_clean"
NEGATIVES = ROOT / "own_neg"
DEFAULT_OUT = ROOT / "work/obb/seg_data_candidate"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate data: {args.out}")

    shutil.copytree(SOURCE, args.out)
    negatives = sorted(p for p in NEGATIVES.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(negatives) != 358:
        raise ValueError(f"Expected 358 audited background images, found {len(negatives)}")

    counts = {"train": 0, "test": 0}
    manifest = []
    for source in negatives:
        relative = source.relative_to(NEGATIVES).as_posix()
        held_out = int(hashlib.md5(relative.encode()).hexdigest(), 16) % 5 == 0
        split = "test" if held_out else "train"
        digest = hashlib.sha256(relative.encode()).hexdigest()[:16]
        stem = f"negative__{digest}"
        image_dest = args.out / split / "images" / f"{stem}{source.suffix.lower()}"
        label_dest = args.out / split / "labels" / f"{stem}.txt"
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_dest)
        label_dest.write_text("")
        counts[split] += 1
        manifest.append({"source": str(source.relative_to(ROOT)), "split": split,
                         "image": str(image_dest.relative_to(args.out))})

    if counts != {"train": 287, "test": 71}:
        raise ValueError(f"Unexpected background split: {counts}")

    (args.out / "seg.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        "names:\n  0: banana\n  1: apple\n  2: orange\n"
    )
    report = {
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "background_split": counts,
        "background_split_rule": "md5(relative path) % 5 == 0 is held out in test",
        "warning": "Fruit polygons are SAM pseudo-labels; inspect per-instance boundaries before production use.",
        "backgrounds": manifest,
    }
    (args.out / "candidate_manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(args.out), "background_split": counts,
                      "images": {split: len(list((args.out / split / "images").glob("*")))
                                 for split in ("train", "val", "test")}}, indent=2))


if __name__ == "__main__":
    main()
