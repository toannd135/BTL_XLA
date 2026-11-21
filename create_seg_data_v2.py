from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import shutil
import re

ROOT = Path.cwd()
SRC = ROOT / "work/obb/seg_data_candidate"
DST = ROOT / "work/obb/seg_data_v2"
ZIP = ROOT / "fruit_cvat_annotation_v2.zip"

assert SRC.is_dir(), f"Missing source dataset: {SRC}"
assert ZIP.is_file(), f"Missing CVAT ZIP: {ZIP}"
assert not DST.exists(), f"Destination already exists: {DST}"

with ZipFile(ZIP) as z:
    archive_names = set(z.namelist())

    assert "train.txt" in archive_names
    assert "seg.yaml" not in archive_names

    image_paths = [
        x.strip()
        for x in z.read("train.txt").decode().splitlines()
        if x.strip()
    ]

    label_paths = sorted(
        x for x in archive_names
        if re.fullmatch(r"labels/train/[^/]+\.txt", x)
    )

    image_stems = [Path(x).stem for x in image_paths]
    label_stems = {Path(x).stem for x in label_paths}

    assert len(image_stems) == len(set(image_stems)), "Duplicate image stems"
    assert label_stems.issubset(set(image_stems)), "Orphan CVAT labels"

    # Verify every exported image belongs to the old training split.
    for image_path in image_paths:
        filename = Path(image_path).name
        assert (SRC / "train/images" / filename).is_file(), (
            f"Image missing from old train split: {filename}"
        )

    # Validate CVAT annotation format and class IDs.
    class_counts = Counter()

    for label_path in label_paths:
        content = z.read(label_path).decode().strip()
        assert content, f"Unexpected empty CVAT label: {label_path}"

        for line_no, line in enumerate(content.splitlines(), 1):
            parts = line.split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))

            assert cls in (0, 1, 2), (
                f"Invalid class: {label_path}:{line_no}"
            )
            assert len(coords) >= 6 and len(coords) % 2 == 0, (
                f"Invalid polygon: {label_path}:{line_no}"
            )
            assert all(0 <= v <= 1 for v in coords), (
                f"Out-of-range coordinate: {label_path}:{line_no}"
            )

            class_counts[cls] += 1

    # Missing CVAT labels must correspond to existing empty negative labels.
    missing_labels = set(image_stems) - label_stems

    for stem in missing_labels:
        old_label = SRC / "train/labels" / f"{stem}.txt"
        assert old_label.is_file(), f"Missing old negative label: {stem}"
        assert not old_label.read_text().strip(), (
            f"CVAT omitted a non-empty old label: {stem}"
        )

    print("Preflight passed.")
    print("CVAT images:", len(image_paths))
    print("CVAT annotation files:", len(label_paths))
    print("Preserved empty negatives:", len(missing_labels))
    print("Instances by class:", dict(class_counts))

    # Copy the complete old dataset; never modify the source.
    shutil.copytree(SRC, DST)

    # Replace only the 120 exported annotation files.
    for label_path in label_paths:
        filename = Path(label_path).name
        target = DST / "train/labels" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(label_path))

# Write a dataset config pointing to the new dataset.
(DST / "seg.yaml").write_text(
    f"""path: {DST}
train: train/images
val: val/images
test: test/images
names:
  0: banana
  1: apple
  2: orange
""",
    encoding="utf-8",
)

# Verify every image has a corresponding label in all splits.
for split in ("train", "val", "test"):
    images_dir = DST / split / "images"
    labels_dir = DST / split / "labels"

    images = [
        p for p in images_dir.iterdir()
        if p.is_file()
    ]

    missing = [
        p.name for p in images
        if not (labels_dir / f"{p.stem}.txt").is_file()
    ]

    assert not missing, (
        f"{split}: images without labels: {missing[:10]}"
    )

    print(f"{split}: {len(images)} images, labels complete")

print("\nDataset v2 created successfully:", DST)
print("Original dataset was not modified.")
