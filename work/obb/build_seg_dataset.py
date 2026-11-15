"""Build a YOLO-seg dataset from reviewed SAM instance polygons.

The source polygons are pseudo-labels from SAM prompted by existing boxes.
Review them before using the output for a final experiment.
"""
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work" / "obb" / "instances.jsonl"
OUT = ROOT / "work" / "obb" / "seg_data"


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing {SOURCE}; run sam_instances.py first")
    rows = defaultdict(list)
    for line in SOURCE.read_text().splitlines():
        item = json.loads(line)
        if item.get("poly"):
            rows[(item["split"], Path(item["image"]))].append(item)
    if not rows:
        raise ValueError("No polygon instances found")

    for split, image in rows:
        if not image.exists():
            raise FileNotFoundError(f"Image referenced by instances.jsonl is missing: {image}")
        target_image = OUT / split / "images" / image.name
        target_label = OUT / split / "labels" / f"{image.stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target_image)
        w, h = next(item["w"] for item in rows[(split, image)]), next(item["h"] for item in rows[(split, image)])
        lines = []
        for item in rows[(split, image)]:
            points = item["poly"]
            if len(points) < 3:
                continue
            coords = []
            for x, y in points:
                coords.extend((max(0.0, min(1.0, x / w)), max(0.0, min(1.0, y / h))))
            lines.append(" ".join([str(item["cls"]), *[f"{v:.6f}" for v in coords]]))
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""))

    (OUT / "seg.yaml").write_text(
        f"path: {OUT.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: banana\n"
        "  1: apple\n"
        "  2: orange\n"
        "pseudo_labels: SAM2 prompted by existing boxes; review before final training\n"
    )
    print(f"wrote {len(rows)} images to {OUT}")


if __name__ == "__main__":
    main()
