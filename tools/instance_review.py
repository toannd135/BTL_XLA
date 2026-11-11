"""Prepare an offline per-fruit label review and build a dataset from reviewed copies.

No model inference, training, or original-label writes. Unreviewed images are never
exported as negatives. Diagnostic images are never exported into a training split.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from PIL import Image, ImageOps

NAMES = {0: "banana", 1: "apple", 2: "orange"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_boxes(boxes):
    if not isinstance(boxes, list):
        raise ValueError("Boxes must be a list")
    result = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 5:
            raise ValueError("Expected [class, center_x, center_y, width, height]")
        cls, x, y, w, h = box
        if isinstance(cls, bool) or cls not in NAMES:
            raise ValueError("Invalid fruit class")
        if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
                   for v in (x, y, w, h)):
            raise ValueError("Invalid coordinate")
        if not (w > 0 and h > 0 and x-w/2 >= -1e-6 and y-h/2 >= -1e-6 and
                x+w/2 <= 1+1e-6 and y+h/2 <= 1+1e-6):
            raise ValueError("Box outside image or empty")
        normalized = [int(cls), float(x), float(y), float(w), float(h)]
        if any(old[0] == cls and all(abs(a-b) < 1e-6 for a, b in zip(old[1:], normalized[1:]))
               for old in result):
            raise ValueError("Duplicate annotation of the same fruit")
        result.append(normalized)
    return result


def read_labels(path):
    boxes, warnings = [], []
    if not path.is_file():
        return [], ["missing-label"]
    for row in path.read_text(encoding="utf-8-sig").splitlines():
        if not row.strip():
            continue
        values = list(map(float, row.split()))
        if len(values) == 5:
            boxes.append(values)
        elif len(values) >= 7 and len(values) % 2 == 1:
            cls, *points = values
            xs, ys = points[::2], points[1::2]
            boxes.append([cls, (min(xs)+max(xs))/2, (min(ys)+max(ys))/2,
                          max(xs)-min(xs), max(ys)-min(ys)])
            warnings.append("polygon-preview-converted-to-box")
        else:
            raise ValueError("Malformed original label")
    return validate_boxes(boxes), warnings


def source_group(stem):
    return re.sub(r"(_(?:jpg|jpeg|png|bmp|webp))?\.rf\.[a-z0-9]+$", "", stem, flags=re.I)


def prepare(source, output, diagnostics=None):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Output exists; choose a new review folder")
    flags = {}
    audit = source / "work/audit/det_label_audit.csv"
    if audit.is_file():
        with audit.open(encoding="utf-8-sig", newline="") as stream:
            flags = {(r["split"], r["image"]): r["flags"] for r in csv.DictReader(stream)}
    quarantine_file = source / "work/audit/quarantine_train.txt"
    quarantine = set(quarantine_file.read_text(encoding="utf-8-sig").split()) if quarantine_file.is_file() else set()
    entries = []
    for role in ("train", "val", "test"):
        folder = source / "local_dataset/det" / role
        for image in sorted((folder / "images").iterdir()):
            if image.suffix.lower() not in EXTENSIONS:
                continue
            label = folder / "labels" / (image.stem + ".txt")
            warning = flags.get((role, image.name), "")
            if role == "train" and image.name in quarantine:
                warning += ";quarantine"
            entries.append((role, image, label, warning.strip(";"),
                            source_group(image.stem), image.relative_to(source).as_posix()))
    if diagnostics:
        for image in sorted(diagnostics.resolve().iterdir()):
            if image.suffix.lower() in EXTENSIONS:
                entries.append(("diagnostic", image, None, "diagnostic-only;needs-manual-labels",
                                "diagnostic:"+image.stem, "diagnostic/"+image.name))
    if not entries:
        raise ValueError("No source images found")
    (output / "images").mkdir(parents=True)
    records = []
    for role, image, label, warning, group, original in entries:
        try:
            boxes, label_warnings = read_labels(label) if label else ([], [])
        except (ValueError, OSError) as exc:
            boxes, label_warnings = [], ["label-parse-error:"+str(exc)]
        identifier = hashlib.sha256((role+":"+original).encode()).hexdigest()[:24]
        target = output / "images" / (identifier + image.suffix.lower())
        with Image.open(image) as im:
            orientation = im.getexif().get(274, 1)
            if orientation != 1:
                target = target.with_suffix(".jpg")
                upright = ImageOps.exif_transpose(im)
                upright.convert("RGB").save(target, quality=95)
                width, height = upright.size
                label_warnings.append("exif-normalized:check-all-boxes")
            else:
                width, height = im.size
                shutil.copy2(image, target)
        warnings = [s for s in warning.split(";") if s] + label_warnings
        priority = (0 if role == "diagnostic" else
                    1 if any(s.startswith(("bunch", "nested", "quarantine", "label-parse")) for s in warnings) else
                    2 if warnings else 3 if len(boxes) > 1 else 4 if boxes else 5)
        records.append({"id": identifier, "role": role, "source_image": original,
                        "source_group": group, "image": target.relative_to(output).as_posix(),
                        "width": width, "height": height, "boxes": boxes,
                        "source_image_sha256": sha256(image), "display_image_sha256": sha256(target),
                        "source_label_sha256": sha256(label) if label and label.is_file() else None,
                        "warnings": warnings, "priority": priority})
    records.sort(key=lambda r: (r["priority"], r["role"], r["source_image"]))
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    report = {"schema": 1, "dataset_id": fingerprint, "names": NAMES,
              "policy": "one visible fruit per annotation; all labels initially unreviewed; original splits preserved",
              "counts": dict(Counter(r["role"] for r in records)), "items": records}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest.js").write_text("window.INSTANCE_DATA = " + json.dumps(report, ensure_ascii=False) + ";\n", encoding="utf-8")
    shutil.copy2(Path(__file__).with_name("instance_review.html"), output / "review.html")
    print(json.dumps({"review": str(output / "review.html"), "counts": report["counts"],
                      "automated_flags_are_unverified": True}, indent=2))
    return report


def build(bundle, reviews_file, output):
    bundle, output = bundle.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Output exists; choose a new dataset version")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    reviews = json.loads(reviews_file.read_text(encoding="utf-8-sig"))
    if reviews.get("dataset_id") != manifest["dataset_id"] or reviews.get("schema") != 1:
        raise ValueError("Reviews belong to another dataset")
    catalog = {item["id"]: item for item in manifest["items"]}
    verified, seen = [], set()
    for review in reviews.get("reviews", []):
        identifier = review.get("id")
        if identifier not in catalog or identifier in seen:
            raise ValueError("Unknown or duplicate review id")
        seen.add(identifier)
        item = catalog[identifier]
        if review.get("status") != "approved" or item["role"] == "diagnostic":
            continue
        count = review.get("expected_count")
        boxes = validate_boxes(review.get("boxes"))
        if isinstance(count, bool) or not isinstance(count, int) or count != len(boxes):
            raise ValueError("Manual fruit count must equal the number of individual boxes")
        image = (bundle / item["image"]).resolve()
        if not image.is_relative_to(bundle) or sha256(image) != item["display_image_sha256"]:
            raise ValueError("Review image missing, changed, or outside bundle")
        verified.append((item, boxes, image))
    for role in ("train", "val"):
        classes = {b[0] for item, boxes, _ in verified if item["role"] == role for b in boxes}
        if classes != set(NAMES):
            raise ValueError(f"Reviewed {role} needs all three classes; unreviewed images cannot be used")
    groups, image_splits = {}, {}
    for item, _, _ in verified:
        role = item["role"]
        for mapping, key in ((groups, item["source_group"]), (image_splits, item["source_image_sha256"])):
            previous = mapping.setdefault(key, role)
            if previous != role:
                raise ValueError("Source/image leakage between reviewed dataset splits")
    output.mkdir(parents=True)
    totals = {role: {"images": 0, "backgrounds": 0, "boxes": Counter()} for role in ("train", "val", "test")}
    for item, boxes, image in verified:
        role, identifier = item["role"], item["id"]
        dest_image = output / role / "images" / (identifier + image.suffix)
        dest_label = output / role / "labels" / (identifier + ".txt")
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, dest_image)
        dest_label.write_text("".join(f"{int(c)} {x:.7f} {y:.7f} {w:.7f} {h:.7f}\n" for c, x, y, w, h in boxes), encoding="utf-8")
        totals[role]["images"] += 1
        totals[role]["backgrounds"] += not boxes
        totals[role]["boxes"].update(b[0] for b in boxes)
    # Omitting path lets Ultralytics resolve splits relative to the YAML file.
    yaml = "train: train/images\nval: val/images\n"
    if totals["test"]["images"]:
        yaml += "test: test/images\n"
    yaml += "names:\n  0: banana\n  1: apple\n  2: orange\n"
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    summary = {"dataset_id": manifest["dataset_id"], "splits": totals,
               "policy": "approved annotations only; diagnostics excluded; pending images are not negatives",
               "provenance": [{"id": item["id"], "source": item["source_image"], "role": item["role"],
                               "image_sha256": item["source_image_sha256"], "boxes": len(boxes)}
                              for item, boxes, _ in verified]}
    (output / "reviewed_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "splits": totals}, indent=2))
    return summary


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--source-root", type=Path,
                             default=repo if (repo / "local_dataset/det").is_dir() else repo.parent)
    preparation.add_argument("--out", type=Path, default=repo / "out/instance_review_v1")
    preparation.add_argument("--diagnostics", type=Path)
    compiler = sub.add_parser("build")
    compiler.add_argument("--bundle", type=Path, default=repo / "out/instance_review_v1")
    compiler.add_argument("--reviews", type=Path, required=True)
    compiler.add_argument("--out", type=Path, default=repo / "out/instance_clean_v1")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source_root, args.out, args.diagnostics)
    else:
        build(args.bundle, args.reviews, args.out)


if __name__ == "__main__":
    main()
