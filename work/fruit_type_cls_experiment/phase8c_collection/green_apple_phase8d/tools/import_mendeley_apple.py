#!/usr/bin/env python3
"""Import licensed, polygon-annotated green-apple candidates from Mendeley V1.

The original ZIP and VIA JSON are preserved. This writes derived YOLO polygon
labels to a separate staging directory and never marks review decisions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parents[1]
ARCHIVE = HERE / "source_archives/gfcmdbvw65_v1_dataset.zip"
EXPECTED_SHA256 = "3b8d3c4f8d6788fd6af8c5fbf92c33755f2130f1159d94ea869543f91e2a1c51"
OUT = HERE / "staging/mendeley_v1"
ATTRIBUTION = "Wang, Dandan; Wang, Bo (2025), A Multi-Stage, Pixel-Level Annotated Apple Dataset for Precision Agriculture Research, Mendeley Data V1, DOI: 10.17632/gfcmdbvw65.1, CC BY 4.0."


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not ARCHIVE.is_file() or sha256(ARCHIVE) != EXPECTED_SHA256:
        raise SystemExit("Source archive missing or SHA-256 mismatch; refusing extraction")
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty staging directory: {OUT}")
    images_out = OUT / "images"
    labels_out = OUT / "labels"
    source_ann_out = OUT / "source_via"
    for d in (images_out, labels_out, source_ann_out):
        d.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    invalid: list[str] = []
    with zipfile.ZipFile(ARCHIVE) as zf:
        if zf.testzip() is not None:
            raise SystemExit("ZIP CRC check failed; refusing extraction")
        for source_split in ("train", "test"):
            ann_name = f"dataset/{source_split}/{source_split}.json"
            (source_ann_out / f"{source_split}.json").write_bytes(zf.read(ann_name))
            annotations = json.loads(zf.read(ann_name))
            for entry in annotations.values():
                filename = entry.get("filename")
                if not filename:
                    continue
                regions = entry.get("regions") or []
                labels = []
                has_immature = False
                for region in regions:
                    attrs = region.get("region_attributes") or {}
                    stage = str(attrs.get("name") or attrs.get("Name") or "").strip().lower()
                    if stage == "immature apple":
                        has_immature = True
                    shape = region.get("shape_attributes") or {}
                    xs, ys = shape.get("all_points_x") or [], shape.get("all_points_y") or []
                    if len(xs) != len(ys) or len(xs) < 3:
                        if stage:
                            invalid.append(f"{source_split}/{filename}: malformed polygon for {stage}")
                        continue
                    if len(set(zip(xs, ys))) < 3:
                        invalid.append(f"{source_split}/{filename}: degenerate polygon")
                        continue
                    # This detector predicts Apple independent of ripeness. Keep all
                    # apple maturity-stage masks in a selected green-containing image.
                    labels.append((stage or "stage_unknown", xs, ys))
                if not has_immature:
                    continue

                src_name = f"dataset/{source_split}/{filename}"
                raw = zf.read(src_name)
                stem = f"mendeley_v1_{source_split}_{Path(filename).stem}"
                image_path = images_out / f"{stem}.jpg"
                label_path = labels_out / f"{stem}.txt"
                image_path.write_bytes(raw)
                with Image.open(image_path) as im:
                    im.verify()
                with Image.open(image_path) as im:
                    width, height = im.size
                lines = []
                for _stage, xs, ys in labels:
                    points = []
                    for x, y in zip(xs, ys):
                        if x < 0 or y < 0 or x > width or y > height:
                            invalid.append(f"{source_split}/{filename}: point outside {width}x{height}")
                            points = []
                            break
                        points.extend((x / width, y / height))
                    if points:
                        lines.append("1 " + " ".join(f"{v:.8f}" for v in points))
                if not lines:
                    image_path.unlink()
                    invalid.append(f"{source_split}/{filename}: no usable apple polygons")
                    continue
                label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                records.append({
                    "asset_id": "mendeley_v1_" + hashlib.sha256(raw).hexdigest()[:24],
                    "original_filename": filename,
                    "source_archive_path": str(ARCHIVE.relative_to(HERE.parents[2])),
                    "source_archive_sha256": EXPECTED_SHA256,
                    "source_split_in_archive": source_split,
                    "source_dataset_doi": "10.17632/gfcmdbvw65.1",
                    "source_url": "https://data.mendeley.com/datasets/gfcmdbvw65/1",
                    "source_provider": "Mendeley Data",
                    "authors": "Dandan Wang; Bo Wang",
                    "license": "CC BY 4.0",
                    "attribution": ATTRIBUTION,
                    "source_family_id": "",
                    "source_family_policy": "unknown provenance; all selected images kept in training only",
                    "class_label": "Apple",
                    "green_status": "contains_source_annotation_immature_apple",
                    "review_status": "PENDING_SOURCE_ANNOTATION_NOT_INDEPENDENTLY_REVIEWED",
                    "width": width,
                    "height": height,
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                    "image_relative_path": str(image_path.relative_to(OUT)),
                    "label_relative_path": str(label_path.relative_to(OUT)),
                    "apple_polygon_count": len(lines),
                    "annotation_source": f"{ann_name} (VIA polygon annotations)",
                })

    fields = list(records[0]) if records else ["asset_id"]
    with (OUT / "mendeley_green_apple_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    (OUT / "import_report.json").write_text(json.dumps({
        "source_archive_sha256": EXPECTED_SHA256,
        "source_doi": "10.17632/gfcmdbvw65.1",
        "license": "CC BY 4.0",
        "extracted_images": len(records),
        "polygon_labels": sum(int(r["apple_polygon_count"]) for r in records),
        "source_family_ids_assigned": 0,
        "review_status": "pending; source annotations retained but not independently verified",
        "all_imported_images_assigned_to_train_only": True,
        "invalid_annotation_notes": invalid,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"extracted_images": len(records), "polygon_labels": sum(int(r["apple_polygon_count"]) for r in records), "invalid_notes": len(invalid), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
