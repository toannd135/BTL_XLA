#!/usr/bin/env python3
"""Create a PENDING-only, per-polygon Green Apple review queue from VIA JSON."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
COLLECTION = HERE.parent / "phase8c_collection/green_apple_phase8d/staging/mendeley_v1"
OUT = HERE / "review_queue.csv"


def item_id(split: str, filename: str, index: int) -> str:
    key = f"gfcmdbvw65.1\0{split}\0{filename}\0{index}"
    return "via_" + hashlib.sha256(key.encode()).hexdigest()[:24]


def main() -> None:
    manifest = {}
    with (COLLECTION / "mendeley_green_apple_manifest.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest[(row["source_split_in_archive"], row["original_filename"])] = row
    rows = []
    for split in ("train", "test"):
        data = json.loads((COLLECTION / "source_via" / f"{split}.json").read_text(encoding="utf-8"))
        for entry in data.values():
            filename = entry.get("filename")
            mrow = manifest.get((split, filename))
            if not mrow:
                continue
            for idx, region in enumerate(entry.get("regions") or []):
                attrs = region.get("region_attributes") or {}
                stage = str(attrs.get("name") or attrs.get("Name") or "").strip()
                shape = region.get("shape_attributes") or {}
                xs, ys = shape.get("all_points_x") or [], shape.get("all_points_y") or []
                rows.append({
                    "polygon_id": item_id(split, filename, idx),
                    "asset_id": mrow["asset_id"],
                    "source_split_original": split,
                    "original_filename": filename,
                    "region_index": idx,
                    "source_label": stage or "UNKNOWN",
                    "shape_type": shape.get("name", "UNKNOWN"),
                    "polygon_points_json": json.dumps([[x, y] for x, y in zip(xs, ys)], separators=(",", ":")),
                    "image_width": mrow["width"],
                    "image_height": mrow["height"],
                    "image_relative_path": mrow["image_relative_path"],
                    "image_sha256": mrow["image_sha256"],
                    "source_doi": "10.17632/gfcmdbvw65.1",
                    "source_url": "https://data.mendeley.com/datasets/gfcmdbvw65/1",
                    "license": "CC BY 4.0",
                    "review_status": "PENDING",
                })
    if len(rows) != 1017:
        raise SystemExit(f"Expected 1,017 source polygons, got {len(rows)}; refusing queue creation")
    fields = list(rows[0])
    with OUT.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Created {len(rows)} PENDING polygon records at {OUT}")


if __name__ == "__main__":
    main()
