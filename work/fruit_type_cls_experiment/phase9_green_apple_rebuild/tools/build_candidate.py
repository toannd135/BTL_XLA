#!/usr/bin/env python3
"""Build an unassigned candidate from latest reviewer-confirmed Green Apple masks."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import runpy
import shutil
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
COLLECTION = BASE.parent / "phase8c_collection/green_apple_phase8d/staging/mendeley_v1"
QUEUE = BASE / "review_queue.csv"
HISTORY = BASE / "review_history.jsonl"
DEFAULT_OUTPUT = BASE / "candidate_dataset"

# Keep this historical command from bypassing the stricter review/mask/duplicate gates.
if __name__ == "__main__":
    print("Using strict Green Apple batch exporter. Specify --output for a new empty directory.")
    runpy.run_path(str(BASE / "tools/export_reviewed_batch.py"), run_name="__main__")
    raise SystemExit(0)


def read_latest():
    latest = {}
    if HISTORY.exists():
        for line in HISTORY.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            latest[event["polygon_id"]] = event
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Candidate output directory; it must be absent or empty.")
    out = parser.parse_args().output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty candidate: {out}")
    (out / "unassigned/images").mkdir(parents=True, exist_ok=True)
    (out / "unassigned/labels").mkdir(parents=True, exist_ok=True)
    queue = list(csv.DictReader((BASE / "review_queue.csv").open(encoding="utf-8")))
    latest = read_latest()
    groups = defaultdict(list)
    for row in queue:
        groups[(row["source_split_original"], row["original_filename"])].append(row)
    manifest_rows = []
    excluded = []
    for (source_split, filename), items in sorted(groups.items()):
        decisions = [latest.get(x["polygon_id"], {}).get("decision", "PENDING") for x in items]
        if source_split == "test":
            excluded.append({"original_filename": filename, "source_split_original": source_split,
                             "reason": "EXCLUDE_SOURCE_DATASET_TEST_FROM_TRAIN_CANDIDATE", "polygon_count": len(items)})
            continue
        if not all(d == "GREEN_APPLE" for d in decisions):
            excluded.append({"original_filename": filename, "source_split_original": source_split,
                             "reason": "IMAGE_REQUIRES_ALL_POLYGONS_REVIEWED_GREEN_APPLE", "polygon_count": len(items)})
            continue
        hashes = {x["image_sha256"] for x in items}
        if len(hashes) != 1:
            excluded.append({"original_filename": filename, "source_split_original": source_split,
                             "reason": "INCONSISTENT_IMAGE_HASH", "polygon_count": len(items)})
            continue
        image_sha = next(iter(hashes))
        duplicate_items = [x for x in queue if x["image_sha256"] == image_sha]
        if len({x["original_filename"] for x in duplicate_items}) > 1:
            excluded.append({"original_filename": filename, "source_split_original": source_split,
                             "reason": "EXACT_DUPLICATE_CANONICAL_SELECTION_REQUIRED", "polygon_count": len(items)})
            continue
        src = COLLECTION / items[0]["image_relative_path"]
        stem = "mendeley_" + hashlib.sha256((source_split + "\0" + filename).encode()).hexdigest()[:20]
        dst = out / "unassigned/images" / f"{stem}.jpg"
        shutil.copy2(src, dst)
        with (out / "unassigned/labels" / f"{stem}.txt").open("w", encoding="utf-8") as f:
            for row in items:
                points = json.loads(row["polygon_points_json"])
                w, h = int(row["image_width"]), int(row["image_height"])
                norm = [coord for x, y in points for coord in (min(1.0, max(0.0, x / w)), min(1.0, max(0.0, y / h)))]
                f.write("1 " + " ".join(f"{v:.8f}" for v in norm) + "\n")
        manifest_rows.append({
            "asset_id": items[0]["asset_id"], "original_filename": filename, "source_split_original": source_split,
            "reviewed_polygon_count": len(items), "class_id": 1, "class_name": "apple",
            "source_doi": items[0]["source_doi"], "source_url": items[0]["source_url"],
            "license": items[0]["license"], "image_sha256": image_sha,
            "source_family_id": "", "dataset_split": "UNASSIGNED",
            "review_event_ids": ";".join(latest[x["polygon_id"]]["event_id"] for x in items),
            "image_path": str(dst.relative_to(out)), "label_path": str((out / "unassigned/labels" / f"{stem}.txt").relative_to(out)),
        })

    def write_csv(path, rows, fields):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    manifest_fields = ["asset_id","original_filename","source_split_original","reviewed_polygon_count","class_id","class_name","source_doi","source_url","license","image_sha256","source_family_id","dataset_split","review_event_ids","image_path","label_path"]
    write_csv(out / "candidate_manifest.csv", manifest_rows, manifest_fields)
    write_csv(out / "excluded_pending_or_leakage.csv", excluded, ["original_filename","source_split_original","reason","polygon_count"])
    report = {
        "candidate_status": "EMPTY_PENDING_HUMAN_REVIEW" if not manifest_rows else "UNASSIGNED_REVIEWED_CANDIDATE",
        "included_source_images": len(manifest_rows),
        "included_polygons": sum(int(r["reviewed_polygon_count"]) for r in manifest_rows),
        "source_families_assigned": 0,
        "train_val_test_assignments": 0,
        "all_source_test_records_excluded_from_training_candidate": True,
        "images_excluded_pending_or_due_to_integrity": len(excluded),
        "evaluation_ready": False,
        "note": "Images enter the unassigned candidate only if every apple polygon in the image has latest decision GREEN_APPLE. Human mask review and source-family-based splitting are still required.",
    }
    (out / "candidate_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text("# Green Apple reviewed candidate\n\nImages are stored in `unassigned/`; no train/validation/test split or source family is assigned. Only images whose every polygon has a latest `GREEN_APPLE` review decision are eligible. Original Mendeley test-split images are never put in this training candidate. This directory is not a training-ready dataset until source-family provenance, split design, and independent evaluation are completed.\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
