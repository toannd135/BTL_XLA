#!/usr/bin/env python3
"""Export only explicitly reviewed Green Apple images into a new unassigned batch."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
COLLECTION = BASE.parent / "phase8c_collection/green_apple_phase8d/staging/mendeley_v1"


def read_latest():
    latest = {}
    p = BASE / "review_history.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = json.loads(line)
                latest[event["polygon_id"]] = event
    return latest


def read_names(path, a, b):
    result = set()
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                for col in (a, b):
                    n = r[col]
                    if n.startswith("mendeley_v1_"):
                        result.add(n.removeprefix("mendeley_v1_"))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True, help="New empty directory for this reviewed batch")
    args = ap.parse_args()
    out = args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {out}")
    queue = list(csv.DictReader((BASE / "review_queue.csv").open(encoding="utf-8", newline="")))
    latest = read_latest()
    exact_names = read_names(BASE / "exact_duplicate_pairs.csv", "image_a", "image_b")
    near_names = read_names(BASE / "near_duplicate_candidates_full_dataset.csv", "image_a", "image_b")
    grouped = defaultdict(list)
    for r in queue:
        grouped[(r["source_split_original"], r["original_filename"])].append(r)
    (out / "unassigned/images").mkdir(parents=True, exist_ok=True)
    (out / "unassigned/labels").mkdir(parents=True, exist_ok=True)
    included, excluded = [], []
    for (split, filename), rows in sorted(grouped.items()):
        decisions = [latest.get(r["polygon_id"], {}) for r in rows]
        reasons = []
        if split != "train": reasons.append("SOURCE_TEST_EXCLUDED_FROM_TRAINING")
        if any(e.get("decision") != "GREEN_APPLE" for e in decisions): reasons.append("NOT_ALL_POLYGONS_HUMAN_CONFIRMED_GREEN_APPLE")
        if any(e.get("mask_valid") != "yes" for e in decisions): reasons.append("MASK_VALIDITY_NOT_CONFIRMED_FOR_EVERY_POLYGON")
        if any(e.get("image_quality_ok") != "yes" for e in decisions): reasons.append("IMAGE_QUALITY_NOT_CONFIRMED_FOR_EVERY_POLYGON")
        name_key = f"{split}_{filename}"
        if name_key in exact_names: reasons.append("EXACT_DUPLICATE_REQUIRES_HUMAN_DISPOSITION")
        if name_key in near_names: reasons.append("PHASH_NEAR_DUPLICATE_NOT_VISUALLY_RESOLVED")
        shas = {r["image_sha256"] for r in rows}
        if len(shas) != 1: reasons.append("INCONSISTENT_IMAGE_SHA256")
        w, h = int(rows[0]["image_width"]), int(rows[0]["image_height"])
        for r in rows:
            pts = json.loads(r["polygon_points_json"])
            if len(set(map(tuple, pts))) < 3 or any(not (0 <= x < w and 0 <= y < h) for x, y in pts):
                reasons.append("INVALID_POLYGON_GEOMETRY")
                break
        if reasons:
            excluded.append({"original_filename": filename, "source_split_original": split,
                             "polygon_count": len(rows), "reasons": ";".join(sorted(set(reasons)))})
            continue
        src = COLLECTION / rows[0]["image_relative_path"]
        stem = "mendeley_" + hashlib.sha256((split + "\0" + filename).encode()).hexdigest()[:20]
        dst_img = out / "unassigned/images" / f"{stem}.jpg"
        shutil.copy2(src, dst_img)
        dst_lbl = out / "unassigned/labels" / f"{stem}.txt"
        with dst_lbl.open("w", encoding="utf-8") as f:
            for r in rows:
                pts = json.loads(r["polygon_points_json"])
                values = [v for x, y in pts for v in (x / w, y / h)]
                f.write("0 " + " ".join(f"{v:.8f}" for v in values) + "\n")
        included.append({"asset_id": rows[0]["asset_id"], "original_filename": filename,
            "source_split_original": split, "polygon_count": len(rows), "class_id": 0,
            "class_name": "green_apple", "source_doi": rows[0]["source_doi"],
            "source_url": rows[0]["source_url"], "license": rows[0]["license"],
            "image_sha256": next(iter(shas)), "source_family_id": "", "dataset_split": "UNASSIGNED",
            "image_path": str(dst_img.relative_to(out)), "label_path": str(dst_lbl.relative_to(out)),
            "review_event_ids": ";".join(e["event_id"] for e in decisions),
            "reviewer_ids": ";".join(sorted({e["reviewer_id"] for e in decisions}))})
    def write(path, rows, fields):
        with path.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    manifest_fields = ["asset_id","original_filename","source_split_original","polygon_count","class_id","class_name","source_doi","source_url","license","image_sha256","source_family_id","dataset_split","image_path","label_path","review_event_ids","reviewer_ids"]
    write(out / "candidate_manifest.csv", included, manifest_fields)
    write(out / "excluded_images.csv", excluded, ["original_filename","source_split_original","polygon_count","reasons"])
    summary = {"status": "EMPTY_NO_ELIGIBLE_REVIEWED_IMAGES" if not included else "REVIEWED_UNASSIGNED_BATCH",
        "included_images": len(included), "included_green_apple_polygons": sum(x["polygon_count"] for x in included),
        "excluded_images": len(excluded), "source_families_assigned": 0, "dataset_splits_assigned": 0,
        "evaluation_ready": False, "rule": "Every polygon must have latest human GREEN_APPLE, mask_valid=yes, image_quality_ok=yes; source-test, exact duplicates, pHash candidates, and invalid geometry are withheld."}
    (out / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
