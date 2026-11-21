#!/usr/bin/env python3
"""Rank review rows by early-candidate eligibility without assigning labels."""
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    return abs(sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1]
                   for i in range(len(points))) / 2.0)


def read_image_names(path, col_a, col_b):
    names = set()
    if not path.exists():
        return names
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for key in (col_a, col_b):
                name = row[key]
                if name.startswith("mendeley_v1_"):
                    names.add(name.removeprefix("mendeley_v1_"))
    return names


def main():
    with (BASE / "review_queue.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    exact_names = read_image_names(BASE / "exact_duplicate_pairs.csv", "image_a", "image_b")
    near_names = read_image_names(BASE / "near_duplicate_candidates_full_dataset.csv", "image_a", "image_b")
    ordered = []
    for row in rows:
        w, h = int(row["image_width"]), int(row["image_height"])
        pts = json.loads(row["polygon_points_json"])
        valid_geometry = len(set(map(tuple, pts))) >= 3 and all(0 <= x < w and 0 <= y < h for x, y in pts)
        ratio = polygon_area(pts) / max(1, w * h)
        plausible_geometry = valid_geometry and 0.005 <= ratio <= 0.75
        duplicate_exact = f"{row['source_split_original']}_{row['original_filename']}" in exact_names
        duplicate_near = f"train_{row['original_filename']}" in near_names
        source_test = row["source_split_original"] == "test"
        reasons = []
        if source_test:
            reasons.append("SOURCE_TEST_CANNOT_ENTER_TRAINING")
        if duplicate_exact:
            reasons.append("EXACT_DUPLICATE_REVIEW_REQUIRED")
        if duplicate_near:
            reasons.append("PHASH_NEAR_DUPLICATE_VISUAL_REVIEW_REQUIRED")
        if not plausible_geometry:
            reasons.append("MASK_GEOMETRY_NEEDS_CAREFUL_REVIEW")
        if row["source_label"].strip().lower() == "immature apple":
            reasons.append("SOURCE_STAGE_CLUE_ONLY_NOT_GREEN_APPLE_APPROVAL")
        if source_test or duplicate_exact or duplicate_near:
            tier = "HOLD_PROVENANCE_OR_DUPLICATE"
        elif not plausible_geometry:
            tier = "REVIEW_MASK_CAREFULLY"
        else:
            tier = "EARLY_CANDIDATE_PRIORITY"
        ordered.append({**row, "priority_tier": tier, "priority_reason": ";".join(reasons) or "TRAIN_SPLIT_HASH_UNIQUE_NO_PHASH_CANDIDATE;HUMAN_REVIEW_REQUIRED",
                        "mask_area_fraction": f"{ratio:.6f}", "geometry_plausible_heuristic": str(bool(plausible_geometry)).lower()})
    rank = {"EARLY_CANDIDATE_PRIORITY": 0, "REVIEW_MASK_CAREFULLY": 1, "HOLD_PROVENANCE_OR_DUPLICATE": 2}
    # Keep every polygon from one source image adjacent and under the same priority tier.
    image_tiers = {}
    for r in ordered:
        key = (r["source_split_original"], r["original_filename"])
        image_tiers[key] = max(image_tiers.get(key, 0), rank[r["priority_tier"]])
    tier_names = {v: k for k, v in rank.items()}
    for r in ordered:
        key = (r["source_split_original"], r["original_filename"])
        r["priority_tier"] = tier_names[image_tiers[key]]
    ordered.sort(key=lambda r: (rank[r["priority_tier"]], r["source_split_original"] != "train", r["original_filename"], int(r["region_index"])))
    fields = list(ordered[0])
    with (BASE / "review_priority_queue.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(ordered)
    counts = {tier: sum(r["priority_tier"] == tier for r in ordered) for tier in rank}
    image_counts = {tier: len({(r["source_split_original"], r["original_filename"]) for r in ordered if r["priority_tier"] == tier}) for tier in rank}
    (BASE / "review_priority_summary.json").write_text(json.dumps({"polygon_count": len(ordered), "priority_counts": counts,
        "unique_image_counts": image_counts,
        "exact_duplicate_image_names": len(exact_names), "phash_candidate_image_names": len(near_names),
        "note": "Priority is only a review order. Source label and geometric heuristics never approve a label or mask."}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
