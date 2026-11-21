#!/usr/bin/env python3
"""Reconcile source-test exclusions and exact/perceptual duplicates read-only."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
AUDIT = BASE.parents[2] / "runs/seg/seg_green_v5_root_cause_audit"
HASHES = AUDIT / "v5_split_hash_manifest.csv"
QUEUE = BASE / "review_queue.csv"


def main():
    rows = list(csv.DictReader(HASHES.open(encoding="utf-8")))
    by_hash = defaultdict(list)
    parsed = []
    for r in rows:
        by_hash[r["sha256"]].append(r)
        parsed.append((r["split"], r["image"], r["sha256"], int(r["phash"], 16)))
    exact = []
    for sha, group in by_hash.items():
        if len(group) > 1:
            for i, a in enumerate(group):
                for b in group[i+1:]:
                    exact.append({"sha256":sha,"split_a":a["split"],"image_a":a["image"],"split_b":b["split"],"image_b":b["image"],"cross_split":a["split"]!=b["split"]})
    near = []
    for i, a in enumerate(parsed):
        for b in parsed[i+1:]:
            dist=(a[3]^b[3]).bit_count()
            if dist<=6:
                near.append({"split_a":a[0],"image_a":a[1],"split_b":b[0],"image_b":b[1],"phash_hamming":dist,"cross_split":a[0]!=b[0],"exact_sha_match":a[2]==b[2]})
    def write(name, records, fields):
        with (BASE/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    write("exact_duplicate_pairs.csv",exact,["sha256","split_a","image_a","split_b","image_b","cross_split"])
    write("near_duplicate_candidates_full_dataset.csv",near,["split_a","image_a","split_b","image_b","phash_hamming","cross_split","exact_sha_match"])
    q=list(csv.DictReader(QUEUE.open(encoding="utf-8")))
    source_test={}
    for r in q:
        if r["source_split_original"]=="test":
            source_test[(r["original_filename"],r["asset_id"])]=r
    exclusions=[]
    for (fn,aid),r in sorted(source_test.items()):
        exclusions.append({"asset_id":aid,"original_filename":fn,"source_split_original":"test","image_sha256":r["image_sha256"],"action":"EXCLUDE_FROM_TRAINING","reason":"Mendeley source-test record was included in V5 train; preserve it as unavailable for future independent evaluation"})
    write("source_test_training_exclusions.csv",exclusions,["asset_id","original_filename","source_split_original","image_sha256","action","reason"])
    result={"v5_images_scanned":len(rows),"sha256_duplicate_pairs_all_splits":len(exact),"sha256_duplicate_cross_split_pairs":sum(x["cross_split"] for x in exact),"near_duplicate_pairs_hamming_le_6_all_splits":len(near),"near_duplicate_pairs_cross_split":sum(x["cross_split"] for x in near),"near_duplicate_pairs_same_split":sum(not x["cross_split"] for x in near),"mendeley_source_test_images_to_exclude_from_training":len(exclusions),"near_duplicate_warning":"pHash distance <= 6 yields candidates requiring visual/provenance review, not automatic duplicate decisions"}
    (BASE/"leakage_audit_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
