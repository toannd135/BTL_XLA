#!/usr/bin/env python3
"""Read-only validation of Phase 1 annotation and Phase 1.2 leakage decisions."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_CSV = ROOT / "work/audit/phase1_review_manifest.csv"
LEAKAGE_CSV = ROOT / "work/audit/phase1.2_leakage_review_manifest.csv"
SOURCE_AUDIT = ROOT / "work/audit/phase1_dataset_audit.json"
REVIEW_DATASET = ROOT / "work/obb/seg_data_review_v1"
DEFAULT_OUTPUT = ROOT / "work/audit/phase1.3_review_validation.json"
SPLITS = {"train", "val", "test"}
ANNOTATION_STATUSES = {"pending", "accepted", "corrected", "needs_review"}
LEAKAGE_STATUSES = {"pending", "same_capture", "near_duplicate", "similar_scene", "unrelated", "needs_review"}
LEAKAGE_DECISIONS = LEAKAGE_STATUSES - {"pending"}
CLASSES = {"banana", "apple", "orange"}


def read_csv(path, expected_columns, issues, category):
    if not path.is_file():
        issues.append({"category": category, "code": "manifest_missing", "path": str(path)})
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = sorted(set(expected_columns) - set(reader.fieldnames or []))
        if missing:
            issues.append({"category": category, "code": "missing_columns", "columns": missing})
        return list(reader)


def resolve_path(value, base):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def image_split_from_path(path):
    parts = path.parts
    for i in range(len(parts) - 1):
        if parts[i] in SPLITS and parts[i + 1] == "images":
            return parts[i]
    return None


def validate(args):
    issues = []
    annotations = read_csv(args.annotation_manifest,
        ("image_path", "source_image_path", "split", "class", "instance_id", "review_status", "reviewer_notes"),
        issues, "annotation")
    leakage = read_csv(args.leakage_manifest,
        ("pair_id", "image_a", "split_a", "image_b", "split_b", "phash_distance",
         "image_a_sha256", "image_b_sha256", "review_status", "reviewer_decision", "reviewer_notes"),
        issues, "leakage")

    try:
        audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
        source_images = {str(Path(x["image"]).resolve()): x for x in audit.get("images", [])}
        canonical_pairs = audit["leakage"]["near_duplicate_image_candidates_phash_le4"]
    except Exception as e:
        issues.append({"category": "source", "code": "source_audit_unreadable", "message": str(e)})
        source_images, canonical_pairs = {}, []
    expected_pairs = {
        f"leakage_pair_{i:04d}": p for i, p in enumerate(canonical_pairs, 1)
    }

    annotation_status_counts = Counter()
    annotation_images = set()
    annotation_instances = set()
    for row_no, row in enumerate(annotations, 2):
        status = (row.get("review_status") or "").strip()
        annotation_status_counts[status or "<blank>"] += 1
        if status not in ANNOTATION_STATUSES:
            issues.append({"category": "annotation", "code": "invalid_review_status", "row": row_no, "value": status})
        decision = (row.get("reviewer_decision") or "").strip()
        if decision and decision not in ANNOTATION_STATUSES - {"pending"}:
            issues.append({"category": "annotation", "code": "invalid_reviewer_decision", "row": row_no, "value": decision})
        if decision and status != decision:
            issues.append({"category": "annotation", "code": "decision_status_mismatch", "row": row_no,
                           "review_status": status, "reviewer_decision": decision})
        note = (row.get("reviewer_notes") or "").strip()
        if status == "needs_review" and not note:
            issues.append({"category": "annotation", "code": "needs_review_requires_notes", "row": row_no})

        split = (row.get("split") or "").strip()
        if split not in SPLITS:
            issues.append({"category": "annotation", "code": "invalid_split", "row": row_no, "value": split})
        review_rel = (row.get("image_path") or "").strip()
        source_raw = (row.get("source_image_path") or "").strip()
        if not review_rel:
            issues.append({"category": "annotation", "code": "missing_image_path", "row": row_no, "field": "image_path"})
        else:
            image_path = resolve_path(review_rel, args.review_dataset_dir)
            annotation_images.add(review_rel)
            if not image_path.is_file():
                issues.append({"category": "annotation", "code": "image_path_not_found", "row": row_no, "path": review_rel})
            if image_split_from_path(image_path) != split:
                issues.append({"category": "annotation", "code": "split_path_mismatch", "row": row_no,
                               "declared_split": split, "path_split": image_split_from_path(image_path), "path": review_rel})
        if not source_raw:
            issues.append({"category": "annotation", "code": "missing_image_path", "row": row_no, "field": "source_image_path"})
        else:
            source_path = resolve_path(source_raw, ROOT)
            source_key = str(source_path)
            canonical = source_images.get(source_key)
            if not source_path.is_file():
                issues.append({"category": "annotation", "code": "source_image_not_found", "row": row_no, "path": source_raw})
            if canonical is None:
                issues.append({"category": "annotation", "code": "source_image_not_in_audit", "row": row_no, "path": source_raw})
            else:
                original_split = canonical["split"]
                if original_split != split:
                    code = "test_image_marked_train" if original_split == "test" and split == "train" else "source_split_changed"
                    issues.append({"category": "split_protection", "code": code, "row": row_no,
                                   "source_split": original_split, "declared_split": split, "path": source_raw})
                if image_split_from_path(source_path) != original_split:
                    issues.append({"category": "split_protection", "code": "source_path_split_mismatch", "row": row_no,
                                   "source_split": original_split, "path": source_raw})

        cls = (row.get("class") or "").strip()
        instance_id = (row.get("instance_id") or "").strip()
        if cls and cls not in CLASSES:
            issues.append({"category": "annotation", "code": "invalid_class", "row": row_no, "value": cls})
        if bool(cls) != bool(instance_id):
            issues.append({"category": "annotation", "code": "class_instance_id_mismatch", "row": row_no})
        if instance_id:
            try:
                if int(instance_id) < 1:
                    raise ValueError
            except ValueError:
                issues.append({"category": "annotation", "code": "invalid_instance_id", "row": row_no, "value": instance_id})
            key = (review_rel, instance_id)
            if key in annotation_instances:
                issues.append({"category": "annotation", "code": "duplicate_image_instance_id", "row": row_no,
                               "image_path": review_rel, "instance_id": instance_id})
            annotation_instances.add(key)

    # Include images represented by multiple instance rows only once in pending image totals.
    annotation_pair_pending = {x for x in annotation_images}

    leakage_status_counts = Counter()
    decision_counts = Counter()
    pair_id_counts = Counter((row.get("pair_id") or "").strip() for row in leakage)
    pair_ids = set(pair_id_counts) - {""}
    duplicate_pair_ids = sorted(pair_id for pair_id, count in pair_id_counts.items() if pair_id and count > 1)
    for pair_id in duplicate_pair_ids:
        issues.append({"category": "leakage", "code": "duplicate_pair_id", "pair_id": pair_id})
    blank_id_rows = pair_id_counts.get("", 0)
    if blank_id_rows:
        issues.append({"category": "leakage", "code": "missing_pair_id", "row_count": blank_id_rows})
    missing_pair_ids = sorted(set(expected_pairs) - pair_ids)
    unexpected_pair_ids = sorted(pair_ids - set(expected_pairs))
    for pair_id in missing_pair_ids:
        issues.append({"category": "leakage", "code": "missing_pair_id", "pair_id": pair_id})
    for pair_id in unexpected_pair_ids:
        issues.append({"category": "leakage", "code": "unexpected_pair_id", "pair_id": pair_id})

    same_capture_train_test = []
    same_capture_train_val = []
    retain_current_split = []
    for row_no, row in enumerate(leakage, 2):
        pair_id = (row.get("pair_id") or "").strip()
        canonical = expected_pairs.get(pair_id)
        status = (row.get("review_status") or "").strip()
        decision = (row.get("reviewer_decision") or "").strip()
        note = (row.get("reviewer_notes") or "").strip()
        leakage_status_counts[status or "<blank>"] += 1
        decision_counts[decision or "<blank>"] += 1
        if status not in LEAKAGE_STATUSES:
            issues.append({"category": "leakage", "code": "invalid_review_status", "row": row_no, "value": status})
        if decision and decision not in LEAKAGE_DECISIONS:
            issues.append({"category": "leakage", "code": "invalid_reviewer_decision", "row": row_no, "value": decision})
        if status == "pending" and decision:
            issues.append({"category": "leakage", "code": "pending_has_reviewer_decision", "row": row_no, "decision": decision})
        if status != "pending" and status in LEAKAGE_STATUSES and decision != status:
            issues.append({"category": "leakage", "code": "decision_status_mismatch", "row": row_no,
                           "review_status": status, "reviewer_decision": decision})
        if status == "needs_review" and not note:
            issues.append({"category": "leakage", "code": "needs_review_requires_notes", "row": row_no, "pair_id": pair_id})
        if status not in {"pending", "needs_review"} and status in LEAKAGE_STATUSES and not note:
            issues.append({"category": "leakage", "code": "final_decision_requires_evidence_note", "row": row_no, "pair_id": pair_id})

        split_a, split_b = (row.get("split_a") or "").strip(), (row.get("split_b") or "").strip()
        if split_a not in SPLITS:
            issues.append({"category": "leakage", "code": "invalid_split", "row": row_no, "side": "a", "value": split_a})
        if split_b not in SPLITS:
            issues.append({"category": "leakage", "code": "invalid_split", "row": row_no, "side": "b", "value": split_b})
        for side, split in (("a", split_a), ("b", split_b)):
            rel = (row.get(f"image_{side}") or "").strip()
            if not rel:
                issues.append({"category": "leakage", "code": "missing_image_path", "row": row_no, "side": side})
                continue
            path = resolve_path(rel, ROOT)
            if not path.is_file():
                issues.append({"category": "leakage", "code": "image_path_not_found", "row": row_no, "side": side, "path": rel})
            path_split = image_split_from_path(path)
            if path_split != split:
                code = "test_image_marked_train" if path_split == "test" and split == "train" else "split_path_mismatch"
                issues.append({"category": "split_protection", "code": code, "row": row_no, "side": side,
                               "declared_split": split, "path_split": path_split, "path": rel})
        if canonical:
            for side, path_key, split_key, hash_key in (
                ("a", "image_a", "split_a", "image_a_sha256"),
                ("b", "image_b", "split_b", "image_b_sha256"),
            ):
                source_abs = str((ROOT / canonical[path_key]).resolve())
                declared_abs = str(resolve_path((row.get(path_key) or "").strip(), ROOT))
                if declared_abs != source_abs or (row.get(split_key) or "").strip() != canonical[f"split_{side}"]:
                    original_split = canonical[f"split_{side}"]
                    declared_split = (row.get(split_key) or "").strip()
                    code = "test_image_marked_train" if original_split == "test" and declared_split == "train" else "canonical_pair_assignment_changed"
                    issues.append({"category": "split_protection", "code": code, "row": row_no,
                                   "pair_id": pair_id, "side": side, "source_split": original_split,
                                   "declared_split": declared_split, "source_path": canonical[path_key],
                                   "declared_path": row.get(path_key)})
                source_record = source_images.get(source_abs)
                expected_sha = source_record.get("image_sha256") if source_record else None
                declared_sha = (row.get(hash_key) or "").strip()
                if expected_sha and declared_sha != expected_sha:
                    issues.append({"category": "leakage", "code": "image_hash_mismatch", "row": row_no,
                                   "pair_id": pair_id, "side": side})
            try:
                if int(row.get("phash_distance", "")) != int(canonical["phash_hamming"]):
                    issues.append({"category": "leakage", "code": "phash_distance_mismatch", "row": row_no, "pair_id": pair_id})
            except (TypeError, ValueError):
                issues.append({"category": "leakage", "code": "invalid_phash_distance", "row": row_no, "pair_id": pair_id})

        # Future corrections are recommendations only. The validator never edits splits.
        if status == "same_capture" and canonical:
            pair = {"pair_id": pair_id, "image_a": row.get("image_a"), "split_a": split_a,
                    "image_b": row.get("image_b"), "split_b": split_b, "action_applied": False}
            split_set = {canonical["split_a"], canonical["split_b"]}
            if split_set == {"train", "test"}:
                pair["future_recommendation"] = "keep both in test; move only the non-test counterpart in a future version; never move test into train"
                same_capture_train_test.append(pair)
            elif split_set == {"train", "val"}:
                pair["future_recommendation"] = "assign both to one common non-test split in a future version"
                same_capture_train_val.append(pair)
        if status in {"similar_scene", "unrelated"} and canonical:
            retain_current_split.append(pair_id)

    # Detect optional proposed/target assignment columns if an edited copy adds them.
    test_to_train_proposals = []
    for row_no, row in enumerate(leakage, 2):
        canonical = expected_pairs.get((row.get("pair_id") or "").strip())
        if not canonical:
            continue
        for side in ("a", "b"):
            proposed = (row.get(f"proposed_split_{side}") or row.get(f"new_split_{side}") or "").strip()
            if canonical[f"split_{side}"] == "test" and proposed == "train":
                test_to_train_proposals.append({"row": row_no, "pair_id": row.get("pair_id"), "side": side,
                                                "code": "test_image_moved_to_train_proposal"})
                issues.append({"category": "split_protection", "code": "test_image_moved_to_train_proposal",
                               "row": row_no, "pair_id": row.get("pair_id"), "side": side})

    errors_by_category = Counter(x["category"] for x in issues)
    report = {
        "schema_version": 1,
        "validator": "phase1.3-human-review-read-only-v1",
        "inputs": {"annotation_manifest": str(args.annotation_manifest),
                   "leakage_manifest": str(args.leakage_manifest),
                   "source_audit": str(args.source_audit),
                   "review_dataset_dir": str(args.review_dataset_dir)},
        "validation_status": "valid" if not issues else "invalid",
        "issues_count": len(issues),
        "issues_by_category": dict(errors_by_category),
        "issues": issues,
        "annotation_review": {
            "manifest_rows": len(annotations),
            "unique_review_images": len(annotation_images),
            "review_status_counts": dict(annotation_status_counts),
            "pending_count": annotation_status_counts.get("pending", 0),
            "pending_unique_image_count": sum(1 for image in annotation_images
                if any((r.get("image_path") or "").strip() == image and (r.get("review_status") or "").strip() == "pending" for r in annotations)),
        },
        "leakage_review": {
            "manifest_rows": len(leakage),
            "canonical_pair_count": len(expected_pairs),
            "unique_pair_ids": len(pair_ids),
            "review_status_counts": dict(leakage_status_counts),
            "reviewer_decision_counts": dict(decision_counts),
            "pending_count": leakage_status_counts.get("pending", 0),
            "missing_pair_ids": missing_pair_ids,
            "duplicate_pair_ids": duplicate_pair_ids,
            "unexpected_pair_ids": unexpected_pair_ids,
        },
        "split_protection": {
            "test_to_train_violations": sum(1 for x in issues if x["code"] == "test_image_marked_train"),
            "test_to_train_proposals": test_to_train_proposals,
            "same_capture_train_test_future_flags": same_capture_train_test,
            "same_capture_train_val_future_flags": same_capture_train_val,
            "similar_or_unrelated_retain_current_pair_ids": retain_current_split,
            "split_changes_applied": 0,
            "validator_modifies_decisions_or_splits": False,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-manifest", type=Path, default=ANNOTATION_CSV)
    parser.add_argument("--leakage-manifest", type=Path, default=LEAKAGE_CSV)
    parser.add_argument("--source-audit", type=Path, default=SOURCE_AUDIT)
    parser.add_argument("--review-dataset-dir", type=Path, default=REVIEW_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_paths = {x.resolve() for x in (args.annotation_manifest, args.leakage_manifest, args.source_audit)}
    if args.output.resolve() in input_paths:
        raise SystemExit("Output path must not overwrite an input manifest/audit")
    report = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "validation_status": report["validation_status"],
                      "issues_count": report["issues_count"],
                      "pending_annotation_rows": report["annotation_review"]["pending_count"],
                      "pending_annotation_images": report["annotation_review"]["pending_unique_image_count"],
                      "pending_leakage_pairs": report["leakage_review"]["pending_count"],
                      "missing_pair_ids": len(report["leakage_review"]["missing_pair_ids"]),
                      "split_correction_flags": len(report["split_protection"]["same_capture_train_test_future_flags"])
                                                + len(report["split_protection"]["same_capture_train_val_future_flags"])}, indent=2))
    raise SystemExit(1 if report["issues_count"] else 0)


if __name__ == "__main__":
    main()
