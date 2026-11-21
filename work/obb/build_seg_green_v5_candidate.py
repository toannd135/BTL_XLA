#!/usr/bin/env python3
"""Build an isolated segmentation candidate with licensed green-apple polygons."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/obb/seg_green_v4_candidate/data"
APPLE = ROOT / "work/fruit_type_cls_experiment/phase8c_collection/green_apple_phase8d/staging/mendeley_v1"
OUT = ROOT / "work/obb/seg_green_v5_candidate/data"
THRESHOLD = 6


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def phash(p: Path) -> int:
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"Unreadable image: {p}")
    small = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    vals = dct.flatten()[1:]
    med = float(np.median(vals))
    bits = vals > med
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def main() -> None:
    if not BASE.is_dir() or not (APPLE / "mendeley_green_apple_manifest.csv").is_file():
        raise SystemExit("Missing base segmentation data or imported Mendeley manifest")
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite candidate: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASE, OUT)

    train_images = OUT / "train/images"
    train_labels = OUT / "train/labels"
    old = []
    for split in ("train", "val", "test"):
        for p in (OUT / split / "images").glob("*"):
            if p.is_file():
                old.append((split, p, sha256(p), phash(p)))
    valtest_hashes = [(s, p, sh, ph) for s, p, sh, ph in old if s in {"val", "test"}]
    old_exact = {sh: (s, p) for s, p, sh, _ in old}

    source_rows = list(csv.DictReader((APPLE / "mendeley_green_apple_manifest.csv").open(encoding="utf-8")))
    imported, held = [], []
    for row in source_rows:
        src = APPLE / row["image_relative_path"]
        label = APPLE / row["label_relative_path"]
        digest = row["image_sha256"]
        ph = phash(src)
        exact_hit = old_exact.get(digest)
        near_hits = [(s, p.name, d) for s, p, sh, old_ph in valtest_hashes if (d := hamming(ph, old_ph)) <= THRESHOLD]
        reason = ""
        if exact_hit:
            reason = f"EXACT_DUPLICATE_OF_{exact_hit[0]}:{exact_hit[1].name}"
        elif near_hits:
            reason = "PENDING_NEAR_DUPLICATE_REVIEW_AGAINST_VAL_TEST"
        if reason:
            held.append({**row, "phash": f"{ph:016x}", "hold_reason": reason,
                         "near_duplicate_matches": json.dumps(near_hits, ensure_ascii=False)})
            continue
        dst_img = train_images / src.name
        dst_label = train_labels / label.name
        if dst_img.exists() or dst_label.exists():
            raise SystemExit(f"Filename collision: {dst_img}")
        shutil.copy2(src, dst_img)
        shutil.copy2(label, dst_label)
        imported.append({**row, "phash": f"{ph:016x}", "assigned_split": "train",
                         "source_family_id": "", "split_reason": "source provenance unavailable; entire source dataset kept training-only"})

    # Rebase absolute green evaluation image lists to the isolated v5 data copy.
    old_root = str(BASE)
    for name in ("green_val.txt", "green_test.txt"):
        p = OUT / name
        content = p.read_text(encoding="utf-8").replace(old_root, str(OUT))
        p.write_text(content, encoding="utf-8")
    for name in ("seg.yaml", "green_eval.yaml"):
        p = OUT / name
        p.write_text(p.read_text(encoding="utf-8").replace(old_root, str(OUT)), encoding="utf-8")

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("asset_id\n", encoding="utf-8")
            return
        fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    write_csv(OUT.parent / "mendeley_green_apple_train_manifest.csv", imported)
    write_csv(OUT.parent / "mendeley_green_apple_held_candidates.csv", held)
    counts = {s: {"images": sum(1 for _ in (OUT / s / "images").glob("*")),
                  "labels": sum(1 for _ in (OUT / s / "labels").glob("*"))}
              for s in ("train", "val", "test")}
    summary = {
        "status": "CANDIDATE_READY_FOR_OFFLINE_TRAINING",
        "base_candidate": "seg_green_v4_candidate (copied; original remains unchanged)",
        "new_green_apple_source": "Mendeley Data V1, DOI 10.17632/gfcmdbvw65.1, CC BY 4.0",
        "source_archive_sha256": "3b8d3c4f8d6788fd6af8c5fbf92c33755f2130f1159d94ea869543f91e2a1c51",
        "new_apple_images_added_to_train": len(imported),
        "new_apple_polygons_added_to_train": sum(int(r["apple_polygon_count"]) for r in imported),
        "held_from_train_exact_or_near_dup_vs_existing_val_test": len(held),
        "source_family_ids_assigned": 0,
        "conservative_provenance_policy": "unresolved per-image families; all source images kept in train only; never used as val/test",
        "annotation_status": "source VIA annotations; not independently reviewed; review status remains pending",
        "split_counts": counts,
        "green_eval_lists_rebased_to_candidate_copy": True,
        "test_not_used_for_training_or_selection": True,
    }
    (OUT.parent / "candidate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
