#!/usr/bin/env python3
"""Create manual comparison sheets for frozen Phase 1 cross-split pHash pairs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "work/audit/phase1_dataset_audit.json"
DEFAULT_CSV = ROOT / "work/audit/phase1.2_leakage_review_manifest.csv"
DEFAULT_SHEETS = ROOT / "work/audit/phase1.2_leakage_contact_sheets"
EXPECTED_PAIR_SHA256 = "d5e2e7043a9ea63c0db01312e034a036d81ffabea70799efe9b01e8bfb7191ba"
ALLOWED_STATUSES = {"pending", "same_capture", "near_duplicate", "similar_scene", "unrelated", "needs_review"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recommendation(split_a, split_b):
    if split_a == split_b:
        return (f"Already in {split_a}; if confirmed same_capture, keep together in {split_a}. "
                "If only visually similar/unrelated, retain the current split.")
    if "test" in (split_a, split_b):
        return ("If confirmed same_capture, keep both in test in a future version by moving only the non-test "
                "counterpart; this keeps the capture out of training and preserves test independence. Never move "
                "a test image into train. If only visually similar/unrelated, retain current splits.")
    return ("If confirmed same_capture, assign both to one common non-test split in a future version. "
            "If only visually similar/unrelated, retain current splits.")


def draw_pair_sheet(records, output_dir, per_sheet=4):
    output_dir.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default()
    panel_w, sheet_h, pair_h = 720, 1480, 370
    sheet_w = 2 * panel_w
    for page_start in range(0, len(records), per_sheet):
        page = records[page_start:page_start + per_sheet]
        canvas = Image.new("RGB", (sheet_w, len(page) * pair_h), "white")
        draw = ImageDraw.Draw(canvas)
        for local_idx, record in enumerate(page):
            y = local_idx * pair_h
            draw.text((8, y + 5), f"{record['pair_id']} | pHash Hamming distance: {record['phash_distance']}",
                      font=font, fill="black")
            paths = [record["_path_a"], record["_path_b"]]
            dims = [record["_dimensions_a"], record["_dimensions_b"]]
            images = [Image.open(p).convert("RGB") for p in paths]
            # A common pixel scale within a pair makes relative source sizes comparable.
            scale = min(700 / max(im.width for im in images), 270 / max(im.height for im in images))
            for side, (path, image, size) in enumerate(zip(paths, images, dims)):
                x = side * panel_w
                split = record["split_a"] if side == 0 else record["split_b"]
                rel = record["image_a"] if side == 0 else record["image_b"]
                label = "Image A" if side == 0 else "Image B"
                draw.text((x + 8, y + 28), f"{label} | split={split} | {size[0]}x{size[1]}",
                          font=font, fill="black")
                draw.text((x + 8, y + 44), rel[:115], font=font, fill="black")
                if len(rel) > 115:
                    draw.text((x + 8, y + 58), rel[115:230], font=font, fill="black")
                resized = image.resize((max(1, round(image.width * scale)),
                                        max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
                top = y + 78
                px = x + (panel_w - resized.width) // 2
                py = top + (280 - resized.height) // 2
                canvas.paste(resized, (px, py))
        page_number = page_start // per_sheet + 1
        canvas.save(output_dir / f"leakage_contact_sheet_{page_number:03d}.jpg", quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--sheets", type=Path, default=DEFAULT_SHEETS)
    parser.add_argument("--pairs-per-sheet", type=int, default=4)
    args = parser.parse_args()
    if not 4 <= args.pairs_per_sheet <= 6:
        raise SystemExit("pairs-per-sheet must be between 4 and 6")
    if args.csv.exists() or args.sheets.exists():
        raise FileExistsError("Review outputs already exist; choose new --csv and --sheets paths to avoid overwriting review work")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    pairs = audit["leakage"]["near_duplicate_image_candidates_phash_le4"]
    pair_sha = hashlib.sha256(json.dumps(pairs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if len(pairs) != 362 or pair_sha != EXPECTED_PAIR_SHA256:
        raise RuntimeError(f"Canonical pair source mismatch: count={len(pairs)} sha256={pair_sha}")
    by_path = {str(Path(item["image"]).resolve()): item for item in audit["images"]}
    records = []
    for idx, pair in enumerate(pairs, 1):
        path_a, path_b = Path(pair["image_a"]).resolve(), Path(pair["image_b"]).resolve()
        if not path_a.is_file() or not path_b.is_file():
            raise FileNotFoundError(f"Missing candidate image for pair {idx}: {path_a} / {path_b}")
        rec_a, rec_b = by_path.get(str(path_a)), by_path.get(str(path_b))
        if rec_a is None or rec_b is None:
            raise RuntimeError(f"Candidate image absent from canonical source manifest: pair {idx}")
        hash_a, hash_b = sha256(path_a), sha256(path_b)
        if hash_a != rec_a["image_sha256"] or hash_b != rec_b["image_sha256"]:
            raise RuntimeError(f"Image SHA256 mismatch for pair {idx}")
        with Image.open(path_a) as im_a, Image.open(path_b) as im_b:
            dimensions_a, dimensions_b = im_a.size, im_b.size
        rel_a, rel_b = path_a.relative_to(ROOT).as_posix(), path_b.relative_to(ROOT).as_posix()
        records.append({
            "pair_id": f"leakage_pair_{idx:04d}", "image_a": rel_a, "split_a": pair["split_a"],
            "image_b": rel_b, "split_b": pair["split_b"], "phash_distance": pair["phash_hamming"],
            "image_a_sha256": hash_a, "image_b_sha256": hash_b,
            "review_status": "pending", "reviewer_decision": "", "reviewer_notes": "",
            "split_recommendation": recommendation(pair["split_a"], pair["split_b"]),
            "_path_a": path_a, "_path_b": path_b,
            "_dimensions_a": dimensions_a, "_dimensions_b": dimensions_b,
        })
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pair_id", "image_a", "split_a", "image_b", "split_b", "phash_distance",
                  "image_a_sha256", "image_b_sha256", "review_status", "reviewer_decision",
                  "reviewer_notes", "split_recommendation"]
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fieldnames})
    draw_pair_sheet(records, args.sheets, args.pairs_per_sheet)
    print(json.dumps({"pair_count": len(records), "pair_list_sha256": pair_sha,
                      "csv": str(args.csv), "contact_sheet_dir": str(args.sheets),
                      "contact_sheet_count": math.ceil(len(records) / args.pairs_per_sheet),
                      "pairs_per_sheet": args.pairs_per_sheet,
                      "review_status_counts": {"pending": len(records)},
                      "image_sha256_verified_against_audit_manifest": True}, indent=2))


if __name__ == "__main__":
    main()
