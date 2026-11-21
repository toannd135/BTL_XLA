"""Export reviewed feedback masks into an isolated YOLO segmentation dataset.

The source files in work/feedback are never changed. Only feedback entries with
the requested verdict are exported, so the result can be inspected before use
in a training run.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


LABELS = {"banana": 0, "apple": 1, "orange": 2}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-dir", type=Path, default=Path("work/feedback"))
    parser.add_argument("--output", type=Path, default=Path("work/feedback_dataset"))
    parser.add_argument("--verdict", choices=("correct", "needs_review"), default="needs_review")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Deterministic validation fraction; groups exact duplicate images together.")
    parser.add_argument("--allow-duplicates", action="store_true",
                        help="Export exact duplicate images; inspect audit.json first.")
    return parser.parse_args()


def load_entry(path, verdict):
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"SKIP {path.name}: {exc}")
        return None
    if entry.get("verdict") != verdict:
        return None
    result = entry.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("objects"), list):
        print(f"SKIP {path.name}: missing result.objects")
        return None
    image = path.with_suffix(".jpg")
    if not image.exists():
        print(f"SKIP {path.name}: missing image")
        return None
    return entry, result, image


def convert_polygon(polygon, width, height):
    if not isinstance(polygon, list) or len(polygon) < 3:
        return None
    try:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
    if cv2.contourArea(clipped.reshape(-1, 1, 2)) <= 0:
        return None
    normalized = clipped / np.asarray([width, height], dtype=np.float32)
    return normalized, not np.array_equal(points, clipped)


def export_entry(entry, result, image_path, output, split):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"SKIP {image_path.name}: unreadable image")
        return False
    height, width = image.shape[:2]
    stem = entry.get("id") or image_path.stem
    target_image = output / split / "images" / f"{stem}.jpg"
    target_label = output / split / "labels" / f"{stem}.txt"
    lines = []
    clipped = 0
    for obj in result["objects"]:
        label = LABELS.get(obj.get("label"))
        converted = convert_polygon(obj.get("polygon"), width, height)
        if label is None or converted is None:
            print(f"SKIP object in {image_path.name}: invalid label or mask polygon")
            continue
        normalized, was_clipped = converted
        if was_clipped:
            clipped += 1
        coords = normalized.reshape(-1)
        lines.append(f"{label} " + " ".join(f"{value:.6f}" for value in coords))
    if not lines:
        print(f"SKIP {image_path.name}: no valid objects")
        return False
    shutil.copyfile(image_path, target_image)
    target_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"image": image_path.name, "split": split, "objects": len(lines), "clipped_masks": clipped}


def main():
    args = parse_args()
    feedback_dir = args.feedback_dir
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be between 0 and 1")
    for split in ("train", "val"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    candidates = []
    for path in sorted(feedback_dir.glob("*.json")):
        loaded = load_entry(path, args.verdict)
        if loaded:
            entry, result, image = loaded
            candidates.append((entry, result, image, hashlib.sha256(image.read_bytes()).hexdigest()))

    groups = {}
    for entry, result, image, digest in candidates:
        groups.setdefault(digest, []).append((entry, result, image))
    ordered = sorted(groups.items(), key=lambda item: item[0])
    eligible = [group for group in ordered if len(group[1]) == 1 or args.allow_duplicates]
    val_count = max(1, round(len(eligible) * args.val_ratio)) if len(eligible) > 1 and args.val_ratio else 0
    report = {"verdict": args.verdict, "groups": len(ordered), "exported": [], "duplicates": []}
    eligible_index = 0
    for digest, items in ordered:
        if len(items) > 1 and not args.allow_duplicates:
            report["duplicates"].append({"sha256": digest, "files": [item[2].name for item in items],
                                         "skipped": True})
            continue
        split = "val" if eligible_index < val_count else "train"
        eligible_index += 1
        for entry, result, image in items:
            exported = export_entry(entry, result, image, output, split)
            if exported:
                report["exported"].append(exported)
        if len(items) > 1:
            report["duplicates"].append({"sha256": digest, "files": [item[2].name for item in items]})

    (output / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\nnames:\n  0: banana\n  1: apple\n  2: orange\n",
        encoding="utf-8",
    )
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(report['exported'])} reviewed images to {output}")
    print(f"Groups: {report['groups']}; validation groups: {val_count}; exact duplicate groups: {len(report['duplicates'])}")


if __name__ == "__main__":
    main()
