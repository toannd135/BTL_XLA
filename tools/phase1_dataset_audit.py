#!/usr/bin/env python3
"""Read-only audit and deterministic review-subset builder for YOLO-seg fruit data."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "work/obb/seg_data_candidate"
DEFAULT_OUT = ROOT / "work/obb/seg_data_review_v1"
CLASSES = {0: "banana", 1: "apple", 2: "orange"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COLORS = {"banana": (255, 220, 0), "apple": (40, 220, 60), "orange": (255, 130, 20)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_label(path, width, height):
    """Parse rows while retaining malformed rows for audit reporting."""
    instances, problems, raw_rows = [], [], []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return instances, [f"unreadable-label:{type(e).__name__}"], raw_rows
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        raw_rows.append(line.strip())
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            problems.append(f"line-{line_no}:too-few-or-odd-coordinates")
            continue
        try:
            class_id = int(parts[0])
            values = np.asarray([float(x) for x in parts[1:]], dtype=np.float64)
        except Exception:
            problems.append(f"line-{line_no}:not-numeric")
            continue
        if class_id not in CLASSES:
            problems.append(f"line-{line_no}:unknown-class-{class_id}")
            continue
        pts = values.reshape(-1, 2)
        if len(pts) < 3:
            problems.append(f"line-{line_no}:fewer-than-3-points")
            continue
        if not np.isfinite(pts).all():
            problems.append(f"line-{line_no}:non-finite")
            continue
        if polygon_self_intersects(pts):
            problems.append(f"line-{line_no}:self-intersecting-polygon")
        out_of_range = bool(((pts < 0) | (pts > 1)).any())
        if out_of_range:
            problems.append(f"line-{line_no}:coordinates-outside-0-1")
        pixel = np.rint(pts * np.array([width - 1, height - 1])).astype(np.int32)
        area = abs(float(cv2.contourArea(pixel)))
        if area <= 0:
            problems.append(f"line-{line_no}:zero-area")
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pixel], 1)
        area_px = int(mask.sum())
        normalized = np.round(pts, 7)
        instances.append({
            "class_id": class_id, "class": CLASSES[class_id], "polygon": pts.tolist(),
            "area_px": area_px, "area_ratio": area_px / max(1, width * height),
            "empty_mask": area_px == 0, "out_of_range": out_of_range,
            "mask": mask, "line_no": line_no,
            "polygon_sha": hashlib.sha256(normalized.tobytes()).hexdigest(),
        })
    return instances, problems, raw_rows


def polygon_self_intersects(points):
    """Sweep over edge x-ranges and test non-adjacent candidates, including collinear touch."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    def on_segment(a, b, p):
        return (min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12 and
                min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12)
    def intersects(a, b, c, d):
        o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
        if ((o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and
                (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0)):
            return True
        return ((abs(o1) <= 1e-12 and on_segment(a, b, c)) or
                (abs(o2) <= 1e-12 and on_segment(a, b, d)) or
                (abs(o3) <= 1e-12 and on_segment(c, d, a)) or
                (abs(o4) <= 1e-12 and on_segment(c, d, b)))
    n = len(points)
    edges = []
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        edges.append((min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1]), i, a, b))
    edges.sort(key=lambda edge: edge[0])
    active = []
    for edge in edges:
        min_x, _max_x, min_y, max_y, i, a, b = edge
        active = [candidate for candidate in active if candidate[1] + 1e-12 >= min_x]
        for other in active:
            if other[3] + 1e-12 < min_y or max_y + 1e-12 < other[2]:
                continue
            j = other[4]
            if j == i or j == (i + 1) % n or (j + 1) % n == i:
                continue
            if intersects(a, b, other[5], other[6]):
                return True
        active.append(edge)
    return False


def phash64(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].reshape(-1)
    median = np.median(dct[1:])
    value = 0
    for bit in (dct > median):
        value = (value << 1) | int(bit)
    return value


def pairwise_geometry(instances):
    pairs = []
    for i in range(len(instances)):
        a = instances[i]
        for j in range(i + 1, len(instances)):
            b = instances[j]
            inter = int(np.logical_and(a["mask"], b["mask"]).sum())
            if not inter:
                continue
            area_a, area_b = a["area_px"], b["area_px"]
            union = area_a + area_b - inter
            pairs.append({
                "instance_a": i + 1, "instance_b": j + 1,
                "class_a": a["class"], "class_b": b["class"], "intersection_px": inter,
                "iou": inter / union if union else 0.0,
                "overlap_of_smaller": inter / min(area_a, area_b) if min(area_a, area_b) else 0.0,
                "mask_overlap": True,
            })
    return pairs


def audit_dataset(data_root):
    result = {
        "schema_version": 1,
        "source_dataset": str(data_root.resolve()),
        "annotation_version": "source-sam-pseudolabel-unreviewed",
        "classes": CLASSES,
        "methodology": {
            "image_label_association": "unique same-stem image and label within each split",
            "mask_rasterization": "OpenCV fillPoly at source image dimensions",
            "small_mask_threshold": "area < 0.01% of image pixels",
            "large_mask_threshold": "area >= 50% of image pixels",
            "duplicate_mask_candidate": "same-class raster mask IoU >= 0.98",
            "severe_overlap_candidate": "intersection >= 50% of the smaller mask",
            "suspicious_instance_count": "> 10 annotations in one image",
            "near_duplicate_screen": "64-bit DCT perceptual hash, cross-split Hamming distance <= 4",
            "self_intersection_check": "sweep over polygon edge x-ranges; non-adjacent crossing or touching is flagged",
            "review_grouping": "exclusive primary groups; 3+ bananas prioritized, with severe overlap taking precedence",
            "interpretation": "geometric and hash flags are candidates for human review, not annotation verdicts",
        },
        "source_metadata": {},
        "splits": {}, "images": [], "issues": [], "leakage": {},
        "class_area_ratio": defaultdict(list),
    }
    for metadata_path in (data_root / "candidate_manifest.json", data_root / "seg.yaml"):
        if metadata_path.is_file():
            result["source_metadata"][metadata_path.name] = {
                "path": str(metadata_path.resolve()), "sha256": sha256(metadata_path)
            }
    image_hash_splits = defaultdict(list)
    phashes = []
    for split in ("train", "val", "test"):
        image_dir, label_dir = data_root / split / "images", data_root / split / "labels"
        split_stats = {"images": 0, "instances": 0, "instances_by_class": Counter(),
                       "images_by_class": Counter(), "negative_images": 0,
                       "missing_labels": [], "orphan_labels": [], "missing_images": [],
                       "invalid_polygon_rows": 0, "empty_masks": 0,
                       "duplicate_annotation_pairs": 0, "overlapping_image_count": 0,
                       "overlap_pairs": 0, "unusually_small_masks": 0,
                       "unusually_large_masks": 0, "suspicious_many_instances": []}
        image_files = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if image_dir.exists() else []
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        images_by_stem = defaultdict(list)
        for img in image_files:
            images_by_stem[img.stem].append(img)
        labels_by_stem = defaultdict(list)
        for lab in labels:
            labels_by_stem[lab.stem].append(lab)
        for stem, imgs in images_by_stem.items():
            if len(imgs) != 1:
                result["issues"].append({"split": split, "stem": stem, "issue": "ambiguous-image-stem", "paths": [str(x) for x in imgs]})
        for stem, labs in labels_by_stem.items():
            if len(labs) != 1:
                result["issues"].append({"split": split, "stem": stem, "issue": "ambiguous-label-stem", "paths": [str(x) for x in labs]})
        split_stats["missing_labels"] = [str(p) for p in image_files if p.stem not in labels_by_stem]
        split_stats["orphan_labels"] = [str(p) for p in labels if p.stem not in images_by_stem]
        split_stats["missing_images"] = list(split_stats["orphan_labels"])
        for image_path in image_files:
            label_matches = labels_by_stem.get(image_path.stem, [])
            if not label_matches:
                continue
            label_path = label_matches[0]
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            split_stats["images"] += 1
            if img is None:
                result["issues"].append({"split": split, "image": str(image_path), "issue": "unreadable-image"})
                result["images"].append({"split": split, "image": str(image_path), "label": str(label_path), "read_error": True})
                continue
            h, w = img.shape[:2]
            instances, problems, raw_rows = read_label(label_path, w, h)
            for problem in problems:
                split_stats["invalid_polygon_rows"] += 1
                result["issues"].append({"split": split, "image": str(image_path), "label": str(label_path), "issue": problem})
            if not instances:
                split_stats["negative_images"] += 1
            image_classes = Counter(x["class"] for x in instances)
            for cls, count in image_classes.items():
                split_stats["instances_by_class"][cls] += count
                split_stats["images_by_class"][cls] += 1
            split_stats["instances"] += len(instances)
            pairs = pairwise_geometry(instances)
            if pairs:
                split_stats["overlapping_image_count"] += 1
                split_stats["overlap_pairs"] += len(pairs)
            duplicate_pairs = []
            for pair in pairs:
                if pair["class_a"] == pair["class_b"] and pair["iou"] >= 0.98:
                    duplicate_pairs.append(pair)
            split_stats["duplicate_annotation_pairs"] += len(duplicate_pairs)
            split_stats["empty_masks"] += sum(x["empty_mask"] for x in instances)
            for inst in instances:
                result["class_area_ratio"][inst["class"]].append(inst["area_ratio"])
            if len(instances) > 10:
                split_stats["suspicious_many_instances"].append({"image": str(image_path), "count": len(instances)})
            image_sha, label_sha = sha256(image_path), sha256(label_path)
            image_hash_splits[image_sha].append({"split": split, "path": str(image_path)})
            phashes.append((phash64(image_path), split, str(image_path)))
            category = classify_image(instances, pairs)
            groups = classify_image_groups(instances, pairs)
            result["images"].append({
                "split": split, "image": str(image_path), "label": str(label_path),
                "width": w, "height": h, "image_sha256": image_sha,
                "label_sha256": label_sha, "instances": [
                    {k: v for k, v in inst.items() if k != "mask"} for inst in instances
                ], "instance_count": len(instances), "class_counts": dict(image_classes),
                "negative": not instances, "group": category, "groups": groups, "overlap_pairs": pairs,
                "duplicate_pairs": duplicate_pairs,
                "large_mask_candidates": [i + 1 for i, x in enumerate(instances) if x["area_ratio"] >= 0.5],
                "small_mask_candidates": [i + 1 for i, x in enumerate(instances) if 0 < x["area_ratio"] < 0.0001],
                "raw_label_rows": raw_rows,
            })
        # Label files with no image are explicitly retained as integrity findings.
        result["splits"][split] = split_stats
    for image_sha, refs in image_hash_splits.items():
        if len({x["split"] for x in refs}) > 1:
            result["leakage"].setdefault("exact_image_hash_cross_split", []).append({"sha256": image_sha, "references": refs})
    valid_hashes = [(h, s, p) for h, s, p in phashes if h is not None]
    near = []
    for i, (ha, sa, pa) in enumerate(valid_hashes):
        for hb, sb, pb in valid_hashes[i + 1:]:
            if sa == sb:
                continue
            dist = (ha ^ hb).bit_count()
            if dist <= 4:
                near.append({"split_a": sa, "image_a": pa, "split_b": sb, "image_b": pb,
                             "phash_hamming": dist})
    result["leakage"]["near_duplicate_image_candidates_phash_le4"] = near
    for cls, values in result["class_area_ratio"].items():
        arr = np.asarray(values, dtype=float)
        result["class_area_ratio"][cls] = {
            "count": int(arr.size), "p01": float(np.quantile(arr, .01)),
            "p05": float(np.quantile(arr, .05)), "median": float(np.median(arr)),
            "p95": float(np.quantile(arr, .95)), "p99": float(np.quantile(arr, .99)),
        }
    return result


def classify_image(instances, overlaps):
    if not instances:
        return "H_negative"
    if not overlaps and len(instances) == 1:
        return "A_single_visible_fruit"
    if any(x["overlap_of_smaller"] >= .5 for x in overlaps):
        return "G_severe_occlusion_candidate"
    classes = {x["class"] for x in instances}
    if len(classes) > 1 and any(x["class_a"] != x["class_b"] for x in overlaps):
        return "F_mixed_fruit_overlap"
    if classes == {"banana"} or sum(x["class"] == "banana" for x in instances) >= 3:
        return "C_partially_overlapping_bananas"
    if classes == {"apple"} and overlaps:
        return "D_partially_overlapping_apples"
    if classes == {"orange"} and overlaps:
        return "E_partially_overlapping_oranges"
    if overlaps:
        return "F_mixed_fruit_overlap"
    return "B_multiple_no_overlap"


def classify_image_groups(instances, overlaps):
    """Return non-exclusive review groups so a dense banana image can be B and C."""
    if not instances:
        return ["H_negative"]
    groups = []
    if len(instances) == 1 and not overlaps:
        groups.append("A_single_visible_fruit")
    if len(instances) > 1 and not overlaps:
        groups.append("B_multiple_no_overlap")
    if (sum(x["class"] == "banana" for x in instances) >= 3 or
            any(p["class_a"] == p["class_b"] == "banana" for p in overlaps)):
        groups.append("C_partially_overlapping_bananas")
    if any(p["class_a"] == p["class_b"] == "apple" for p in overlaps):
        groups.append("D_partially_overlapping_apples")
    if any(p["class_a"] == p["class_b"] == "orange" for p in overlaps):
        groups.append("E_partially_overlapping_oranges")
    if any(p["class_a"] != p["class_b"] for p in overlaps):
        groups.append("F_mixed_fruit_overlap")
    if any(p["overlap_of_smaller"] >= .5 for p in overlaps):
        groups.append("G_severe_occlusion_candidate")
    return groups


def load_prediction_deltas(path):
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text())
    result = {}
    for model in data.get("models", []):
        if Path(model.get("checkpoint", "")).name != "best.pt":
            continue
        for split, record in model.get("splits", {}).items():
            for item in record.get("images", []):
                result[(split, item["image"])] = abs(int(item["gt_count"]) - int(item["pred_count"]))
    return result


def select_review_images(audit, benchmark_path, limit=180):
    deltas = load_prediction_deltas(benchmark_path)
    images = audit["images"]
    def ranking(item):
        cls = item["class_counts"]
        banana_n = cls.get("banana", 0)
        return (banana_n >= 3, banana_n, len(item["overlap_pairs"]),
                len(item["large_mask_candidates"]), deltas.get((item["split"], Path(item["image"]).name), 0),
                item["instance_count"], item["image"])
    selections, reasons = {}, {}
    quotas = [
        ("H_negative", 25, "negative image consistency"),
        ("C_partially_overlapping_bananas", 45, "overlapping banana instance boundaries"),
        ("G_severe_occlusion_candidate", 25, "high geometric overlap; verify visible boundaries"),
        ("D_partially_overlapping_apples", 12, "representative apple overlap"),
        ("E_partially_overlapping_oranges", 12, "representative orange overlap"),
        ("F_mixed_fruit_overlap", 18, "mixed-class overlap"),
        ("B_multiple_no_overlap", 12, "multiple fruit baseline without polygon overlap"),
    ]
    for group, quota, reason in quotas:
        candidates = sorted((x for x in images if x["group"] == group), key=ranking, reverse=True)
        for item in candidates:
            key = (item["split"], item["image"])
            if key in selections:
                continue
            selections[key], reasons[key] = item, reason
            if sum(1 for x in selections.values() if x["group"] == group) >= quota:
                break
    # Add high prediction/annotation count disagreement candidates from Phase 0.5.
    for item in sorted(images, key=lambda x: deltas.get((x["split"], Path(x["image"]).name), 0), reverse=True):
        key = (item["split"], item["image"])
        delta = deltas.get((item["split"], Path(item["image"]).name), 0)
        if delta >= 2 and key not in selections and len(selections) < limit:
            selections[key], reasons[key] = item, f"prediction/annotation count difference {delta} (benchmark; verify manually)"
    # Include large-mask candidates even when their overlap category quota was filled.
    for item in sorted((x for x in images if x["large_mask_candidates"]), key=ranking, reverse=True):
        key = (item["split"], item["image"])
        if key not in selections and len(selections) < limit:
            selections[key], reasons[key] = item, "contains mask covering at least 50% of image; geometric flag only"
    # Enforce the requested maximum deterministically while preserving category balance order.
    ordered = list(selections.items())[:limit]
    return [(item, reasons[key], deltas.get((item["split"], Path(item["image"]).name))) for key, item in ordered]


def make_contact_sheets(selected, output_dir, per_sheet=4):
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    cards = []
    for item, reason, _delta in selected:
        rgb = cv2.cvtColor(cv2.imread(item["image"]), cv2.COLOR_BGR2RGB)
        original = Image.fromarray(rgb)
        overlay = original.copy().convert("RGBA")
        layer = Image.new("RGBA", original.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for idx, inst in enumerate(item["instances"], 1):
            points = [(float(x) * (original.width - 1), float(y) * (original.height - 1)) for x, y in inst["polygon"]]
            color = COLORS[inst["class"]]
            draw.polygon(points, fill=(*color, 72), outline=(*color, 245), width=max(2, original.width // 500))
            cx = sum(x for x, _ in points) / len(points)
            cy = sum(y for _, y in points) / len(points)
            draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=(0, 0, 0, 220))
            draw.text((cx - 4, cy - 6), str(idx), font=font, fill=(255, 255, 255, 255))
        overlay = Image.alpha_composite(overlay, layer).convert("RGB")
        cards.append((item, reason, original, overlay))
    card_w, panel_h, header_h = 640, 250, 80
    card_h = header_h + panel_h
    for offset in range(0, len(cards), per_sheet):
        chunk = cards[offset:offset + per_sheet]
        cols = 2
        rows = math.ceil(len(chunk) / cols)
        sheet = Image.new("RGB", (cols * card_w, rows * card_h), "white")
        sd = ImageDraw.Draw(sheet)
        for j, (item, reason, original, overlay) in enumerate(chunk):
            x, y = (j % cols) * card_w, (j // cols) * card_h
            name = f"{item['split']} | {Path(item['image']).name} | n={item['instance_count']} | {item['group']}"
            sd.text((x + 6, y + 3), name[:110], font=font, fill="black")
            sd.text((x + 6, y + 18), reason[:110], font=font, fill="black")
            id_classes = ", ".join(f"{i}:{inst['class']}" for i, inst in enumerate(item["instances"], 1)) or "no instances"
            mapping = f"Instance ID → class: {id_classes}"
            for line_idx in range(3):
                chunk = mapping[line_idx * 80:(line_idx + 1) * 80]
                if chunk:
                    sd.text((x + 6, y + 33 + line_idx * 12), chunk, font=font, fill="black")
            for k, im in enumerate((original, overlay)):
                im.thumbnail((card_w // 2 - 6, panel_h - 4))
                panel_x = x + k * (card_w // 2)
                px = panel_x + (card_w // 2 - im.width) // 2
                py = y + header_h + (panel_h - im.height) // 2
                sheet.paste(im, (px, py))
                sd.text((panel_x + 6, y + header_h + 2), "Original" if k == 0 else "GT instance overlay", font=font, fill="black")
        sheet.save(output_dir / f"contact_sheet_{offset // per_sheet + 1:03d}.jpg", quality=90)


def write_outputs(audit, data_root, output_dataset, benchmark_path):
    audit["review_groups"] = defaultdict(lambda: {"images": 0, "instances": 0, "class_distribution": Counter(),
                                                     "overlap_pairs": 0, "large_mask_images": 0,
                                                     "potential_duplicate_candidates": 0})
    for item in audit["images"]:
        for group_name in item["groups"]:
            group = audit["review_groups"][group_name]
            group["images"] += 1
            group["instances"] += item["instance_count"]
            group["overlap_pairs"] += len(item["overlap_pairs"])
            group["large_mask_images"] += int(bool(item["large_mask_candidates"]))
            group["potential_duplicate_candidates"] += len(item["duplicate_pairs"])
            group["class_distribution"].update(item["class_counts"])
    audit["review_groups"] = {
        name: {**dict(values), "class_distribution": dict(values["class_distribution"])}
        for name, values in audit["review_groups"].items()
    }
    # Split-level mask-area thresholds and image flags, based on measured class distribution.
    for split, stats in audit["splits"].items():
        candidates = [x for x in audit["images"] if x["split"] == split]
        stats["duplicate_annotation_pairs"] = int(stats["duplicate_annotation_pairs"])
        stats["suspiciously_many_instances_threshold"] = 10
        stats["suspicious_many_instance_image_count"] = len(stats["suspicious_many_instances"])
        ratios = [inst["area_ratio"] for x in candidates for inst in x["instances"]]
        stats["mask_area_ratio_p01"] = float(np.quantile(ratios, .01)) if ratios else None
        stats["mask_area_ratio_p99"] = float(np.quantile(ratios, .99)) if ratios else None
        stats["banana_cluster_images_3plus"] = sum(x["class_counts"].get("banana", 0) >= 3 for x in candidates)
        stats["mask_overlap_image_count"] = sum(any(p["mask_overlap"] for p in x["overlap_pairs"]) for x in candidates)
        for x in candidates:
            stats["unusually_small_masks"] += len(x["small_mask_candidates"])
            stats["unusually_large_masks"] += len(x["large_mask_candidates"])
        for key in ("instances_by_class", "images_by_class"):
            stats[key] = dict(stats[key])
    totals = {"images": len(audit["images"]), "instances": sum(x["instance_count"] for x in audit["images"]),
              "negative_images": sum(x["negative"] for x in audit["images"]),
              "instances_by_class": dict(Counter(i["class"] for x in audit["images"] for i in x["instances"]))}
    audit["totals"] = totals
    selected = select_review_images(audit, benchmark_path)
    review_root = output_dataset / "review_subset"
    selected_keys = {(item["split"], item["image"]) for item, _reason, _delta in selected}
    # Preserve any previously generated review candidates on reruns. Never drop
    # a file from this working version just because ranking/quotas changed.
    source_by_split_name = {(x["split"], Path(x["image"]).name): x for x in audit["images"]}
    for split in ("train", "val", "test"):
        for existing in sorted((review_root / split / "images").glob("*")):
            item = source_by_split_name.get((split, existing.name))
            if item and (item["split"], item["image"]) not in selected_keys:
                selected.append((item, "prior deterministic candidate retained; review still pending", None))
                selected_keys.add((item["split"], item["image"]))
    rows = []
    version_rows = []
    for item, reason, prediction_delta in selected:
        image_src, label_src = Path(item["image"]), Path(item["label"])
        split = item["split"]
        image_dst = review_root / split / "images" / image_src.name
        label_dst = review_root / split / "labels" / label_src.name
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        if not image_dst.exists():
            shutil.copy2(image_src, image_dst)
        if not label_dst.exists():
            shutil.copy2(label_src, label_dst)
        image_rel = str(image_dst.relative_to(output_dataset))
        label_rel = str(label_dst.relative_to(output_dataset))
        for idx, inst in enumerate(item["instances"], 1):
            rows.append({"image_path": image_rel, "source_image_path": str(image_src), "split": split,
                         "class": inst["class"], "instance_id": idx, "review_reason": reason,
                         "review_status": "pending", "reviewer_notes": ""})
        if not item["instances"]:
            rows.append({"image_path": image_rel, "source_image_path": str(image_src), "split": split,
                         "class": "", "instance_id": "", "review_reason": reason,
                         "review_status": "pending", "reviewer_notes": ""})
        version_rows.append({"split": split, "image_path": image_rel, "label_path": label_rel,
                             "source_image_path": str(image_src), "source_label_path": str(label_src),
                             "image_sha256": item["image_sha256"], "label_sha256": item["label_sha256"],
                             "annotation_version": "source-sam-pseudolabel-unreviewed",
                             "review_status": "pending", "review_group": item["group"],
                             "review_reason": reason, "benchmark_count_delta": prediction_delta})
    output_dataset.mkdir(parents=True, exist_ok=True)
    source_manifest_items = []
    for item in audit["images"]:
        is_selected = any(item is selected_item for selected_item, _reason, _delta in selected)
        source_manifest_items.append({
            "split": item["split"], "source_image_path": str(Path(item["image"]).relative_to(data_root)),
            "source_label_path": str(Path(item["label"]).relative_to(data_root)),
            "image_sha256": item["image_sha256"], "label_sha256": item["label_sha256"],
            "annotation_version": "source-sam-pseudolabel-unreviewed",
            "review_status": "pending" if is_selected else "not-selected",
            "review_group": item["group"],
        })
    (output_dataset / "dataset_manifest.json").write_text(json.dumps({
        "dataset_version": "seg_data_review_v1", "source_dataset": str(data_root.resolve()),
        "annotation_version": "source-sam-pseudolabel-unreviewed", "review_status": "pending",
        "review_subset_only": True, "full_dataset_copied": False, "classes": CLASSES,
        "source_splits": {k: {"images": v["images"], "instances": v["instances"]} for k, v in audit["splits"].items()},
        "source_image_label_hash_manifest": source_manifest_items,
        "review_image_count": len(version_rows), "review_items": version_rows,
    }, indent=2) + "\n")
    (output_dataset / "review_subset" / "seg.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: banana\n  1: apple\n  2: orange\n"
    )
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: banana\n  1: apple\n  2: orange\n"
    )
    archive_path = output_dataset / "cvat_import_review_v1.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(review_root.rglob("*")):
            if file.is_file() and file.name != archive_path.name:
                archive.write(file, file.relative_to(review_root))
    with open(ROOT / "work/audit/phase1_review_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "source_image_path", "split", "class", "instance_id", "review_reason", "review_status", "reviewer_notes"])
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheets(selected, ROOT / "work/audit/phase1_contact_sheets")
    audit["review_subset"] = {"selected_images": len(selected), "manifest_csv": "work/audit/phase1_review_manifest.csv",
                              "dataset_version_dir": str(output_dataset),
                              "cvat_import_archive": str(archive_path),
                              "contact_sheets_dir": "work/audit/phase1_contact_sheets",
                              "selection_policy": "deterministic group quotas; geometric and benchmark flags are review leads only"}
    # Keep JSON serializable, and drop large polygon vertex arrays only from any arrays? Preserve complete audit geometry.
    target = ROOT / "work/audit/phase1_dataset_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audit": str(target), "review_images": len(selected),
                      "review_instances": len(rows), "output_dataset": str(output_dataset),
                      "contact_sheets": len(list((ROOT / 'work/audit/phase1_contact_sheets').glob('*.jpg'))),
                      "totals": totals}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dataset", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "work/audit/phase05_benchmark_raw_masks.json")
    args = parser.parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    audit = audit_dataset(args.dataset)
    write_outputs(audit, args.dataset, args.output_dataset, args.benchmark)


if __name__ == "__main__":
    main()
