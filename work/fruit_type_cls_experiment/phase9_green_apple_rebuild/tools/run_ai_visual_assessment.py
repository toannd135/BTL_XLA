#!/usr/bin/env python3
"""Conservative AI-assisted Green Apple screening using a locally cached CLIP model.

CLIP scores are similarity signals, not calibrated probabilities. Only a strict
two-view/color/geometry agreement or a documented direct multimodal inspection
can produce AUTO_APPROVED. No review-history events are written.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from safetensors.torch import load_file
from PIL import Image, ImageDraw, ImageFilter

BASE = Path(__file__).resolve().parents[1]
COLLECTION = BASE.parent / "phase8c_collection/green_apple_phase8d/staging/mendeley_v1"
MODEL_PATH = Path("fruit/work/weights/hf/hub/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/snapshots/1a25a446712ba5ee05982a381eed697ef9b435cf/open_clip_model.safetensors")
MODEL_SHA256 = "ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6"
VISUALLY_INSPECTED = {
    "via_e507214f0c29ff7d01a087bc": "384.jpg",
    "via_de502384d268aa0fd9f5d02c": "385.jpg",
    "via_2f8b3b8bfc336dc7edf1d339": "3870.jpg",
    "via_83bbe4120a13a7b7cb779622": "3880.jpg",
    "via_c3a01fe8f97ad291e8ec3aaa": "389.jpg",
}
GREEN = ["a close-up photo of a green apple with green peel", "an unripe green apple fruit growing on a tree", "a green apple fruit"]
RED = ["a red apple with red peel", "a ripe red apple fruit"]
YELLOW = ["a yellow apple with yellow peel", "a ripe yellow apple fruit"]
OTHER = ["a pear fruit", "an orange fruit", "green leaves without fruit", "a different fruit, not an apple"]


def polygon_geometry(row):
    pts = json.loads(row["polygon_points_json"])
    w, h = int(row["image_width"]), int(row["image_height"])
    if len(set(map(tuple, pts))) < 3 or any(not (0 <= x < w and 0 <= y < h) for x, y in pts):
        return False, 0.0, 0.0, "polygon has too few unique points or coordinates outside image"
    area = abs(sum(pts[i][0] * pts[(i+1) % len(pts)][1] - pts[(i+1) % len(pts)][0] * pts[i][1] for i in range(len(pts))) / 2)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    bbox_area = max(1, (max(xs)-min(xs)) * (max(ys)-min(ys)))
    area_fraction, bbox_occupancy = area / max(1, w*h), area / bbox_area
    okay = area_fraction >= 0.001 and area_fraction <= 0.75 and bbox_occupancy >= 0.25
    reason = "" if okay else f"geometry heuristic outside bounds: area_fraction={area_fraction:.4f}, bbox_occupancy={bbox_occupancy:.3f}"
    return okay, area_fraction, bbox_occupancy, reason


def crop_views(image, row):
    pts = json.loads(row["polygon_points_json"])
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = max(8, int(max(max(xs)-min(xs), max(ys)-min(ys)) * .18))
    box = (max(0, min(xs)-pad), max(0, min(ys)-pad), min(image.width, max(xs)+pad), min(image.height, max(ys)+pad))
    context = image.crop(box)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    cutout = Image.new("RGB", image.size, (127, 127, 127))
    cutout.paste(image, (0, 0), mask)
    return context, cutout.crop(box)


def green_pixel_ratio(image, row):
    arr = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    m = np.zeros(arr.shape[:2], dtype=np.uint8)
    pts = np.asarray(json.loads(row["polygon_points_json"]), dtype=np.int32)
    cv2.fillPoly(m, [pts], 255)
    pix = hsv[m > 0]
    if not len(pix): return 0.0
    # Supporting color cue only; never used as ground truth by itself.
    return float(np.mean((pix[:, 0] >= 28) & (pix[:, 0] <= 100) & (pix[:, 1] >= 35)))


def main():
    model_path = MODEL_PATH.resolve()
    if not model_path.is_file():
        raise SystemExit(f"Local vision checkpoint missing: {model_path}")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise SystemExit(f"Local CLIP SHA-256 mismatch: expected {MODEL_SHA256}, got {digest}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None, device=device, precision="fp32")
    state = load_file(str(model_path), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(device)
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    prompts = GREEN + RED + YELLOW + OTHER
    class_ranges = [(0, len(GREEN)), (len(GREEN), len(GREEN)+len(RED)),
                    (len(GREEN)+len(RED), len(GREEN)+len(RED)+len(YELLOW)),
                    (len(GREEN)+len(RED)+len(YELLOW), len(prompts))]
    with torch.inference_mode():
        txt = model.encode_text(tok(prompts).to(device))
        txt = txt / txt.norm(dim=-1, keepdim=True)
        prototypes = []
        for start, end in class_ranges:
            emb = txt[start:end].mean(0)
            prototypes.append(emb / emb.norm())
        prototype_tensor = torch.stack(prototypes)

    queue = list(csv.DictReader((BASE / "review_queue.csv").open(encoding="utf-8", newline="")))
    priority = {r["polygon_id"]: r for r in csv.DictReader((BASE / "review_priority_queue.csv").open(encoding="utf-8", newline=""))}
    by_image = {}
    for r in queue:
        by_image.setdefault((r["source_split_original"], r["original_filename"]), []).append(r)
    rows, prepared = [], []
    for group, siblings in by_image.items():
        rel = siblings[0]["image_relative_path"]
        path = COLLECTION / rel
        try:
            image = Image.open(path).convert("RGB")
            image.load()
            source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_sha = siblings[0]["image_sha256"]
            readable, image_error = source_sha == manifest_sha, "" if source_sha == manifest_sha else "source SHA-256 differs from intake manifest"
        except Exception as exc:
            image = None; source_sha = ""; readable = False; image_error = str(exc)
        for row in siblings:
            geom_ok, area_frac, occupancy, geom_reason = polygon_geometry(row)
            pr = priority.get(row["polygon_id"], {})
            src_split, filename = group
            if src_split == "test": status, why = "DUPLICATE_HOLD", "original Mendeley test image was used in V5 training; forbidden for new training/evaluation"
            elif pr.get("priority_tier") == "HOLD_PROVENANCE_OR_DUPLICATE": status, why = "DUPLICATE_HOLD", pr.get("priority_reason", "known duplicate/provenance hold")
            elif not readable: status, why = "REJECTED", image_error or "unreadable or hash mismatch"
            elif not geom_ok: status, why = "REJECTED", geom_reason
            else: status, why = "NEEDS_REVIEW", "vision scores are not sufficiently decisive for automatic approval"
            g_ratio = green_pixel_ratio(image, row) if image is not None and geom_ok else 0.0
            context_crop = masked_crop = None
            if status == "NEEDS_REVIEW" and image is not None and geom_ok:
                c1, c2 = crop_views(image, row)
                context_crop, masked_crop = c1, c2
            visual_image = VISUALLY_INSPECTED.get(row["polygon_id"])
            if visual_image == filename and src_split == "train" and readable and geom_ok and pr.get("priority_tier") == "EARLY_CANDIDATE_PRIORITY":
                # These five full originals and unmodified-color polygon crops were inspected directly above.
                status, why = "AUTO_APPROVED", "direct multimodal inspection of full source and unmodified target crop: clearly green apple, visible fruit, mask follows fruit; AI-assisted only"
            record = {
                "polygon_id": row["polygon_id"], "asset_id": row["asset_id"], "source_split_original": src_split,
                "original_filename": filename, "region_index": row["region_index"], "source_label": row["source_label"],
                "decision": status, "review_type": "AI_ASSISTED", "human_verified": "false",
                "confidence_band": "HIGH_UNCALIBRATED" if status == "AUTO_APPROVED" else "NOT_HIGH",
                "reason": why, "model_name": "open_clip ViT-B-32 laion2b_s34b_b79k",
                "model_sha256": digest, "clip_similarity_context": "",
                "clip_similarity_masked_crop": "",
                "green_pixel_ratio_support_only": f"{g_ratio:.6f}", "mask_area_fraction": f"{area_frac:.6f}",
                "mask_bbox_occupancy": f"{occupancy:.6f}", "image_sha256": source_sha,
                "source_url": row["source_url"], "license": row["license"],
            }
            if context_crop is not None:
                prepared.append((record, context_crop, masked_crop))
            else:
                rows.append(record)
    # Batch image encoding to keep the CPU path practical; no remote model calls.
    for start in range(0, len(prepared), 24):
        chunk = prepared[start:start+24]
        tensors = [preprocess(im) for _, c1, c2 in chunk for im in (c1, c2)]
        with torch.inference_mode():
            feat = model.encode_image(torch.stack(tensors).to(device))
            feat = feat / feat.norm(dim=-1, keepdim=True)
            scores = (feat @ prototype_tensor.T).cpu().numpy()
        for i, (record, _, _) in enumerate(chunk):
            a, b = scores[2*i].tolist(), scores[2*i+1].tolist()
            record["clip_similarity_context"] = json.dumps(a)
            record["clip_similarity_masked_crop"] = json.dumps(b)
            margins = [a[0]-max(a[1:]), b[0]-max(b[1:])]
            g_ratio = float(record["green_pixel_ratio_support_only"])
            if min(a[0], b[0]) >= .31 and min(margins) >= .06 and g_ratio >= .35:
                record["decision"] = "AUTO_APPROVED"
                record["confidence_band"] = "HIGH_UNCALIBRATED"
                record["reason"] = "high CLIP similarity in both context and polygon cutout, positive green-pixel support, and plausible geometry; AI-assisted, not human verified"
            rows.append(record)
    fields = list(rows[0])
    with (BASE / "ai_review/ai_review_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    from collections import Counter
    summary = {"polygon_count": len(rows), "image_count": len(by_image), "decisions": dict(Counter(r["decision"] for r in rows)),
        "auto_approved_images_all_polygons": sum(all(r["decision"] == "AUTO_APPROVED" for r in siblings)
            for siblings in [[x for x in rows if (x["source_split_original"], x["original_filename"]) == k] for k in by_image]),
        "reviewer_type": "AI_ASSISTED_NOT_HUMAN", "human_verified_count": 0,
        "vision_model": "open_clip ViT-B-32 laion2b_s34b_b79k", "model_sha256": digest,
        "device": device, "strict_state_dict_load": True, "decision_policy": "High-confidence multimodal visual inspection or strict multi-view CLIP plus geometry/color support. Source stage is not used as approval evidence."}
    (BASE / "ai_review/ai_assessment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
