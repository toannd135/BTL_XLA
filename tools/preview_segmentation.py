"""Save segmentation overlays for visual review; this is not an accuracy benchmark."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import SAM, YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from detection import configure_sam_refiner, compute_mask_metrics, detect, detect_seg, split_fruit_cluster_detections
from quality import assess_detection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "yolo11s-seg.pt")
    parser.add_argument("--obb", type=Path, default=ROOT / "exports/det_obb_s.pt")
    parser.add_argument("--sam", type=Path, default=ROOT / "work/weights/sam2.1_t.pt")
    parser.add_argument("--hbb", type=Path, default=ROOT / "exports/det_B_clean_backup.pt")
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--output", type=Path, default=ROOT / "out/seg_boundary_review")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.image_dir:
        args.images.extend(sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}))
    if not args.images:
        parser.error("Provide image paths or --image-dir")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    for path in (args.model, args.obb, args.hbb, *args.images):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    if not args.no_sam and not args.sam.is_file():
        parser.error(f"Missing SAM checkpoint: {args.sam}; use --no-sam for detector-only review")
    detector, obb, hbb = YOLO(str(args.model)), YOLO(str(args.obb)), YOLO(str(args.hbb))
    if detector.task != "segment":
        parser.error("--model must be a segmentation checkpoint")
    sam = None if args.no_sam else configure_sam_refiner(SAM(str(args.sam)))
    args.output.mkdir(parents=True, exist_ok=True)
    palette = [(235, 99, 37), (234, 51, 147), (12, 65, 194), (140, 127, 8), (119, 39, 219)]
    rows = []
    for path in args.images:
        print(f"Checking {path.name}...", flush=True)
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        height, width = image.shape[:2]
        diagnostics = {}
        start = time.perf_counter()
        polygons, boxes, scores, classes, _ = detect_seg(
            detector, image, conf=args.conf, imgsz=args.imgsz, iou=0.7, compute_metrics=False)
        hbb_detections = detect(hbb,image,conf=args.conf,imgsz=640,iou=.7)
        polygons, boxes, scores, classes = split_fruit_cluster_detections(
            polygons, boxes, scores, classes, obb, image, conf=max(.3, args.conf),
            det_names=detector.names, mask_model=sam, diagnostics=diagnostics,
            hbb_detections=hbb_detections,hbb_names=hbb.names,seg_model=detector)
        metrics = compute_mask_metrics(polygons, boxes, width, height)
        elapsed = (time.perf_counter()-start)*1000
        overlay, objects = image.copy(), []
        for i, (polygon, box, score, cls, metric) in enumerate(zip(polygons, boxes, scores, classes, metrics)):
            item = {"label": detector.names[int(cls)], "confidence": float(score), "bbox": box.tolist(),
                    "polygon": polygon.tolist(), **metric, "mask_overlaps": metric["overlaps"]}
            objects.append(item)
            color = palette[i % len(palette)]
            if len(polygon) >= 3:
                cv2.polylines(overlay, [np.round(polygon).astype(np.int32)], True, color, 2, cv2.LINE_AA)
            else:
                x1, y1, x2, y2 = np.round(box).astype(int)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            x, y = np.round(box[:2]).astype(int)
            cv2.putText(overlay, f"#{i+1} {item['label']} {score:.2f}", (int(x), max(16, int(y))),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(args.output / f"{path.stem}_overlay.jpg"), overlay)
        quality = assess_detection(objects, width, height, diagnostics, conf_threshold=args.conf)
        rows.append({"image": str(path), "counts": dict(Counter(item["label"] for item in objects)),
                     "image_width": width, "image_height": height,
                     "inference_ms_including_first_use": round(elapsed, 1),
                     "diagnostics": diagnostics, "quality": quality, "objects": objects})
        print(path.name, rows[-1]["counts"], diagnostics, flush=True)
    checkpoints = {"seg": args.model, "obb": args.obb, "hbb": args.hbb}
    if sam is not None:
        checkpoints["sam"] = args.sam
    report = {"pipeline": "YOLO-seg + OBB/HBB proposals + crop verification" + (" + SAM" if sam else ""),
              "note": "Visual smoke check only. No reviewed ground truth or accuracy claim; no CLIP/freshness/enhancement.",
              "device": "cuda" if torch.cuda.is_available() else "cpu", "imgsz": args.imgsz,
              "checkpoints": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for key, path in checkpoints.items()}, "images": rows}
    (args.output / "preview.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved overlays and report to {args.output}")


if __name__ == "__main__":
    main()
