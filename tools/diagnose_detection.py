"""Compare the original C and candidate on one image, without changing app settings."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[640, 960])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")
    output = args.output or repo / "out" / "diagnostics" / image_path.stem
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str((repo / "out/ultralytics_config").resolve()))

    import cv2
    import numpy as np
    from ultralytics import YOLO

    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Cannot decode image: {image_path}")
    thresholds = (0.10, 0.25, 0.40, 0.50, 0.75)
    report = {"image": str(image_path), "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
              "shape": list(image.shape[:2]), "device": "cpu", "iou": 0.6,
              "agnostic_nms": True, "prediction_floor": 0.05, "results": []}
    for tag, name in (("C_original", "det_finetune_best.pt"), ("C_candidate", "det_C_candidate.pt")):
        model_path = repo / "exports" / name
        detector = YOLO(str(model_path))
        for size in args.sizes:
            prediction = detector.predict(image, imgsz=size, conf=0.05, iou=0.6, agnostic_nms=True,
                                          device="cpu", quantize=32, verbose=False)[0]
            objects = [{"label": detector.names[int(c)], "confidence": float(p), "bbox": b}
                       for c, p, b in zip(prediction.boxes.cls.tolist(), prediction.boxes.conf.tolist(),
                                          prediction.boxes.xyxy.tolist())]
            objects.sort(key=lambda item: -item["confidence"])
            row = {"model": tag, "imgsz": size,
                   "counts": {f"{threshold:.2f}": sum(o["confidence"] >= threshold for o in objects)
                              for threshold in thresholds},
                   "highest_confidence": objects[0]["confidence"] if objects else None,
                   "objects_at_floor": objects}
            report["results"].append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "objects_at_floor"}))
            for threshold in (0.10, 0.25, 0.40):
                canvas = image.copy()
                for obj in objects:
                    if obj["confidence"] < threshold:
                        continue
                    x1, y1, x2, y2 = map(round, obj["bbox"])
                    color = (0, 200, 0)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(canvas, f'{obj["label"]} {obj["confidence"]:.2f}',
                                (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                target = output / f"{tag}_{size}_conf{int(threshold * 100):02d}.jpg"
                ok, encoded = cv2.imencode(".jpg", canvas)
                if not ok:
                    raise RuntimeError(f"Cannot encode diagnostic image: {target}")
                encoded.tofile(target)
        del detector
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved diagnostics:", str(output))


if __name__ == "__main__":
    main()
