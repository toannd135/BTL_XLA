"""Compare hard/soft NMS on the same pre-NMS YOLO11 predictions, outside the app."""

import argparse
import json
import os
from pathlib import Path

import torch


def overlap(box, others):
    intersection = (torch.minimum(box[2:], others[:, 2:]) -
                    torch.maximum(box[:2], others[:, :2])).clamp(min=0).prod(1)
    area = (box[2:] - box[:2]).clamp(min=0).prod()
    areas = (others[:, 2:] - others[:, :2]).clamp(min=0).prod(1)
    return intersection / (area + areas - intersection).clamp(min=1e-9)


def soft_nms(detections, method="linear", iou_threshold=0.6, sigma=0.5, score_threshold=0.4, max_det=300):
    """Class-agnostic Soft-NMS; return xyxy/decayed-score/class with updated scores."""
    if method not in {"linear", "gaussian"} or sigma <= 0:
        raise ValueError("Expected linear/gaussian and positive sigma")
    remaining = detections.clone()
    remaining = remaining[remaining[:, 4] >= score_threshold]
    selected = []
    while len(remaining) and len(selected) < max_det:
        best = remaining[:, 4].argmax()
        chosen = remaining[best].clone()
        selected.append(chosen)
        keep = torch.arange(len(remaining), device=remaining.device) != best
        remaining = remaining[keep]
        if not len(remaining):
            break
        ious = overlap(chosen[:4], remaining[:, :4])
        if method == "linear":
            weights = torch.where(ious > iou_threshold, 1 - ious, torch.ones_like(ious))
        else:
            weights = torch.exp(-(ious * ious) / sigma)
        remaining[:, 4] *= weights
        remaining = remaining[remaining[:, 4] >= score_threshold]
    return torch.stack(selected) if selected else detections.new_zeros((0, 6))


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[640, 960])
    parser.add_argument("--model", type=Path, default=repo / "exports/det_C_candidate.pt")
    parser.add_argument("--output", type=Path, default=repo / "out/diagnostics/soft_nms")
    args = parser.parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str((repo / "out/ultralytics_config").resolve()))

    import cv2
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import ops
    from ultralytics.utils.nms import TorchNMS

    class CapturePredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            raw = preds[0] if isinstance(preds, (tuple, list)) else preds
            self.raw_prediction = raw.detach().float().cpu().clone()
            self.input_shape = tuple(img.shape[2:])
            return super().postprocess(preds, img, orig_imgs, **kwargs)

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    assert model.task == "detect"
    assert not getattr(model.model, "end2end", False), "This experiment expects standard YOLO11 outputs"
    rows = []
    for source in args.images:
        image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {source}")
        image_output = args.output / source.stem
        image_output.mkdir(parents=True, exist_ok=True)
        for size in args.sizes:
            normal = model.predict(image, predictor=CapturePredictor, device="cpu", quantize=32,
                                   imgsz=size, conf=0.05, iou=0.6, agnostic_nms=True,
                                   max_det=300, verbose=False)[0]
            raw = model.predictor.raw_prediction
            assert raw.ndim == 3 and raw.shape[1] == 4 + len(model.names)
            xywh = raw[0, :4].transpose(0, 1)
            scores, classes = raw[0, 4:].max(0)
            candidates = torch.cat((ops.xywh2xyxy(xywh), scores[:, None], classes[:, None].float()), 1)
            candidates = candidates[candidates[:, 4] > 0.05]
            # Prove the extracted raw boxes reproduce the app's standard NMS before trying alternatives.
            hard_indices = TorchNMS.nms(candidates[:, :4], candidates[:, 4], 0.6)[:300]
            assert len(hard_indices) == len(normal.boxes)
            assert torch.allclose(candidates[hard_indices, 4], normal.boxes.conf.cpu(), atol=1e-6)
            raw_info = {"image": source.name, "imgsz": size, "method": "pre_nms",
                        "count_005": len(candidates),
                        "count_025": int((candidates[:, 4] >= 0.25).sum()),
                        "count_040": int((candidates[:, 4] >= 0.4).sum()),
                        "highest_score": float(candidates[:, 4].max()) if len(candidates) else None}
            rows.append(raw_info)
            print(json.dumps(raw_info))
            for threshold in (0.25, 0.4):
                filtered = candidates[candidates[:, 4] >= threshold]
                methods = {}
                for iou in (0.45, 0.6, 0.8):
                    idx = TorchNMS.nms(filtered[:, :4], filtered[:, 4], iou)[:300]
                    methods[f"hard_iou{int(iou * 100)}"] = filtered[idx].clone()
                for method in ("linear", "gaussian"):
                    methods[f"soft_{method}"] = soft_nms(candidates, method=method, score_threshold=threshold)
                for method, selected in methods.items():
                    selected[:, :4] = ops.scale_boxes(model.predictor.input_shape, selected[:, :4], image.shape)
                    row = {"image": source.name, "imgsz": size, "method": method,
                           "threshold": threshold, "box_count": len(selected), "detections": selected.tolist()}
                    rows.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != "detections"}))
                    canvas = image.copy()
                    for x1, y1, x2, y2, score, cls in selected.tolist():
                        x1, y1, x2, y2 = map(round, (x1, y1, x2, y2))
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        cv2.putText(canvas, f"{model.names[int(cls)]} {score:.2f}",
                                    (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                    path = image_output / f"{size}_{method}_conf{int(threshold * 100)}.jpg"
                    ok, encoded = cv2.imencode(".jpg", canvas)
                    if not ok:
                        raise RuntimeError(f"Cannot save {path}")
                    encoded.tofile(path)
    report = {"model": str(args.model), "agnostic": True, "device": "cpu", "floor": 0.05,
              "soft_linear_iou": 0.6, "soft_gaussian_sigma": 0.5,
              "notes": "Counts are boxes, not verified fruit counts. Soft-NMS scores are decayed.", "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
