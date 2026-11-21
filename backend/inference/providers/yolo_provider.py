"""Adapter for the existing Ultralytics segmentation detector."""

import numpy as np

from detection import detect_seg
from ..base import ProviderResult
from ..normalizer import normalize_instances


class YoloSegProvider:
    name = "local_yolo"

    def __init__(self, model, *, conf=0.4, imgsz=640, iou=0.6,
                 quantize=None, device=None, half=None):
        if getattr(model, "task", None) != "segment":
            raise ValueError("local_yolo provider requires a segmentation checkpoint")
        names = {str(name).lower() for name in model.names.values()}
        if not {"banana", "apple", "orange"}.issubset(names):
            raise ValueError(f"local_yolo checkpoint lacks required fruit classes: {names}")
        self.model = model
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.iou = float(iou)
        self.quantize = quantize
        self.device = device
        self.half = half

    def predict(self, image: np.ndarray) -> ProviderResult:
        polygons, boxes, scores, classes, _ = detect_seg(
            self.model, image, conf=self.conf, imgsz=self.imgsz, iou=self.iou,
            quantize=self.quantize, device=self.device, half=self.half,
            compute_metrics=False,
        )
        instances = normalize_instances(polygons, boxes, scores, classes, self.model.names, self.name)
        height, width = image.shape[:2]
        return ProviderResult(instances, width, height, self.name)

