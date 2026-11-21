"""Regression check for fixed-square YOLO inference on the five smoke images.

This reads the existing v6 checkpoint and candidate images only. It does not
write checkpoints, modify backend code, or touch the source dataset.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import unittest
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[5]
BATCH = ROOT / "fruit/work/fruit_type_cls_experiment/phase9_green_apple_rebuild/candidate_batches/green_apple_smoke_001"
CHECKPOINT = ROOT / "fruit/runs/seg/seg_green_v6_smoke_cpu_20261121/weights/best.pt"


def _paths():
    with (BATCH / "candidate_manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, [str(BATCH / row["image_path"]) for row in rows]


def _polygon_mask(poly, shape):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    if len(poly) >= 3:
        cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 1)
    return mask.astype(bool)


class InferenceConsistencyTest(unittest.TestCase):
 def test_fixed_square_padding_makes_batch_and_single_outputs_consistent(self):
    if not CHECKPOINT.is_file():
        self.skipTest(f"forensic checkpoint unavailable: {CHECKPOINT}")
    rows, paths = _paths()
    model = YOLO(str(CHECKPOINT))
    assert model.task == "segment"
    assert model.names == {0: "green_apple"}

    common = dict(
        imgsz=640,
        device="cpu",
        conf=0.05,
        iou=0.70,
        max_det=300,
        retina_masks=False,
        rect=False,  # fixed square letterbox for mixed aspect ratios
        verbose=False,
        save=False,
    )
    batch_results = model.predict(source=paths, **common)
    single_results = [model.predict(source=path, **common)[0] for path in paths]
    assert len(batch_results) == len(single_results) == len(rows) == 5

    for batch_result, single_result in zip(batch_results, single_results):
        assert batch_result.orig_shape == single_result.orig_shape
        assert batch_result.boxes is not None and single_result.boxes is not None
        assert len(batch_result.boxes) == len(single_result.boxes)
        np.testing.assert_array_equal(
            batch_result.boxes.cls.cpu().numpy(), single_result.boxes.cls.cpu().numpy()
        )
        np.testing.assert_allclose(
            batch_result.boxes.conf.cpu().numpy(),
            single_result.boxes.conf.cpu().numpy(),
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            batch_result.boxes.xyxy.cpu().numpy(),
            single_result.boxes.xyxy.cpu().numpy(),
            rtol=1e-4,
            atol=1e-3,
        )
        assert batch_result.masks is not None and single_result.masks is not None
        assert len(batch_result.masks.xy) == len(single_result.masks.xy)
        for batch_poly, single_poly in zip(batch_result.masks.xy, single_result.masks.xy):
            batch_mask = _polygon_mask(batch_poly, batch_result.orig_shape)
            single_mask = _polygon_mask(single_poly, single_result.orig_shape)
            intersection = np.logical_and(batch_mask, single_mask).sum()
            union = np.logical_or(batch_mask, single_mask).sum()
            assert union == 0 or intersection / union >= 0.995
            h, w = batch_result.orig_shape
            assert np.isfinite(batch_poly).all()
            assert (batch_poly[:, 0] >= 0).all() and (batch_poly[:, 0] <= w).all()
            assert (batch_poly[:, 1] >= 0).all() and (batch_poly[:, 1] <= h).all()
