import unittest

import numpy as np

from inference.normalizer import normalize_instances
from inference.router import InferenceRouter
from postprocessing.counting import count_instances
from postprocessing.quality_gate import assess_instances


class NormalizerTests(unittest.TestCase):
    def test_instances_keep_class_score_geometry_and_source(self):
        polygon = np.asarray([[1, 2], [8, 2], [8, 9], [1, 9]], dtype=np.float32)
        result = normalize_instances(
            [polygon], [[1, 2, 8, 9]], [0.84], [0], {0: "Banana"},
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].instance_id, "local_yolo:1")
        self.assertEqual(result[0].class_name, "banana")
        self.assertEqual(result[0].bbox_xyxy, (1.0, 2.0, 8.0, 9.0))
        self.assertEqual(result[0].source, "local_yolo")
        np.testing.assert_array_equal(result[0].polygon, polygon)

    def test_misaligned_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_instances([], [[0, 0, 1, 1]], [], [0], {0: "banana"})

    def test_invalid_boxes_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_instances([], [[2, 0, 1, 1]], [0.5], [0], {0: "banana"})


class RouterTests(unittest.TestCase):
    def test_unknown_provider_fails_instead_of_silently_falling_back(self):
        with self.assertRaises(ValueError):
            InferenceRouter({"local_yolo": object()}, default_provider="roboflow")

    def test_registered_provider_is_selected(self):
        class Provider:
            def predict(self, image):
                return image.shape

        router = InferenceRouter({"local_yolo": Provider()})
        self.assertEqual(router.predict(np.zeros((3, 4, 3))), (3, 4, 3))


class CountAndQualityTests(unittest.TestCase):
    def test_counts_one_authoritative_list_by_class(self):
        objects = [
            {"label": "banana"}, {"label": "banana"}, {"label": "apple"},
            {"label": "other"},
        ]
        self.assertEqual(count_instances(objects), {
            "total": 3,
            "by_class": {"banana": 2, "apple": 1, "orange": 0},
            "source_instance_count": 4,
        })

    def test_quality_gate_reports_signals_as_heuristics(self):
        objects = [{"label": "banana", "confidence": 0.2, "mask_area": 70_000,
                    "polygon": [[0, 0], [2, 0]]}]
        result = assess_instances(objects, 100, 100)
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(result["signals_are_heuristics"])
        self.assertEqual({x["signal"] for x in result["signals"]},
                         {"low_confidence", "large_mask", "invalid_mask"})


if __name__ == "__main__":
    unittest.main()
