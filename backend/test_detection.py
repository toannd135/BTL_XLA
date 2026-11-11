"""Regression checks for coordinate restoration and cross-view deduplication."""

import unittest
import numpy as np
from detection import (merge_detections, restore_clockwise_boxes, restore_counterclockwise_boxes,
                       remove_group_boxes, refine_group_detections, detect)


class CoordinateTests(unittest.TestCase):
    def test_rectangular_image_round_trip(self):
        original = np.array([[0, 0, 1199, 807], [10, 20, 40, 70], [500, 300, 700, 600]], dtype=np.float32)
        rotated = np.column_stack((807-original[:, 3], original[:, 0], 807-original[:, 1], original[:, 2]))
        np.testing.assert_allclose(restore_clockwise_boxes(rotated, 807), original)

    def test_empty_views(self):
        self.assertEqual(restore_clockwise_boxes([], 807).shape, (0, 4))
        boxes, scores, classes = merge_detections([], [], [])
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(len(scores), 0)
        self.assertEqual(len(classes), 0)

    def test_counterclockwise_rectangular_round_trip(self):
        original = np.array([[0, 0, 1199, 807], [10, 20, 40, 70], [500, 300, 700, 600]], dtype=np.float32)
        rotated = np.column_stack((original[:, 1], 1199-original[:, 2], original[:, 3], 1199-original[:, 0]))
        np.testing.assert_allclose(restore_counterclockwise_boxes(rotated, 1199), original)
        self.assertEqual(restore_counterclockwise_boxes([], 1199).shape, (0, 4))

    def test_duplicate_across_classes_keeps_highest_score(self):
        boxes, scores, classes = merge_detections(
            [[10, 10, 40, 40], [10, 10, 40, 40], [60, 60, 80, 80]], [.8, .9, .7], [0, 1, 0])
        self.assertEqual(len(boxes), 2)
        self.assertEqual(classes.tolist(), [1, 0])
        np.testing.assert_allclose(scores, [.9, .7])

    def test_distinct_overlapping_fruits_are_retained(self):
        boxes, _, _ = merge_detections([[0, 0, 10, 20], [7, 0, 17, 20]], [.9, .8], [0, 0], iou=.6)
        self.assertEqual(len(boxes), 2)


class GroupBoxTests(unittest.TestCase):
    @staticmethod
    def detections(boxes, scores=None, classes=None):
        return (np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                np.asarray(scores if scores is not None else [.8]*len(boxes), dtype=np.float32),
                np.asarray(classes if classes is not None else [0]*len(boxes), dtype=np.int64))

    def test_group_is_removed_even_with_highest_confidence(self):
        boxes, scores, _ = remove_group_boxes(
            [[0, 0, 100, 100], [0, 0, 50, 100], [50, 0, 100, 100]], [.9, .8, .7], [0, 0, 0])
        np.testing.assert_allclose(boxes, [[0, 0, 50, 100], [50, 0, 100, 100]])
        np.testing.assert_allclose(scores, [.8, .7])

    def test_single_nested_child_is_not_evidence_of_a_group(self):
        boxes = [[0, 0, 100, 100], [10, 0, 70, 100], [200, 0, 240, 100]]
        actual, _, _ = remove_group_boxes(boxes, [.9, .8, .7], [0, 0, 0])
        np.testing.assert_allclose(actual, boxes)

    def test_overlapping_duplicates_cannot_support_two_fruits(self):
        boxes = [[0, 0, 100, 100], [0, 0, 70, 100], [10, 0, 80, 100]]
        actual, _, _ = remove_group_boxes(boxes, [.9, .8, .7], [0, 0, 0])
        np.testing.assert_allclose(actual, boxes)

    def test_different_classes_are_not_used_to_drop_a_parent(self):
        boxes = [[0, 0, 100, 100], [0, 0, 50, 100], [50, 0, 100, 100]]
        actual, _, _ = remove_group_boxes(boxes, [.9, .8, .7], [0, 0, 1])
        np.testing.assert_allclose(actual, boxes)

    def test_tiny_or_weak_inner_boxes_do_not_drop_a_real_large_object(self):
        boxes = [[0, 0, 100, 100], [0, 0, 10, 10], [90, 90, 100, 100],
                 [0, 0, 50, 100], [50, 0, 100, 100]]
        actual, _, _ = remove_group_boxes(boxes, [.95, .8, .7, .1, .1], [0]*5)
        np.testing.assert_allclose(actual, boxes)

    def test_partial_coverage_and_outside_objects_do_not_drop_parent(self):
        for children in [[[0, 0, 35, 100], [65, 0, 100, 100]],
                         [[-25, 0, 40, 100], [60, 0, 125, 100]]]:
            boxes = [[0, 0, 100, 100]] + children
            actual, _, _ = remove_group_boxes(boxes, [.9, .8, .7], [0, 0, 0])
            np.testing.assert_allclose(actual, boxes)

    def test_empty_input(self):
        boxes, scores, classes = remove_group_boxes([], [], [])
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(len(scores), 0)
        self.assertEqual(len(classes), 0)

    def test_extra_view_only_contributes_children_of_replaced_group(self):
        primary = self.detections([[0, 0, 100, 100], [0, 0, 50, 100]], [.9, .8])
        extra = self.detections([[50, 0, 100, 100], [200, 200, 220, 220]], [.7, .99])
        boxes, _, _ = refine_group_detections(primary, extra)
        np.testing.assert_allclose(boxes, [[0, 0, 50, 100], [50, 0, 100, 100]])

    def test_extra_view_cannot_add_unrelated_detections_or_change_single_fruit(self):
        primary = self.detections([[0, 0, 100, 100]])
        extra = self.detections([[1, 1, 99, 99], [200, 200, 220, 220]], [.99, .99])
        boxes, scores, _ = refine_group_detections(primary, extra)
        np.testing.assert_allclose(boxes, primary[0])
        np.testing.assert_allclose(scores, primary[1])

    def test_extra_view_cannot_create_detections_in_an_empty_scene(self):
        boxes, _, _ = refine_group_detections(self.detections([]), self.detections([[0, 0, 100, 100]]))
        self.assertEqual(len(boxes), 0)

    def test_pipeline_only_runs_third_view_to_resolve_a_nested_pair(self):
        from types import SimpleNamespace
        class Values:
            def __init__(self, values): self.values = np.asarray(values, dtype=np.float32)
            def cpu(self): return self
            def numpy(self): return self.values
            def tolist(self): return self.values.tolist()

        class Model:
            def __init__(self, views): self.views = iter(views); self.calls = 0
            def predict(self, frame, **kwargs):
                self.calls += 1
                boxes, scores, classes = next(self.views)
                return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Values(boxes), conf=Values(scores), cls=Values(classes)))]

        # Parent/left child in original coordinates; right child in a counterclockwise image.
        model = Model([self.detections([[0, 0, 100, 100], [0, 0, 50, 100]], [.9, .8]),
                       self.detections([]), self.detections([[0, 0, 100, 50]], [.7])])
        boxes, _, _ = detect(model, np.zeros((100, 100, 3), dtype=np.uint8), rotated=True)
        self.assertEqual(model.calls, 3)
        np.testing.assert_allclose(boxes, [[0, 0, 50, 100], [50, 0, 100, 100]])

        model = Model([self.detections([[0, 0, 100, 100]]), self.detections([])])
        detect(model, np.zeros((100, 100, 3), dtype=np.uint8), rotated=True)
        self.assertEqual(model.calls, 2)


if __name__ == "__main__":
    unittest.main()
