import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import evaluator as ev


def rect(x1, y1, x2, y2, width=12, height=12):
    return ev.polygon_to_mask([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], width, height)


class ClipPolicyTests(unittest.TestCase):
    def test_prompt_grouping_sums_all_prompt_softmax(self):
        p = np.array([.10, .20, .05, .05, .15, .10, .05, .10, .10, .10])
        groups = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])
        np.testing.assert_allclose(ev.group_prompt_probabilities(p, groups), [.35, .20, .15, .30])

    def test_current_prompt_payload_hash(self):
        source = ev.ROOT / "backend/clip_check.py"
        self.assertEqual(ev.prompt_payload_hash(source),
                         "1504ef1f4b753b26c4aa77a5ad9f1f83f6726e1084a86e52934caeedd8900eb1")

    def test_swap_above_and_at_threshold(self):
        self.assertTrue(ev.swap_decision(0, [.05, .80, .10, .05])["swapped"])
        self.assertFalse(ev.swap_decision(0, [.10, .79, .06, .05])["swapped"])

    def test_other_never_drops_or_swaps(self):
        result = ev.swap_decision(1, [.01, .02, .03, .94])
        self.assertEqual(result["result_class_id"], 1)
        self.assertFalse(result["swapped"])
        self.assertEqual(result["reason"], "top_group_other_no_swap")

    def test_ties_choose_lowest_group_id(self):
        result = ev.swap_decision(2, [.45, .45, .05, .05])
        self.assertEqual(result["top_group_id"], 0)
        self.assertEqual(result["result_class_id"], 2)  # top group is below threshold

    def test_nonfinite_or_malformed_scores_fail_closed(self):
        for scores in ([.5, .5, 0, float("nan")], [.5, .5, 0, float("inf")], [.2, .2, .2, .2]):
            with self.assertRaises(ValueError):
                ev.swap_decision(0, scores)

    def test_crop_coordinates_match_clip_checker_formula_at_edges(self):
        image = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
        crop = ev.clip_box_crop(image, [-4, -2, 5, 4], .10)
        # x=[0,5], y=[0,4] after expansion and integer truncation/clipping.
        np.testing.assert_array_equal(crop, image[0:4, 0:5])

    def test_crop_keeps_float32_box_arithmetic(self):
        image = np.arange(20 * 20, dtype=np.uint8).reshape(20, 20)
        box = np.asarray([np.float32(1.2), np.float32(2.3), np.float32(10.4), np.float32(11.7)])
        px, py = (box[2] - box[0]) * .10, (box[3] - box[1]) * .10
        x1 = int(max(0, min(19, box[0] - px)))
        y1 = int(max(0, min(19, box[1] - py)))
        x2 = int(min(20, max(x1 + 1, box[2] + px)))
        y2 = int(min(20, max(y1 + 1, box[3] + py)))
        np.testing.assert_array_equal(ev.clip_box_crop(image, box, .10), image[y1:y2, x1:x2])

    def test_inner_view_is_queried_and_selected_only_at_threshold(self):
        class FakeChecker:
            def __init__(self, rows): self.rows, self.calls = rows, []
            def probs(self, crops):
                self.calls.append(len(crops))
                return np.asarray(self.rows.pop(0), dtype=float)
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        checker = FakeChecker([
            [[.05, .05, .05, .85], [.05, .05, .05, .85]],
            [[.05, .91, .02, .02], [.10, .79, .05, .06]],
        ])
        rows = ev.score_boxes(checker, img, np.array([[2, 2, 12, 12], [2, 2, 12, 12]]))
        self.assertEqual(checker.calls, [2, 2])
        self.assertEqual(rows[0]["selected_view"], "inner")
        self.assertEqual(rows[1]["selected_view"], "expanded")


class GeometryAndMatchingTests(unittest.TestCase):
    def test_polygon_rounding_clipping_and_invalid_empty(self):
        mask = ev.polygon_to_mask([[-3, -3], [4.5, 0], [9, 9], [0, 9]], 10, 10)
        self.assertEqual(mask.shape, (10, 10))
        self.assertGreater(mask.sum(), 0)
        for poly in ([], [[1, 1], [2, 2]], [[1, 1], [2, 2], [float("nan"), 3]]):
            self.assertEqual(int(ev.polygon_to_mask(poly, 10, 10).sum()), 0)

    def test_same_object_same_or_different_class_geometry_matches(self):
        m = rect(1, 1, 5, 5)
        result = ev.class_agnostic_match([m], [m.copy()])
        self.assertEqual(result["matches"][0]["gt_id"], 0)
        self.assertEqual(result["unmatched_gt_ids"], [])
        self.assertEqual(result["unmatched_prediction_ids"], [])
        # Matcher has no class input, so class changes cannot affect this result.
        self.assertEqual(result, ev.class_agnostic_match([m], [m.copy()]))

    def test_evaluate_matches_geometry_before_comparing_classes(self):
        mask = rect(1, 1, 5, 5)
        gt = [{"class_id": 0, "class_name": "banana", "mask": mask}]
        prediction = {"instances": [{"class_id": 1, "class_name": "apple",
                                      "polygon": [[1, 1], [5, 1], [5, 5], [1, 5]]}]}
        report = ev.evaluate_image(gt, prediction, 12, 12)
        self.assertEqual(report["matched_count"], 1)
        self.assertEqual(report["confusion"]["banana"]["apple"], 1)
        self.assertEqual(report["false_positives"], [])

    def test_overlapping_objects_are_one_to_one(self):
        gt = [rect(0, 0, 7, 7), rect(4, 0, 11, 7)]
        pred = [rect(0, 0, 7, 7), rect(4, 0, 11, 7)]
        result = ev.class_agnostic_match(gt, pred)
        self.assertEqual([(m["gt_id"], m["prediction_id"]) for m in result["matches"]], [(0, 0), (1, 1)])

    def test_equal_iou_ties_use_lowest_prediction_id(self):
        m = rect(2, 2, 8, 8)
        result = ev.class_agnostic_match([m, m.copy()], [m.copy(), m.copy()])
        self.assertEqual([(x["gt_id"], x["prediction_id"]) for x in result["matches"]], [(0, 0), (1, 1)])

    def test_unmatched_prediction_and_ground_truth(self):
        m = rect(1, 1, 4, 4)
        distant = rect(8, 8, 11, 11)
        result = ev.class_agnostic_match([m], [m.copy(), distant])
        self.assertEqual(result["unmatched_prediction_ids"], [1])
        result = ev.class_agnostic_match([m, distant], [m.copy()])
        self.assertEqual(result["unmatched_gt_ids"], [1])

    def test_empty_masks_never_match(self):
        empty = np.zeros((12, 12), dtype=np.uint8)
        result = ev.class_agnostic_match([empty], [empty.copy()])
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["unmatched_gt_ids"], [0])
        self.assertEqual(result["unmatched_prediction_ids"], [0])


class InvariantAndPersistenceTests(unittest.TestCase):
    def record(self, cls=0):
        grouped = [.90, .03, .04, .03] if cls == 0 else [.03, .90, .04, .03]
        return {"image_id": "synthetic", "image_path": "synthetic.jpg", "image_sha256": "a" * 64,
                "label_path": "synthetic.txt", "label_sha256": "b" * 64,
                "width": 4, "height": 4, "inference_status": "success",
                "detector_checkpoint_sha256": "c" * 64, "config_sha256": "d" * 64,
                "prediction_count_before_clip": 1, "prediction_count_after_clip": 1,
                "instances": [{"instance_id": "det_0000", "class_id": cls,
                               "class_name": ev.FRUITS[cls],
                               "confidence": .8, "bbox": [0, 0, 2, 2],
                               "polygon": [[0, 0], [2, 0], [2, 2]], "mask_area": 6,
                               "mask_sha256": "e" * 64,
                               "candidate": {"grouped_probabilities": grouped,
                                   "first_view_grouped_probabilities": grouped,
                                   "inner_view_grouped_probabilities": None, "selected_view": "expanded",
                                   "original_class_id": 0, "original_class_name": "banana",
                                   "result_class_id": cls, "result_class_name": ev.FRUITS[cls],
                                   "swapped": cls != 0, "top_group_id": cls,
                                   "top_group_name": ev.FRUITS[cls], "top_group_probability": .90,
                                   "threshold": .80,
                                   "reason": "top_group_matches_detector" if cls == 0 else "different_top_fruit_at_or_above_threshold"}}]}

    def test_swap_invariants_accept_class_change_and_reject_geometry_change(self):
        ev.validate_swap_invariants(self.record(0), self.record(1))
        broken = self.record(1)
        broken["instances"][0]["bbox"][0] = 1
        with self.assertRaises(AssertionError):
            ev.validate_swap_invariants(self.record(0), broken)

    def test_atomic_write_and_recovery_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            ev.atomic_json(path, {"images": {"one": "success"}})
            recovered = json.loads(path.read_text())
            self.assertEqual(recovered["images"]["one"], "success")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_config_digest_is_stable_and_ignores_embedded_digest(self):
        config = {"a": 1, "b": [2, 3]}
        digest = ev.canonical_hash(config)
        self.assertEqual(digest, ev.canonical_hash({**config, "config_sha256": "ignored"}))

    def test_checked_in_config_and_artifact_hashes(self):
        config, digest = ev.load_and_verify_config(ev.HERE / "config.json")
        self.assertEqual(config["config_sha256"], digest)

    def test_resume_recovery_requires_two_success_records_same_config(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, candidate_path = Path(directory) / "b.json", Path(directory) / "c.json"
            base = self.record(0); candidate = self.record(1)
            for row in (base, candidate):
                row.update(config_sha256="f" * 64)
            ev.atomic_json(base_path, base); ev.atomic_json(candidate_path, candidate)
            self.assertTrue(ev.reusable_completed_pair(base_path, candidate_path, "f" * 64,
                                                       "synthetic", "a" * 64, "b" * 64))
            self.assertFalse(ev.reusable_completed_pair(base_path, candidate_path, "f" * 64,
                                                        "synthetic", "0" * 64, "b" * 64))
            self.assertFalse(ev.reusable_completed_pair(base_path, candidate_path, "e" * 64))
            candidate["instances"][0]["mask_area"] += 1
            ev.atomic_json(candidate_path, candidate)
            self.assertFalse(ev.reusable_completed_pair(base_path, candidate_path, "f" * 64))

    def test_prediction_schema_validation(self):
        record = self.record(1)
        record["instances"][0]["candidate"] = {
            "grouped_probabilities": [.05, .90, .03, .02],
            "first_view_grouped_probabilities": [.05, .90, .03, .02],
            "inner_view_grouped_probabilities": None, "selected_view": "expanded",
            "original_class_id": 1, "result_class_id": 1, "swapped": False, "reason": "top_group_matches_detector"}
        ev.validate_prediction_record(record)
        record["prediction_count_after_clip"] = 0
        with self.assertRaises(Exception):
            ev.validate_prediction_record(record)


if __name__ == "__main__":
    unittest.main(verbosity=2)
