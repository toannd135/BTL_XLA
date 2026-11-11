import unittest

from evaluate_instance_count import match_instances, score_image, summarize


class InstanceCountTests(unittest.TestCase):
    def test_group_plus_two_individuals_is_an_extra_detection(self):
        truth = [[0, 0, 0, .5, 1], [0, .5, 0, 1, 1]]
        score = score_image(truth, [[0, 0, 0, 1, 1]] + truth)
        self.assertEqual(score["absolute_count_error"], 1)
        self.assertEqual(score["iou_75"]["tp"], 2)
        self.assertEqual(score["iou_75"]["fp"], 1)
        self.assertFalse(score["iou_75"]["perfect_instances"])

    def test_equal_count_does_not_hide_duplicate_and_missing_fruit(self):
        truth = [[0, 0, 0, .4, 1], [0, .6, 0, 1, 1]]
        score = score_image(truth, [truth[0], truth[0]])
        self.assertTrue(score["exact_count"])
        self.assertFalse(score["iou_50"]["perfect_instances"])
        self.assertEqual((score["iou_50"]["tp"], score["iou_50"]["fp"], score["iou_50"]["fn"]), (1, 1, 1))

    def test_misclassification_is_not_a_correct_instance(self):
        score = score_image([[0, 0, 0, 1, 1]], [[1, 0, 0, 1, 1]])
        self.assertEqual(score["iou_50"]["tp"], 0)
        self.assertEqual(score["per_class_absolute_error"], {0: 1, 1: 1, 2: 0})

    def test_matching_can_reassign_to_avoid_greedy_under_matching(self):
        truth = [[0, 0, 0, .7, 1], [0, .3, 0, 1, 1]]
        predictions = [[0, .1, 0, .8, 1], [0, 0, 0, .5, 1]]
        self.assertEqual(match_instances(truth, predictions, .5)["tp"], 2)

    def test_empty_scene_and_summary(self):
        score = score_image([], [])
        self.assertTrue(score["iou_75"]["perfect_instances"])
        result = summarize([score, score_image([[0, 0, 0, 1, 1]], [])])
        self.assertEqual(result["count_mae"], .5)
        self.assertEqual(result["exact_count_rate"], .5)
        self.assertEqual(result["iou_50"]["perfect_instances_rate"], .5)


if __name__ == "__main__":
    unittest.main()
