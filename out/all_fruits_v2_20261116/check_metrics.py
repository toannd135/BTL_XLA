"""Meaningful regression checks: duplicate detections, wrong class and assignment."""
import numpy as np
from metrics_all3 import match_boxes, summarize

assert match_boxes(np.zeros((0, 2)), [], [0, 1]) == []
assert match_boxes(np.zeros((2, 0)), [0, 1], []) == []
assert match_boxes([[.99]], [0], [1]) == []  # Location alone cannot fix a wrong class.
assert len(match_boxes([[.8, .9]], [0], [0, 0])) == 1  # Duplicate is one TP, one FP.
assert len(match_boxes([[.8], [.9]], [0, 0], [0])) == 1  # A bunch box cannot count twice.
# Greedy top-overlap matching would miss one valid object; maximize number first.
assert set(match_boxes([[.9, .8], [.7, .1]], [0, 0], [0, 0])) == {(0, 1), (1, 0)}
rows = [{"truth_counts": [0, 1, 0], "pred_counts": [2, 0, 0],
         "tp": [0, 0, 0], "fp": [2, 0, 0], "fn": [0, 1, 0]}]
result = summarize(rows)
assert result["count_MAE_all_classes"] == 3
assert result["per_class"]["banana"]["count_MAE_all_images"] == 2
assert result["per_class"]["banana"]["count_MAE_when_present"] is None
assert result["per_class"]["apple"]["FN"] == 1
assert result["exact_all_classes"] == 0
print("Matching and all-three-class count checks passed.")
