"""Các bộ prompt CLIP để so (tune_prompts.py). "v0" = bộ đang dùng trong backend/clip_check.py lúc bắt đầu chỉnh."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from clip_check import FRUIT_PROMPTS as F0, OTHER_PROMPTS as O0, TEMPLATE as T0  # noqa: E402

F1 = {
    "banana": ["a banana", "a bunch of bananas", "a ripe yellow banana", "a green banana", "an overripe banana with brown spots",
               "a rotten black banana"],
    "apple": ["an apple", "a red apple", "a green apple", "a yellow apple", "a rotten apple", "a bruised apple with brown spots",
              "a moldy apple"],
    "orange": ["an orange", "an orange citrus fruit", "a mandarin orange", "a tangerine", "a pale yellow orange", "a rotten orange",
               "a moldy orange", "a sliced orange"],
}
O_SPECIFIC = [x for x in O0 if x not in ("an object", "food", "a toy", "an empty background", "a table", "a vegetable")]
O1 = O_SPECIFIC + ["a ball", "a toy", "an empty background", "a table"]
O2 = O_SPECIFIC + ["an empty background", "a table", "a vegetable", "an object", "food"]

SETS = {
    "v0": (F0, O0, T0),
    "v1": (F1, O0, T0),                    # thêm prompt quả (hỏng, vàng nhạt), giữ vật khác
    "v2": (F1, O1, T0),                    # bỏ mục chung "an object"/"food"
    "v3": (F1, [x for x in O1 if x not in ("a lemon",)] + ["a bright yellow lemon with pointed ends"], T0),
    "v4": (F1, O2, "a close-up photo of {}."),
}
