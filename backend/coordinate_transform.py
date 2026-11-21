"""Geometry transforms for image preprocessing and API coordinate restoration."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ImageTransform:
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    scale_x: float
    scale_y: float
    crop_left: float = 0.0
    crop_top: float = 0.0
    pad_left: float = 0.0
    pad_top: float = 0.0
    pad_right: float = 0.0
    pad_bottom: float = 0.0

    @classmethod
    def from_sizes(cls, original_size, processed_size, *, crop=(0, 0, 0, 0), pad=(0, 0, 0, 0)):
        """Build from `(width, height)` sizes; crop/pad order is left, top, right, bottom."""
        ow, oh = map(int, original_size)
        pw, ph = map(int, processed_size)
        cl, ct, cr, cb = map(float, crop)
        pl, pt, pr, pb = map(float, pad)
        cw, ch = ow - cl - cr, oh - ct - cb
        content_w, content_h = pw - pl - pr, ph - pt - pb
        if min(ow, oh, pw, ph, cw, ch, content_w, content_h) <= 0:
            raise ValueError("Image dimensions after crop/padding must be positive")
        return cls(ow, oh, pw, ph, content_w / cw, content_h / ch,
                   cl, ct, pl, pt, pr, pb)

    def map_points_to_original(self, points):
        """Map Nx2 processed pixel-edge coordinates back and clip to original edges."""
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must have shape (N, 2)")
        mapped = points.copy()
        mapped[:, 0] = (mapped[:, 0] - self.pad_left) / self.scale_x + self.crop_left
        mapped[:, 1] = (mapped[:, 1] - self.pad_top) / self.scale_y + self.crop_top
        mapped[:, 0] = np.clip(mapped[:, 0], 0, self.original_width)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, self.original_height)
        return mapped

    def map_box_to_original(self, box):
        """Map an xyxy box using its two pixel-edge corners."""
        x1, y1, x2, y2 = np.asarray(box, dtype=np.float32).reshape(4)
        corners = self.map_points_to_original([[x1, y1], [x2, y2]])
        return np.asarray([corners[0, 0], corners[0, 1], corners[1, 0], corners[1, 1]], dtype=np.float32)

    def to_dict(self):
        return asdict(self)
