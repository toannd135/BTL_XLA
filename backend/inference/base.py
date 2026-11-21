"""Provider-neutral inference records."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Instance:
    instance_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    polygon: np.ndarray | None
    source: str
    coordinate_space: str = "image_pixels"
    attributes: dict = field(default_factory=dict)


@dataclass
class ProviderResult:
    instances: list[Instance]
    image_width: int
    image_height: int
    provider: str
    diagnostics: dict = field(default_factory=dict)


class InferenceProvider(Protocol):
    name: str

    def predict(self, image: np.ndarray) -> ProviderResult:
        """Run a provider and return normalized instances in image pixel space."""

