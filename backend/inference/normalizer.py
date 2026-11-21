"""Convert detector arrays into the single internal instance representation."""

import numpy as np

from .base import Instance


def normalize_instances(polygons, boxes, scores, classes, names, source="local_yolo"):
    lengths = {len(boxes), len(scores), len(classes)}
    if len(lengths) != 1:
        raise ValueError("boxes, scores, and classes must have the same number of instances")
    if polygons is not None and len(polygons) != len(boxes):
        raise ValueError("polygons must align with detector instances")

    normalized = []
    for index, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        coords = np.asarray(box, dtype=np.float32).reshape(4)
        x1, y1, x2, y2 = map(float, coords)
        if not np.isfinite(coords).all() or x2 < x1 or y2 < y1:
            raise ValueError(f"invalid xyxy box for instance {index}")
        class_id = int(class_id)
        class_name = str(names.get(class_id, f"class_{class_id}")).lower()
        polygon = None if polygons is None else np.asarray(polygons[index], dtype=np.float32).reshape(-1, 2)
        normalized.append(Instance(
            instance_id=f"{source}:{index + 1}",
            class_id=class_id,
            class_name=class_name,
            confidence=float(score),
            bbox_xyxy=(x1, y1, x2, y2),
            polygon=polygon,
            source=source,
        ))
    return normalized

