"""Bounded sequential inference for standard and rotated-image detection."""

import cv2
import numpy as np


def restore_clockwise_boxes(boxes, original_height):
    """Map xyxy edge coordinates from a 90-degree clockwise image to its original."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    restored = boxes.copy()
    restored[:, 0] = boxes[:, 1]
    restored[:, 1] = original_height - boxes[:, 2]
    restored[:, 2] = boxes[:, 3]
    restored[:, 3] = original_height - boxes[:, 0]
    return restored


def restore_counterclockwise_boxes(boxes, original_width):
    """Map xyxy edge coordinates from a 90-degree counterclockwise image back."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.column_stack((original_width - boxes[:, 3], boxes[:, 0],
                            original_width - boxes[:, 1], boxes[:, 2]))


def _group_removal_plan(boxes, scores, classes):
    """Return retained indices and the child pair supporting each removed group."""
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    kept, support = [], {}
    for parent in np.argsort(area, kind="stable"):
        children = np.asarray(kept, dtype=np.int64)
        ratio = area[children] / max(float(area[parent]), 1e-9)
        children = children[(classes[children] == classes[parent]) &
                            (ratio >= 0.25) & (ratio <= 0.85) &
                            (scores[children] >= scores[parent] * 0.5)]
        if len(children) >= 2:
            low = np.maximum(boxes[children, :2], boxes[parent, :2])
            high = np.minimum(boxes[children, 2:], boxes[parent, 2:])
            inside = np.maximum(0, high - low).prod(1)
            valid = inside / np.maximum(area[children], 1e-9) >= 0.8
            children, low, high, inside = children[valid], low[valid], high[valid], inside[valid]
            if len(children) >= 2:
                a, b = np.triu_indices(len(children), k=1)
                intersection = np.maximum(0, np.minimum(boxes[children[a], 2:], boxes[children[b], 2:]) -
                                          np.maximum(boxes[children[a], :2], boxes[children[b], :2])).prod(1)
                pair_iou = intersection / np.maximum(area[children[a]] + area[children[b]] - intersection, 1e-9)
                shared_inside = np.maximum(0, np.minimum(high[a], high[b]) - np.maximum(low[a], low[b])).prod(1)
                coverage = (inside[a] + inside[b] - shared_inside) / max(float(area[parent]), 1e-9)
                valid_pairs = np.flatnonzero((pair_iou <= 0.3) & (coverage >= 0.85))
                if len(valid_pairs):
                    pair_strength = np.minimum(scores[children[a[valid_pairs]]], scores[children[b[valid_pairs]]])
                    pair = valid_pairs[np.argmax(pair_strength)]
                    support[parent] = (children[a[pair]], children[b[pair]])
                    continue
        kept.append(parent)
    mask = np.zeros(len(boxes), dtype=bool)
    mask[kept] = True
    return mask, support


def remove_group_boxes(boxes, scores, classes):
    """Prefer two distinct supported detections over an enclosing group hypothesis.

    A single nested box never removes its parent. Children must be the same class,
    largely inside the parent, substantially smaller, and together cover its area.
    This is a geometric heuristic, not proof of how many fruit are in the pixels.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int64)
    keep, _ = _group_removal_plan(boxes, scores, classes)
    return boxes[keep], scores[keep], classes[keep]


def _has_nested_candidates(boxes, scores, classes):
    """A possible enclosing pair justifies another view, but never deletion by itself."""
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    for parent in range(len(boxes)):
        ratio = area / max(float(area[parent]), 1e-9)
        intersection = np.maximum(0, np.minimum(boxes[:, 2:], boxes[parent, 2:]) -
                                  np.maximum(boxes[:, :2], boxes[parent, :2])).prod(1)
        if np.any((classes == classes[parent]) & (ratio >= 0.25) & (ratio <= 0.85) &
                  (scores >= scores[parent]*0.5) & (intersection / np.maximum(area, 1e-9) >= 0.8)):
            return True
    return False


def refine_group_detections(primary, extra, iou=0.6):
    """Admit extra-view boxes only when they support replacing an existing group.

    Unrelated new detections and alternative boxes of a single fruit are ignored.
    Nothing is recovered when the original views have no detections.
    """
    count = len(primary[0])
    combined = tuple(np.concatenate((a, b)) for a, b in zip(primary, extra))
    boxes, scores, classes = combined
    _, support = _group_removal_plan(boxes, scores, classes)
    keep = np.zeros(len(boxes), dtype=bool)
    keep[:count] = True
    for parent, children in support.items():
        if parent < count:
            keep[parent] = False
            keep[list(children)] = True
    return merge_detections(boxes[keep], scores[keep], classes[keep], iou=iou)


def merge_detections(boxes, scores, classes, iou=0.6, max_det=300):
    """Score-ordered, class-agnostic NMS across views; never add scores from views."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int64)
    if not len(boxes):
        return boxes, scores, classes
    order = np.argsort(-scores, kind="stable")
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    kept = []
    while len(order) and len(kept) < max_det:
        best = order[0]
        kept.append(best)
        rest = order[1:]
        intersection = np.maximum(0, np.minimum(boxes[best, 2:], boxes[rest, 2:]) -
                                  np.maximum(boxes[best, :2], boxes[rest, :2])).prod(1)
        overlap = intersection / np.maximum(area[best] + area[rest] - intersection, 1e-9)
        order = rest[overlap <= iou]
    return boxes[kept], scores[kept], classes[kept]


def detect(model, image, conf=0.4, imgsz=640, iou=0.6, quantize=None, rotated=False):
    """Two views for images; a third only refines existing ambiguous group boxes."""
    height, width = image.shape[:2]

    def view(angle):
        rotation = cv2.ROTATE_90_CLOCKWISE if angle == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE
        frame = image if angle == 0 else cv2.rotate(image, rotation)
        result = model.predict(frame, conf=conf, imgsz=imgsz, quantize=quantize,
                               agnostic_nms=True, iou=iou, max_det=300, verbose=False)[0]
        view_boxes = result.boxes.xyxy.cpu().numpy().copy()
        if angle == 90:
            view_boxes = restore_clockwise_boxes(view_boxes, height)
        elif angle == 270:
            view_boxes = restore_counterclockwise_boxes(view_boxes, width)
        view_boxes[:, (0, 2)] = np.clip(view_boxes[:, (0, 2)], 0, width)
        view_boxes[:, (1, 3)] = np.clip(view_boxes[:, (1, 3)], 0, height)
        return (view_boxes, np.asarray(result.boxes.conf.tolist(), dtype=np.float32),
                np.asarray(result.boxes.cls.tolist(), dtype=np.int64))

    boxes, scores, classes = [], [], []
    for angle in ((0, 90) if rotated else (0,)):
        b, s, c = view(angle)
        boxes.extend(b.tolist())
        scores.extend(s.tolist())
        classes.extend(c.tolist())
    merged = merge_detections(boxes, scores, classes, iou=iou)
    if not rotated:
        return merged
    primary = remove_group_boxes(*merged)
    if _has_nested_candidates(*primary):
        return refine_group_detections(primary, view(270), iou=iou)
    return primary
