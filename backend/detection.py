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


def detect_obb(model, image, conf=0.4, imgsz=640, iou=0.6, quantize=None):
    """Model khung xoay (YOLO-OBB): NMS dùng IoU xoay nên quả nằm chéo sát nhau không bị xoá nhầm.
    Trả (polygons Nx4x2, xyxy Nx4 bao ngoài, scores, classes)."""
    r = model.predict(image, conf=conf, imgsz=imgsz, quantize=quantize, agnostic_nms=True, iou=iou,
                      max_det=300, verbose=False)[0]
    o = r.obb
    return (o.xyxyxyxy.cpu().numpy(), o.xyxy.cpu().numpy(),
            np.asarray(o.conf.tolist(), dtype=np.float32), np.asarray(o.cls.tolist(), dtype=np.int64))


def fuse_teacher_boxes(polygons, boxes, scores, classes, teacher_boxes, teacher_scores,
                       teacher_classes, class_map, thresholds, overlap=0.3):
    """Add only independent high-confidence teacher boxes not covered by the detector."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    polygons = np.asarray(polygons, dtype=np.float32).reshape(-1, 4, 2)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int64)
    added_boxes, added_scores, added_classes = [], [], []
    for box, score, teacher_class in zip(teacher_boxes, teacher_scores, teacher_classes):
        if int(teacher_class) not in class_map:
            continue
        cls = class_map[int(teacher_class)]
        if float(score) < thresholds.get(cls, 1.0):
            continue
        same_class = classes == cls
        if len(boxes) and np.any(same_class):
            same = boxes[same_class]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            center_inside = (
                (same[:, 0] <= cx) & (cx <= same[:, 2]) &
                (same[:, 1] <= cy) & (cy <= same[:, 3])
            )
            if np.any(center_inside):
                continue
        added_boxes.append(box)
        added_scores.append(score)
        added_classes.append(cls)
    if not added_boxes:
        return polygons, boxes, scores, classes
    added_boxes = np.asarray(added_boxes, dtype=np.float32)
    added_polygons = np.asarray([
        [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        for b in added_boxes
    ], dtype=np.float32)
    return (
        np.concatenate((polygons, added_polygons)),
        np.concatenate((boxes, added_boxes)),
        np.concatenate((scores, np.asarray(added_scores, dtype=np.float32))),
        np.concatenate((classes, np.asarray(added_classes, dtype=np.int64))),
    )


def deduplicate_boxes(polygons, boxes, scores, classes, iou_threshold=0.65):
    """Remove near-identical same-class boxes while preserving crowded bananas."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    polygons = np.asarray(polygons, dtype=np.float32).reshape(-1, 4, 2)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    keep = []
    for index in order:
        if classes[index] == 0:
            keep.append(index)
            continue
        duplicate = False
        for kept in keep:
            if classes[kept] != classes[index]:
                continue
            a, b = boxes[kept], boxes[index]
            inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
            area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
            area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
            iou = inter / max(area_a + area_b - inter, 1e-9)
            covered = inter / min(area_a, area_b)
            if iou >= iou_threshold or covered >= 0.85:
                duplicate = True
                break
        if not duplicate:
            keep.append(index)
    keep = np.asarray(sorted(keep), dtype=np.int64)
    return polygons[keep], boxes[keep], scores[keep], classes[keep]


def crop_rotated(img, poly, pad=0.10):
    """Cắt vùng khung xoay và xoay thẳng (cạnh dài nằm ngang), nới mỗi chiều pad."""
    rect = cv2.minAreaRect(np.asarray(poly, dtype=np.float32))
    (cx, cy), (w, h), ang = rect
    w, h = max(w * (1 + 2 * pad), 2), max(h * (1 + 2 * pad), 2)
    if w < h:
        w, h, ang = h, w, ang + 90
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    M[0, 2] += w / 2 - cx
    M[1, 2] += h / 2 - cy
    return cv2.warpAffine(img, M, (int(round(w)), int(round(h))), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def refine_types(det_names, classes, type_names, type_probs, t_other=0.6, t_swap=0.7,
                 scores=None, conf=None, t_rescue=0.9, protected_min_conf=0.30):
    """Kiểm tra lại loại quả bằng bộ phân loại crop (banana/apple/orange/other).
    - p(other) >= t_other: bỏ box (vật không phải chuối/táo/cam, vd. dâu, nho).
    - loại quả khác detector và p >= t_swap: đổi nhãn theo bộ phân loại.
    - scores/conf: box có điểm detector < conf (chạy detector ở ngưỡng thấp hơn) chỉ giữ khi p(quả) >= t_rescue.
    Trả (keep: mảng bool, classes mới theo chỉ số của detector)."""
    det_idx = {n: i for i, n in det_names.items()}
    keep = np.ones(len(classes), dtype=bool)
    out = np.asarray(classes, dtype=np.int64).copy()
    fruit_ids = [i for i, n in type_names.items() if n != "other"]
    for k, p in enumerate(type_probs):
        p = np.asarray(p, dtype=np.float32)
        # box dưới ngưỡng detector (conf) chỉ giữ khi bộ kiểm tra chắc chắn là quả (>= t_rescue)
        if scores is not None and conf is not None and scores[k] < conf and max(p[i] for i in fruit_ids) < t_rescue:
            keep[k] = False
            continue
        top = int(p.argmax())
        name = type_names[top]
        if name == "other":
            banana_id = det_idx.get("banana")
            protected_banana = (
                scores is not None and banana_id is not None and out[k] == banana_id
                and scores[k] >= protected_min_conf
            )
            if p[top] >= t_other and not protected_banana:
                keep[k] = False
        elif name in det_idx and det_idx[name] != out[k] and p[top] >= t_swap:
            out[k] = det_idx[name]
    return keep, out


def simplify_polygon(poly, epsilon=0.75):
    """Bound contour displacement in image pixels without shrinking narrow tips."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return pts
    area = cv2.contourArea(pts)
    approx = cv2.approxPolyDP(pts, epsilon=max(0.0, epsilon), closed=True).reshape(-1, 2)
    if len(approx) < 3 or abs(cv2.contourArea(approx) - area) > max(1e-6, area * 0.02):
        return pts
    return approx


def _polygon_mask(poly, width, height):
    """Rasterize only the contour's bounding region, not an entire image per fruit."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3 or not np.isfinite(pts).all():
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8)
    pts = pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    if cv2.contourArea(pts) <= 0:
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(np.round(pts).astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts - [x, y]).astype(np.int32)], 1)
    return (x, y, x + w, y + h), mask


def compute_mask_metrics(polygons, boxes, width, height):
    """Measure visible-mask overlap; this cannot estimate the hidden part of fruit."""
    n = len(polygons)
    if n == 0:
        return []
    regions = [_polygon_mask(p, width, height) for p in polygons]
    bounds, masks = zip(*regions)
    areas = [float(m.sum()) for m in masks]
    metrics = []
    for i in range(n):
        p = polygons[i]
        b = bounds[i]
        area = areas[i]
        solidity = 1.0
        if len(p) >= 3 and area > 0:
            cnt = np.asarray(p, dtype=np.float32).reshape(-1, 1, 2)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(min(1.0, max(0.0, cv2.contourArea(cnt) / hull_area)))
        covered = np.zeros_like(masks[i])
        overlap_details = []
        for j in range(n):
            if i == j:
                continue
            bj = bounds[j]
            x1, y1, x2, y2 = max(b[0], bj[0]), max(b[1], bj[1]), min(b[2], bj[2]), min(b[3], bj[3])
            if x1 < x2 and y1 < y2:
                intersection = (masks[i][y1-b[1]:y2-b[1], x1-b[0]:x2-b[0]] &
                                masks[j][y1-bj[1]:y2-bj[1], x1-bj[0]:x2-bj[0]])
                inter = float(intersection.sum())
                if inter > 0 and area > 0:
                    overlap_ratio = inter / area
                    covered[y1-b[1]:y2-b[1], x1-b[0]:x2-b[0]] |= intersection
                    overlap_details.append({
                        "other_idx": j,
                        "overlap_ratio": round(float(overlap_ratio), 3),
                        "intersection_px": int(inter),
                        "iou": round(inter / max(area + areas[j] - inter, 1), 3),
                    })
        overlap = float(covered.sum()) / max(area, 1)
        metrics.append({
            "mask_area": int(area),
            "mask_valid": bool(len(p) >= 3 and area > 0),
            "solidity": round(float(solidity), 3),
            "overlap_ratio": round(overlap, 3),
            "has_mask_overlap": bool(overlap >= 0.25),
            # Compatibility with previously saved results. These aliases measure overlap too.
            "occlusion_ratio": round(overlap, 3),
            "is_occluded": bool(overlap >= 0.25),
            "overlaps": overlap_details,
        })
    return metrics


def deduplicate_masks(polygons, boxes, scores, classes, width, height,
                       iou_threshold=0.60, coverage_threshold=0.85):
    """Remove near-identical same-class masks while preserving touching fruit.

    A box overlap is not sufficient here: adjacent fruit can have heavily
    overlapping boxes while their visible masks remain distinct.
    """
    polygons = list(polygons)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int64)
    if not len(polygons):
        return polygons, boxes, scores, classes

    regions = []
    areas = []
    for polygon in polygons:
        bounds, mask = _polygon_mask(polygon, width, height)
        regions.append((bounds, mask))
        areas.append(float(mask.sum()))

    order = np.argsort(-scores, kind="stable")
    keep = []
    for index in order:
        if len(polygons[index]) < 3 or areas[index] <= 0:
            keep.append(index)
            continue
        duplicate = False
        for previous in keep:
            if classes[previous] != classes[index] or areas[previous] <= 0:
                continue
            first_bounds, first_mask = regions[previous]
            second_bounds, second_mask = regions[index]
            x1 = max(first_bounds[0], second_bounds[0])
            y1 = max(first_bounds[1], second_bounds[1])
            x2 = min(first_bounds[2], second_bounds[2])
            y2 = min(first_bounds[3], second_bounds[3])
            if x1 >= x2 or y1 >= y2:
                intersection = 0.0
            else:
                intersection = float((
                    first_mask[y1-first_bounds[1]:y2-first_bounds[1],
                               x1-first_bounds[0]:x2-first_bounds[0]] &
                    second_mask[y1-second_bounds[1]:y2-second_bounds[1],
                                x1-second_bounds[0]:x2-second_bounds[0]]
                ).sum())
            union = areas[previous] + areas[index] - intersection
            iou = intersection / max(union, 1.0)
            coverage = intersection / max(min(areas[previous], areas[index]), 1.0)
            if iou >= iou_threshold or coverage >= coverage_threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(index)

    keep = np.asarray(sorted(keep), dtype=np.int64)
    return ([polygons[index] for index in keep], boxes[keep], scores[keep], classes[keep])


def detect_seg(model, image, conf=0.4, imgsz=640, iou=0.6, quantize=None, simplify_eps=0.75,
               retina_masks=True, compute_metrics=True):
    """YOLO Instance Segmentation.
    Runs prediction and returns (polygons, boxes, scores, classes, mask_metrics).
    Handles both custom 3-class fruit models (banana/apple/orange) and 80-class COCO seg models.
    """
    height, width = image.shape[:2]
    classes_arg = None
    if len(model.names) >= 80 and 46 in model.names and 47 in model.names:
        classes_arg = [46, 47, 49]

    result = model.predict(
        image, conf=conf, imgsz=imgsz, quantize=quantize,
        agnostic_nms=True, iou=iou, max_det=300, verbose=False,
        classes=classes_arg, retina_masks=retina_masks,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return [], np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64), []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = np.asarray(result.boxes.conf.tolist(), dtype=np.float32)
    classes = np.asarray(result.boxes.cls.tolist(), dtype=np.int64)

    polygons = []
    raw_polygons = result.masks.xy if result.masks is not None else []
    for index in range(len(boxes)):
        poly = np.asarray(raw_polygons[index], dtype=np.float32).copy() if index < len(raw_polygons) else np.zeros((0, 2), dtype=np.float32)
        if len(poly) >= 3 and np.isfinite(poly).all():
            poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
            poly = simplify_polygon(poly, epsilon=simplify_eps)
            if cv2.contourArea(poly) <= 0:
                poly = np.zeros((0, 2), dtype=np.float32)
        else:
            poly = np.zeros((0, 2), dtype=np.float32)
        # Missing masks stay missing; do not claim a rectangular fallback is a mask.
        polygons.append(poly)

    mask_metrics = compute_mask_metrics(polygons, boxes, width, height) if compute_metrics else []
    return polygons, boxes, scores, classes, mask_metrics


def get_obb_spine(pts):
    """Compute the centerline spine segment (s1, s2) of an oriented bounding box."""
    pts = np.asarray(pts, dtype=np.float32)
    d01 = np.linalg.norm(pts[0] - pts[1])
    d12 = np.linalg.norm(pts[1] - pts[2])
    if d01 < d12:
        return (pts[0] + pts[1]) / 2.0, (pts[2] + pts[3]) / 2.0
    else:
        return (pts[1] + pts[2]) / 2.0, (pts[3] + pts[0]) / 2.0


def dist_to_segment(p, s1, s2):
    """Euclidean distance from point p to line segment (s1, s2)."""
    v = s2 - s1
    l2 = float(np.dot(v, v))
    if l2 == 0:
        return float(np.linalg.norm(p - s1))
    t = max(0.0, min(1.0, float(np.dot(p - s1, v) / l2)))
    proj = s1 + t * v
    return float(np.linalg.norm(p - proj))


def _sam_non_overlapping(predictor):
    # SAM2 can choose the strongest image-derived logit for each pixel before
    # thresholding. Do not turn overlapping binary masks into a geometric cut.
    if hasattr(predictor.model, "_apply_non_overlapping_constraints"):
        predictor.non_overlap_masks = True


def configure_sam_refiner(model):
    model.add_callback("on_predict_start", _sam_non_overlapping)
    return model


def _sam_instance_parts(mask_model, image, parent, candidates, simplify_eps=0.75):
    """Use one image-encoder pass with positive/negative prompts for a cluster.

    OBBs suggest seeds, never the final boundary. Reject the whole replacement
    when any child is missing, duplicated, disconnected, or loses parent coverage.
    """
    if mask_model is None or not 2 <= len(candidates) <= 12:
        return None
    height, width = parent.shape
    seeds, seed_sets, prompt_boxes = [], [], []
    for i, candidate in enumerate(candidates):
        region = parent & candidate["mask"]
        exclusive = region.copy()
        for j, other in enumerate(candidates):
            if i != j:
                exclusive &= 1 - other["mask"]
        if exclusive.sum() < max(16, region.sum() * 0.05):
            return None
        # A seed safely inside this fruit is more useful than an OBB's centre,
        # which can lie on the background of a curved banana.
        distance = cv2.distanceTransform(exclusive, cv2.DIST_L2, 5)
        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        if distance[y, x] < 2:
            return None
        seeds.append([int(x), int(y)])
        # Cover more of the fruit's length. A single negative point near a
        # neighbour's tip does not prevent SAM from taking its other end.
        ys, xs = np.where(distance >= max(2, float(distance[y, x]) * 0.35))
        s1, s2 = get_obb_spine(candidate["pts"])
        direction = (s2-s1) / max(float(np.linalg.norm(s2-s1)), 1e-6)
        projection = xs*direction[0] + ys*direction[1]
        anchors = [[int(x), int(y)]]
        for quantile in (0.15, 0.85):
            target = np.quantile(projection, quantile)
            band = np.abs(projection-target) <= max(2, float(np.ptp(projection)) * 0.08)
            choice = np.argmax(np.where(band, distance[ys, xs], -1))
            anchors.append([int(xs[choice]), int(ys[choice])])
        seed_sets.append(anchors)
        lo, hi = candidate["pts"].min(axis=0), candidate["pts"].max(axis=0)
        pad = (hi - lo) * 0.08
        prompt_boxes.append([max(0, float(lo[0]-pad[0])), max(0, float(lo[1]-pad[1])),
                             min(width-1, float(hi[0]+pad[0])), min(height-1, float(hi[1]+pad[1]))])

    points = [[seeds[i]] + [point for j, group in enumerate(seed_sets) if i != j for point in group]
              for i in range(len(seeds))]
    labels = [[1] + [0]*3*(len(seeds)-1) for _ in seeds]
    result = mask_model.predict(image, bboxes=prompt_boxes, points=points, labels=labels,
                                conf=0.0, verbose=False)[0]
    if result.masks is None or result.boxes is None:
        return None
    raw_masks = result.masks.data.cpu().numpy()
    prompt_ids = np.asarray(result.boxes.cls.tolist(), dtype=np.int64)
    if len(raw_masks) != len(seeds) or sorted(prompt_ids.tolist()) != list(range(len(seeds))):
        return None
    quality = np.asarray(result.boxes.conf.tolist(), dtype=np.float32)
    if not np.isfinite(quality).all() or np.any(quality < 0.55):
        return None

    # Permit a small correction of the outer contour, without spilling into a
    # different fruit. Preserve SAM's image-derived internal boundaries.
    allowed = cv2.dilate(parent, np.ones((7, 7), np.uint8))
    parts = [None] * len(seeds)
    masks = [None] * len(seeds)
    parent_area = float(parent.sum())
    for raw, prompt_id in zip(raw_masks, prompt_ids):
        mask = (raw > 0.5).astype(np.uint8)
        if mask.shape != parent.shape:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask &= allowed
        area = float(mask.sum())
        seed = seeds[prompt_id]
        if area < max(64, parent_area * 0.08) or not mask[seed[1], seed[0]]:
            return None
        if any(mask[y, x] for j, (x, y) in enumerate(seeds) if j != prompt_id):
            return None
        candidate_mask = candidates[prompt_id]["mask"]
        candidate_intersection = float((mask & candidate_mask).sum())
        if (candidate_intersection / max(area, 1.0) < 0.35 or
                candidate_intersection / max(float(candidate_mask.sum()), 1.0) < 0.15):
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        # A single polygon cannot faithfully encode multiple visible islands.
        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 1, cv2.FILLED)
        if float((mask & component).sum()) / area < 0.95:
            return None
        if float((mask & component).sum()) / max(float(component.sum()), 1) < 0.98:
            return None
        polygon = simplify_polygon(contour, epsilon=simplify_eps)
        if len(polygon) < 3 or cv2.contourArea(polygon) < 32:
            return None
        masks[prompt_id] = mask
        parts[prompt_id] = polygon
    union = np.zeros_like(parent)
    for i, mask in enumerate(masks):
        for other in masks[:i]:
            inter = float((mask & other).sum())
            if inter / max(min(float(mask.sum()), float(other.sum())), 1) > 0.10:
                return None
        union |= mask
    if float((union & parent).sum()) / max(parent_area, 1) < 0.85:
        return None
    return parts


def _proposal_overlap(first, second):
    intersection = float((first["mask"] & second["mask"]).sum())
    return intersection / max(first["area"]+second["area"]-intersection, 1)


def _fruit_proposals(obb_model, image, labels, conf, quantize, hbb_detections=None,
                     hbb_names=None, diagnostics=None):
    """Collect class-aware hypotheses; a box is not itself a segmented fruit."""
    height, width = image.shape[:2]
    candidates = []
    obb_models = list(obb_model) if isinstance(obb_model, (list, tuple)) else (
        [obb_model] if obb_model is not None else []
    )

    def add(polygon, score, label, source):
        if label not in labels:
            return
        polygon = np.asarray(polygon, np.float32).reshape(-1, 2)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
        area = float(mask.sum())
        if area:
            candidates.append({"pts": polygon, "mask": mask, "area": area, "conf": float(score),
                               "label": label, "sources": {source}})

    for model in obb_models:
        selected = [index for index, label in model.names.items() if label in labels]
        if not selected:
            continue
        result = model.predict(image, conf=conf, imgsz=1280, quantize=quantize,
                               agnostic_nms=False, iou=0.5, verbose=False,
                               classes=selected, max_det=64)[0]
        if getattr(result, "obb", None) is not None:
            for cls, score, polygon in zip(result.obb.cls, result.obb.conf, result.obb.xyxyxyxy):
                add(polygon.cpu().numpy(), score, model.names[int(cls)], "obb")
    unique_obb_bananas = []
    for candidate in sorted(
        (item for item in candidates
         if item["label"] == "banana" and "obb" in item["sources"]),
        key=lambda item: -item["conf"],
    ):
        if not any(_proposal_overlap(candidate, previous) >= 0.70
                   for previous in unique_obb_bananas):
            unique_obb_bananas.append(candidate)
    obb_bananas = len(unique_obb_bananas)
    if diagnostics is not None:
        diagnostics["obb_ensemble_models"] = len(obb_models)
        diagnostics["obb_proposals"] = sum("obb" in item["sources"] for item in candidates)
        diagnostics["obb_unique_banana_proposals"] = obb_bananas
    if hbb_detections is not None:
        for box, score, cls in zip(*hbb_detections):
            # HBB banana labels can enclose a bunch. Keep the established OBB
            # proposals when it already finds individual bananas; use HBB as
            # a proposal fallback when OBB sees only one or no banana.
            if hbb_names[int(cls)] == "banana" and obb_bananas >= 2:
                continue
            x1, y1, x2, y2 = box
            add([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], score, hbb_names[int(cls)], "hbb")

    candidates.sort(key=lambda item: -item["conf"])
    unique = []
    for candidate in candidates:
        duplicate = None
        for previous in unique:
            if previous["label"] != candidate["label"]:
                continue
            ratio = min(previous["area"], candidate["area"]) / max(previous["area"], candidate["area"])
            centres = np.linalg.norm(previous["pts"].mean(axis=0)-candidate["pts"].mean(axis=0))
            scale = max(1, min(np.ptp(previous["pts"], axis=0).min(), np.ptp(candidate["pts"], axis=0).min()))
            iou = _proposal_overlap(previous, candidate)
            covered = float((previous["mask"] & candidate["mask"]).sum()) / min(previous["area"],candidate["area"])
            same_banana = candidate["label"] == "banana" and covered >= .90
            same_round_fruit = candidate["label"] != "banana" and ratio >= .65 and iou >= .45 and centres <= scale*.30
            if iou >= .70 or same_banana or same_round_fruit:
                duplicate = previous
                break
        if duplicate is None:
            unique.append(candidate)
        else:
            duplicate["sources"].update(candidate["sources"])
    if len(unique) > 1:
        boxes = np.asarray([np.concatenate((p["pts"].min(axis=0), p["pts"].max(axis=0))) for p in unique])
        names = {label: index for index, label in enumerate(sorted(labels))}
        keep, _ = _group_removal_plan(boxes, np.asarray([p["conf"] for p in unique]),
                                      np.asarray([names[p["label"]] for p in unique]))
        unique = [p for p, retain in zip(unique, keep) if retain or (p["label"] == "banana" and obb_bananas >= 2)]
    # A group can contain more than two children. The legacy pair-only check
    # misses bunches whose individual boxes each cover less than 25% of it.
    removed = set()
    for parent_index in sorted(range(len(unique)),key=lambda i: unique[i]["area"]):
        parent = unique[parent_index]
        if parent["label"] == "banana" and obb_bananas >= 2:
            continue
        low = np.maximum(0,np.floor(parent["pts"].min(axis=0))).astype(int)
        high = np.minimum([width,height],np.ceil(parent["pts"].max(axis=0))+1).astype(int)
        x1,y1 = low
        x2,y2 = high
        parent_roi = parent["mask"][y1:y2,x1:x2]
        union = np.zeros_like(parent_roi)
        children = []
        for i,child in enumerate(unique):
            ratio = child["area"]/parent["area"]
            if i == parent_index or i in removed or child["label"] != parent["label"]:
                continue
            if not .12 <= ratio <= .80 or child["conf"] < parent["conf"]*.5:
                continue
            child_roi = child["mask"][y1:y2,x1:x2]
            if float((parent_roi & child_roi).sum())/child["area"] < .85:
                continue
            if any(float((child_roi & previous["mask"][y1:y2,x1:x2]).sum()) /
                   min(child["area"],previous["area"]) > .75 for previous in children):
                continue
            children.append(child)
            union |= child_roi
        if len(children) >= 2 and float((union & parent_roi).sum())/parent["area"] >= .85:
            removed.add(parent_index)
    unique = [candidate for i,candidate in enumerate(unique) if i not in removed]
    return unique[:64]


def _crop_fruit_instance(model, image, candidate, label, conf, quantize):
    """Re-detect a proposal at a larger effective scale with the semantic model.

    SAM does not classify fruit. A new auxiliary box is admitted only if YOLO-seg
    independently finds this class in the crop and provides a usable mask.
    """
    height, width = image.shape[:2]
    bounds = candidate["pts"].astype(np.float64)
    lo, hi = bounds.min(axis=0), bounds.max(axis=0)
    pad = (hi-lo) * .30
    x1, y1 = np.maximum(0, np.floor(lo-pad)).astype(int)
    x2, y2 = np.minimum([width,height], np.ceil(hi+pad)).astype(int)
    if x2-x1 < 8 or y2-y1 < 8:
        return None
    polygons, boxes, scores, classes, _ = detect_seg(
        model, image[y1:y2,x1:x2], conf=max(.5, conf), imgsz=640, iou=.7,
        quantize=quantize, compute_metrics=False)
    choices = []
    for polygon, score, cls in zip(polygons, scores, classes):
        if model.names[int(cls)] != label or len(polygon) < 3:
            continue
        polygon = polygon + [x1,y1]
        mask = np.zeros((height,width), np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
        area = float(mask.sum())
        intersection = float((mask & candidate["mask"]).sum())
        if area < 64 or intersection / max(area,1) < .60:
            continue
        if intersection / candidate["area"] < .15:
            continue
        # A crop boundary must not become an invented fruit edge.
        local_mask = mask[y1:y2,x1:x2]
        edge_length = ((int(local_mask[:,:2].sum()) if x1 > 0 else 0) +
                       (int(local_mask[:,-2:].sum()) if x2 < width else 0) +
                       (int(local_mask[:2,:].sum()) if y1 > 0 else 0) +
                       (int(local_mask[-2:,:].sum()) if y2 < height else 0))
        if edge_length > max(4, cv2.arcLength(polygon.astype(np.float32),True)*.10):
            continue
        choices.append((intersection/max(area+candidate["area"]-intersection,1), polygon, float(score)))
    if not choices:
        return None
    _, polygon, score = max(choices, key=lambda choice: choice[0])
    return polygon.astype(np.float32), score


def _validated_crop_parts(parts, parent):
    if not parts or any(part is None for part in parts):
        return None
    masks = []
    union = np.zeros_like(parent)
    parent_area = float(parent.sum())
    for polygon, _ in parts:
        mask = np.zeros_like(parent)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
        area = float(mask.sum())
        if area < max(64, parent_area*.08) or float((mask & parent).sum()) / max(area,1) < .85:
            return None
        for previous in masks:
            if float((mask & previous).sum()) / max(min(area,float(previous.sum())),1) > .10:
                return None
        union |= mask
        masks.append(mask)
    if float((union & parent).sum()) / max(parent_area,1) < .85:
        return None
    return [part[0] for part in parts]


def split_fruit_cluster_detections(polygons, boxes, scores, classes, obb_model, image,
                                   conf=.4, det_names=None, quantize=None, mask_model=None,
                                   diagnostics=None, hbb_detections=None, hbb_names=None,
                                   seg_model=None, allowed_labels=None, max_crop_checks=12):
    """Refine grouped bananas, apples and oranges; recover verified missed fruit.

    Auxiliary detections only propose instances. SAM or crop segmentation must
    supply the boundaries. Each proposal belongs to at most one existing mask.
    Failed replacements remain atomic and are marked for review.
    """
    diagnostics = diagnostics if diagnostics is not None else {}
    for key in ("fruit_split_groups", "fruit_split_pending", "fruit_recovered", "banana_split_groups", "banana_split_pending"):
        diagnostics.setdefault(key, 0)
    per_class = diagnostics.setdefault("instance_refinement", {})
    names = det_names or {0:"banana",1:"apple",2:"orange",46:"banana",47:"apple",49:"orange"}
    labels = set(allowed_labels or {"banana","apple","orange"})
    indices = [i for i,c in enumerate(classes) if names.get(int(c)) in labels and len(polygons[i]) >= 3]
    if not indices:
        return polygons, boxes, scores, classes
    candidates = _fruit_proposals(
        obb_model, image, labels, conf, quantize, hbb_detections, hbb_names, diagnostics
    )
    if not candidates:
        return polygons, boxes, scores, classes
    height,width = image.shape[:2]
    parents = {}
    for i in indices:
        mask = np.zeros((height,width), np.uint8)
        cv2.fillPoly(mask,[np.round(polygons[i]).astype(np.int32)],1)
        parents[i] = mask
    assignments = {i:[] for i in indices}
    unassigned = []
    for candidate in candidates:
        matches = []
        for i,mask in parents.items():
            if names[int(classes[i])] != candidate["label"]:
                continue
            intersection = float((mask & candidate["mask"]).sum())
            coverage = intersection / candidate["area"]
            area = float(mask.sum())
            iou = intersection/max(area+candidate["area"]-intersection,1)
            if coverage >= .25 and (candidate["label"] == "banana" or iou >= .30 or coverage >= .60):
                matches.append((iou,-area,float(scores[i]),i))
        if matches:
            assignments[max(matches)[-1]].append(candidate)
        else:
            unassigned.append(candidate)
    replacements = {}
    checks_left = max_crop_checks

    def verify_crop(candidate, label):
        try:
            return _crop_fruit_instance(seg_model,image,candidate,label,conf,quantize)
        except (RuntimeError,ValueError) as exc:
            diagnostics["fruit_refine_error"] = type(exc).__name__
            return None

    for i,matched in assignments.items():
        if len(matched) < 2:
            continue
        label = names[int(classes[i])]
        state = per_class.setdefault(label,{"split_groups":0,"pending_groups":0,"recovered":0})
        try:
            parts = _sam_instance_parts(mask_model,image,parents[i],matched)
        except (RuntimeError,ValueError) as exc:
            diagnostics["fruit_refine_error"] = type(exc).__name__
            parts = None
        if parts is None and seg_model is not None and checks_left >= len(matched) + max_crop_checks//2:
            checks_left -= len(matched)
            crop_parts = [verify_crop(candidate,label) for candidate in matched]
            parts = _validated_crop_parts(crop_parts,parents[i])
        if parts is None:
            state["pending_groups"] += 1
            diagnostics["fruit_split_pending"] += 1
            if label == "banana":
                diagnostics["banana_split_pending"] += 1
        else:
            state["split_groups"] += 1
            diagnostics["fruit_split_groups"] += 1
            if label == "banana":
                diagnostics["banana_split_groups"] += 1
            replacements[i] = list(zip(parts,matched))

    new_polys, new_boxes, new_scores, new_classes = [], [], [], []
    for i in range(len(boxes)):
        if i not in replacements:
            new_polys.append(polygons[i])
            new_boxes.append(boxes[i])
            new_scores.append(scores[i])
            new_classes.append(classes[i])
        else:
            for polygon,candidate in replacements[i]:
                new_polys.append(polygon)
                new_boxes.append(np.concatenate((polygon.min(axis=0),polygon.max(axis=0))))
                new_scores.append(min(float(scores[i]),candidate["conf"]))
                new_classes.append(classes[i])

    if seg_model is not None:
        class_ids = {label:next((int(c) for c in classes if names.get(int(c)) == label),
                               next((int(c) for c,name in names.items() if name == label),None)) for label in labels}
        medians = {label:float(np.median([mask.sum() for i,mask in parents.items() if names[int(classes[i])] == label]))
                   for label in labels if any(names[int(classes[i])] == label for i in parents)}
        scene_area = float(np.median([mask.sum() for mask in parents.values()]))
        for candidate in unassigned:
            label = candidate["label"]
            # Very small OBB-only "apples" among grapes must not become fruit
            # merely because a 3-class detector gives them a high score.
            if checks_left <= 0 or class_ids.get(label) is None:
                continue
            if label not in medians and ("hbb" not in candidate["sources"] or candidate["conf"] < .65):
                continue
            area_floor = medians[label]*.20 if label in medians else scene_area*.10
            if candidate["area"] < area_floor or candidate["conf"] < max(.5,conf):
                continue
            checks_left -= 1
            recovered = verify_crop(candidate,label)
            if recovered is None:
                continue
            polygon,score = recovered
            mask = np.zeros((height,width),np.uint8)
            cv2.fillPoly(mask,[np.round(polygon).astype(np.int32)],1)
            area = float(mask.sum())
            duplicate = False
            for old_polygon in new_polys:
                if len(old_polygon) < 3:
                    continue
                existing = np.zeros_like(mask)
                cv2.fillPoly(existing,[np.round(old_polygon).astype(np.int32)],1)
                other_area = float(existing.sum())
                intersection = float((mask & existing).sum())
                if intersection/max(area+other_area-intersection,1) >= .4 or intersection/max(min(area,other_area),1) >= .65:
                    duplicate = True
                    break
            if duplicate:
                continue
            new_polys.append(polygon)
            new_boxes.append(np.concatenate((polygon.min(axis=0),polygon.max(axis=0))))
            new_scores.append(score)
            new_classes.append(class_ids[label])
            diagnostics["fruit_recovered"] += 1
            per_class.setdefault(label,{"split_groups":0,"pending_groups":0,"recovered":0})["recovered"] += 1
    new_polys, new_boxes, new_scores, new_classes = deduplicate_masks(
        new_polys, np.asarray(new_boxes, np.float32).reshape(-1, 4),
        np.asarray(new_scores, np.float32), np.asarray(new_classes, np.int64),
        width, height,
    )
    return new_polys, new_boxes, new_scores, new_classes


def split_banana_cluster_detections(polygons, boxes, scores, classes, obb_model, image,
                                    conf=.35, det_names=None, quantize=None, mask_model=None, diagnostics=None):
    """Compatibility wrapper for previous banana-only experiments."""
    result = split_fruit_cluster_detections(polygons,boxes,scores,classes,obb_model,image,
        conf=conf,det_names=det_names,quantize=quantize,mask_model=mask_model,
        diagnostics=diagnostics,allowed_labels={"banana"})
    if diagnostics is not None and "fruit_refine_error" in diagnostics:
        diagnostics["banana_refine_error"] = diagnostics["fruit_refine_error"]
    return result
