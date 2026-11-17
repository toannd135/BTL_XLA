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
    keep = np.asarray(keep, dtype=np.int64)
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


from scipy.ndimage import gaussian_filter1d


def smooth_contour_curve(pts, sigma=1.8, max_points=100):
    """Smooth closed 2D polygon contour coordinates using Gaussian filtering along parameter curve.
    Removes proto-mask 160x160 scaling staircase artifacts and sharp jagged corners,
    producing an organic, visually pleasing fruit outline while preserving real area.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 6:
        return pts
    x = gaussian_filter1d(pts[:, 0], sigma=sigma, mode="wrap")
    y = gaussian_filter1d(pts[:, 1], sigma=sigma, mode="wrap")
    smoothed = np.column_stack([x, y])
    if len(smoothed) > max_points:
        step = len(smoothed) / float(max_points)
        idx = [int(i * step) for i in range(max_points)]
        smoothed = smoothed[idx]
    return np.round(smoothed, 1)


def simplify_polygon(poly, epsilon=1.5):
    """Simplify polygon contour points to keep JSON payload compact and smooth."""
    pts = np.asarray(poly, dtype=np.float32)
    if len(pts) < 3:
        return pts
    approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), epsilon=epsilon, closed=True).reshape(-1, 2)
    return approx if len(approx) >= 3 else pts


def compute_mask_metrics(polygons, boxes, width, height):
    """Calculate mask area, solidity, and pairwise occlusion/overlap between instances."""
    n = len(polygons)
    if n == 0:
        return []
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(n)]
    for i, p in enumerate(polygons):
        if len(p) >= 3:
            cv2.fillPoly(masks[i], [np.round(p).astype(np.int32)], 1)
    areas = [float(m.sum()) for m in masks]
    metrics = []
    for i in range(n):
        p = polygons[i]
        b = boxes[i]
        area = areas[i]
        solidity = 1.0
        if len(p) >= 3 and area > 0:
            cnt = np.asarray(p, dtype=np.float32).reshape(-1, 1, 2)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(min(1.0, max(0.0, cv2.contourArea(cnt) / hull_area)))
        max_overlap = 0.0
        overlap_details = []
        for j in range(n):
            if i == j:
                continue
            bj = boxes[j]
            if max(b[0], bj[0]) < min(b[2], bj[2]) and max(b[1], bj[1]) < min(b[3], bj[3]):
                inter = float((masks[i] & masks[j]).sum())
                if inter > 0 and area > 0:
                    overlap_ratio = inter / area
                    if overlap_ratio > max_overlap:
                        max_overlap = overlap_ratio
                    overlap_details.append({
                        "other_idx": j,
                        "overlap_ratio": round(float(overlap_ratio), 3),
                        "intersection_px": int(inter),
                    })
        metrics.append({
            "mask_area": int(area),
            "solidity": round(float(solidity), 3),
            "occlusion_ratio": round(float(max_overlap), 3),
            "is_occluded": bool(max_overlap >= 0.25),
            "overlaps": overlap_details,
        })
    return metrics


def detect_seg(model, image, conf=0.4, imgsz=640, iou=0.6, quantize=None, simplify_eps=1.5):
    """YOLO Instance Segmentation.
    Runs prediction and returns (polygons, boxes, scores, classes, mask_metrics).
    Handles both custom 3-class fruit models (banana/apple/orange) and 80-class COCO seg models.
    """
    height, width = image.shape[:2]
    classes_arg = None
    coco_fruit_ids = {46: 0, 47: 1, 49: 2}
    is_coco = False
    if len(model.names) >= 80 and 46 in model.names and 47 in model.names:
        is_coco = True
        classes_arg = [46, 47, 49]

    result = model.predict(
        image, conf=conf, imgsz=imgsz, quantize=quantize,
        agnostic_nms=True, iou=iou, max_det=300, verbose=False,
        classes=classes_arg,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return [], np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64), []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = np.asarray(result.boxes.conf.tolist(), dtype=np.float32)
    raw_classes = np.asarray(result.boxes.cls.tolist(), dtype=np.int64)
    classes = raw_classes

    polygons = []
    if result.masks is not None and len(result.masks.xy) > 0:
        for poly in result.masks.xy:
            if len(poly) >= 3:
                # Apply smooth Gaussian curve FIRST to remove pixel-grid staircase artifacts
                # from YOLO's 160x160 proto-mask upscaling, then simplify
                smooth = smooth_contour_curve(poly, sigma=1.8, max_points=120)
                simp = simplify_polygon(smooth, epsilon=simplify_eps)
                simp[:, 0] = np.clip(simp[:, 0], 0, width)
                simp[:, 1] = np.clip(simp[:, 1], 0, height)
                polygons.append(simp)
            else:
                polygons.append(np.zeros((0, 2), dtype=np.float32))
    else:
        for b in boxes:
            polygons.append(np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], dtype=np.float32))

    mask_metrics = compute_mask_metrics(polygons, boxes, width, height)
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


def split_banana_cluster_detections(polygons, boxes, scores, classes, obb_model, image,
                                    conf=0.35, det_names=None, quantize=None):
    """Partitions grouped banana cluster masks into individual banana instances using OBB guidance.

    Pipeline:
    1. Run OBB detector (det_obb_s.pt) to find individual banana orientations.
    2. Deduplicate OBB bananas by mask IoU (keeps distinct non-overlapping boxes only).
    3. For each seg banana mask overlapping >= 2 OBB bananas:
       - Assign each foreground pixel to the nearest OBB spine (Voronoi partitioning).
       - Morphological close + Gaussian blur smoothing on each sub-mask.
       - Extract clean smoothed contour polygon for each separated banana.
    4. Non-banana objects are passed through unchanged.
    """
    if len(boxes) == 0 or obb_model is None:
        return polygons, boxes, scores, classes

    height, width = image.shape[:2]
    banana_ids = {k for k, v in det_names.items() if v == "banana"} if det_names else {0, 46}
    obb_banana_ids = {k for k, v in obb_model.names.items() if v == "banana"}

    # Run OBB prediction to detect individual banana orientations
    obb_res = obb_model.predict(
        image, conf=conf, imgsz=1280, quantize=quantize,
        agnostic_nms=True, iou=0.5, verbose=False,
    )[0]
    if not hasattr(obb_res, "obb") or obb_res.obb is None or len(obb_res.obb) == 0:
        return polygons, boxes, scores, classes

    # Collect all OBB banana candidates sorted by confidence (highest first)
    raw_obb = []
    for cls, c_conf, poly in zip(obb_res.obb.cls, obb_res.obb.conf, obb_res.obb.xyxyxyxy):
        if int(cls) in obb_banana_ids:
            pts = poly.cpu().numpy()
            b_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(b_mask, [np.round(pts).astype(np.int32)], 1)
            raw_obb.append({"conf": float(c_conf), "pts": pts, "mask": b_mask, "area": float(b_mask.sum())})
    raw_obb.sort(key=lambda x: -x["conf"])

    # Deduplicate: remove OBBs that overlap heavily with a higher-confidence OBB
    obb_bananas = []
    for ob in raw_obb:
        dup = False
        for kept in obb_bananas:
            inter = float((ob["mask"] & kept["mask"]).sum())
            union = float((ob["mask"] | kept["mask"]).sum())
            iou = inter / max(union, 1.0)
            overlap_smaller = inter / max(min(ob["area"], kept["area"]), 1.0)
            if iou >= 0.35 or overlap_smaller >= 0.50:
                dup = True
                break
        if not dup:
            ob["spine"] = get_obb_spine(ob["pts"])
            obb_bananas.append(ob)

    if len(obb_bananas) <= 1:
        return polygons, boxes, scores, classes

    new_polys, new_boxes, new_scores, new_classes = [], [], [], []
    default_banana_cls = next(iter(banana_ids))

    for idx in range(len(boxes)):
        c = int(classes[idx])
        poly = polygons[idx] if idx < len(polygons) else None
        b = boxes[idx]
        sc = scores[idx]

        if c not in banana_ids or poly is None or len(poly) < 3:
            new_polys.append(poly)
            new_boxes.append(b)
            new_scores.append(sc)
            new_classes.append(c)
            continue

        # Render the segmentation banana mask
        cur_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(cur_mask, [np.round(poly).astype(np.int32)], 1)

        # Match OBB bananas that overlap significantly with this seg mask
        matched = []
        for ob in obb_bananas:
            inter = int((cur_mask & ob["mask"]).sum())
            if ob["area"] > 0 and inter / ob["area"] >= 0.15:
                matched.append(ob)

        if len(matched) <= 1:
            # No splitting needed — pass through unchanged
            new_polys.append(poly)
            new_boxes.append(b)
            new_scores.append(sc)
            new_classes.append(c)
            continue

        # --- Spine-distance Voronoi partitioning ---
        y_idxs, x_idxs = np.where(cur_mask > 0)
        coords = np.column_stack([x_idxs, y_idxs]).astype(np.float32)
        dists = []
        for ob in matched:
            s1, s2 = ob["spine"]
            dists.append([dist_to_segment(p, s1, s2) for p in coords])
        assigned = np.argmin(np.array(dists), axis=0)

        for m_idx, ob in enumerate(matched):
            sub_mask = np.zeros((height, width), dtype=np.uint8)
            sub_mask[y_idxs[assigned == m_idx], x_idxs[assigned == m_idx]] = 255

            # Morphological close to fill seam gaps, then Gaussian blur to smooth jagged boundary
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_CLOSE, kernel)
            sub_mask = cv2.GaussianBlur(sub_mask, (7, 7), 2.0)
            _, sub_mask = cv2.threshold(sub_mask, 127, 255, cv2.THRESH_BINARY)

            cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) < 100:
                continue

            # Smooth the contour with Gaussian curve to remove staircase artifacts
            smooth = smooth_contour_curve(cnt.reshape(-1, 2), sigma=2.0, max_points=100)
            smooth[:, 0] = np.clip(smooth[:, 0], 0, width)
            smooth[:, 1] = np.clip(smooth[:, 1], 0, height)
            x, y, bw, bh = cv2.boundingRect(cnt)
            new_polys.append(smooth)
            new_boxes.append(np.array([x, y, x + bw, y + bh], dtype=np.float32))
            new_scores.append(float(ob["conf"]))
            new_classes.append(default_banana_cls)

    return (
        new_polys,
        np.array(new_boxes, dtype=np.float32) if len(new_boxes) else np.zeros((0, 4), dtype=np.float32),
        np.array(new_scores, dtype=np.float32) if len(new_scores) else np.zeros(0, dtype=np.float32),
        np.array(new_classes, dtype=np.int64) if len(new_classes) else np.zeros(0, dtype=np.int64),
    )


