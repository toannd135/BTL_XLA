"""Heuristic review signals; none of these signals proves an instance error."""


def assess_instances(instances, image_width, image_height, *, min_confidence=0.4):
    reasons = []
    image_area = max(int(image_width) * int(image_height), 1)
    for index, instance in enumerate(instances, start=1):
        if hasattr(instance, "confidence"):
            confidence = instance.confidence
            polygon = instance.polygon
            area = instance.attributes.get("mask_area", 0)
        else:
            confidence = instance.get("confidence", 1.0)
            polygon = instance.get("polygon")
            area = instance.get("mask_area", 0)
        if confidence < min_confidence:
            reasons.append({"instance_id": index, "signal": "low_confidence", "value": round(float(confidence), 4)})
        if area / image_area > 0.60:
            reasons.append({"instance_id": index, "signal": "large_mask", "value": round(area / image_area, 4)})
        if polygon is not None and len(polygon) < 3:
            reasons.append({"instance_id": index, "signal": "invalid_mask", "value": len(polygon)})
    return {"status": "needs_review" if reasons else "ready",
            "signals_are_heuristics": True, "signals": reasons}

