"""Quality checks shared by the API and the offline count benchmark."""


def assess_detection(objects, width, height, diagnostics=None, conf_threshold=0.4):
    reasons = []
    automatic_count_allowed = True
    image_area = max(width * height, 1)
    for index, obj in enumerate(objects):
        label = obj["label"]
        x1, y1, x2, y2 = obj["bbox"]
        box_area = max(x2-x1, 0) * max(y2-y1, 0)
        area = obj["mask_area"] if obj.get("mask_valid") else box_area
        confidence = obj.get("confidence", 1.0)
        if confidence < conf_threshold and not obj.get("rescued"):
            reasons.append(f"quả #{index+1} ({label}) có độ tin cậy thấp ({confidence:.2f} < {conf_threshold})")
        if area / image_area > 0.60:
            reasons.append(f"vùng #{index+1} quá lớn ({area/image_area:.0%}), cần kiểm tra số quả")
        if obj.get("mask_valid") is False:
            reasons.append(f"quả #{index+1} ({label}) chưa có mask hợp lệ, đang hiển thị khung bao")
        if obj.get("mask_valid") and label in ("apple", "orange") and obj.get("solidity", 1.0) < 0.65:
            reasons.append(f"quả #{index+1} ({label}) có biên lõm nhiều, cần kiểm tra mask")
        overlap = obj.get("overlap_ratio", obj.get("occlusion_ratio", 0.0))
        if overlap >= 0.55:
            reasons.append(f"mask quả #{index+1} ({label}) chồng lấp với mask khác ({overlap:.0%})")

    for i, first in enumerate(objects):
        for j in range(i+1, len(objects)):
            second = objects[j]
            if first.get("mask_valid") and second.get("mask_valid"):
                iou = next((item["iou"] for item in first.get("mask_overlaps", [])
                            if item["other_idx"] == j), 0.0)
                shape = "mask"
            else:
                a, b = first["bbox"], second["bbox"]
                intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
                area_a = max(a[2]-a[0], 0) * max(a[3]-a[1], 0)
                area_b = max(b[2]-b[0], 0) * max(b[3]-b[1], 0)
                iou = intersection / max(area_a+area_b-intersection, 1e-9)
                shape = "khung"
            same_class = first["label"] == second["label"]
            if iou >= (0.40 if same_class else 0.50):
                reasons.append(f"hai {shape} #{i+1} và #{j+1} chồng lấp lớn (IoU {iou:.0%}), cần kiểm tra trùng quả")

    if diagnostics:
        names = {"banana":"chuối", "apple":"táo", "orange":"cam"}
        detail = diagnostics.get("instance_refinement", {})
        for label, state in detail.items():
            name = names.get(label,label)
            if state.get("split_groups",0):
                reasons.append(f"vùng {name} vừa được tách; cần đối chiếu biên và số quả")
            if state.get("pending_groups",0):
                reasons.append(f"có vùng {name} nghi gộp nhiều quả; chưa xác định được biên từng quả, cần duyệt")
                if label == "banana":
                    automatic_count_allowed = False
            if state.get("recovered",0):
                reasons.append(f"đã bổ sung {state['recovered']} quả {name} từ vùng kiểm tra lại; cần đối chiếu")
        if not detail:
            if diagnostics.get("banana_split_groups",0):
                reasons.append("vùng chuối vừa được tách; cần đối chiếu biên và số quả")
            if diagnostics.get("banana_split_pending",0):
                reasons.append("có vùng chuối nghi gộp nhiều quả; chưa xác định được biên từng quả, cần duyệt")
                automatic_count_allowed = False
        if diagnostics.get("fruit_refine_error", diagnostics.get("banana_refine_error")):
            reasons.append("bước tinh chỉnh biên chưa hoàn tất; đã giữ mask gốc")
        if diagnostics.get("fallback_used"):
            reasons.append("mô hình kiểm chứng cho kết quả khác biệt, cần đối chiếu số lượng")
        primary, fallback = diagnostics.get("primary_count"), diagnostics.get("fallback_count")
        discrepancy = 1 if diagnostics.get("primary_task") == "segment" else 2
        if primary is not None and fallback is not None and abs(primary-fallback) >= discrepancy:
            reasons.append(f"số lượng quả giữa các mô hình lệch nhau (Detector chính: {primary}, Kiểm chứng: {fallback})")

    reasons = list(dict.fromkeys(reasons))
    if diagnostics and diagnostics.get("banana_split_pending", 0):
        automatic_count_allowed = False
    return {
        "status": "needs_review" if reasons else "ready",
        "is_uncertain": bool(reasons),
        "automatic_count_allowed": automatic_count_allowed,
        "decision": "count" if automatic_count_allowed else "do_not_auto_count",
        "reasons": reasons,
    }
