"""Class-aware one-to-one OBB matching and per-class count metrics."""
from collections import Counter
import numpy as np

NAMES = ("banana", "apple", "orange")


def match_boxes(overlap, gt_classes, pred_classes, threshold=.5):
    overlap = np.asarray(overlap, dtype=np.float32)
    gt_classes = np.asarray(gt_classes, dtype=np.int64)
    pred_classes = np.asarray(pred_classes, dtype=np.int64)
    assert overlap.shape == (len(gt_classes), len(pred_classes))
    if not len(gt_classes) or not len(pred_classes):
        return []
    valid = (overlap >= threshold) & (gt_classes[:, None] == pred_classes[None, :])
    # Augmenting paths maximize the number of valid one-to-one matches. Visit
    # high-overlap edges first, without claiming globally optimal summed overlap.
    edges = [sorted(np.flatnonzero(valid[i]).tolist(), key=lambda j: -float(overlap[i, j])) for i in range(len(gt_classes))]
    owner = {}

    def augment(gt, visited):
        for pred in edges[gt]:
            if pred in visited:
                continue
            visited.add(pred)
            if pred not in owner or augment(owner[pred], visited):
                owner[pred] = gt
                return True
        return False

    for gt in sorted(range(len(gt_classes)), key=lambda i: -max((float(overlap[i, j]) for j in edges[i]), default=0)):
        augment(gt, set())
    return sorted((int(gt), int(pred)) for pred, gt in owner.items())


def summarize(rows):
    classes = {}
    for cls, name in enumerate(NAMES):
        tp = sum(r["tp"][cls] for r in rows)
        fp = sum(r["fp"][cls] for r in rows)
        fn = sum(r["fn"][cls] for r in rows)
        p, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        errors = [abs(r["truth_counts"][cls] - r["pred_counts"][cls]) for r in rows]
        present = [i for i, r in enumerate(rows) if r["truth_counts"][cls] > 0]
        classes[name] = {"TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": recall,
                         "F1": 2 * p * recall / max(p + recall, 1e-12),
                         "count_MAE_all_images": float(np.mean(errors)) if errors else None,
                         "count_MAE_when_present": float(np.mean([errors[i] for i in present])) if present else None,
                         "exact_count_rate_all_images": float(np.mean([e == 0 for e in errors])) if errors else None,
                         "images_with_truth": len(present)}
    errors_all = [sum(abs(g - p) for g, p in zip(r["truth_counts"], r["pred_counts"])) for r in rows]
    return {"images": len(rows), "per_class": classes,
            "macro_F1": sum(c["F1"] for c in classes.values()) / 3,
            "count_MAE_all_classes": float(np.mean(errors_all)) if rows else None,
            "exact_all_classes": float(np.mean([e == 0 for e in errors_all])) if rows else None}


def evaluate_counts(model, root, records, imgsz, batch=2):
    import torch
    from ultralytics.utils.metrics import batch_probiou
    from ultralytics.utils.ops import xyxyxyxy2xywhr
    from PIL import Image
    rows = []
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        results = model.predict([str(root / r["image"]) for r in chunk], imgsz=imgsz, batch=batch,
                                conf=.4, iou=.7, agnostic_nms=True, device=0,
                                max_det=300, verbose=False)
        for record, result in zip(chunk, results):
            with Image.open(root / record["image"]) as image:
                w, h = image.size
            gt = [list(map(float, ln.split())) for ln in (root / record["label"]).read_text().splitlines() if ln.strip()]
            gc = [int(v[0]) for v in gt]
            points = np.asarray([v[1:] for v in gt], dtype=np.float32).reshape(-1, 4, 2)
            points *= np.asarray([w, h], dtype=np.float32)
            xywhr = xyxyxyxy2xywhr(points).astype(np.float32) if len(gt) else np.zeros((0, 5), np.float32)
            pred = result.obb
            pc = [int(c) for c in pred.cls.tolist()]
            overlap = batch_probiou(torch.from_numpy(xywhr), pred.xywhr.cpu()).numpy() if len(gc) and len(pc) else np.zeros((len(gc), len(pc)))
            matched = match_boxes(overlap, gc, pc)
            gt_count, pred_count = Counter(gc), Counter(pc)
            tp = Counter(gc[g] for g, _ in matched)
            rows.append({"image": record["image"], "source": record["source"],
                         "truth_counts": [gt_count[c] for c in range(3)], "pred_counts": [pred_count[c] for c in range(3)],
                         "tp": [tp[c] for c in range(3)], "fp": [pred_count[c] - tp[c] for c in range(3)],
                         "fn": [gt_count[c] - tp[c] for c in range(3)],
                         "predictions": [{"class": int(c), "score": float(s), "corners": p}
                                         for c, s, p in zip(pc, pred.conf.tolist(), pred.xyxyxyxy.cpu().tolist())]})
        if start % 50 < batch:
            print(f"Count evaluation: {min(start + batch, len(records))}/{len(records)}", flush=True)
    return {"summary": summarize(rows), "rows": rows}


def save_previews(root, records, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image
    lookup = {r["image"]: r for r in records}
    # Worst two count errors for every class; union avoids favoring bananas.
    selected = set()
    for cls in range(3):
        ordered = sorted(range(len(rows)), key=lambda i: abs(rows[i]["truth_counts"][cls] - rows[i]["pred_counts"][cls]), reverse=True)
        selected.update(ordered[:2])
    output.mkdir(exist_ok=True, parents=True)
    colors = ("#2563eb", "#dc2626", "#d97706")
    for index in sorted(selected):
        row = rows[index]
        record = lookup[row["image"]]
        with Image.open(root / record["image"]) as im:
            image = im.convert("RGB")
        gt = [list(map(float, ln.split())) for ln in (root / record["label"]).read_text().splitlines() if ln.strip()]
        truth = [{"class": int(v[0]), "corners": (np.array(v[1:]).reshape(4, 2) * image.size).tolist()} for v in gt]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=110)
        for ax, boxes, title in zip(axes, (truth, row["predictions"]), ("Legacy/reviewed labels", "Predictions conf .40")):
            ax.imshow(image)
            for j, box in enumerate(boxes):
                points = np.asarray(box["corners"])
                cls = box["class"]
                ax.add_patch(Polygon(points, closed=True, fill=False, edgecolor=colors[cls], linewidth=1.5))
                ax.text(*points.mean(0), f"{j + 1}:{NAMES[cls]}", fontsize=7, color="white",
                        bbox={"facecolor": colors[cls], "pad": 1, "alpha": .8})
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"GT {row['truth_counts']} / predicted {row['pred_counts']} (banana, apple, orange)")
        fig.tight_layout()
        fig.savefig(output / f"scene_{index:04d}.png")
        plt.close(fig)
