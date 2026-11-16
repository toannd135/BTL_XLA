"""Bounded, read-only CPU inference on the four reviewed validation scenes."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REPORT_SOURCE = Path(r"C:\Users\LENOVO\Downloads\comparison_val.json")
MODELS = {"baseline": REPO / "exports/det_obb_s.pt",
          "candidate": REPO / "exports/det_obb_trial_v1.pt"}
COLORS = ["#2563eb", "#9333ea", "#e11d48", "#059669", "#d97706", "#0891b2"]
torch.set_num_threads(2)
torch.set_num_interop_threads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, choices=(640, 1280), default=640)
    args = parser.parse_args()
    submitted = json.loads(REPORT_SOURCE.read_text(encoding="utf-8-sig"))
    for row in submitted["results"]:
        assert sha(MODELS[row["model"]]) == row["sha256"], row["model"]
    manifest = json.loads((HERE / "input_manifest.json").read_text(encoding="utf-8"))
    records = [r for r in manifest["files"] if r["origin"] == "submitted_review_geometry_padded" and r["split"] == "val"]
    assert len(records) == 4
    output = HERE / ("comparison_750cadad" + ("_1280" if args.imgsz == 1280 else ""))
    output.mkdir(exist_ok=False)
    (output / "submitted_comparison_val.json").write_text(json.dumps(submitted, indent=2), encoding="utf-8")
    with zipfile.ZipFile(HERE / "fruit_accuracy_trial_20261116.zip") as archive:
        for record in records:
            for key in ("image", "label"):
                data = archive.read(record[key])
                assert hashlib.sha256(data).hexdigest() == record[key + "_sha256"]
                target = output / "inputs" / record[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)

    rows = []
    for record in records:
        image_path = output / "inputs" / record["image"]
        label_path = output / "inputs" / record["label"]
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        gt = np.array([list(map(float, line.split())) for line in label_path.read_text().splitlines() if line.strip()], dtype=np.float32)
        corners = gt[:, 1:].reshape(-1, 4, 2) * np.array([w, h], dtype=np.float32)
        rows.append({"source": record["source"], "image_path": str(image_path),
                     "truth": [{"class": int(c), "corners": p.tolist()} for c, p in zip(gt[:, 0], corners)],
                     "models": {}})
    for name, path in MODELS.items():
        model = YOLO(str(path))
        assert model.task == "obb" and model.names == {0: "banana", 1: "apple", 2: "orange"}
        for i, row in enumerate(rows):
            started = time.perf_counter()
            result = model.predict(row["image_path"], imgsz=args.imgsz, conf=.1, iou=.7,
                                   device="cpu", half=False, agnostic_nms=True, max_det=300, verbose=False)[0]
            boxes = result.obb
            predictions = [{"class": int(c), "score": float(s), "corners": p.tolist(), "xywhr": b.tolist()}
                           for c, s, p, b in zip(boxes.cls.tolist(), boxes.conf.tolist(),
                                                boxes.xyxyxyxy.cpu().numpy(), boxes.xywhr.cpu().numpy())]
            selected = [p for p in predictions if p["score"] >= .4]
            counts = {str(t): dict(Counter(p["class"] for p in predictions if p["score"] >= t)) for t in (.4, .25, .1)}
            gtc = np.array([g["corners"] for g in row["truth"]], dtype=np.float32)
            gtbox = torch.from_numpy(xyxyxyxy2xywhr(gtc).astype(np.float32))
            if selected:
                overlaps = batch_probiou(gtbox, torch.tensor([p["xywhr"] for p in selected], dtype=torch.float32)).numpy()
            else:
                overlaps = np.empty((len(gtbox), 0))
            row["models"][name] = {"sha256": sha(path), "elapsed_seconds": time.perf_counter() - started,
                                  "counts_by_threshold": counts, "predictions_at_conf_0_1": predictions,
                                  "gt_vs_prediction_probIoU_at_conf_0_4": overlaps.tolist()}
            expected = submitted["results"][0 if name == "baseline" else 1]["reviewed_banana_val"]["rows"][i]
            assert row["source"] == expected["source"]
            local_counts = {str(k): v for k, v in Counter(p["class"] for p in selected).items()}
            row["models"][name]["reproduces_colab_count"] = local_counts == expected["predicted"] if args.imgsz == 640 else None
            print(name, i + 1, "imgsz", args.imgsz, counts, flush=True)
        del model

    overview, overview_axes = plt.subplots(4, 3, figsize=(15, 18), dpi=130)
    for i, row in enumerate(rows):
        image = Image.open(row["image_path"]).convert("RGB")
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=130)
        sets = [("Reviewed labels", row["truth"])]
        for name in MODELS:
            sets.append((name, [p for p in row["models"][name]["predictions_at_conf_0_1"] if p["score"] >= .4]))
        for axlist in (axes, overview_axes[i]):
            for axis, (name, boxes) in zip(axlist, sets):
                axis.imshow(image)
                for j, box in enumerate(boxes):
                    points = np.array(box["corners"])
                    color = COLORS[j % len(COLORS)]
                    axis.add_patch(Polygon(points, closed=True, fill=False, edgecolor=color, linewidth=2))
                    label = f"#{j + 1}" + (f" {box['score']:.2f}" if "score" in box else "")
                    axis.text(*points.mean(axis=0), label, color="white", fontsize=9, ha="center",
                              bbox={"facecolor": color, "alpha": .85, "pad": 2, "edgecolor": "none"})
                axis.set_title(f"Scene {i + 1} | {name}: {len(boxes)} boxes | imgsz{args.imgsz}", fontsize=10)
                axis.axis("off")
        fig.tight_layout()
        fig.savefig(output / f"scene_{i + 1}.png")
        plt.close(fig)
    overview.tight_layout()
    overview.savefig(output / "overview.png")
    plt.close(overview)
    report = {"protocol": f"CPU 2 threads; raw OBB; imgsz{args.imgsz}; agnostic NMS iou.7; proposals conf.1; displayed/evaluated conf.4. No train or app model changes.",
              "thresholds_are_development_diagnostics_not_a_selected_setting": True,
              "rows": rows}
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved:", output, flush=True)


if __name__ == "__main__":
    main()
