"""Train a candidate YOLO segmentation model without changing production.

The generated dataset contains SAM-derived polygons and synthetic crowded
bananas. It must be reviewed before a candidate can replace the API detector.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/obb/seg_data_clean/seg.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pseudo-labels", action="store_true",
                        help="Explicitly allow training on the current unreviewed polygons")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="seg_candidate")
    args = parser.parse_args()
    if not args.allow_pseudo_labels:
        raise SystemExit(
            "Refusing to train: segmentation polygons are not manually reviewed. "
            "Use --allow-pseudo-labels only for an offline candidate."
        )
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    model = YOLO(str(ROOT / "yolo11s-seg.pt"))
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=42,
        deterministic=False,
        degrees=15,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        patience=12,
        workers=2,
        project=str(ROOT / "runs/seg"),
        name=args.name,
        exist_ok=False,
        plots=True,
    )


if __name__ == "__main__":
    main()
