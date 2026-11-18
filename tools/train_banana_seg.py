"""Train an offline banana-only instance-segmentation candidate.

The source dataset must contain reviewed YOLO segmentation polygons. This
script never changes production weights or model configuration.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="data.yaml for a banana-only segmentation dataset")
    parser.add_argument("--base", type=Path, default=ROOT / "yolo11s-seg.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--name", default="banana_seg_candidate")
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if not args.base.is_file():
        raise FileNotFoundError(args.base)
    model = YOLO(str(args.base))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=42,
        deterministic=False,
        degrees=20,
        translate=0.10,
        scale=0.35,
        shear=4,
        perspective=0.0005,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.25,
        close_mosaic=15,
        patience=20,
        workers=2,
        project=str(ROOT / "runs" / "banana_seg"),
        name=args.name,
        exist_ok=False,
        plots=True,
    )


if __name__ == "__main__":
    main()
