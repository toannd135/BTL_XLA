"""Create reproducible crowded-banana segmentation data from reviewed labels.

This is deliberately offline: it never changes the source dataset and never
creates labels unless a source image already has segmentation polygons.
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_segments(path, width, height):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 7 or int(fields[0]) != 0:
            continue
        points = np.asarray([float(value) for value in fields[1:]], np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        rows.append(points)
    return rows


def write_segments(path, segments, width, height):
    lines = []
    for points in segments:
        normalized = points.copy()
        normalized[:, 0] /= width
        normalized[:, 1] /= height
        values = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
        lines.append(f"0 {values}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="YOLO segmentation dataset with train/images and train/labels")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    source_images = args.source / "train" / "images"
    source_labels = args.source / "train" / "labels"
    images = sorted(p for p in source_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise FileNotFoundError(f"No training images found in {source_images}")
    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read {image_path}")
        height, width = image.shape[:2]
        segments = read_segments(label_path, width, height)
        if not segments:
            continue
        for copy_index in range(args.copies + 1):
            name = f"{image_path.stem}_{copy_index:02d}"
            out_image = args.out / "train" / "images" / f"{name}.jpg"
            out_label = args.out / "train" / "labels" / f"{name}.txt"
            out_image.parent.mkdir(parents=True, exist_ok=True)
            out_label.parent.mkdir(parents=True, exist_ok=True)
            if copy_index == 0:
                augmented, output_segments = image, segments
            else:
                augmented = image.copy()
                output_segments = []
                angle = rng.uniform(-18, 18)
                scale = rng.uniform(0.85, 1.15)
                matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
                augmented = cv2.warpAffine(
                    augmented, matrix, (width, height),
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                for points in segments:
                    transformed = cv2.transform(points[None], matrix)[0]
                    if len(transformed) >= 3:
                        output_segments.append(transformed)
                if rng.random() < 0.5:
                    augmented = cv2.flip(augmented, 1)
                    output_segments = [
                        np.column_stack((width - 1 - points[:, 0], points[:, 1]))
                        for points in output_segments
                    ]
            cv2.imwrite(str(out_image), augmented, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_segments(out_label, output_segments, width, height)
    shutil.copyfile(args.source / "data.yaml", args.out / "data.yaml")
    print(f"Generated augmented banana dataset at {args.out}")


if __name__ == "__main__":
    main()
