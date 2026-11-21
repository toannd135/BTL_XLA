from pathlib import Path
from collections import Counter
import cv2
import numpy as np

ROOT = Path("work/obb/seg_data_v2")
OLD = Path("work/obb/seg_data_candidate/train/labels")
NEW = ROOT / "train/labels"
IMAGES = ROOT / "train/images"
OUT = Path("work/obb/seg_data_v2_count_review")
OUT.mkdir(parents=True, exist_ok=True)

NAMES = {0: "banana", 1: "apple", 2: "orange"}
COLORS = {0: (0, 255, 255), 1: (0, 165, 255), 2: (0, 140, 0)}

def counts(path):
    result = Counter()
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if p:
                result[int(p[0])] += 1
    return result

def draw(image, label_path):
    canvas = image.copy()
    h, w = image.shape[:2]

    if label_path.exists():
        for i, line in enumerate(label_path.read_text().splitlines(), 1):
            p = line.split()
            if not p:
                continue

            cls = int(p[0])
            xy = np.array(list(map(float, p[1:])), dtype=np.float32)
            points = np.round(xy.reshape(-1, 2) * [w, h]).astype(np.int32)
            color = COLORS[cls]

            cv2.fillPoly(canvas, [points], color)
            cv2.polylines(canvas, [points], True, color, 2)

            x, y = points.mean(axis=0).astype(int)
            cv2.putText(canvas, str(i), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

    return cv2.addWeighted(image, 0.55, canvas, 0.45, 0)

changed = []

for new_file in sorted(NEW.glob("*.txt")):
    old_file = OLD / new_file.name
    if counts(old_file) != counts(new_file):
        changed.append((new_file, old_file))

print("Images requiring count review:", len(changed))

for new_file, old_file in changed:
    stem = new_file.stem
    candidates = list(IMAGES.glob(stem + ".*"))

    if not candidates:
        print("IMAGE NOT FOUND:", stem)
        continue

    image = cv2.imread(str(candidates[0]))
    if image is None:
        print("CANNOT READ:", stem)
        continue

    old_counts = counts(old_file)
    new_counts = counts(new_file)

    left = draw(image, old_file)
    right = draw(image, new_file)

    h, w = image.shape[:2]
    header_h = 65
    comparison = np.zeros((h + header_h, w * 2, 3), dtype=np.uint8)

    comparison[header_h:, :w] = left
    comparison[header_h:, w:] = right

    cv2.putText(comparison, "OLD: " + str(dict(old_counts)),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "NEW: " + str(dict(new_counts)),
                (w + 10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    output = OUT / f"{stem}_comparison.jpg"
    cv2.imwrite(str(output), comparison)

    print(stem, "OLD:", dict(old_counts),
          "NEW:", dict(new_counts), "->", output)

print("\nOutput directory:", OUT.resolve())
