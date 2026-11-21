from pathlib import Path
import cv2
import numpy as np

ROOT = Path("work/obb/seg_data_v2")
OUT = Path("work/obb/seg_data_v2_review")
OUT.mkdir(parents=True, exist_ok=True)

samples = [
    "mbk__apple_24",
    "mbk__banana_66",
    "mbk__orange_1",
]

names = {0: "banana", 1: "apple", 2: "orange"}
colors = {
    0: (0, 255, 255),
    1: (0, 165, 255),
    2: (0, 140, 0),
}

for stem in samples:
    image_path = ROOT / "train/images" / f"{stem}.jpg"
    label_path = ROOT / "train/labels" / f"{stem}.txt"

    if not image_path.exists():
        candidates = list((ROOT / "train/images").glob(stem + ".*"))
        if not candidates:
            print("IMAGE NOT FOUND:", stem)
            continue
        image_path = candidates[0]

    image = cv2.imread(str(image_path))
    if image is None:
        print("CANNOT READ:", image_path)
        continue

    overlay = image.copy()
    h, w = image.shape[:2]
    counts = {name: 0 for name in names.values()}

    for idx, line in enumerate(label_path.read_text().splitlines(), 1):
        parts = line.split()
        cls = int(parts[0])
        coords = np.array(list(map(float, parts[1:])), dtype=np.float32)
        points = coords.reshape(-1, 2)
        points[:, 0] *= w
        points[:, 1] *= h
        points = np.round(points).astype(np.int32)

        color = colors[cls]
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(overlay, [points], True, color, 2)

        x, y = points.mean(axis=0).astype(int)
        cv2.putText(
            overlay, f"{names[cls]} #{idx}",
            (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2
        )
        counts[names[cls]] += 1

    result = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
    output_path = OUT / f"{stem}_review.jpg"
    cv2.imwrite(str(output_path), result)

    print(f"\n{stem}: {sum(counts.values())} instances")
    print("Class counts:", counts)
    print("Saved:", output_path)

print("\nReview images:", OUT.resolve())
