"""Prepare a versioned, non-destructive rotation experiment; never train locally."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import zipfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_boxes(path):
    """Normalize legacy polygon rows to enclosing detection boxes on the train copy only."""
    boxes = []
    polygons = 0
    for row in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not row.strip():
            continue
        values = list(map(float, row.split()))
        if len(values) < 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Invalid label: {path}")
        cls, *coords = values
        if not cls.is_integer() or int(cls) not in range(3) or not all(0 <= x <= 1 for x in coords):
            raise ValueError(f"Invalid class or coordinate: {path}")
        if len(coords) == 4:
            x, y, width, height = coords
        else:
            if len(coords) < 6 or len(coords) % 2:
                raise ValueError(f"Invalid polygon: {path}")
            xs, ys = coords[::2], coords[1::2]
            x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            width, height = max(xs) - min(xs), max(ys) - min(ys)
            polygons += 1
        if width <= 0 or height <= 0:
            raise ValueError(f"Empty box: {path}")
        boxes.append((int(cls), x, y, width, height))
    return boxes, polygons


def write_boxes(path, boxes):
    Path(path).write_text("".join(f"{c} {x:.7f} {y:.7f} {w:.7f} {h:.7f}\n"
                                 for c, x, y, w, h in boxes), encoding="utf-8")


def quarter_turn(boxes, turns):
    transformed = []
    for c, x, y, w, h in boxes:
        for _ in range(turns % 4):
            x, y, w, h = 1 - y, x, h, w
        transformed.append((c, x, y, w, h))
    return transformed


def prepare(repo, sources, output, rotations=180, synthetic=240):
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    rng = random.Random(42)
    if output.exists():
        raise ValueError(f"Output already exists. Choose a new version directory: {output}")
    output.mkdir(parents=True)
    dataset = output / "dataset"
    for split in ("train", "val", "val_orientation", "test", "synthetic", "neg_test"):
        for kind in ("images", "labels"):
            (dataset / split / kind).mkdir(parents=True)
    quarantine = set((sources / "work/audit/quarantine_train.txt").read_text().split())
    report = {"seed": 42, "excluded_quarantine": [], "excluded_empty_banana_group": [],
              "converted_train_polygon_rows": 0, "rotated_sources": [], "train_files": [],
              "draft_synthetic_files": [], "source_hashes": {}, "split_counts": {},
              "policy": "Only train copies modified. Original val/test retained. User diagnostic images not used in training.",
              "synthetic_review_approved": False,
              "synthetic_status": "Draft only: excluded from training. A curated replacement with verified labels is required."}

    def load_image(path):
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        return image

    def save_image(path, image):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError(f"Cannot encode {path}")
        encoded.tofile(path)

    positive_pool = {0: [], 1: [], 2: []}
    original_positives = []
    for split in ("train", "val", "test"):
        folder = sources / "local_dataset/det" / split
        for image in sorted((folder / "images").glob("*.jpg")):
            label = folder / "labels" / (image.stem + ".txt")
            if split == "train" and image.name in quarantine:
                report["excluded_quarantine"].append(image.name)
                continue
            boxes, polygons = read_boxes(label)
            # Verified missed-label group, identified by source image; do not relabel it automatically.
            if split == "train" and image.name.startswith("rf_v7__b_f011_") and not boxes:
                report["excluded_empty_banana_group"].append(image.name)
                continue
            dest_image = dataset / split / "images" / image.name
            dest_label = dataset / split / "labels" / label.name
            shutil.copy2(image, dest_image)
            if split == "train":
                write_boxes(dest_label, boxes)
                report["converted_train_polygon_rows"] += polygons
                report["train_files"].append(dest_image.relative_to(output).as_posix())
                if boxes:
                    original_positives.append((dest_image, boxes))
                    if len(boxes) == 1:
                        positive_pool[boxes[0][0]].append((dest_image, boxes[0]))
            else:
                # Byte-for-byte unchanged evaluation labels, including known limitations.
                shutil.copy2(label, dest_label)
        report["split_counts"][split] = len(list((dataset / split / "images").glob("*.jpg")))

    # Validation also needs held-out orientations, otherwise early stopping only rewards old layouts.
    for index, source in enumerate(sorted((dataset / "val/images").glob("*.jpg"))):
        turn = index % 3 + 1
        frame = load_image(source)
        boxes, _ = read_boxes(dataset / "val/labels" / (source.stem + ".txt"))
        transformed = cv2.rotate(frame, {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180,
                                        3: cv2.ROTATE_90_COUNTERCLOCKWISE}[turn])
        dest = dataset / "val_orientation/images" / f"valrot{turn}_{index:04d}.jpg"
        save_image(dest, transformed)
        write_boxes(dataset / "val_orientation/labels" / (dest.stem + ".txt"), quarter_turn(boxes, turn))
    report["augmented_validation_images"] = index + 1

    # Keep the established own x4 replay, but normalize only its copied training labels.
    for image in sorted((sources / "local_dataset/det/train_ft/images").glob("own-*.jpg")):
        label = sources / "local_dataset/det/train_ft/labels" / (image.stem + ".txt")
        boxes, _ = read_boxes(label)
        name = "ownreplay__" + image.name
        dest = dataset / "train/images" / name
        shutil.copy2(image, dest)
        write_boxes(dataset / "train/labels" / (dest.stem + ".txt"), boxes)
        report["train_files"].extend([dest.relative_to(output).as_posix()] * 4)

    train_backgrounds = []
    for image in sorted((sources / "own_neg").rglob("*.jpg")):
        relative = image.relative_to(sources / "own_neg").as_posix()
        held = int(hashlib.md5(relative.encode()).hexdigest(), 16) % 5 == 0
        split = "neg_test" if held else "train"
        dest = dataset / split / "images" / ("negative__" + hashlib.sha256(relative.encode()).hexdigest()[:16] + ".jpg")
        shutil.copy2(image, dest)
        write_boxes(dataset / split / "labels" / (dest.stem + ".txt"), [])
        if not held:
            train_backgrounds.append(dest)
            report["train_files"].extend([dest.relative_to(output).as_posix()] * 2)
    assert len(train_backgrounds) == 287
    assert len(list((dataset / "neg_test/images").glob("*.jpg"))) == 71

    # Sample by class to avoid making all extra rotations bananas or the majority class.
    rng.shuffle(original_positives)
    chosen = []
    per_class = Counter()
    for image, boxes in original_positives:
        classes = {b[0] for b in boxes}
        if all(per_class[c] >= max(1, rotations // 3) for c in classes):
            continue
        chosen.append((image, boxes))
        per_class.update(classes)
        if len(chosen) >= rotations:
            break
    for index, (image, boxes) in enumerate(chosen):
        frame = load_image(image)
        report["rotated_sources"].append(image.name)
        for turn in (1, 2, 3):
            transformed = cv2.rotate(frame, {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}[turn])
            dest = dataset / "train/images" / f"rotation{turn}_{index:04d}.jpg"
            save_image(dest, transformed)
            write_boxes(dataset / "train/labels" / (dest.stem + ".txt"), quarter_turn(boxes, turn))
            report["train_files"].append(dest.relative_to(output).as_posix())

    # Build a small bounded cutout pool using train-only objects and two CPU threads.
    cutouts = {0: [], 1: [], 2: []}
    for cls, entries in positive_pool.items():
        rng.shuffle(entries)
        for path, (_, x, y, w, h) in entries:
            if len(cutouts[cls]) >= 12:
                break
            if not 0.08 < w * h < 0.95:
                continue
            image = load_image(path)
            height, width = image.shape[:2]
            x1, y1, x2, y2 = max(0, int((x-w/2)*width)), max(0, int((y-h/2)*height)), min(width, int((x+w/2)*width)), min(height, int((y+h/2)*height))
            crop = image[y1:y2, x1:x2]
            if min(crop.shape[:2]) < 12:
                continue
            scale = min(1, 200 / max(crop.shape[:2]))
            crop = cv2.resize(crop, None, fx=scale, fy=scale)
            mask = np.zeros(crop.shape[:2], np.uint8)
            try:
                cv2.grabCut(crop, mask, (1, 1, crop.shape[1]-2, crop.shape[0]-2),
                            np.zeros((1, 65)), np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_RECT)
            except cv2.error:
                continue
            mask = np.isin(mask, (1, 3)).astype(np.uint8)
            if not 0.25 < mask.mean() < 0.98:
                continue
            ys, xs = np.where(mask)
            cutout = np.dstack([crop, mask * 255])[ys.min():ys.max()+1, xs.min():xs.max()+1]
            cutouts[cls].append(cutout)
    report["cutout_pool_counts"] = {str(c): len(v) for c, v in cutouts.items()}
    if synthetic and not all(cutouts.values()):
        raise ValueError("Insufficient train-only cutouts; cannot make a balanced draft synthetic set")

    for index in range(synthetic):
        image = load_image(rng.choice(train_backgrounds))
        image = cv2.resize(image, (640, 640))
        occupied = []
        masks = []
        # Half scenes contain bananas only; half contain all classes.
        classes = [0] if index % 2 == 0 else [0, 1, 2]
        for obj_index in range(rng.randint(4, 9)):
            cls = classes[obj_index % len(classes)]
            rgba = rng.choice(cutouts[cls]).copy()
            rgba = np.ascontiguousarray(np.rot90(rgba, rng.randrange(4)))
            length = rng.randint(140, 280)
            scale = length / max(rgba.shape[:2])
            rgba = cv2.resize(rgba, None, fx=scale, fy=scale)
            h, w = rgba.shape[:2]
            for attempt in range(25):
                x, y = rng.randrange(640-w), rng.randrange(640-h)
                box = (x, y, x+w, y+h)
                if all(max(0, min(box[2], b[2])-max(x,b[0])) * max(0,min(box[3],b[3])-max(y,b[1]))
                       < 0.35 * min(w*h,(b[2]-b[0])*(b[3]-b[1])) for b in occupied):
                    break
            else:
                continue
            alpha = rgba[:, :, 3] > 127
            if not alpha.any():
                continue
            for _, previous in masks:
                previous[y:y+h,x:x+w][alpha] = False
            mask = np.zeros((640,640),bool)
            mask[y:y+h,x:x+w] = alpha
            image[y:y+h,x:x+w][alpha] = rgba[:, :, :3][alpha]
            masks.append((cls,mask))
            occupied.append(box)
        boxes = []
        for cls, mask in masks:
            ys,xs = np.where(mask)
            if len(xs):
                x1,y1,x2,y2 = xs.min(),ys.min(),xs.max()+1,ys.max()+1
                boxes.append((cls,(x1+x2)/1280,(y1+y2)/1280,(x2-x1)/640,(y2-y1)/640))
        dest = dataset / "synthetic/images" / f"draft_{index:04d}.jpg"
        save_image(dest,image)
        write_boxes(dataset / "synthetic/labels" / (dest.stem+".txt"),boxes)
        report["draft_synthetic_files"].append(dest.relative_to(output).as_posix())

    # Preview boxes for human review; these draft labels are not automatically trusted.
    preview_paths = [dataset / "synthetic/images" / f"draft_{i:04d}.jpg" for i in range(min(12,synthetic))]
    preview = np.full((3*240,4*240,3),245,np.uint8)
    for i,path in enumerate(preview_paths):
        tile = load_image(path)
        boxes,_ = read_boxes(path.parent.parent/"labels"/(path.stem+".txt"))
        for cls,x,y,w,h in boxes:
            cv2.rectangle(tile,(round((x-w/2)*640),round((y-h/2)*640)),
                          (round((x+w/2)*640),round((y+h/2)*640)),(0,0,255),2)
        tile = cv2.resize(tile,(240,240))
        cv2.putText(tile,path.stem,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
        preview[(i//4)*240:(i//4+1)*240,(i%4)*240:(i%4+1)*240] = tile
    save_image(output/"synthetic_preview.jpg",preview)
    (output/"exports").mkdir()
    shutil.copy2(repo/"exports/det_C_candidate.pt",output/"exports/base.pt")
    shutil.copy2(repo/"tools/train_rotation.py",output/"train_rotation.py")
    report["source_hashes"]["base_model"] = digest(output/"exports/base.pt")
    report["training_entries"] = len(report["train_files"])
    report["rotation_count"] = len(chosen)*3
    report["generated_train_images"] = len(list((dataset/"train/images").glob("*.jpg")))
    report["val_test_limitation"] = "Legacy labels unchanged, including empty labels on some banana images; not a new independent test."
    report["file_hashes"] = {p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()}
    (output/"rotation_manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    archive = output.parent/"detector_rotation_colab_input.zip"
    if archive.exists():
        raise ValueError(f"Archive exists; choose a new output parent: {archive}")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in output.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps({k:v for k,v in report.items() if k not in {'file_hashes','train_files','draft_synthetic_files','rotated_sources'}},indent=2),flush=True)
    print('Prepared ZIP:',archive,'bytes:',archive.stat().st_size,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    repo=Path(__file__).resolve().parents[1]
    parser.add_argument('--sources',type=Path,default=repo if (repo/'local_dataset/det').is_dir() else repo.parent)
    parser.add_argument('--output',type=Path,default=repo/'out/rotation_v1/data')
    parser.add_argument('--rotations',type=int,default=180)
    parser.add_argument('--synthetic',type=int,default=240)
    args=parser.parse_args()
    if args.rotations<0 or args.synthetic<0: parser.error('Counts must be nonnegative')
    prepare(repo,args.sources.resolve(),args.output.resolve(),args.rotations,args.synthetic)
