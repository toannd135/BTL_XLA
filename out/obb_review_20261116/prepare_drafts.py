"""Prepare unverified OBB drafts; never alter app code or original annotations."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import shutil
import zipfile

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BUNDLE = ROOT / "out/review_first_batch_20261115"
DATA = OUT / "draft_dataset"
WEIGHTS = ROOT / "exports/det_obb_s.pt"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DATA.exists() or (OUT / "obb_drafts_UNREVIEWED.zip").exists():
        raise ValueError("Draft output already exists; do not overwrite a reviewed version")
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    selected = [x for x in manifest["items"] if x["role"] in ("train", "val")]
    if not selected:
        raise ValueError("No train/validation images")
    groups, hashes = {}, {}
    for item in selected:
        image = (BUNDLE / item["image"]).resolve()
        if not image.is_relative_to(BUNDLE.resolve()) or digest(image) != item["display_image_sha256"]:
            raise ValueError("Review image changed or is outside the bundle")
        for mapping, key in ((groups, item["source_group"]), (hashes, item["source_image_sha256"])):
            if mapping.setdefault(key, item["role"]) != item["role"]:
                raise ValueError("Source group or exact image appears in both splits")

    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    model = YOLO(str(WEIGHTS))
    if model.task != "obb" or model.names != {0:"banana", 1:"apple", 2:"orange"}:
        raise ValueError("Expected the current three-class OBB model")
    lists = {"train": [], "val": []}
    records = []
    for item in selected:
        image = BUNDLE / item["image"]
        frame = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Cannot decode image")
        result = model.predict(frame, device="cpu", imgsz=640, conf=.25,
                               iou=.7, max_det=300, verbose=False)[0]
        torch.set_num_threads(2)
        role, identifier = item["role"], item["id"]
        target = DATA / "images" / role / (identifier + image.suffix)
        label = DATA / "labels" / role / (identifier + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target)
        points = result.obb.xyxyxyxyn.cpu().numpy()
        classes = result.obb.cls.cpu().numpy().astype(int)
        scores = result.obb.conf.cpu().numpy()
        lines = [str(int(cls)) + " " + " ".join(f"{v:.7f}" for v in np.clip(poly, 0, 1).ravel())
                 for cls, poly in zip(classes, points)]
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        lists[role].append("./" + target.relative_to(DATA).as_posix())
        records.append({"id":identifier, "role":role, "source_image":item["source_image"],
                        "source_group":item["source_group"], "display_image_sha256":digest(target),
                        "image":target.relative_to(DATA).as_posix(), "label":label.relative_to(DATA).as_posix(),
                        "status":"pending", "expected_count":None, "verified":False,
                        "draft_counts":dict(Counter(int(c) for c in classes)), "scores":scores.tolist()})
    for role, paths in lists.items():
        (DATA / (role + ".txt")).write_text("\n".join(paths) + "\n", encoding="utf-8")
    # Omit path so Ultralytics can resolve this dataset after moving it to Colab.
    (DATA / "data.yaml").write_text("train: train.txt\nval: val.txt\nnames:\n  0: banana\n  1: apple\n  2: orange\n", encoding="utf-8")
    provenance = {"schema":1, "parent_dataset_id":manifest["dataset_id"],
                  "model_sha256":digest(WEIGHTS), "counts":dict(Counter(x["role"] for x in records)),
                  "conf":.25,"iou":.7,"imgsz":640,"device":"cpu","threads":2,
                  "policy":"Unverified model drafts only. Empty predictions are NOT confirmed negatives. "
                           "All target fruits must be checked. Diagnostics/test are excluded. Do not train this draft archive.",
                  "items":records}
    (OUT / "draft_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = OUT / "obb_drafts_UNREVIEWED.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(DATA.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(DATA).as_posix())
    print(json.dumps({"archive":str(archive),"counts":provenance["counts"],
                      "review_required":True,"automatic_approvals":0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
