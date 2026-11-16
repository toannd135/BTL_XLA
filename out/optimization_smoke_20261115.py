"""Bounded offline CPU checks; these diagnostic images are not ground truth."""
import asyncio
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(CPU_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HALF="0",
                  CLIP_CHECK="1", CLIP_BATCH_SIZE="4", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="-1",
                  MODEL_PATH=str(ROOT / "exports/det_obb_s.pt"))
sys.path.insert(0, str(ROOT / "backend"))
import cv2
import numpy as np
from fastapi import UploadFile
from starlette.datastructures import Headers
import main


def label_fingerprint():
    value = hashlib.sha256()
    for split in ("val", "test"):
        for file in sorted((ROOT / "local_dataset/det" / split / "labels").glob("*.txt")):
            value.update(file.name.encode())
            value.update(file.read_bytes())
    return value.hexdigest()


async def check():
    baseline_labels = label_fingerprint()
    bundle = ROOT / "out/review_first_batch_20261115"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    report = {"ground_truth_reviewed": False, "purpose": "functional CPU smoke, not accuracy measurement",
              "label_fingerprint_before": baseline_labels, "rows": []}
    async with main.lifespan(main.app):
        report["health"] = main.health()
        print(json.dumps(report["health"]), flush=True)
        assert report["health"]["clip_check"], main.clip_error
        for name in ("cup.jpg", "singleBanana.jpg", "test.webp"):
            item = next(x for x in manifest["items"] if x["source_image"] == "diagnostic/"+name)
            data = (bundle / item["image"]).read_bytes()
            mime = "image/webp" if name.endswith(".webp") else "image/jpeg"
            upload = UploadFile(file=io.BytesIO(data), filename=name, headers=Headers({"content-type": mime}))
            result = await main.predict_api(upload, "careful")
            assert result["type_verification"] == "active"
            assert all("polygon" in o and "type_check" in o for o in result["objects"])
            row = {"image": name, "counts": dict(Counter(o["label"] for o in result["objects"])), **result}
            report["rows"].append(row)
            print(json.dumps({k: row[k] for k in ("image", "counts", "inference_ms", "detection_counts")}), flush=True)
        item = next(x for x in manifest["items"] if x["source_image"] == "diagnostic/singleBanana.jpg")
        image = cv2.imdecode(np.fromfile(bundle/item["image"],dtype=np.uint8), cv2.IMREAD_COLOR)
        crops = [image, image[:,::-1].copy(), image.copy(), image[::-1].copy(), image.copy()]
        checker = main.models["clip"]
        small = checker.probs(crops)
        single = checker.probs(crops,batch=1)
        report["clip_batch_max_abs_difference"] = float(np.max(np.abs(small-single)))
        assert np.allclose(small,single,atol=1e-4)
    report["label_fingerprint_after"] = label_fingerprint()
    assert baseline_labels == report["label_fingerprint_after"]
    suffix = "_b32" if report["health"]["clip_arch"] == "ViT-B-32" else ""
    target = ROOT/("out/optimization_smoke_20261115"+suffix+".json")
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("SMOKE OK", flush=True)


asyncio.run(check())
