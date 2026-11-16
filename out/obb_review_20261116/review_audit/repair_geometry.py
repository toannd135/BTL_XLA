"""Audit submitted labels and pad copies to preserve out-of-frame OBB geometry.

No semantic labels are corrected automatically; generated data stays NOT_READY.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "obb_reviewed_v1.zip"
EXCLUDE = {"7fc208b0975b8a85053b0559", "85caa7b72160f48981c54944"}
FLAGS = {
    "95369e3dc95bb75bb39d7499": "train frame10: object1 is a banana but its class is orange (2).",
    "81d5f64b7d06b9ac1ccea6f0": "val frame4: only 6 boxes for a scene with many visible target fruits; incomplete annotation.",
    "fd1fb0e99efd248fdd05be40": "val frame7: boxes on foliage/background and overlapping parts of apples need per-instance review.",
}


def main():
    source_sha = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    output = ROOT / ("geometry_fixed_" + source_sha[:8] + "_NOT_READY")
    if output.exists():
        raise ValueError("Output already exists; preserve this audited version")
    draft = json.loads((ROOT / "draft_provenance.json").read_text(encoding="utf-8"))
    records, bounds = [], []
    classes = {"train":Counter(), "val":Counter()}
    groups, hashes = {}, {}
    pending = []
    with zipfile.ZipFile(ARCHIVE) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Corrupt ZIP")
        for record in draft["items"]:
            if record["id"] in EXCLUDE:
                continue
            for mapping, key in ((groups,record["source_group"]), (hashes,record["display_image_sha256"])):
                if mapping.setdefault(key,record["role"]) != record["role"]:
                    raise ValueError("Cross-split source or image leakage")
            if record["label"] not in zipped.namelist():
                raise ValueError("Missing labels for a retained image")
            text = zipped.read(record["label"]).decode("utf-8-sig")
            labels=[]
            for line_number,line in enumerate(text.splitlines(),1):
                if not line.strip():
                    continue
                values=list(map(float,line.split()))
                if len(values)!=9 or values[0] not in (0,1,2) or not all(math.isfinite(x) for x in values):
                    raise ValueError("Malformed OBB annotation")
                points=np.array(values[1:]).reshape(4,2)
                if np.any(points<0) or np.any(points>1):
                    bounds.append({"id":record["id"],"role":record["role"],"line":line_number,
                                   "minimum":float(points.min()),"maximum":float(points.max())})
                labels.append((int(values[0]),points))
            source=ROOT/"draft_dataset"/record["image"]
            if hashlib.sha256(source.read_bytes()).hexdigest()!=record["display_image_sha256"]:
                raise ValueError("Source image hash mismatch")
            with Image.open(source) as opened:
                image=opened.convert("RGB")
                width,height=image.size
                all_points=np.concatenate([p for _,p in labels])*[width,height] if labels else np.array([[0,0],[width,height]])
                low,high=all_points.min(0),all_points.max(0)
                left=max(0,math.ceil(-low[0])+1) if low[0]<0 else 0
                top=max(0,math.ceil(-low[1])+1) if low[1]<0 else 0
                right=max(0,math.ceil(high[0]-width)+1) if high[0]>width else 0
                bottom=max(0,math.ceil(high[1]-height)+1) if high[1]>height else 0
                padded=ImageOps.expand(image,border=(left,top,right,bottom),fill=(0,0,0))
                new_size=np.array(padded.size)
                shift=np.array([left,top])
                destination=output/"images"/record["role"]/(record["id"]+".png")
                destination.parent.mkdir(parents=True,exist_ok=True)
                padded.save(destination)
            updated=[]
            for cls,points in labels:
                original=points*[width,height]
                normalized=(original+shift)/new_size
                if np.any(normalized<0) or np.any(normalized>1):
                    raise ValueError("Padding did not contain the OBB")
                serialized=np.array([float(f"{v:.7f}") for v in normalized.ravel()]).reshape(4,2)
                if not np.allclose(serialized*new_size-shift,original,rtol=0,atol=.001):
                    raise ValueError("Padding changed the original pixel geometry")
                updated.append(str(cls)+" "+" ".join(f"{v:.7f}" for v in normalized.ravel()))
                classes[record["role"]][cls]+=1
            target=output/record["label"]
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text("\n".join(updated)+("\n" if updated else ""),encoding="utf-8")
            if record["id"] in FLAGS:
                pending.append({"id":record["id"],"role":record["role"],"issue":FLAGS[record["id"]]})
            records.append({"id":record["id"],"role":record["role"],"source":record["source_image"],
                            "image":destination.relative_to(output).as_posix(),"label":record["label"],
                            "source_image_sha256":record["display_image_sha256"],
                            "padding_ltrb":[left,top,right,bottom],"size":new_size.tolist(),
                            "boxes":len(labels),"semantic_review":"needs_correction" if record["id"] in FLAGS else "submitted_by_user"})
    report={"source_zip_sha256":source_sha,"output":str(output),"training_ready":False,
            "counts":dict(Counter(r["role"] for r in records)),"boxes":sum(r["boxes"] for r in records),
            "boxes_by_class":classes,"out_of_bounds_boxes":len(bounds),
            "out_of_bounds_images":len({r["id"] for r in bounds}),"coordinate_issues":bounds,
            "user_confirmed_exclusions":[{"id":i,"reason":"User requested exclusion of the two difficult unlabelled validation images on 2026-11-16"} for i in sorted(EXCLUDE)],
            "pending_semantic_corrections":pending,"items":records,
            "policy":"Original ZIP, images and labels unchanged. Copies padded with black border to contain OBBs; "
                     "all corners preserve original pixel geometry within .001px. No class or fruit counts automatically corrected. "
                     "No data.yaml provided because semantic annotation errors remain. Pilot set only, not the complete fine-tuning dataset."}
    (output/"AUDIT_NOT_READY.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"review_audit"/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("output","training_ready","counts","boxes","boxes_by_class","out_of_bounds_boxes","out_of_bounds_images","pending_semantic_corrections")},ensure_ascii=False))


if __name__=="__main__":
    main()
