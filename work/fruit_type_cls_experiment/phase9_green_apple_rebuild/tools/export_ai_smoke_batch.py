#!/usr/bin/env python3
"""Export only AI AUTO_APPROVED, non-leaking Green Apple source-train images."""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image

BASE=Path(__file__).resolve().parents[1]
COLLECTION=BASE.parent/"phase8c_collection/green_apple_phase8d/staging/mendeley_v1"

def rows(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();out=args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):raise SystemExit(f"Refusing to overwrite non-empty output: {out}")
    queue=rows(BASE/"review_queue.csv"); assessments=rows(BASE/"ai_review/ai_review_manifest.csv")
    by_asset=defaultdict(list)
    for r in queue:by_asset[r["asset_id"]].append(r)
    assess={r["polygon_id"]:r for r in assessments}
    groups=defaultdict(list)
    for r in queue:groups[(r["source_split_original"],r["original_filename"])].append(r)
    (out/"images/train").mkdir(parents=True,exist_ok=True);(out/"labels/train").mkdir(parents=True,exist_ok=True)
    included=[];excluded=[]
    for (split,fn),items in sorted(groups.items()):
        states=[assess.get(r["polygon_id"],{}) for r in items]
        reasons=[]
        if split!="train":reasons.append("ORIGINAL_SOURCE_TEST_WAS_USED_IN_V5_TRAIN")
        if not states or any(s.get("decision")!="AUTO_APPROVED" or s.get("confidence_band")!="HIGH_UNCALIBRATED" or s.get("human_verified")!="false" for s in states):
            reasons.append("NOT_ALL_INSTANCES_AI_AUTO_APPROVED_AT_HIGH_CONFIDENCE")
        if any(s.get("image_sha256")!=items[0]["image_sha256"] for s in states):reasons.append("SOURCE_IMAGE_HASH_MISMATCH")
        if any(s.get("decision")=="DUPLICATE_HOLD" for s in states):reasons.append("DUPLICATE_OR_PHASH_HOLD")
        if reasons:
            excluded.append({"original_filename":fn,"source_split_original":split,"polygon_count":len(items),"decision_categories":";".join(sorted({s.get("decision","MISSING") for s in states})),"reason":";".join(sorted(set(reasons)))})
            continue
        src=COLLECTION/items[0]["image_relative_path"]
        try:
            with Image.open(src) as im:im.verify()
        except Exception as exc:
            excluded.append({"original_filename":fn,"source_split_original":split,"polygon_count":len(items),"decision_categories":"AUTO_APPROVED","reason":f"IMAGE_UNREADABLE:{exc}"});continue
        digest=hashlib.sha256(src.read_bytes()).hexdigest()
        if digest!=items[0]["image_sha256"]:
            excluded.append({"original_filename":fn,"source_split_original":split,"polygon_count":len(items),"decision_categories":"AUTO_APPROVED","reason":"SHA256_MISMATCH"});continue
        stem="mendeley_"+hashlib.sha256((split+"\0"+fn).encode()).hexdigest()[:20]
        dst_img=out/"images/train"/f"{stem}.jpg";shutil.copy2(src,dst_img)
        label=out/"labels/train"/f"{stem}.txt";w,h=int(items[0]["image_width"]),int(items[0]["image_height"])
        with label.open("w",encoding="utf-8") as f:
            for r in items:
                pts=json.loads(r["polygon_points_json"]);vals=[v for x,y in pts for v in (x/w,y/h)]
                f.write("0 "+" ".join(f"{v:.8f}" for v in vals)+"\n")
        included.append({"asset_id":items[0]["asset_id"],"original_filename":fn,"source_split_original":split,
            "instance_count":len(items),"class_id":0,"class_name":"green_apple","image_sha256":digest,
            "source_url":items[0]["source_url"],"license":items[0]["license"],"source_family_id":"",
            "dataset_split":"UNASSIGNED_SMOKE_TRAIN_ONLY","review_type":"AI_ASSISTED_NOT_HUMAN",
            "human_verified":"false","confidence_band":"HIGH_UNCALIBRATED",
            "assessment_polygon_ids":";".join(r["polygon_id"] for r in items),
            "assessment_reasons":" | ".join(s["reason"] for s in states),
            "clip_checkpoint_sha256":states[0]["model_sha256"],"image_path":str(dst_img.relative_to(out)),
            "label_path":str(label.relative_to(out))})
    # Verify no exact duplicate survived in this exported batch.
    hashes=[r["image_sha256"] for r in included]
    if len(hashes)!=len(set(hashes)):raise SystemExit("Internal error: duplicate SHA survived candidate export")
    def write(path,data,fields):
        with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    fields=["asset_id","original_filename","source_split_original","instance_count","class_id","class_name","image_sha256","source_url","license","source_family_id","dataset_split","review_type","human_verified","confidence_band","assessment_polygon_ids","assessment_reasons","clip_checkpoint_sha256","image_path","label_path"]
    write(out/"candidate_manifest.csv",included,fields)
    write(out/"excluded_images.csv",excluded,["original_filename","source_split_original","polygon_count","decision_categories","reason"])
    write(out/"ai_review_manifest.csv",assessments,list(assessments[0]))
    data_yaml=(f"path: {out}\ntrain: images/train\nval: images/train\nnames:\n  0: green_apple\n")
    (out/"green_apple.yaml").write_text(data_yaml,encoding="utf-8")
    report={"status":"AI_ASSISTED_SMOKE_CANDIDATE_READY" if included else "EMPTY_NO_AI_AUTO_APPROVED_IMAGES",
        "training_images":len(included),"training_instances":sum(r["instance_count"] for r in included),
        "excluded_source_images":len(excluded),"source_families_assigned":0,"train_validation_test_split_created":False,
        "human_verified":False,"label_review":"AI-assisted only; not human ground truth",
        "validation_note":"For smoke pipeline only, val points to the same training images because smoke config disables validation; do not interpret metrics as validation.",
        "green_apple_test_set":False,"generalization_claim_allowed":False}
    (out/"batch_summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
