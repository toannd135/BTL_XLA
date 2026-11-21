#!/usr/bin/env python3
"""Run one explicitly non-evaluative YOLO11s-seg pipeline smoke epoch."""
from __future__ import annotations
import csv, hashlib, json, os, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[5]
PHASE=Path(__file__).resolve().parents[1]
BATCH=PHASE/"candidate_batches/green_apple_smoke_001"
CHECKER=PHASE/"training_readiness_report.json"
INITIAL=ROOT/"fruit/runs/seg/seg_green_v4_candidate/weights/best.pt"
INITIAL_SHA="e9f807bd4a92a7b59047ae399898bc37c63ba715b030163662e76e35f215479c"
RUN_NAME="seg_green_v6_smoke_cpu_20261121"
RUN_DIR=ROOT/"fruit/runs/seg"/RUN_NAME
REPORT=PHASE/"GREEN_APPLE_AUTONOMOUS_SMOKE_TRAINING_REPORT.md"
LOG_DIR=PHASE/"logs"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def write_report(status, info):
    readiness=info.get("readiness",{})
    summary=info.get("assessment_summary",{})
    lines=["# Green Apple Autonomous Smoke Training Report","",f"## Final status: `{status}`","",
      "## A. Dataset audit","",
      f"- Source images assessed: {summary.get('image_count','NOT_AVAILABLE')}",
      f"- Polygon rows assessed: {summary.get('polygon_count','NOT_AVAILABLE')}",
      f"- AI AUTO_APPROVED polygons: {summary.get('decisions',{}).get('AUTO_APPROVED',0)}; human-verified: 0",
      f"- NEEDS_REVIEW polygons: {summary.get('decisions',{}).get('NEEDS_REVIEW',0)}",
      f"- REJECTED polygons: {summary.get('decisions',{}).get('REJECTED',0)}",
      f"- DUPLICATE_HOLD polygons: {summary.get('decisions',{}).get('DUPLICATE_HOLD',0)}",
      "- Rejection causes are structural geometry / unreadable or hash mismatch; held records include the source-test split and known exact/pHash candidates.",
      "- The 77 original Mendeley test images were excluded from training. AI review is not human ground truth.","",
      "## B. Dataset used","",
      f"- Training images / instances: {info.get('training_images',0)} / {info.get('training_instances',0)}",
      f"- Source: Mendeley Data V1, DOI 10.17632/gfcmdbvw65.1, CC BY 4.0.",
      f"- Candidate directory: `{BATCH}`",
      f"- SHA-256 image/annotation manifest: `{BATCH/'candidate_manifest.csv'}`",
      "- Provenance: source-family IDs unknown and intentionally blank; smoke does not claim independent sampling.",
      "- Review type: AI-assisted. `human_verified=false` for all included rows.","",
      "## C. Training","",
      f"- Model: YOLO11s-seg; initialized from local V4 checkpoint `{INITIAL}`.",
      f"- Initial checkpoint SHA-256: `{info.get('initial_checkpoint_sha256',INITIAL_SHA)}`",
      f"- Device: {info.get('device','NOT_RUN')}; GPU: {info.get('gpu','NOT_AVAILABLE')}; VRAM: {info.get('vram','NOT_AVAILABLE')}.",
      f"- Batch size: 1; image size: 640; epochs completed: {info.get('epochs_completed',0)}; workers: 0.",
      f"- Training duration seconds: {info.get('duration_seconds','NOT_RUN')}.",
      f"- Console log: `{LOG_DIR/'smoke_training.log'}`",
      f"- Results CSV: `{RUN_DIR/'results.csv'}`",
      f"- Loss history: `{info.get('loss_history',[])}`",
      f"- Saved checkpoint: `{info.get('checkpoint','NOT_SAVED')}`",
      f"- Saved checkpoint SHA-256: `{info.get('checkpoint_sha256','NOT_AVAILABLE')}`",
      f"- NaN/Inf loss detected: {info.get('nonfinite_loss','NOT_VERIFIED')}","",
      "## D. Checkpoint and inference","",
      f"- Reload checkpoint: {info.get('checkpoint_reload','NOT_RUN')}",
      f"- Inference: {info.get('inference_status','NOT_RUN')}; images attempted: {info.get('inference_images',0)}; masks returned: {info.get('inference_masks',0)}.",
      f"- Output mask validity: {info.get('inference_mask_validation','NOT_RUN')}.",
      f"- Saved visualizations: `{info.get('inference_output','NOT_CREATED')}`",
      f"- Inference error: {info.get('inference_error','None')}.","",
      "## E. Readiness","",
      f"- SMOKE_TEST_READY: {readiness.get('SMOKE_TEST_READY',{}).get('ready',False)}",
      f"- PILOT_READY: {readiness.get('PILOT_READY',{}).get('ready',False)}",
      f"- EVALUATION_READY: {readiness.get('EVALUATION_READY',{}).get('ready',False)}",
      "- Smoke results are not validation metrics, accuracy, mAP, or evidence of generalization.","",
      "## F. Limitations and conclusion","",
      "This run only checks that reviewed image/mask data can pass through the YOLO training and inference pipeline. Five AI-assisted images are not human-verified and do not constitute a reliable Green Apple dataset. Do not promote this checkpoint or claim Green Apple improvement. The 77 leaked source-test images and the six-image prior Green Apple test set were not used for evaluation.","",
      f"Smoke training run directory: `{RUN_DIR}`.",
      "No V4/V5 checkpoint, original dataset, or backend configuration was modified."]
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    info={"device":"CPU","gpu":"unavailable (torch.cuda.is_available=False; nvidia-smi driver communication failed)","vram":"not available"}
    try:
        readiness=json.loads(CHECKER.read_text(encoding="utf-8"))
        if not readiness.get("readiness",{}).get("SMOKE_TEST_READY",{}).get("ready"):
            raise RuntimeError("Readiness checker did not report SMOKE_TEST_READY")
        batch_summary=json.loads((BATCH/"batch_summary.json").read_text(encoding="utf-8"))
        assessment=json.loads((PHASE/"ai_review/ai_assessment_summary.json").read_text(encoding="utf-8"))
        if batch_summary.get("human_verified") is not False:
            raise RuntimeError("Candidate must remain explicitly AI-assisted, not human verified")
        if not INITIAL.is_file() or sha(INITIAL)!=INITIAL_SHA:
            raise RuntimeError("V4 initialization checkpoint missing or SHA-256 mismatch")
        if RUN_DIR.exists():
            raise RuntimeError(f"Refusing to overwrite existing run directory: {RUN_DIR}")
        if not torch.cuda.is_available():torch.set_num_threads(min(4,os.cpu_count() or 1))
        data=str((BATCH/"green_apple.yaml").resolve())
        started=time.time()
        with (LOG_DIR/"smoke_training.log").open("w",encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            model=YOLO(str(INITIAL))
            model.train(data=data,epochs=1,patience=1,batch=1,imgsz=640,workers=0,device="cpu",
                project=str(RUN_DIR.parent),name=RUN_DIR.name,exist_ok=False,pretrained=True,
                seed=20261120,deterministic=True,amp=False,val=False,save=True,save_period=-1,
                optimizer="AdamW",lr0=0.0001,plots=False,cache=False,close_mosaic=0,
                mosaic=0.0,fliplr=0.0,scale=0.0,verbose=True)
        info["duration_seconds"]=round(time.time()-started,2)
        info["initial_checkpoint_sha256"]=INITIAL_SHA
        info["training_images"]=batch_summary["training_images"]
        info["training_instances"]=batch_summary["training_instances"]
        info["assessment_summary"]=assessment
        info["readiness"]=readiness["readiness"]
        results_csv=RUN_DIR/"results.csv"
        losses=[]
        if results_csv.exists():
            with results_csv.open(encoding="utf-8",newline="") as f:
                for row in csv.DictReader(f):
                    losses.append(row)
        info["loss_history"]=losses
        loss_keys=[k for r in losses for k in r if "loss" in k.lower()]
        info["nonfinite_loss"]=any(str(r[k]).lower() in {"nan","inf","-inf"} for r in losses for k in loss_keys)
        info["epochs_completed"]=len(losses)
        checkpoint=RUN_DIR/"weights/best.pt"
        if not checkpoint.is_file():checkpoint=RUN_DIR/"weights/last.pt"
        if not checkpoint.is_file():raise RuntimeError("Training completed without a readable best.pt or last.pt checkpoint")
        info["checkpoint"]=str(checkpoint);info["checkpoint_sha256"]=sha(checkpoint)
        reloaded=YOLO(str(checkpoint));info["checkpoint_reload"]="success";info["model_task"]=reloaded.task
        image_paths=sorted((BATCH/"images/train").glob("*.jpg"))
        pred_dir=RUN_DIR/"smoke_inference"
        if pred_dir.exists():raise RuntimeError(f"Refusing to overwrite inference output: {pred_dir}")
        preds=reloaded.predict(source=[str(p) for p in image_paths],imgsz=640,conf=0.05,device="cpu",save=True,
            project=str(RUN_DIR),name="smoke_inference",exist_ok=False,verbose=False)
        mask_count=0;valid=True
        for result in preds:
            if result.masks is None:continue
            data_t=result.masks.data
            if not torch.isfinite(data_t).all():valid=False
            h,w=result.orig_shape
            for polygon in result.masks.xy:
                if len(polygon)<3 or (polygon[:,0]<0).any() or (polygon[:,0]>w).any() or (polygon[:,1]<0).any() or (polygon[:,1]>h).any():valid=False
                mask_count+=1
        info.update({"inference_status":"success","inference_images":len(preds),"inference_masks":mask_count,
            "inference_mask_validation":"finite, >=3 vertices and coordinates within source bounds" if valid else "invalid output mask geometry",
            "inference_output":str(pred_dir),"inference_error":"None"})
        if not valid:raise RuntimeError("Inference produced non-finite or out-of-bounds masks")
        final_status="SMOKE_TRAINING_COMPLETED"
        write_report(final_status,info)
        print(json.dumps({"status":final_status,"run_dir":str(RUN_DIR),"checkpoint":str(checkpoint),"checkpoint_sha256":info["checkpoint_sha256"],"epochs_completed":info["epochs_completed"],"inference_images":info["inference_images"],"inference_masks":mask_count},indent=2))
    except Exception as exc:
        info["error"]=str(exc);info["traceback"]=traceback.format_exc()
        if RUN_DIR.exists():
            info["partial_run_dir"]=str(RUN_DIR)
            candidate=RUN_DIR/"weights/last.pt"
            if candidate.is_file():info["partial_checkpoint"]=str(candidate);info["partial_checkpoint_sha256"]=sha(candidate)
        info.setdefault("assessment_summary",json.loads((PHASE/"ai_review/ai_assessment_summary.json").read_text()) if (PHASE/"ai_review/ai_assessment_summary.json").exists() else {})
        info.setdefault("readiness",json.loads(CHECKER.read_text()).get("readiness",{}) if CHECKER.exists() else {})
        try:info.setdefault("training_images",json.loads((BATCH/"batch_summary.json").read_text()).get("training_images",0))
        except Exception:pass
        status="SMOKE_TRAINING_FAILED" if RUN_DIR.exists() else "SMOKE_TRAINING_BLOCKED"
        write_report(status,info)
        print(json.dumps({"status":status,"error":str(exc),"run_dir":str(RUN_DIR) if RUN_DIR.exists() else None},indent=2),file=sys.stderr)
        raise


if __name__=="__main__":main()
