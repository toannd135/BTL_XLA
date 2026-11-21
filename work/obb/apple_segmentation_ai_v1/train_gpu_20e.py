#!/usr/bin/env python3
"""GPU fine-tuning for the three-class YOLO segmentation candidate."""
from pathlib import Path
import hashlib
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[4]
DATA=ROOT/'fruit/work/obb/apple_segmentation_ai_v1/dataset/data.yaml'
SOURCE=ROOT/'fruit/runs/seg/apple_segmentation_ai_v1/weights/last.pt'
PROJECT=ROOT/'fruit/runs/seg'
NAME='apple_segmentation_ai_gpu20'
OUT=PROJECT/NAME
if not SOURCE.is_file():raise SystemExit(f'missing source checkpoint {SOURCE}')
if OUT.exists():raise SystemExit(f'refusing to overwrite existing output {OUT}')
if not torch.cuda.is_available():raise SystemExit('TRAINING_BLOCKED: CUDA unavailable in this runtime')
print('device',torch.cuda.get_device_name(0),flush=True)
print('source_checkpoint',SOURCE,flush=True)
print('source_sha256',hashlib.sha256(SOURCE.read_bytes()).hexdigest(),flush=True)
model=YOLO(str(SOURCE),task='segment')
model.train(data=str(DATA),epochs=20,patience=4,imgsz=512,batch=4,workers=4,device=0,amp=True,cache=False,
            optimizer='AdamW',lr0=5e-5,lrf=.01,weight_decay=.0005,seed=20261121,deterministic=True,
            project=str(PROJECT),name=NAME,exist_ok=False,plots=False,save=True,val=True,verbose=True)
