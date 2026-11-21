#!/usr/bin/env python3
"""Run the experimental YOLO11s-seg checkpoint on one image or a folder."""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[5]
DEFAULT_WEIGHTS=ROOT/'fruit/runs/seg/apple_segmentation_ai_gpu20/weights/best.pt'
DEFAULT_PROJECT=ROOT/'fruit/work/obb/apple_segmentation_ai_v1/inference_outputs'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('source',help='image file or directory')
    p.add_argument('--weights',default=str(DEFAULT_WEIGHTS))
    p.add_argument('--device',default='0',help="'0' for CUDA or 'cpu'")
    p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--iou',type=float,default=.7)
    p.add_argument('--imgsz',type=int,default=512)
    p.add_argument('--project',default=str(DEFAULT_PROJECT))
    p.add_argument('--name',default='predict')
    a=p.parse_args()
    weights=Path(a.weights); source=Path(a.source)
    if not weights.is_file():p.error(f'checkpoint not found: {weights}')
    if not source.exists():p.error(f'source not found: {source}')
    model=YOLO(str(weights),task='segment')
    results=model.predict(source=str(source),imgsz=a.imgsz,device=a.device,conf=a.conf,iou=a.iou,rect=False,
                          save=True,project=a.project,name=a.name,exist_ok=False,verbose=True)
    print('classes:',model.names)
    print('images:',len(results))
    print('saved to:',Path(a.project)/a.name)

if __name__=='__main__':main()
