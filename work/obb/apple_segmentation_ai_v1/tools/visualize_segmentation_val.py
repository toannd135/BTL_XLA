#!/usr/bin/env python3
"""Validate the current best checkpoint on val and save GT/prediction panels."""
import json, sys
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[5]
DATA=ROOT/'fruit/work/obb/apple_segmentation_ai_v1/dataset'
WEIGHTS=Path(os.environ.get('PHASE9_SEG_WEIGHTS',ROOT/'fruit/runs/seg/apple_segmentation_ai_gpu20/weights/best.pt'))
OUT=Path(os.environ.get('PHASE9_SEG_VISUALS',ROOT/'fruit/work/obb/apple_segmentation_ai_v1/VISUAL_RESULTS'))
NAMES={0:'banana',1:'apple',2:'orange'}
COLORS={0:(60,220,60),1:(40,120,255),2:(0,220,255)}

def overlay_gt(image,label_path):
    canvas=image.copy();h,w=image.shape[:2]
    if not label_path.exists():return canvas
    for line in label_path.read_text().splitlines():
        p=line.split()
        if len(p)<7:continue
        cls=int(float(p[0])); coords=np.array([float(x) for x in p[5:]],dtype=np.float32).reshape(-1,2)
        coords[:,0]*=w;coords[:,1]*=h;pts=coords.astype(np.int32)
        color=COLORS.get(cls,(255,255,255)); cv2.fillPoly(canvas,[pts],color)
        cv2.polylines(canvas,[pts],True,(255,255,255),2)
        x,y=pts.mean(axis=0).astype(int);cv2.putText(canvas,f'GT {NAMES.get(cls,cls)}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3,cv2.LINE_AA);cv2.putText(canvas,f'GT {NAMES.get(cls,cls)}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    return cv2.addWeighted(image,.55,canvas,.45,0)

def overlay_pred(image,result):
    canvas=image.copy()
    if result.masks is None:return canvas
    for i,poly in enumerate(result.masks.xy):
        cls=int(result.boxes.cls[i]); conf=float(result.boxes.conf[i]);pts=np.asarray(poly,dtype=np.int32)
        if len(pts)<3:continue
        color=COLORS.get(cls,(255,255,255));cv2.fillPoly(canvas,[pts],color);cv2.polylines(canvas,[pts],True,(255,255,255),2)
        x,y=pts.mean(axis=0).astype(int);label=f'{NAMES.get(cls,cls)} {conf:.2f}'
        cv2.putText(canvas,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3,cv2.LINE_AA);cv2.putText(canvas,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    return cv2.addWeighted(image,.55,canvas,.45,0)

def main():
    if not WEIGHTS.is_file():raise SystemExit(f'missing checkpoint: {WEIGHTS}')
    OUT.mkdir(parents=True,exist_ok=True)
    model=YOLO(str(WEIGHTS),task='segment')
    metrics=model.val(data=str(DATA/'data.yaml'),split='val',imgsz=512,device='0' if __import__('torch').cuda.is_available() else 'cpu',batch=4,workers=0,plots=False,project=str(OUT),name='metric_eval',exist_ok=True,verbose=False)
    per_class={}
    for i,class_id in enumerate(metrics.seg.ap_class_index):
        precision,recall,map50,map5095=metrics.seg.class_result(i)
        per_class[NAMES[int(class_id)]]={'precision':float(precision),'recall':float(recall),'mask_map50':float(map50),'mask_map50_95':float(map5095)}
    report={'checkpoint':str(WEIGHTS),'split':'val','image_count':236,'box_precision':float(metrics.box.mp),'box_recall':float(metrics.box.mr),'box_map50':float(metrics.box.map50),'box_map50_95':float(metrics.box.map),'mask_precision':float(metrics.seg.mp),'mask_recall':float(metrics.seg.mr),'mask_map50':float(metrics.seg.map50),'mask_map50_95':float(metrics.seg.map),'mask_per_class':per_class,'names':NAMES,'annotation_caveat':'Validation masks are inherited/AI-assisted, not human-verified.'}
    (OUT/'validation_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    images=sorted((DATA/'val/images').glob('*'))[:12]
    preds=model.predict([str(p) for p in images],imgsz=512,rect=False,device='0' if __import__('torch').cuda.is_available() else 'cpu',batch=4,conf=.25,iou=.7,verbose=False)
    for path,res in zip(images,preds):
        im=cv2.imread(str(path));gt=overlay_gt(im,DATA/'val/labels'/f'{path.stem}.txt');pr=overlay_pred(im,res)
        panel=np.concatenate([cv2.resize(im,(512,512)),cv2.resize(gt,(512,512)),cv2.resize(pr,(512,512))],axis=1)
        cv2.putText(panel,'ORIGINAL',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA);cv2.putText(panel,'GROUND TRUTH',(524,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA);cv2.putText(panel,'PREDICTION',(1036,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA)
        cv2.imwrite(str(OUT/f'{path.stem}_comparison.jpg'),panel)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
