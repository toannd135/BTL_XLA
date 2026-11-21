#!/usr/bin/env python3
"""Evaluate a frozen classifier on the untouched test crop split."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import cv2, numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from ultralytics import YOLO
import yaml

ROOT=Path(__file__).resolve().parents[2]
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--config',type=Path,default=ROOT/'work/fruit_type_cls_experiment/config.yaml'); ap.add_argument('--split',choices=['val','test'],default='test'); args=ap.parse_args()
    cfg=yaml.safe_load(args.config.read_text()); root=ROOT/cfg['data_dir']/args.split; names=cfg['classes']
    paths=[]; truth=[]
    for i,name in enumerate(names):
        for p in sorted((root/name).glob('*.jpg')): paths.append(p); truth.append(i)
    if not paths: raise SystemExit(f'No crop files in {root}')
    model=YOLO(str(args.checkpoint)); device='0' if __import__('torch').cuda.is_available() else 'cpu'
    # warm-up excluded from latency; all images evaluated once without tuning.
    sample=cv2.imread(str(paths[0])); model.predict(sample,imgsz=cfg['imgsz'],device=device,verbose=False)
    start=time.perf_counter(); results=model.predict([str(x) for x in paths],imgsz=cfg['imgsz'],device=device,stream=True,verbose=False)
    pred=[]
    for r in results: pred.append(int(r.probs.top1))
    elapsed=(time.perf_counter()-start)*1000
    cm=confusion_matrix(truth,pred,labels=list(range(len(names))))
    p,r,f,_=precision_recall_fscore_support(truth,pred,labels=list(range(len(names))),zero_division=0)
    payload={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':digest(args.checkpoint),'split':args.split,'sample_count':len(paths),'class_names':names,
        'accuracy':float(accuracy_score(truth,pred)),'macro_precision':float(p.mean()),'macro_recall':float(r.mean()),'macro_f1':float(f.mean()),
        'per_class':{n:{'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i]),'support':int(cm[i].sum())} for i,n in enumerate(names)},
        'confusion_matrix_rows_true_cols_pred':cm.tolist(),'inference_total_ms':elapsed,'mean_ms_per_crop':elapsed/len(paths),'test_split_used_for_tuning':False}
    out=ROOT/'work/fruit_type_cls_experiment'; (out/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(cm,cmap='Blues'); ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,xlabel='Predicted',ylabel='True',title=f"Frozen {args.split} confusion matrix");
    for i in range(len(names)):
        for j in range(len(names)): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=160); plt.close(fig)
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
