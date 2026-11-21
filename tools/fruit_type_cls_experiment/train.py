#!/usr/bin/env python3
"""Train an isolated YOLO11 fruit-type classifier; requires a working CUDA GPU."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import torch, yaml
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/'work/fruit_type_cls_experiment/config.yaml'
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,default=DEFAULT_CONFIG); ap.add_argument('--model'); args=ap.parse_args()
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8')); cfg['model']=args.model or cfg['model']
    if not torch.cuda.is_available(): raise SystemExit('STOP: CUDA is unavailable. Training was not started; install/fix a CUDA driver or schedule this unchanged config on a CUDA host.')
    data=ROOT/cfg['data_dir']; out=ROOT/cfg['output_dir']; run=ROOT/cfg['run_dir']
    for split in ('train','val'):
        for name in cfg['classes']:
            if not list((data/split/name).glob('*.jpg')): raise SystemExit(f'Missing crops: {data/split/name}')
    slug=Path(str(cfg['model'])).stem.replace('-','_')
    dest=out/f'{slug}_best.pt'
    if dest.exists(): raise SystemExit(f'Refusing to overwrite {dest}')
    run.mkdir(parents=True,exist_ok=True); name=f"{slug}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model=YOLO(cfg['model'])
    model.train(data=str(data),imgsz=cfg['imgsz'],epochs=cfg['epochs'],batch=cfg['batch'],workers=cfg['workers'],
        patience=cfg['patience'],optimizer=cfg['optimizer'],lr0=cfg['lr0'],seed=cfg['seed'],deterministic=cfg['deterministic'],
        pretrained=cfg['pretrained'],cos_lr=cfg['cos_lr'],device=cfg['device'],project=str(run),name=name,exist_ok=False,verbose=True)
    result=Path(model.trainer.save_dir)
    best=result/'weights/best.pt'
    if not best.is_file(): raise RuntimeError(f'Ultralytics did not produce {best}')
    out.mkdir(parents=True,exist_ok=True)
    if dest.exists(): raise RuntimeError(f'Refusing to overwrite {dest}')
    shutil.copy2(best,dest)
    metadata={'config':cfg,'config_sha256':digest(args.config),'dataset_manifest_sha256':digest(ROOT/'work/fruit_type_cls_experiment/dataset_manifest.csv'),
        'best_checkpoint_sha256':digest(dest),'training_run':str(result.relative_to(ROOT)),'checkpoint':str(dest.relative_to(ROOT)),
        'trained_utc':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
    (out/'training_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    import pandas as pd
    import matplotlib.pyplot as plt
    history=pd.read_csv(result/'results.csv')
    columns=[c for c in history.columns if any(k in c.lower() for k in ('loss','acc/top1','acc/top5'))]
    if columns:
        ax=history[columns].plot(figsize=(9,5),grid=True); ax.set_xlabel('Epoch'); ax.set_ylabel('Metric'); ax.figure.tight_layout(); ax.figure.savefig(out/'training_curves.png',dpi=160); plt.close(ax.figure)
    print(json.dumps(metadata,indent=2))
if __name__=='__main__': main()
