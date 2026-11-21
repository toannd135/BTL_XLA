#!/usr/bin/env python3
"""Train exploratory GreenApple vs NonGreenApple MobileNetV3 using AI color labels."""
import csv, hashlib, json, os, random, shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "fruit/work/fruit_type_cls_experiment/phase9_2_green_apple_expansion"
RUN = Path(os.environ.get("PHASE9_CLASSIFIER_RUN", BASE / "green_apple_classifier/run_v2"))
DATA = Path(os.environ.get("PHASE9_CLASSIFIER_DATASET", BASE / "green_apple_classifier/dataset_v2"))
WEIGHTS = Path(os.environ.get("PHASE7_PRETRAINED_WEIGHTS", "/home/toannd/models/mobilenet/mobilenet_v3_large-5c1a4163.pth")).expanduser()
EXPECTED = "5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997"
SEED = 20261121

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    if RUN.exists() and any(RUN.iterdir()): raise SystemExit(f"refusing to overwrite non-empty run: {RUN}")
    if not WEIGHTS.is_file(): raise SystemExit(f"TRAINING_BLOCKED: missing checkpoint {WEIGHTS}")
    digest=sha256(WEIGHTS)
    if digest != EXPECTED: raise SystemExit(f"TRAINING_BLOCKED: checkpoint digest mismatch {digest}")
    for split in ('train','val','test'):
        for cls in ('GreenApple','NonGreenApple'):
            if not list((DATA/split/cls).glob('*')): raise SystemExit(f"empty class folder {DATA/split/cls}")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
    train_tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(0.78,1.0)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)])
    eval_tf=transforms.Compose([transforms.Resize(232),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean,std)])
    train_ds=datasets.ImageFolder(DATA/'train',transform=train_tf)
    val_ds=datasets.ImageFolder(DATA/'val',transform=eval_tf)
    test_ds=datasets.ImageFolder(DATA/'test',transform=eval_tf)
    if train_ds.class_to_idx != {'GreenApple':0,'NonGreenApple':1}: raise SystemExit(f"unexpected mapping {train_ds.class_to_idx}")
    train_dl=DataLoader(train_ds,batch_size=16,shuffle=True,num_workers=0)
    val_dl=DataLoader(val_ds,batch_size=32,shuffle=False,num_workers=0)
    test_dl=DataLoader(test_ds,batch_size=32,shuffle=False,num_workers=0)
    model=mobilenet_v3_large(weights=None)
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    model.classifier[3]=nn.Linear(model.classifier[3].in_features,2)
    device=torch.device('cpu'); model.to(device)
    RUN.mkdir(parents=True,exist_ok=False)
    (RUN/'logs').mkdir()
    config={'architecture':'mobilenet_v3_large','weights_enum':'IMAGENET1K_V2','checkpoint':str(WEIGHTS),'checkpoint_sha256':digest,'class_to_idx':train_ds.class_to_idx,'train':len(train_ds),'val':len(val_ds),'test':len(test_ds),'device':str(device),'batch_size':16,'input_size':224,'eval_resize':232,'normalization':{'mean':mean,'std':std},'labels':'AI-assisted CLIP plus green-pixel-support triage; not human verified','selection':'minimum validation cross-entropy; test evaluated once after selection','epochs_max':30,'patience':7,'stage_a_frozen_epochs':3,'seed':SEED}
    (RUN/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    criterion=nn.CrossEntropyLoss(); history=[]; best=float('inf'); best_epoch=0; wait=0
    optimizer_a=torch.optim.AdamW(model.classifier.parameters(),lr=2e-4,weight_decay=1e-4)
    optimizer_b=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4)
    for epoch in range(30):
        freeze=epoch<3
        for p in model.features.parameters(): p.requires_grad=not freeze
        optimizer=optimizer_a if freeze else optimizer_b
        model.train(); train_loss=0.; correct=0; seen=0
        for x,y in train_dl:
            x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step()
            train_loss+=loss.item()*y.numel(); correct+=(logits.argmax(1)==y).sum().item(); seen+=y.numel()
        model.eval(); val_loss=0.; val_correct=0; n=0
        with torch.no_grad():
            for x,y in val_dl:
                logits=model(x.to(device)); y=y.to(device); val_loss+=criterion(logits,y).item()*y.numel(); val_correct+=(logits.argmax(1)==y).sum().item(); n+=y.numel()
        row={'epoch':epoch+1,'stage':'A_frozen' if freeze else 'B_unfrozen','train_loss':train_loss/seen,'train_accuracy':correct/seen,'val_loss':val_loss/n,'val_accuracy':val_correct/n}
        history.append(row)
        with (RUN/'training_history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
        print(json.dumps(row),flush=True)
        if row['val_loss']<best:
            best=row['val_loss'];best_epoch=epoch+1;wait=0
            torch.save({'state_dict':model.state_dict(),'epoch':best_epoch,'best_val_loss':best,'class_to_idx':train_ds.class_to_idx,'checkpoint_sha256':digest},RUN/'best_model.pt')
        else: wait+=1
        if wait>=7: break
    saved=torch.load(RUN/'best_model.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict']); model.eval()
    ys=[]; ps=[]
    with torch.no_grad():
        for x,y in test_dl: ys.extend(y.tolist()); ps.extend(model(x).argmax(1).tolist())
    cm=[[sum(1 for a,b in zip(ys,ps) if a==i and b==j) for j in range(2)] for i in range(2)]
    metrics={}
    fs=[]
    for i,name in enumerate(('GreenApple','NonGreenApple')):
        tp=cm[i][i]; fp=sum(cm[r][i] for r in range(2) if r!=i); fn=sum(cm[i][c] for c in range(2) if c!=i)
        prec=tp/(tp+fp) if tp+fp else 0.; rec=tp/(tp+fn) if tp+fn else 0.; f1=2*prec*rec/(prec+rec) if prec+rec else 0.
        metrics[name]={'precision':prec,'recall':rec,'f1':f1,'support':sum(cm[i])};fs.append(f1)
    result={'best_epoch':best_epoch,'best_val_loss':best,'test_count':len(ys),'test_accuracy':sum(a==b for a,b in zip(ys,ps))/len(ys),'test_macro_f1':sum(fs)/2,'test_confusion_matrix':cm,'per_class':metrics,'interpretation':'Agreement with AI-generated color labels only; exploratory, not ground-truth accuracy.'}
    (RUN/'test_metrics_ai_labels.json').write_text(json.dumps(result,indent=2)+'\n')
    with (RUN/'test_predictions_ai_labels.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['path','target_idx','prediction_idx','target_name','prediction_name']);
        for (path,_),y,p in zip(test_ds.samples,ys,ps): w.writerow([path,y,p,test_ds.classes[y],test_ds.classes[p]])
    result['best_checkpoint_sha256']=sha256(RUN/'best_model.pt')
    (RUN/'test_metrics_ai_labels.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FINAL',json.dumps(result),flush=True)

if __name__=='__main__': main()
