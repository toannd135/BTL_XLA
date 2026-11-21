#!/usr/bin/env python3
"""Reproducible controlled MobileNetV3-Large fruit-type baseline."""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, platform, random, shutil, subprocess, sys, time, traceback
from collections import Counter
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[4]
EXP = ROOT / 'work' / 'fruit_type_cls_experiment'
RUN = pathlib.Path(os.environ.get('PHASE7_RUN_DIR', str(EXP / 'phase7_baseline_v2'))).expanduser().resolve()
DATA = EXP / 'phase6b_candidate'
MANIFEST = DATA / 'manifests' / 'included.csv'
CLASS_TO_IDX = {'Banana': 0, 'Apple': 1, 'Orange': 2}
IDX_TO_CLASS = {v:k for k,v in CLASS_TO_IDX.items()}
SEED = 20261120
EXPECTED_PRETRAINED_SHA256 = '5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997'
DEFAULT_PRETRAINED_CHECKPOINT = pathlib.Path('/home/toannd/models/mobilenet/mobilenet_v3_large-5c1a4163.pth')


def sha256(path: pathlib.Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def resolve_pretrained_checkpoint(path=None):
    """Resolve and verify the one approved local ImageNet1K_V2 checkpoint."""
    candidate=path or os.environ.get('PHASE7_PRETRAINED_WEIGHTS') or str(DEFAULT_PRETRAINED_CHECKPOINT)
    checkpoint=pathlib.Path(candidate).expanduser().resolve(strict=False)
    if not checkpoint.is_file():
        raise RuntimeError(f'Pretrained checkpoint is missing or not a regular file: {checkpoint}')
    try:
        with checkpoint.open('rb') as stream:
            if not stream.read(1):raise RuntimeError(f'Pretrained checkpoint is empty: {checkpoint}')
    except OSError as exc:
        raise RuntimeError(f'Pretrained checkpoint is not readable: {checkpoint}: {exc}') from exc
    actual=sha256(checkpoint)
    if actual!=EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError(f'Pretrained checkpoint SHA-256 mismatch for {checkpoint}: expected {EXPECTED_PRETRAINED_SHA256}, got {actual}')
    return checkpoint,actual

def load_verified_pretrained_model(num_classes=3, checkpoint_path=None):
    """Strictly load verified local ImageNet weights before replacing the head."""
    import torch
    from torch import nn
    from torchvision.models import mobilenet_v3_large

    checkpoint,digest=resolve_pretrained_checkpoint(checkpoint_path)
    try:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    except Exception as exc:
        raise RuntimeError(f'Failed to load checkpoint safely with weights_only=True: {type(exc).__name__}: {exc}') from exc
    if not isinstance(state,dict) or not state or not all(isinstance(k,str) for k in state):
        raise RuntimeError('Checkpoint is corrupt or is not a plain model state_dict.')

    # weights=None constructs the architecture without invoking torchvision's
    # cache/download helper. Do not replace the classifier until strict loading
    # and tensor-by-tensor backbone verification have succeeded.
    model=mobilenet_v3_large(weights=None)
    expected=model.state_dict()
    missing=sorted(set(expected)-set(state))
    unexpected=sorted(set(state)-set(expected))
    shape_mismatches=sorted(k for k in (set(expected)&set(state)) if not hasattr(state[k],'shape') or tuple(state[k].shape)!=tuple(expected[k].shape))
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(f'Checkpoint state_dict structure mismatch: missing_keys={missing}; unexpected_keys={unexpected}; shape_mismatches={shape_mismatches}')
    try:
        load_result=model.load_state_dict(state,strict=True)
    except Exception as exc:
        raise RuntimeError(f'Strict pretrained state_dict load failed: {type(exc).__name__}: {exc}') from exc
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f'Strict state_dict load returned incompatible keys: missing_keys={load_result.missing_keys}; unexpected_keys={load_result.unexpected_keys}')
    backbone_keys=[k for k in expected if k.startswith('features.')]
    mismatched=[k for k in backbone_keys if not torch.equal(model.state_dict()[k].cpu(),state[k].cpu())]
    if mismatched:
        raise RuntimeError(f'Loaded backbone tensors differ from verified checkpoint: {mismatched}')
    pretrained_finite=all(torch.isfinite(p.detach()).all().item() for p in model.parameters() if p.is_floating_point())
    if not pretrained_finite:
        raise RuntimeError('Loaded model contains non-finite floating-point parameters.')
    verification={'checkpoint_path':str(checkpoint),'checkpoint_sha256':digest,'expected_sha256':EXPECTED_PRETRAINED_SHA256,
                  'strict_load':True,'missing_keys':[],'unexpected_keys':[],'backbone_tensors_compared':len(backbone_keys),
                  'all_backbone_tensors_match':True,'all_floating_parameters_finite':pretrained_finite}
    model.classifier[3]=nn.Linear(model.classifier[3].in_features,num_classes)
    if model.classifier[3].out_features!=num_classes:
        raise RuntimeError(f'Classifier head has {model.classifier[3].out_features} outputs; expected {num_classes}.')
    if not all(torch.isfinite(p.detach()).all().item() for p in model.parameters() if p.is_floating_point()):
        raise RuntimeError('Model contains non-finite floating-point parameters after classifier initialization.')
    verification['classifier_outputs']=model.classifier[3].out_features
    return model,verification

def validate_run_directory(run_dir):
    run_dir=pathlib.Path(run_dir).expanduser().resolve(strict=False)
    original=(EXP/'phase7_baseline').resolve()
    if run_dir==original:
        raise RuntimeError(f'Refusing to use original Phase 7 output directory: {run_dir}')
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f'Refusing to overwrite non-empty run directory: {run_dir}')
    return run_dir

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields:fields.append(k)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def git_commit():
    try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return 'unavailable'
def make_environment():
    import torch, torchvision, sklearn, PIL, numpy
    cpu_count=os.cpu_count()
    try:
        import psutil
        ram={'total_gb':round(psutil.virtual_memory().total/1024**3,2),'available_gb':round(psutil.virtual_memory().available/1024**3,2)}
    except Exception:ram={'status':'psutil unavailable'}
    gpu=[]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop=torch.cuda.get_device_properties(i)
            gpu.append({'index':i,'name':torch.cuda.get_device_name(i),'vram_gb':round(prop.total_memory/1024**3,2)})
    d=shutil.disk_usage(RUN)
    return {'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_available':torch.cuda.is_available(),'torch_cuda_build':torch.version.cuda,'gpu':gpu,'cpu_cores':cpu_count,'ram':ram,'disk_free_gb_at_start':round(d.free/1024**3,2),'scikit_learn':sklearn.__version__,'pillow':PIL.__version__,'numpy':numpy.__version__,'git_commit':git_commit(),'seed':SEED}
def preflight():
    problems=[]
    if not DATA.is_dir():problems.append(f'missing dataset directory: {DATA}')
    if not MANIFEST.is_file():problems.append(f'missing included manifest: {MANIFEST}')
    if problems:return None,problems
    rows=read_csv(MANIFEST)
    if len(rows)!=2080:problems.append(f'included manifest count {len(rows)} != 2080')
    ids=[r.get('review_id','') for r in rows]
    if not all(ids) or len(set(ids))!=len(ids):problems.append('missing or duplicate crop IDs')
    counts=Counter((r.get('split_v2',''),r.get('fruit_class','')) for r in rows)
    expected={('train','banana'):618,('train','apple'):437,('train','orange'):337,('val','banana'):168,('val','apple'):105,('val','orange'):89,('test','banana'):134,('test','apple'):103,('test','orange'):89}
    if dict(counts)!=expected:problems.append(f'class/split mismatch: {dict(counts)}')
    for r in rows:
        c=r.get('fruit_class','').capitalize()
        if c not in CLASS_TO_IDX:problems.append(f"unknown class {r.get('fruit_class')} ({r.get('review_id')})")
        if r.get('decision')!='VERIFIED_CORRECT' or r.get('build_group')!='included':problems.append(f"non-included decision in included manifest: {r.get('review_id')}")
        if r.get('split_v2') not in {'train','val','test'}:problems.append(f"unknown split: {r.get('review_id')}")
        img=DATA/r.get('candidate_copy_path','')
        if not img.is_file():problems.append(f"missing included image {r.get('review_id')}: {img}");continue
        if sha256(img)!=r.get('candidate_copy_sha256') or sha256(img)!=r.get('crop_sha256'):
            problems.append(f"hash mismatch {r.get('review_id')}")
        try:
            from PIL import Image
            with Image.open(img) as im:im.verify()
        except Exception as e:problems.append(f"unreadable image {r.get('review_id')}: {e}")
    complete=read_csv(DATA/'manifests'/'complete_manifest.csv')
    if len(complete)!=2327:problems.append(f'complete manifest count {len(complete)} != 2327')
    if any(r.get('build_group') in {'excluded','quarantine'} and r.get('decision')=='VERIFIED_CORRECT' for r in complete):problems.append('excluded/quarantine entry marked VERIFIED_CORRECT')
    excluded_ids={r['review_id'] for r in complete if r.get('build_group') in {'excluded','quarantine'}}
    if any(r['review_id'] in excluded_ids for r in rows):problems.append('excluded or quarantine ID is present in included manifest')
    return rows,problems

def save_blocked_report(env,problems,reason):
    text=['# Phase 7 Controlled Baseline','',f'**Status: TRAINING_BLOCKED**','',f'Updated: {datetime.now().astimezone().isoformat(timespec="seconds")}','',f'## Reason','',reason,'','## Preflight findings','']+[f'- {x}' for x in problems]
    (RUN/'reports'/'phase7_baseline_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    write_json(RUN/'metrics'/'preflight_report.json',{'passed':not problems,'problems':problems,'environment':env})

def main():
    # Only this output tree is written. Its existence is checked by the launcher before run.
    env=make_environment()
    (RUN/'logs'/'environment.txt').write_text(json.dumps(env,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (RUN/'configs'/'resolved_config.json').write_text(json.dumps({'class_to_idx':CLASS_TO_IDX,'seed':SEED,'environment':env},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows,problems=preflight()
    if problems:
        save_blocked_report(env,problems,'Dataset preflight did not pass. Training was not started.')
        return 2
    write_json(RUN/'metrics'/'preflight_report.json',{'passed':True,'rows':len(rows),'class_to_idx':CLASS_TO_IDX,'class_split_counts':{f'{s}/{c}':n for (s,c),n in Counter((r['split_v2'],r['fruit_class']) for r in rows).items()},'environment':env})
    try:
        import torch
        from torch import nn
        from torch.utils.data import Dataset,DataLoader
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix
        from PIL import Image,ImageDraw
        import numpy as np
    except Exception as e:
        save_blocked_report(env,[],f'Required dependency import failed: {type(e).__name__}: {e}')
        return 2
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size=32 if device.type=='cuda' else 16
    # Avoid CPU oversubscription on smaller machines.
    if device.type=='cpu':torch.set_num_threads(min(12,os.cpu_count() or 1))
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    train_tf=transforms.Compose([
        transforms.RandomResizedCrop(224,scale=(0.85,1.0),ratio=(0.9,1.1),interpolation=InterpolationMode.BILINEAR,antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10,interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.08,hue=0.01),
        transforms.ToTensor(),transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])
    val_tf=MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()
    class CropSet(Dataset):
        def __init__(self,subset,tf):self.items=[r for r in rows if r['split_v2']==subset];self.tf=tf
        def __len__(self):return len(self.items)
        def __getitem__(self,i):
            r=self.items[i];p=DATA/r['candidate_copy_path']
            with Image.open(p) as im:x=self.tf(im.convert('RGB'))
            return x,CLASS_TO_IDX[r['fruit_class'].capitalize()],i
    def seed_worker(worker_id):
        seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)
    generator=torch.Generator().manual_seed(SEED)
    default_workers=min(4,os.cpu_count() or 1)
    workers=int(os.environ.get('PHASE7_NUM_WORKERS',default_workers))
    if workers<0 or workers>(os.cpu_count() or 1):
        raise ValueError(f'PHASE7_NUM_WORKERS must be between 0 and {os.cpu_count() or 1}; got {workers}')
    train_ds=CropSet('train',train_tf);val_ds=CropSet('val',val_tf);test_ds=CropSet('test',val_tf)
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker,generator=generator,persistent_workers=workers>0)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker,persistent_workers=workers>0)
    test_loader=DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker,persistent_workers=workers>0)
    try:
        model,pretrained_verification=load_verified_pretrained_model(num_classes=3)
    except Exception as e:
        reason=f'Verified local pretrained checkpoint could not be loaded; training was stopped before epoch 1. No random backbone fallback or download was attempted. {type(e).__name__}: {e}'
        save_blocked_report(env,[],reason)
        write_json(RUN/'reports'/'pretrained_checkpoint_verification.json',{'status':'TRAINING_BLOCKED','reason':reason,'expected_sha256':EXPECTED_PRETRAINED_SHA256,'checkpoint_path':os.environ.get('PHASE7_PRETRAINED_WEIGHTS',str(DEFAULT_PRETRAINED_CHECKPOINT)),'training_started':False})
        return 3
    model.to(device)
    weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2
    write_json(RUN/'reports'/'pretrained_checkpoint_verification.json',{'status':'VERIFIED','architecture':'mobilenet_v3_large','weights_identifier':'MobileNet_V3_Large_Weights.IMAGENET1K_V2','training_started':False,**pretrained_verification})
    # Record exact local checkpoint provenance and the official transform metadata.
    resolved={'model':'torchvision.models.mobilenet_v3_large','weights_enum':'MobileNet_V3_Large_Weights.IMAGENET1K_V2','checkpoint_path':pretrained_verification['checkpoint_path'],'checkpoint_sha256':pretrained_verification['checkpoint_sha256'],'pretrained_verification':pretrained_verification,'weights_transforms':str(weights.transforms()),'num_classes':3,'class_to_idx':CLASS_TO_IDX,'image_size':224,'device':str(device),'batch_size':batch_size,'num_workers':workers,'optimizer':'AdamW','loss':'CrossEntropyLoss','weight_decay':0.0001,'head_lr':0.0003,'backbone_lr_stage_b':0.00003,'max_epochs':40,'stage_a_epochs':5,'scheduler':'ReduceLROnPlateau(factor=0.5, patience=3, monitor=val_loss)','early_stopping':{'patience':8,'min_delta':0.0001},'augmentations':{'horizontal_flip':0.5,'rotation_degrees':10,'random_resized_crop_scale':[0.85,1.0],'aspect_ratio':[0.9,1.1],'color_jitter':{'brightness':0.1,'contrast':0.1,'saturation':0.08,'hue':0.01}},'seed':SEED,'environment':env,'data_preflight':'PASS'}
    write_json(RUN/'configs'/'resolved_config.json',resolved)
    criterion=nn.CrossEntropyLoss()
    log=[];best_val=float('inf');best_epoch=0;best_state=None;start=time.time();stale=0
    log_path=RUN/'logs'/'training_log.csv'
    def evaluate(loader,with_predictions=False):
        model.eval();total_loss=0;ys=[];preds=[];probs=[]
        with torch.no_grad():
            for xb,yb,_ in loader:
                xb=xb.to(device);yb=yb.to(device);out=model(xb);loss=criterion(out,yb)
                total_loss+=loss.item()*yb.size(0);pr=torch.softmax(out,dim=1)
                ys.extend(yb.cpu().tolist());preds.extend(out.argmax(1).cpu().tolist());probs.extend(pr.cpu().tolist())
        p,r,f,_=precision_recall_fscore_support(ys,preds,labels=[0,1,2],zero_division=0)
        metrics={'loss':total_loss/len(ys),'accuracy':accuracy_score(ys,preds),'macro_precision':float(p.mean()),'macro_recall':float(r.mean()),'macro_f1':float(f.mean()),'weighted_f1':precision_recall_fscore_support(ys,preds,labels=[0,1,2],average='weighted',zero_division=0)[2]}
        return metrics,(ys,preds,probs) if with_predictions else None
    def loader_train_epoch():
        model.train()
        if all(not p.requires_grad for p in model.features.parameters()):model.features.eval()
        total=0;correct=0;loss_sum=0
        for xb,yb,_ in train_loader:
            xb=xb.to(device);yb=yb.to(device);optimizer.zero_grad(set_to_none=True);out=model(xb);loss=criterion(out,yb);loss.backward();optimizer.step()
            loss_sum+=loss.item()*yb.size(0);correct+=(out.argmax(1)==yb).sum().item();total+=yb.size(0)
        return loss_sum/total,correct/total
    try:
        for epoch in range(1,41):
            stage='A_feature_extraction' if epoch<=5 else 'B_finetuning'
            if epoch==1:
                for name,p in model.named_parameters():p.requires_grad=name.startswith('classifier.')
                optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=0.0003,weight_decay=0.0001)
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3,min_lr=1e-6)
            elif epoch==6:
                for p in model.parameters():p.requires_grad=True
                optimizer=torch.optim.AdamW([{'params':model.features.parameters(),'lr':0.00003},{'params':model.classifier.parameters(),'lr':0.0003}],weight_decay=0.0001)
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3,min_lr=1e-6)
            t=time.time();train_loss,train_acc=loader_train_epoch();val_metrics,_=evaluate(val_loader)
            scheduler.step(val_metrics['loss'])
            lrs=[g['lr'] for g in optimizer.param_groups]
            row={'epoch':epoch,'stage':stage,'train_loss':train_loss,'train_accuracy':train_acc,'val_loss':val_metrics['loss'],'val_accuracy':val_metrics['accuracy'],'val_macro_f1':val_metrics['macro_f1'],'val_weighted_f1':val_metrics['weighted_f1'],'learning_rates':json.dumps(lrs),'epoch_seconds':round(time.time()-t,3)};log.append(row)
            write_csv(log_path,log)
            if val_metrics['loss']<best_val-0.0001:
                best_val=val_metrics['loss'];best_epoch=epoch;stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save({'model_state_dict':best_state,'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'epoch':epoch,'best_val_loss':best_val,'class_to_idx':CLASS_TO_IDX,'weights_enum':str(weights),'config':resolved},RUN/'checkpoints'/'best_model.pt')
            else:stale+=1
            torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'epoch':epoch,'best_epoch':best_epoch,'best_val_loss':best_val,'class_to_idx':CLASS_TO_IDX,'weights_enum':str(weights),'config':resolved},RUN/'checkpoints'/'last_model.pt')
            print(f"epoch {epoch:02d}/40 [{stage}] train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} best={best_epoch} ({best_val:.4f})",flush=True)
            if epoch>=6 and stale>=8:
                print(f'Early stopping after {stale} epochs without val_loss improvement.',flush=True);break
    except Exception as e:
        save_blocked_report(env,[],f'Training failed: {type(e).__name__}: {e}\n{traceback.format_exc()}')
        return 4
    elapsed=time.time()-start
    if best_state is None:
        save_blocked_report(env,[],'No best validation checkpoint was produced.');return 5
    model.load_state_dict(best_state);model.to(device)
    val_metrics,val_raw=evaluate(val_loader,True)
    val_ys,val_preds,_=val_raw
    val_report=classification_report(val_ys,val_preds,labels=[0,1,2],target_names=[IDX_TO_CLASS[i] for i in range(3)],output_dict=True,zero_division=0)
    val_cm=confusion_matrix(val_ys,val_preds,labels=[0,1,2])
    write_json(RUN/'metrics'/'validation_metrics.json',{'best_epoch':best_epoch,'selection_metric':'minimum val_loss','metrics':val_metrics,'per_class':val_report,'confusion_matrix':val_cm.tolist(),'class_order':[IDX_TO_CLASS[i] for i in range(3)],'class_to_idx':CLASS_TO_IDX,'sample_count':len(val_ds)})
    # Test is evaluated once, only after checkpoint/configuration are fixed by validation.
    fixed_checkpoint_sha=sha256(RUN/'checkpoints'/'best_model.pt')
    test_metrics,test_raw=evaluate(test_loader,True)
    ys,preds,probs=test_raw
    test_report=classification_report(ys,preds,labels=[0,1,2],target_names=[IDX_TO_CLASS[i] for i in range(3)],output_dict=True,zero_division=0)
    cm=confusion_matrix(ys,preds,labels=[0,1,2])
    write_json(RUN/'metrics'/'test_metrics.json',{'best_epoch':best_epoch,'checkpoint_sha256':fixed_checkpoint_sha,'metrics':test_metrics,'sample_count':len(test_ds),'test_evaluations':1,'interpretation':'Exploratory only: source provenance warning remains unresolved.'})
    write_json(RUN/'metrics'/'classification_report.json',test_report)
    write_csv(RUN/'metrics'/'confusion_matrix.csv',[{'true_class':IDX_TO_CLASS[i],**{IDX_TO_CLASS[j]:int(cm[i,j]) for j in range(3)}} for i in range(3)])
    test_items=test_ds.items;prediction_rows=[];errors=[]
    for n,(item,y,pred,prob) in enumerate(zip(test_items,ys,preds,probs)):
        row={'review_id':item['review_id'],'crop_path':item['candidate_copy_path'],'true_class':IDX_TO_CLASS[y],'predicted_class':IDX_TO_CLASS[pred],'correct':pred==y,'confidence':max(prob),'source_image_path':item['source_image_path'],'source_family_id':item['source_family_id'],'source_split':item['split'],'candidate_split_v2':item['split_v2']}
        for i,name in IDX_TO_CLASS.items():row[f'prob_{name}']=prob[i]
        prediction_rows.append(row)
        if pred!=y:errors.append(row)
    write_csv(RUN/'metrics'/'predictions.csv',prediction_rows)
    write_csv(RUN/'error_analysis'/'misclassified_samples.csv',errors)
    # Contact sheets for every error, at most 32 panels per sheet.
    for start_i in range(0,len(errors),32):
        subset=errors[start_i:start_i+32];tw,th=240,275;cols=4;rows_n=(len(subset)+cols-1)//cols
        sheet=Image.new('RGB',(cols*tw,rows_n*th),'white');draw=ImageDraw.Draw(sheet)
        for j,item in enumerate(subset):
            x=(j%cols)*tw;y=(j//cols)*th;im=Image.open(RUN/item['crop_path']).convert('RGB');im.thumbnail((tw-12,th-56))
            sheet.paste(im,(x+6,y+6));draw.text((x+6,y+th-44),f"{item['review_id']}\n{item['true_class']} -> {item['predicted_class']} ({item['confidence']:.2f})",fill='black')
        sheet.save(RUN/'error_analysis'/'contact_sheets'/f'misclassified_{start_i+1:04d}_{start_i+len(subset):04d}.jpg',quality=90)
    # Sensitivity is N/A if the unresolved pair's test image is not an evaluated crop.
    unresolved_names={'rf_v7__o_r069_png.rf.09e46ebd0725c16915dba94640b97920.jpg','rf_v7__o_r093_png.rf.5f02696874d5f6ce4ce0aa073dcbd1a0.jpg'}
    pair_rows=[r for r in prediction_rows if pathlib.Path(r['source_image_path']).name in unresolved_names]
    write_json(RUN/'metrics'/'provenance_sensitivity.json',{'pair_images':sorted(unresolved_names),'matching_test_prediction_rows':len(pair_rows),'sensitivity_result':'not computable: neither unresolved source image is represented in included classifier manifest' if not pair_rows else 'see prediction rows','test_set_still_exploratory':True})
    # Loss curve (train + validation), saved without reading test data.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5));plt.plot([r['epoch'] for r in log],[r['train_loss'] for r in log],label='train loss');plt.plot([r['epoch'] for r in log],[r['val_loss'] for r in log],label='validation loss');plt.axvline(best_epoch,color='gray',linestyle='--',label=f'best epoch {best_epoch}');plt.xlabel('Epoch');plt.ylabel('Cross-entropy loss');plt.title('MobileNetV3-Large fruit-type baseline');plt.legend();plt.tight_layout();plt.savefig(RUN/'metrics'/'loss_curve.png',dpi=150);plt.close()
    report=['# Phase 7 — Controlled Baseline Training','', '**Status: TRAINING_SUCCESS**','',f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",'','## Dataset and preflight','',f'- Included crops: {len(rows)}; class map fixed as `{CLASS_TO_IDX}`.',f'- Split counts: train {len(train_ds)}, validation {len(val_ds)}, test {len(test_ds)}.', '- Dataset preflight passed: included manifest, unique crop IDs, class/split counts, file existence, image decoding, image hashes, and excluded/quarantine separation. Phase 6B dataset was not modified.',f'- Active assignment is existing `split_v2`; original `split` and source-family metadata remain recorded.','', '## Model and configuration','',f'- Torchvision MobileNetV3-Large, pretrained weights `{weights}`.',f'- Official weight URL: {weights.url}.', '- Final classifier output: 3 classes (Banana=0, Apple=1, Orange=2).','- Input 224×224; ImageNet normalization; validation/test use the official weights transform (resize 232, center crop 224).','- Training-only augmentation: horizontal flip, rotation ±10°, conservative random resized crop, mild brightness/contrast/saturation and hue jitter.',f'- Optimizer AdamW; batch size {batch_size}; stage A freezes features for 5 epochs; stage B fine-tunes all layers.', '- Stage A head LR 3e-4; Stage B backbone LR 3e-5 and head LR 3e-4; weight decay 1e-4; ordinary CrossEntropyLoss, no class weighting.', '- ReduceLROnPlateau monitors validation loss; early stopping patience 8; best checkpoint selected by minimum validation loss.','', '## Environment and duration','',f'- Device: {device}; CPU cores: {env["cpu_cores"]}; RAM: {env["ram"]}; GPU: {env["gpu"] if env["gpu"] else "none (CUDA unavailable)"}.',f'- Python {platform.python_version()}, PyTorch {torch.__version__}, torchvision {env["torchvision"]}.',f'- Training duration: {elapsed:.1f} seconds; epochs completed: {len(log)}; best epoch: {best_epoch}.',f'- Git commit: `{env["git_commit"]}`.','', '## Training and validation','', '| Epoch | Stage | Train loss | Val loss | Val accuracy | Val macro F1 |','|---:|---|---:|---:|---:|---:|']
    for r in log:report.append(f"| {r['epoch']} | {r['stage']} | {r['train_loss']:.4f} | {r['val_loss']:.4f} | {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} |")
    report += ['',f"Best validation metrics: `{json.dumps(val_metrics)}`.",'','![Train and validation loss](../metrics/loss_curve.png)','', '## Test evaluation','',f"Test was evaluated once after fixing the best checkpoint; checkpoint SHA-256 `{fixed_checkpoint_sha}`.",f"Test metrics: `{json.dumps(test_metrics)}`.",'','Confusion matrix (rows=true, columns=predicted):','', '| True \\ Predicted | Banana | Apple | Orange |','|---|---:|---:|---:|']
    for i,name in IDX_TO_CLASS.items():report.append('| '+name+' | '+' | '.join(str(int(cm[i,j])) for j in range(3))+' |')
    report += ['',f'- Misclassified crops: {len(errors)}/{len(test_ds)}; see `error_analysis/misclassified_samples.csv` and contact sheets.','- Per-class precision/recall/F1 are in `metrics/classification_report.json`; per-crop confidence scores are in `metrics/predictions.csv`.','- No test-based tuning was performed.','', '## Limitations','', '- Test metrics are exploratory only: Phase 6B retains an unresolved source-level `o_r069/o_r093` train/test provenance pair. Neither source has a crop row in this classifier’s included manifest, so the requested with/without-`o_r093` sensitivity cannot be calculated from this dataset.', '- Green Banana has only 10 verified-green crops from one train-only source family; there is no independent val/test green-banana family. No green-banana generalization claim is supported.', '- Dataset crops within source families are correlated; crop count overstates independent source count.', '', '## Conclusion and next experiment','',('The baseline exceeds chance-level 3-class accuracy on validation and provides an initial fruit-type signal; interpret class-specific errors and test results in light of the provenance and source-family limits.' if val_metrics['accuracy']>1/3 else 'Validation accuracy does not exceed the 3-class chance reference; this baseline does not yet demonstrate reliable fruit-type separation. Review errors and data coverage before a follow-up experiment.'),'The next experiment should resolve the source provenance issue, add independent green-banana families to validation/test, and compare a single controlled change at a time (for example class weighting or stronger source diversity). Do not tune against this test result.','']
    (RUN/'reports'/'phase7_baseline_report.md').write_text('\n'.join(report),encoding='utf-8')
    write_json(RUN/'metrics'/'run_summary.json',{'status':'TRAINING_SUCCESS','best_epoch':best_epoch,'epochs_completed':len(log),'training_seconds':elapsed,'best_checkpoint_sha256':fixed_checkpoint_sha,'test_evaluations':1,'device':str(device),'metrics_test':test_metrics,'misclassified_count':len(errors)})
    return 0

if __name__=='__main__':
    try:
        RUN=validate_run_directory(RUN)
    except Exception as e:
        print(f'Output directory safety check failed before any write: {e}',file=sys.stderr);sys.exit(10)
    for d in ['configs','checkpoints','logs','metrics','error_analysis/contact_sheets','scripts','reports']:(RUN/d).mkdir(parents=True,exist_ok=True)
    source_config=pathlib.Path(__file__).resolve().parents[1]/'configs'/'baseline_config.yaml'
    target_config=RUN/'configs'/'baseline_config.yaml'
    if source_config.is_file() and not target_config.exists():shutil.copyfile(source_config,target_config)
    sys.exit(main())
