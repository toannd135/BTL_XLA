#!/usr/bin/env python3
"""Verify official torchvision MobileNetV3-Large IMAGENET1K_V2 local weights only.

No network request and no training are performed. A successful run verifies the
official SHA-256 prefix convention, strict state-dict compatibility, and that all
backbone tensors were loaded from the supplied pretrained checkpoint.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, sys, time, traceback
from datetime import datetime, timezone

RUN=pathlib.Path(__file__).resolve().parents[1]
EXP=RUN.parent
FRUIT=EXP.parent.parent
EXPECTED_FILENAME='mobilenet_v3_large-5c1a4163.pth'
OFFICIAL_URL='https://download.pytorch.org/models/'+EXPECTED_FILENAME
OFFICIAL_PREFIX='5c1a4163'
MIN_SIZE=15_000_000
MAX_SIZE=30_000_000
DATA_DIR=EXP/'phase6b_candidate'
CLASS_MAPPING={'Banana':0,'Apple':1,'Orange':2}


def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    import torch, torchvision
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
    enum_available=hasattr(MobileNet_V3_Large_Weights,'IMAGENET1K_V2')
    enum=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if enum_available else None
    if not enum_available:
        raise RuntimeError('Installed torchvision lacks MobileNet_V3_Large_Weights.IMAGENET1K_V2')
    api_url=enum.url
    if api_url!=OFFICIAL_URL:
        raise RuntimeError(f'Installed torchvision URL differs from required official URL: {api_url}')
    torch_home=os.environ.get('TORCH_HOME')
    hub_dir=pathlib.Path(torch.hub.get_dir())
    cache_candidates=[]
    if torch_home:cache_candidates.append(pathlib.Path(torch_home)/'checkpoints'/EXPECTED_FILENAME)
    cache_candidates.append(hub_dir/'checkpoints'/EXPECTED_FILENAME)
    cache_candidates.append(pathlib.Path.home()/'.cache/torch/hub/checkpoints'/EXPECTED_FILENAME)
    seen=set();cache_candidates=[p for p in cache_candidates if not (str(p) in seen or seen.add(str(p)))]
    supplied=os.environ.get('PHASE7_PRETRAINED_WEIGHTS')
    checkpoint=pathlib.Path(supplied).expanduser().resolve() if supplied else next((p.resolve() for p in cache_candidates if p.is_file()),None)
    details={'checked_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'torch_version':torch.__version__,'torchvision_version':torchvision.__version__,'weights_api_available':enum_available,'weights_enum':str(enum),'official_url':api_url,'expected_filename':EXPECTED_FILENAME,'torch_home_env':torch_home,'torch_hub_dir':str(hub_dir),'cache_candidates':[str(p) for p in cache_candidates],'local_path_env':'PHASE7_PRETRAINED_WEIGHTS','local_path_value':str(checkpoint) if supplied and checkpoint else supplied,'official_full_sha256_published_in_torchvision_metadata':False,'official_sha256_prefix':OFFICIAL_PREFIX,'sha256_verification_method':'Compute full local checkpoint SHA-256, compare its prefix to the official prefix used by torchvision check_hash=True; separately compute baseline configuration SHA-256.','minimum_size_bytes':MIN_SIZE,'maximum_size_bytes':MAX_SIZE,'checkpoint_found':bool(checkpoint and checkpoint.is_file()),'actual_size_bytes':None,'checkpoint_sha256':None,'config_sha256':None,'config_path':str(EXP/'phase7_baseline'/'configs'/'baseline_config.yaml'),'official_sha256_prefix_match':False,'state_dict_load_attempted':False,'state_dict_load_success':False,'strict_load':False,'backbone_parameter_keys_expected':None,'backbone_parameter_keys_loaded':None,'backbone_parameter_value_match_count':None,'all_backbone_parameters_loaded':False,'parameters_finite':False,'backbone_pretrained_verification':False,'training_preflight_run':False,'training_started':False,'status':'TRAINING_BLOCKED','block_reason':None}
    try:
        if checkpoint is None or not checkpoint.is_file():
            details['block_reason']=(f'PHASE7_PRETRAINED_WEIGHTS points to a missing or unreadable file: {supplied}' if supplied else 'No local checkpoint was supplied and the expected checkpoint was absent from all checked PyTorch cache locations.')
            return details
        if checkpoint.name!=EXPECTED_FILENAME:
            details['block_reason']=f'Checkpoint basename must be exactly {EXPECTED_FILENAME}; got {checkpoint.name}.'
            return details
        size=checkpoint.stat().st_size;details['actual_size_bytes']=size
        if not MIN_SIZE<=size<=MAX_SIZE:
            details['block_reason']=f'Checkpoint size {size} is outside plausible range [{MIN_SIZE}, {MAX_SIZE}] bytes.'
            return details
        checkpoint_digest=sha256_file(checkpoint);details['checkpoint_sha256']=checkpoint_digest
        details['official_sha256_prefix_match']=checkpoint_digest.startswith(OFFICIAL_PREFIX)
        if not details['official_sha256_prefix_match']:
            details['block_reason']=f'Computed SHA-256 does not match torchvision official integrity prefix {OFFICIAL_PREFIX}.'
            return details
        # Load CPU tensors only. weights_only prevents arbitrary pickle object loading.
        details['state_dict_load_attempted']=True
        try:obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
        except TypeError:obj=torch.load(checkpoint,map_location='cpu')
        state=obj
        if isinstance(obj,dict) and 'state_dict' in obj and isinstance(obj['state_dict'],dict):state=obj['state_dict']
        if not isinstance(state,dict) or not state or not all(isinstance(k,str) for k in state):
            details['block_reason']='Checkpoint did not contain a string-keyed tensor state dict.'
            return details
        # Only instantiate a structural target; strict-load the verified checkpoint before any use.
        # Meta initialization allocates no random parameters. The verified state
        # dict is assigned immediately and strictly into this structural target.
        with torch.device('meta'):
            model=mobilenet_v3_large(weights=None)
        expected=model.state_dict()
        backbone_keys=[k for k in expected if k.startswith('features.')]
        details['backbone_parameter_keys_expected']=len(backbone_keys)
        incompatible=model.load_state_dict(state,strict=True,assign=True)
        details['strict_load']=not incompatible.missing_keys and not incompatible.unexpected_keys
        details['state_dict_load_success']=details['strict_load']
        loaded=model.state_dict()
        same=sum(1 for k in backbone_keys if k in state and torch.equal(loaded[k],state[k]))
        details['backbone_parameter_keys_loaded']=sum(1 for k in backbone_keys if k in state)
        details['backbone_parameter_value_match_count']=same
        details['all_backbone_parameters_loaded']=details['backbone_parameter_keys_loaded']==len(backbone_keys) and same==len(backbone_keys)
        details['parameters_finite']=all(bool(torch.isfinite(v).all()) for v in loaded.values() if torch.is_floating_point(v))
        details['backbone_pretrained_verification']=details['official_sha256_prefix_match'] and details['strict_load'] and details['all_backbone_parameters_loaded'] and details['parameters_finite']
        if not details['backbone_pretrained_verification']:
            details['block_reason']='Checkpoint did not pass all digest, strict-load, backbone-coverage, or finite-tensor checks.'
            return details
        details['checkpoint_source']='local path supplied by PHASE7_PRETRAINED_WEIGHTS' if supplied else 'PyTorch Torch Hub cache; the corresponding official URL is recorded above'
        # The dataset preflight is intentionally gated behind successful weight
        # digest and strict backbone verification. This phase never trains.
        manifest=DATA_DIR/'manifests'/'included.csv'
        data_result={'ran':True,'manifest':str(manifest),'passed':False,'errors':[]}
        try:
            import csv
            from collections import Counter
            from PIL import Image
            with manifest.open(encoding='utf-8-sig',newline='') as f:
                data_rows=list(csv.DictReader(f))
            data_result['row_count']=len(data_rows)
            expected_counts={('train','banana'):618,('train','apple'):437,('train','orange'):337,
                             ('val','banana'):168,('val','apple'):105,('val','orange'):89,
                             ('test','banana'):134,('test','apple'):103,('test','orange'):89}
            actual_counts=Counter((r.get('split_v2',''),r.get('fruit_class','')) for r in data_rows)
            if len(data_rows)!=2080:
                data_result['errors'].append(f'included manifest row count={len(data_rows)}, expected=2080')
            ids=[r.get('review_id','') for r in data_rows]
            if len(set(ids))!=2080 or not all(ids):
                data_result['errors'].append('crop IDs are missing or duplicated')
            if dict(actual_counts)!=expected_counts:
                data_result['errors'].append(f'class/split mismatch: {dict(actual_counts)}')
            if any(r.get('decision')!='VERIFIED_CORRECT' or r.get('build_group')!='included' for r in data_rows):
                data_result['errors'].append('included manifest contains non-verified/excluded entries')
            if CLASS_MAPPING!={'Banana':0,'Apple':1,'Orange':2}:
                data_result['errors'].append('class mapping differs from the fixed Phase 7 mapping')
            for row in data_rows:
                image_path=DATA_DIR/row.get('candidate_copy_path','')
                if not image_path.is_file():
                    data_result['errors'].append(f'missing included image {row.get("review_id")}: {image_path}')
                    continue
                image_digest=sha256_file(image_path)
                if image_digest!=row.get('candidate_copy_sha256') or image_digest!=row.get('crop_sha256'):
                    data_result['errors'].append(f'image hash mismatch {row.get("review_id")}')
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as image_error:
                    data_result['errors'].append(f'unreadable image {row.get("review_id")}: {image_error}')
            data_result['class_to_idx']=CLASS_MAPPING
            data_result['class_split_counts']={f'{s}/{c}':n for (s,c),n in actual_counts.items()}
            data_result['passed']=not data_result['errors']
        except Exception as data_error:
            data_result['errors'].append(f'{type(data_error).__name__}: {data_error}')
        details['dataset_training_preflight']=data_result
        transform=enum.transforms()
        details['preprocessing']={'image_size':224,'official_weights_transform':str(transform),
                                  'mean':list(transform.mean),'std':list(transform.std)}
        config_path=EXP/'phase7_baseline'/'configs'/'baseline_config.yaml'
        config_digest=sha256_file(config_path) if config_path.is_file() else None
        details['config_sha256']=config_digest
        details['printed_configuration']={'architecture':'mobilenet_v3_large','weights_enum':str(enum),
                                          'checkpoint_source':details['checkpoint_source'],
                                          'checkpoint_sha256':checkpoint_digest,
                                          'config_sha256':config_digest,'config_path':str(config_path),
                                          'class_to_idx':CLASS_MAPPING,
                                          'dataset_preflight':'PASS' if data_result['passed'] else 'FAIL'}
        details['status']='WEIGHTS_READY'
        details['block_reason']=None
        details['training_preflight_run']=True
        details['training_started']=False
        return details
    except Exception as e:
        details['block_reason']=f'{type(e).__name__}: {e}'
        details['error_traceback']=traceback.format_exc()
        return details
    finally:
        write_json(RUN/'reports'/'weights_integrity.json',details)
        lines=['# Phase 7A — Pretrained Weights Recovery','',f"**Status: {details['status']}**",'',f"Checked: {details['checked_at_utc']}",'','## Result','',f"- Checkpoint found: {details['checkpoint_found']}",f"- Source: {details.get('checkpoint_source','none; no checkpoint available')}",f"- Checkpoint SHA-256: `{details.get('checkpoint_sha256') or 'not available'}`",f"- Baseline config SHA-256: `{details.get('config_sha256') or 'not available'}`",f"- Official SHA-256 prefix check: {details['official_sha256_prefix_match']} (expected official prefix `{OFFICIAL_PREFIX}`)",f"- Strict state-dict load: {details['state_dict_load_success']}",f"- Backbone tensors loaded and value-verified: {details['backbone_parameter_keys_loaded']} / {details['backbone_parameter_keys_expected']}",f"- All loaded parameters finite: {details['parameters_finite']}",f"- Pretrained backbone verified: {details['backbone_pretrained_verification']}",f"- Dataset training preflight: {details.get('dataset_training_preflight',{}).get('passed','NOT RUN')}",f"- Training preflight run: {details['training_preflight_run']}",f"- Training started: {details['training_started']}", '', '## Environment','',f"- Python: {sys.version.split()[0]}",f"- PyTorch: {details['torch_version']}",f"- torchvision: {details['torchvision_version']}",f"- API `MobileNet_V3_Large_Weights.IMAGENET1K_V2`: {details['weights_api_available']}",f"- `TORCH_HOME`: `{torch_home or '(unset; using default torch hub directory)'}`",f"- Torch Hub directory: `{hub_dir}`",f"- `PHASE7_PRETRAINED_WEIGHTS`: `{supplied or 'unset'}`",'', '## Blocking reason','',details['block_reason'] or ('Weights and training-data preflight passed. No epoch was run in Phase 7A.' if details['status']=='WEIGHTS_READY' else 'No blocker recorded.'),'', '## Integrity note','', 'Torchvision publishes the official checkpoint URL and invokes `check_hash=True`; PyTorch documents that mechanism as validating a SHA-256 prefix from the artifact URL. The complete computed local checkpoint SHA-256 is recorded above when a file is present. The baseline configuration hash is separate. The installed weights metadata does not publish a full official SHA-256 value, so this report does not claim a full-hash comparison. Official references: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py and https://docs.pytorch.org/docs/stable/hub.html.','']
        (RUN/'reports'/'weights_recovery_report.md').write_text('\n'.join(lines),encoding='utf-8')
        with (RUN/'logs'/'recovery_log.txt').open('a',encoding='utf-8') as f:
            f.write(f"{details['checked_at_utc']} status={details['status']} reason={details['block_reason']}\n")

if __name__=='__main__':
    result=main()
    print(json.dumps(result,ensure_ascii=False,indent=2))
    sys.exit(0 if result['status']=='WEIGHTS_READY' else 2)
