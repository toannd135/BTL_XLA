#!/usr/bin/env python3
"""AI-assisted green/non-green apple crop triage; does not modify source crops."""
import csv, hashlib, json
from pathlib import Path
import cv2
import numpy as np
import torch, open_clip
from PIL import Image
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[5]
MAN=ROOT/'fruit/work/fruit_type_cls_experiment/phase6b_candidate/manifests/complete_manifest.csv'
MODEL=ROOT/'fruit/work/weights/hf/hub/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/snapshots/1a25a446712ba5ee05982a381eed697ef9b435cf/open_clip_model.safetensors'
EXPECTED='ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6'
OUT=ROOT/'fruit/work/fruit_type_cls_experiment/phase9_2_green_apple_expansion/green_apple_classifier_ai_scores.csv'
PROMPTS={
 'green':['a close-up photo of an apple with clearly green peel','a green apple fruit','an apple with green skin on the whole fruit','a ripe green apple'],
 'red':['a close-up photo of a red apple','an apple with clearly red peel','a ripe apple with red skin'],
 'yellow':['a close-up photo of a yellow apple','an apple with clearly yellow peel','a ripe yellow apple'],
 'other':['an apple with mixed red and green peel','an apple with brown or russet peel','a non-green apple']
}
def main():
 h=hashlib.sha256(MODEL.read_bytes()).hexdigest()
 if h!=EXPECTED:raise SystemExit('Local CLIP checkpoint SHA mismatch')
 device='cuda' if torch.cuda.is_available() else 'cpu'
 model,_,pre=open_clip.create_model_and_transforms('ViT-B-32',pretrained=None,device=device,precision='fp32')
 model.load_state_dict(load_file(str(MODEL),device='cpu'),strict=True);model.eval()
 tok=open_clip.get_tokenizer('ViT-B-32')
 protos=[]
 with torch.inference_mode():
  for name,ps in PROMPTS.items():
   t=model.encode_text(tok(ps).to(device));t=t/t.norm(dim=-1,keepdim=True);v=t.mean(0);protos.append(v/v.norm())
  text=torch.stack(protos)
 rows=list(csv.DictReader(MAN.open(encoding='utf-8-sig',newline='')))
 rows=[r for r in rows if r['fruit_class'].lower()=='apple' and r['decision']=='VERIFIED_CORRECT' and r['build_group']=='included' and r['source_family_id'] and r['split_v2'] in {'train','val','test'}]
 out=[]
 for start in range(0,len(rows),32):
  chunk=rows[start:start+32];views=[]
  for r in chunk:
   p=ROOT/'fruit/work/fruit_type_cls_experiment/phase6b_candidate'/r['candidate_copy_path']
   im=Image.open(p).convert('RGB');im.load()
   # Full reviewed crop plus central focus crop are separate model views.
   w,h=im.size;cx0=int(w*.12);cx1=int(w*.88);cy0=int(h*.12);cy1=int(h*.88)
   center=im.crop((cx0,cy0,cx1,cy1))
   views.extend([pre(im),pre(center)])
  with torch.inference_mode():
   x=torch.stack(views).to(device);feat=model.encode_image(x);feat=feat/feat.norm(dim=-1,keepdim=True);scores=(feat@text.T).cpu().numpy()
  for i,r in enumerate(chunk):
   s1=scores[2*i];s2=scores[2*i+1]
   # Support-only color cue in central crop; not treated as ground truth.
   cp=ROOT/'fruit/work/fruit_type_cls_experiment/phase6b_candidate'/r['candidate_copy_path'];arr=cv2.cvtColor(cv2.imread(str(cp)),cv2.COLOR_BGR2HSV)
   hh,ww=arr.shape[:2];cx0=int(ww*.12);cx1=int(ww*.88);cy0=int(hh*.12);cy1=int(hh*.88);a=arr[cy0:cy1,cx0:cx1]
   green=float(np.mean((a[:,:,0]>=28)&(a[:,:,0]<=100)&(a[:,:,1]>=35)))
   # Aggregate non-green prototype with maximum red/yellow/other affinity.
   ng1=float(np.max(s1[1:]));ng2=float(np.max(s2[1:]));green_score=float(min(s1[0],s2[0]));non_score=float(min(ng1,ng2));margin=green_score-non_score
   class1='AI_GREEN_CANDIDATE' if margin>=.02 and green>=.18 else ('AI_NON_GREEN_CANDIDATE' if margin<=-.02 and green<=.30 else 'AMBIGUOUS_AI_REVIEW')
   record={k:r[k] for k in ['image','source_image_path','source_image_sha256','crop_path','crop_sha256','crop_width','crop_height','fruit_class','split_v2','source','source_family_id','decision','review_id','reviewer_id']}
   record.update({'clip_model':'ViT-B-32 laion2b_s34b_b79k','clip_sha256':h,'full_view_scores':json.dumps([float(x) for x in s1]),'center_view_scores':json.dumps([float(x) for x in s2]),'green_min_view_score':green_score,'non_green_min_view_score':non_score,'green_minus_non_green_margin':margin,'center_green_pixel_ratio_support_only':green,'ai_color_decision':class1,'human_verified_color':'false','decision_note':'AI-assisted color triage; fruit identity review is inherited from Phase6B; green color is not human verified'})
   out.append(record)
 OUT.parent.mkdir(parents=True,exist_ok=True)
 with OUT.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print('scored',len(out),'device',device,'output',OUT)
if __name__=='__main__':main()
