#!/usr/bin/env python3
"""Rebuild a read-only banana instance audit from the pinned 345-image run."""
from __future__ import annotations
import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "work/benchmarks/candidate_eval"
RUN = HERE / "full_runs/fresh_clip_swap_only_345"
OUT = ROOT / "work/audit/banana_instance_audit"
sys.path.insert(0, str(HERE))
import evaluator as ev

def load(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def decode(r):
    h,w=r['size']; flat=np.empty(h*w,np.uint8); off=0; val=0
    for n in r['counts']:
        flat[off:off+n]=val; off+=n; val=1-val
    if off != h*w: raise ValueError('bad RLE length')
    return flat.reshape(h,w)
def iou(a,b):
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
    return float(inter/union) if union else 0.0
def boxmask(box,w,h):
    m=np.zeros((h,w),np.uint8); x1,y1,x2,y2=np.rint(box).astype(int)
    x1,x2=np.clip([x1,x2],0,w-1); y1,y2=np.clip([y1,y2],0,h-1)
    if x2>=x1 and y2>=y1: m[y1:y2+1,x1:x2+1]=1
    return m
def rect_iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=x*y; u=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return float(inter/u) if u else 0.0
def contour(mask,color,img):
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,contours,-1,color,2)
def label(img,xy,s,color):
    x,y=map(int,xy); cv2.rectangle(img,(x,y-20),(x+max(65,len(s)*9),y+2),color,-1); cv2.putText(img,s,(x+2,y-4),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)

def main():
    err=load(HERE/'error_analysis.json'); recdir=RUN/'records'; lookup={}
    for role in ('baseline','ground_truth'):
        lookup[role]={p.stem:load(p) for p in (recdir/role).glob('*.json')}
    fns={}; fps={}
    for d in err['unmatched_ground_truth_details']:
        if d['class_name']=='banana': fns.setdefault(d['image'],[]).append(d)
    for d in err['unmatched_prediction_details']:
        if d['class_name']=='banana': fps.setdefault(d['image'],[]).append(d)
    reviewed=[]; categories=Counter(); cluster_files=[]; all_img=[]; expected_hashes=[]
    for image_info in err['images']:
        rel=image_info['image']; key=hashlib.sha256(rel.encode()).hexdigest()
        gt=lookup['ground_truth'][key]; pr=lookup['baseline'][key]; w,h=gt['width'],gt['height']
        if gt['image_sha256']!=pr['image_sha256'] or gt['label_sha256']!=pr['label_sha256']: raise ValueError(f'record mismatch: {rel}')
        img_path=ROOT/gt['image_path']; lab_path=ROOT/gt['label_path']
        actual_img=sha(img_path); actual_lab=sha(lab_path)
        expected_hashes.append((actual_img==gt['image_sha256'],actual_lab==gt['label_sha256']))
        gt_items=[]
        for x in gt['instances']:
            m=decode(x['mask_rle']); gt_items.append((x,m))
        pred_items=[]
        for x in pr['instances']:
            m=ev.polygon_to_mask(x['polygon'],w,h); pred_items.append((x,m))
        bgt=[x for x,m in gt_items if x['class_name']=='banana']; bms=[m for x,m in gt_items if x['class_name']=='banana']
        pair_ov=[]
        for i in range(len(bms)):
            for j in range(i+1,len(bms)):
                ov=iou(bms[i],bms[j])
                if ov>0: pair_ov.append({'gt_ids':[bgt[i]['instance_id'],bgt[j]['instance_id']],'mask_iou':ov})
        cluster=len(bms)>=3 or bool(pair_ov)
        if cluster: cluster_files.append(rel)
        image_categories=[]; entities=[]
        matches=image_info.get('match_records',[])
        # GT unmatched: geometric candidate flags are cues for review, not adjudications.
        for item in fns.get(rel,[]):
            idx=int(item['instance_id'].split('_')[-1]); gx,gm=gt_items[idx]
            overlaps=[(x,m,iou(gm,m)) for x,m in pred_items]
            best=max((z[2] for z in overlaps),default=0)
            touching=[(x,m,v) for x,m,v in overlaps if v>0]
            flags=[]
            if not touching: flags.append('missed_detection_candidate')
            elif item.get('max_prediction_bbox_iou',0)>=.5 and best<.5: flags.append('incorrect_segmentation_or_box_candidate')
            if cluster: flags.append('occlusion_or_cluster_context')
            if any(x['class_name']!='banana' and v>=.5 for x,m,v in overlaps): flags.append('incorrect_class_assignment')
            image_categories += flags; entities.append({'kind':'unmatched_gt','id':gx['instance_id'],'class':'banana','flags':flags,'best_prediction_mask_iou':best,'visual_review_required':True})
        # FP banana cues: multi-GT coverage may indicate merge; duplicates are mask overlap between preds.
        for item in fps.get(rel,[]):
            idx=int(item['instance_id'].split('_')[-1]); px,pm=pred_items[idx]; flags=[]
            banana_over=[(x,m,iou(pm,m)) for x,m in gt_items if x['class_name']=='banana']
            significant=[z for z in banana_over if z[2]>=.05]
            if len(significant)>=2: flags.append('possible_merged_instances')
            if not banana_over or max((z[2] for z in banana_over),default=0)==0: flags.append('possible_missing_annotation_or_false_positive')
            dups=[]
            for j,(qx,qm) in enumerate(pred_items):
                if j!=idx and qx['class_name']=='banana' and iou(pm,qm)>=.5: dups.append(qx['instance_id'])
            if dups: flags.append('duplicate_prediction_candidate')
            if cluster: flags.append('occlusion_or_cluster_context')
            image_categories += flags; entities.append({'kind':'unmatched_prediction','id':px['instance_id'],'class':'banana','confidence':px['confidence'],'flags':flags,'overlapping_banana_gt':[{ 'id':x['instance_id'],'mask_iou':v} for x,m,v in significant],'duplicate_prediction_ids':dups,'visual_review_required':True})
        # matched class errors involving banana
        class_err=[]
        for m in matches:
            gx,px=gt_items[m['gt_id']][0],pred_items[m['prediction_id']][0]
            if (gx['class_name']=='banana') != (px['class_name']=='banana'):
                class_err.append({'gt_id':gx['instance_id'],'gt_class':gx['class_name'],'prediction_id':px['instance_id'],'pred_class':px['class_name'],'mask_iou':m['mask_iou']})
                image_categories.append('incorrect_class_assignment')
        # Render every difficult image and every banana cluster; leave originals untouched.
        if entities or class_err or cluster:
            canvas=cv2.imread(str(img_path),cv2.IMREAD_COLOR)
            if canvas is None: raise ValueError(f'cannot read {img_path}')
            for x,m in gt_items:
                if x['class_name']=='banana':
                    contour(m,(30,180,40),canvas); ys,xs=np.where(m)
                    if len(xs): label(canvas,(xs.min(),ys.min()),'GT '+x['instance_id'],(30,140,35))
            for x,m in pred_items:
                if x['class_name']=='banana':
                    contour(m,(20,20,235),canvas); ys,xs=np.where(m)
                    if len(xs): label(canvas,(xs.min(),ys.min()),'P '+x['instance_id'],(20,20,210))
            cv2.putText(canvas,'GT banana=green | predicted banana=red',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(40,40,40),2,cv2.LINE_AA)
            target=OUT/'visualizations'/Path(rel).with_suffix('.jpg'); target.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(target),canvas)
        for c in set(image_categories): categories[c]+=1
        all_img.append({'image':rel,'image_id':image_info['image_id'],'image_sha256':gt['image_sha256'],'label_sha256':gt['label_sha256'],'gt_banana_count':len(bms),'pred_banana_count':sum(x['class_name']=='banana' for x,m in pred_items),'unmatched_gt_banana':len(fns.get(rel,[])),'unmatched_pred_banana':len(fps.get(rel,[])),'banana_gt_overlap_pairs':pair_ov,'banana_cluster':cluster,'class_errors':class_err,'entities':entities,'category_flags':sorted(set(image_categories)),'visualization':str((Path('visualizations')/Path(rel).with_suffix('.jpg'))) if entities or class_err or cluster else None})
    if not all(a and b for a,b in expected_hashes): raise ValueError('source data hashes differ from pinned benchmark')
    difficult=sorted(all_img,key=lambda x:(x['unmatched_gt_banana']+x['unmatched_pred_banana']+2*len(x['class_errors'])+len(x['banana_gt_overlap_pairs']),x['gt_banana_count']),reverse=True)[:30]
    result={'title':'Banana instance detection root cause audit','protocol':{'benchmark':'fresh_clip_swap_only_345','images':len(all_img),'matching':'class-agnostic one-to-one maximum eligible mask IoU','iou_threshold':.50,'source_hashes_verified':len(expected_hashes),'dataset_modified':False,'model_modified':False,'retrained':False,'diagnosis_limit':'Geometry-derived categories are triage heuristics. Missing annotation vs false positive and visually ambiguous merge/box errors require human review.'},'benchmark_counts':{'banana_gt':156,'matched_banana_predictions':108,'banana_gt_classified_as_apple':1,'unmatched_banana_gt':sum(x['unmatched_gt_banana'] for x in all_img),'unmatched_banana_predictions':sum(x['unmatched_pred_banana'] for x in all_img),'banana_clusters_or_overlaps':len(cluster_files)},'category_image_counts':dict(categories),'banana_cluster_images':cluster_files,'ranked_difficult_images_top30':difficult,'images':all_img,'recommendations':['Manually adjudicate high-ranked cluster and merged-instance candidates against original pixels before editing labels.','Repair only confirmed missing or merged banana instances; preserve untouched source split and record image/label hashes.','Measure instance separation on dense/occluded clusters; current OBB and watershed validation results are worse than raw segmentation candidate, so do not promote them.','Use verified corrected instances for a controlled fine-tuning experiment only after split leakage and label quality review.','Keep CLIP classification separate from instance recall: it cannot recover a banana instance absent from detector masks.']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'banana_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (OUT/'banana_audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['rank','image','gt_banana_count','pred_banana_count','unmatched_gt_banana','unmatched_pred_banana','banana_cluster','cluster_overlap_pairs','class_error_count','category_flags','visualization']
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader()
        ranks={x['image']:i for i,x in enumerate(difficult,1)}
        for x in all_img:
            wr.writerow({'rank':ranks.get(x['image'],''),'image':x['image'],'gt_banana_count':x['gt_banana_count'],'pred_banana_count':x['pred_banana_count'],'unmatched_gt_banana':x['unmatched_gt_banana'],'unmatched_pred_banana':x['unmatched_pred_banana'],'banana_cluster':x['banana_cluster'],'cluster_overlap_pairs':len(x['banana_gt_overlap_pairs']),'class_error_count':len(x['class_errors']),'category_flags':';'.join(x['category_flags']),'visualization':x['visualization'] or ''})
    print(json.dumps({'output':str(OUT),'images':len(all_img),'unmatched_gt_banana':result['benchmark_counts']['unmatched_banana_gt'],'unmatched_pred_banana':result['benchmark_counts']['unmatched_banana_predictions'],'cluster_images':len(cluster_files),'category_image_counts':dict(categories),'top30':len(difficult)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
