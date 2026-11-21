#!/usr/bin/env python3
"""Build the immutable-input Fruit Human Review Package v1. Refuses to overwrite a built manifest."""
from __future__ import annotations
import csv, hashlib, json, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[3]
DATA = ROOT / 'work/obb/seg_data_candidate'
REVIEW = ROOT / 'work/obb/seg_data_review_v1'
AUDIT = ROOT / 'work/audit/phase1_dataset_audit.json'
ANN_CSV = ROOT / 'work/audit/phase1_review_manifest.csv'
LEAK_CSV = ROOT / 'work/audit/phase1.2_leakage_review_manifest.csv'
DATASET_MANIFEST = REVIEW / 'dataset_manifest.json'
CHECKPOINT = ROOT / 'runs/seg/seg_boundary_candidate_gpu/weights/best.pt'
CLASSES = {0:'banana',1:'apple',2:'orange'}
COLORS = {0:(0,210,255),1:(60,220,60),2:(255,115,30)}
CAT_NAMES = {
 'A':'Banana clusters','B':'Orange occlusion','C':'Apple overlap',
 'D':'Mixed fruit overlap','E':'Annotation topology issues','F':'Negative images'
}

def sha(path: Path) -> str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def atomic_json(path: Path, obj):
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 tmp.replace(path)

def image_id(rel: str) -> str:
 return 'img_'+hashlib.sha256(rel.encode()).hexdigest()[:14]

def read_rows(path):
 with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
 if (PACKAGE/'manifest.json').exists() or (PACKAGE/'review_history.jsonl').exists():
  raise SystemExit('Refusing to rebuild over an existing review package/history. Use its existing manifest and decisions.')
 for p in [DATASET_MANIFEST,AUDIT,ANN_CSV,LEAK_CSV]:
  if not p.exists():raise FileNotFoundError(p)
 for d in ['assets/originals','assets/labels','assets/overlays','assets/instances','assets/leakage']:
  (PACKAGE/d).mkdir(parents=True,exist_ok=True)
 source_manifest=json.loads(DATASET_MANIFEST.read_text())
 audit=json.loads(AUDIT.read_text())
 audit_by_path={str(Path(x['image']).resolve()):x for x in audit['images']}
 issues_by_label=defaultdict(list)
 for issue in audit.get('issues',[]):issues_by_label[str(Path(issue.get('label','')).resolve())].append(issue.get('issue','unknown'))
 ann_rows=read_rows(ANN_CSV); grouped=defaultdict(list)
 for row in ann_rows:grouped[row['image_path']].append(row)
 review_items={x['image_path']:x for x in source_manifest['review_items']}
 if len(grouped)!=194 or set(grouped)!=set(review_items):raise ValueError('Review CSV and dataset manifest image inventory differ.')
 report_images=[]; review_manifest_rows=[]; generation_errors=[]; image_asset_by_source={}
 review_copy_matches=0; instance_total=0; source_ref_hash_checks=0
 for rel in sorted(grouped):
  rows=grouped[rel]; meta=review_items[rel]; split=meta['split']; src_image=Path(meta['source_image_path']); src_label=Path(meta['source_label_path'])
  iid=image_id(rel)
  if not src_image.is_file() or not src_label.is_file():raise FileNotFoundError(f'Missing source pair: {src_image} / {src_label}')
  if sha(src_image)!=meta['image_sha256'] or sha(src_label)!=meta['label_sha256']:raise ValueError(f'Source hash mismatch against review dataset manifest: {rel}')
  src_review_image=REVIEW/meta['image_path']; src_review_label=REVIEW/meta['label_path']
  if not src_review_image.is_file() or not src_review_label.is_file():raise FileNotFoundError(f'Missing review subset pair for {rel}')
  if sha(src_review_image)!=meta['image_sha256'] or sha(src_review_label)!=meta['label_sha256']:raise ValueError(f'Review subset hash mismatch: {rel}')
  review_copy_matches+=1
  bgr=cv2.imread(str(src_image),cv2.IMREAD_COLOR)
  if bgr is None:raise ValueError(f'Unreadable image: {src_image}')
  h,w=bgr.shape[:2]
  local_image=f'assets/originals/{iid}{src_image.suffix.lower()}'
  local_label=f'assets/labels/{iid}.txt'
  shutil.copyfile(src_image,PACKAGE/local_image);shutil.copyfile(src_label,PACKAGE/local_label)
  image_asset_by_source[str(src_image.resolve())]=(iid,local_image,w,h)
  if sha(PACKAGE/local_image)!=meta['image_sha256'] or sha(PACKAGE/local_label)!=meta['label_sha256']:raise ValueError(f'Copied asset hash mismatch: {rel}')
  parsed=[]
  raw_lines=src_label.read_text(encoding='utf-8').splitlines()
  labeled_rows=[r for r in rows if r.get('instance_id','').strip()]
  if len(raw_lines)!=len(labeled_rows):raise ValueError(f'Instance-row count mismatch for {rel}: label={len(raw_lines)} csv={len(labeled_rows)}')
  for row_idx,line in enumerate(raw_lines,1):
   row=labeled_rows[row_idx-1]
   try:
    parts=line.split(); cls=int(parts[0]); coords=np.asarray([float(x) for x in parts[1:]],dtype=np.float64)
    if cls not in CLASSES or len(parts)<7 or coords.size%2 or not np.isfinite(coords).all() or len(coords)<6:raise ValueError('malformed YOLO-seg polygon')
    pts=coords.reshape(-1,2)
    if cls!=int({'banana':0,'apple':1,'orange':2}[row['class']]) or int(row['instance_id'])!=row_idx:raise ValueError(f'instance id/class order mismatch at row {row_idx}')
    pix=np.rint(pts*np.array([w-1,h-1])).astype(np.int32)
    parsed.append({'instance_id':str(row_idx),'class_id':cls,'class_name':CLASSES[cls],'polygon':pts,'pixels':pix,'source_line':row_idx,'confidence':None})
   except Exception as e:generation_errors.append({'image_id':iid,'filename':src_image.name,'line':row_idx,'error':str(e)})
  info=audit_by_path.get(str(src_image.resolve()))
  if info is None:raise ValueError(f'No per-image audit record for {src_image}')
  overlap_pairs=info.get('overlap_pairs',[])
  class_counts=Counter(x['class_name'] for x in parsed)
  image_issue_flags=sorted(set(x for p in parsed for x in issues_by_label.get(str(src_label.resolve()),[]) if f'line-{p["source_line"]}:' in x))
  # Group audit issue strings are keyed by label and include line-<n>:<reason>.
  reasons={k:[] for k in CAT_NAMES}
  bananas=[x for x in parsed if x['class_id']==0]; apples=[x for x in parsed if x['class_id']==1]; oranges=[x for x in parsed if x['class_id']==2]
  banana_overlaps=[p for p in overlap_pairs if p.get('class_a')==p.get('class_b')=='banana']
  orange_overlaps=[p for p in overlap_pairs if p.get('class_a')==p.get('class_b')=='orange']
  apple_overlaps=[p for p in overlap_pairs if p.get('class_a')==p.get('class_b')=='apple']
  cross_overlaps=[p for p in overlap_pairs if p.get('class_a')!=p.get('class_b')]
  def severe(p):return float(p.get('overlap_of_smaller',0))>=0.5
  if len(bananas)>=2: reasons['A'].append(f'{len(bananas)} banana instances in one image')
  if len(bananas)>=3: reasons['A'].append('three-or-more banana instances')
  if banana_overlaps:reasons['A'].append(f'{len(banana_overlaps)} intersecting banana mask pair(s)')
  if any(severe(p) for p in banana_overlaps):reasons['A'].append('severe banana mask overlap candidate')
  if orange_overlaps:reasons['B'].append(f'{len(orange_overlaps)} intersecting orange mask pair(s)')
  if len(oranges)>=2 and not orange_overlaps:reasons['B'].append(f'{len(oranges)} orange instances; inspect stacking/partial visibility')
  boundary_oranges=[]
  for x in oranges:
   q=x['polygon'];
   if np.any(q<=0.01) or np.any(q>=0.99):boundary_oranges.append(x['instance_id'])
  if boundary_oranges:reasons['B'].append('orange polygon touches/approaches image boundary: '+','.join(boundary_oranges))
  if apple_overlaps:reasons['C'].append(f'{len(apple_overlaps)} intersecting apple mask pair(s)')
  if len(apples)>=2 and not apple_overlaps:reasons['C'].append(f'{len(apples)} apple instances; inspect touching boundaries')
  if cross_overlaps:reasons['D'].append(f'{len(cross_overlaps)} cross-class intersecting mask pair(s)')
  if len(class_counts)>=2 and not cross_overlaps:reasons['D'].append('multiple fruit classes in image; visually verify touching/overlap')
  row_groups=[r.get('review_reason','') for r in rows]
  if image_issue_flags:reasons['E'].append('polygon topology candidate(s): '+'; '.join(image_issue_flags))
  if info.get('large_mask_candidates'):reasons['E'].append('unusually large mask candidate instance(s): '+','.join(map(str,info['large_mask_candidates'])))
  if info.get('small_mask_candidates'):reasons['E'].append('unusually small mask candidate instance(s): '+','.join(map(str,info['small_mask_candidates'])))
  if info.get('duplicate_pairs'):reasons['E'].append('potential duplicate mask candidate(s)')
  if info.get('suspicious_many_instances'):reasons['E'].append('suspicious instance-count candidate')
  if not parsed:reasons['F'].append('empty source label; human must confirm negative image')
  cats=[k for k,v in reasons.items() if v]
  # Combined overlay with per-class color and explicit row instance ID.
  overlay=bgr.copy()
  for x in parsed:cv2.fillPoly(overlay,[x['pixels']],COLORS[x['class_id']])
  combined=cv2.addWeighted(overlay,0.26,bgr,0.74,0)
  for x in parsed:
   cv2.polylines(combined,[x['pixels']],True,COLORS[x['class_id']],max(2,round(min(w,h)/500)))
   center=tuple(np.clip(np.mean(x['pixels'],axis=0).astype(int),[0,0],[w-1,h-1]))
   cv2.putText(combined,f"{x['class_name']} #{x['instance_id']} (id {x['class_id']})",center,cv2.FONT_HERSHEY_SIMPLEX,max(.45,min(w,h)/1000),COLORS[x['class_id']],2,cv2.LINE_AA)
   one=bgr.copy(); mask=np.zeros((h,w),dtype=np.uint8);cv2.fillPoly(mask,[x['pixels']],1)
   tint=np.zeros_like(bgr);tint[:]=COLORS[x['class_id']]; one[mask>0]=cv2.addWeighted(bgr[mask>0],.48,tint[mask>0],.52,0)
   cv2.polylines(one,[x['pixels']],True,COLORS[x['class_id']],max(2,round(min(w,h)/400)))
   center=tuple(np.clip(np.mean(x['pixels'],axis=0).astype(int),[0,0],[w-1,h-1]))
   cv2.putText(one,f"{x['class_name']} #{x['instance_id']} (id {x['class_id']})",center,cv2.FONT_HERSHEY_SIMPLEX,max(.55,min(w,h)/850),COLORS[x['class_id']],2,cv2.LINE_AA)
   instpath=f"assets/instances/{iid}_instance_{x['instance_id']}.jpg"
   if not cv2.imwrite(str(PACKAGE/instpath),one,[cv2.IMWRITE_JPEG_QUALITY,91]):raise IOError(f'Could not write {instpath}')
   x['visualization']=instpath
  combpath=f'assets/overlays/{iid}.jpg'
  if not cv2.imwrite(str(PACKAGE/combpath),combined,[cv2.IMWRITE_JPEG_QUALITY,91]):raise IOError(f'Could not write {combpath}')
  instances=[]
  for x in parsed:
   iissues=[z for z in issues_by_label.get(str(src_label.resolve()),[]) if f"line-{x['source_line']}:" in z]
   instances.append({'instance_id':x['instance_id'],'class_id':x['class_id'],'class_name':x['class_name'],'source_line':x['source_line'],'confidence':None,'polygon_topology_flags':iissues,'polygon_points':len(x['pixels']),'visualization':x['visualization'],'decision_key':f"{iid}::{x['instance_id']}"})
   instance_total+=1
   original_row=labeled_rows[x['source_line']-1]
   review_manifest_rows.append({'image_id':iid,'filename':src_image.name,'image_path':local_image,'source_image_path':str(src_image),'label_path':local_label,'source_label_path':str(src_label),'split':split,'class_id':x['class_id'],'class_name':x['class_name'],'instance_id':x['instance_id'],'source_line':x['source_line'],'image_width':w,'image_height':h,'overlap_pair_count':len(overlap_pairs),'topology_flags':';'.join(iissues),'review_categories':';'.join(cats),'category_reasons_json':json.dumps(reasons,ensure_ascii=False),'annotation_status':'pending','human_decision':'','reviewer_id':'','reviewed_at':''})
  if not parsed:
   rr=rows[0]
   review_manifest_rows.append({'image_id':iid,'filename':src_image.name,'image_path':local_image,'source_image_path':str(src_image),'label_path':local_label,'source_label_path':str(src_label),'split':split,'class_id':'','class_name':'','instance_id':'','source_line':'','image_width':w,'image_height':h,'overlap_pair_count':len(overlap_pairs),'topology_flags':'','review_categories':';'.join(cats),'category_reasons_json':json.dumps(reasons,ensure_ascii=False),'annotation_status':'pending','human_decision':'','reviewer_id':'','reviewed_at':''})
  report_images.append({'image_id':iid,'filename':src_image.name,'image_path':local_image,'source_label_copy':local_label,'source_image_path':str(src_image),'source_label_path':str(src_label),'split':split,'width':w,'height':h,'instances_by_class':{CLASSES[k]:class_counts.get(CLASSES[k],0) for k in CLASSES},'instance_count':len(parsed),'overlap_pair_count':len(overlap_pairs),'overlap_pairs':overlap_pairs,'polygon_topology_flags':image_issue_flags,'topology_flag_count':len(image_issue_flags),'large_mask_candidates':info.get('large_mask_candidates',[]),'small_mask_candidates':info.get('small_mask_candidates',[]),'categories':cats,'category_reasons':reasons,'review_status':'pending','combined_overlay':combpath,'instances':instances,'negative':len(parsed)==0,'priority_example':src_image.name in {'mbk__banana_66.jpg','mbk__orange_1.jpg','mbk__apple_24.jpg'},'saved_model_prediction_available':False})
  source_ref_hash_checks+=2
 # Copy leakage endpoints. Reuse image asset if already included in annotation subset.
 leak_rows=read_rows(LEAK_CSV); pair_records=[];leak_asset_map={}
 for row in leak_rows:
  pair={'pair_id':row['pair_id'],'image_a':row['image_a'],'split_a':row['split_a'],'image_b':row['image_b'],'split_b':row['split_b'],'phash_distance':int(row['phash_distance']),'image_a_sha256':row['image_a_sha256'],'image_b_sha256':row['image_b_sha256'],'review_status':'pending','reviewer_decision':'','reviewer_notes':'','split_recommendation':row['split_recommendation']}
  for side in ['a','b']:
   path=Path(pair[f'image_{side}']).resolve(); key=str(path)
   if key not in image_asset_by_source:
    if not path.is_file():raise FileNotFoundError(f'Leakage image missing: {path}')
    im=cv2.imread(str(path));
    if im is None:raise ValueError(f'Unreadable leakage image: {path}')
    h,w=im.shape[:2];aid='leak_'+hashlib.sha256(key.encode()).hexdigest()[:14]
    local=f'assets/originals/{aid}{path.suffix.lower()}'
    shutil.copyfile(path,PACKAGE/local)
    if sha(PACKAGE/local)!=pair[f'image_{side}_sha256']:raise ValueError(f'Leakage copy hash mismatch: {path}')
    image_asset_by_source[key]=(aid,local,w,h)
   aid,local,w,h=image_asset_by_source[key]
   if sha(path)!=pair[f'image_{side}_sha256']:raise ValueError(f'Leakage source hash mismatch: {path}')
   pair[f'asset_id_{side}']=aid;pair[f'asset_path_{side}']=local;pair[f'width_{side}']=w;pair[f'height_{side}']=h
  pair_records.append(pair)
 # Persist deterministic initial manifests and empty decision stores.
 manifest={'schema_version':1,'package':'human_review_package_v1','class_names':{str(k):v for k,v in CLASSES.items()},'source_manifest':'work/obb/seg_data_review_v1/dataset_manifest.json','source_annotation_manifest':'work/audit/phase1_review_manifest.csv','source_leakage_manifest':'work/audit/phase1.2_leakage_review_manifest.csv','source_audit':'work/audit/phase1_dataset_audit.json','annotation_images':report_images,'leakage_pairs':pair_records,'categories':CAT_NAMES,'decision_defaults':{'instance':None,'image':None,'negative_image':None,'leakage':None},'prediction_artifacts_available':False}
 atomic_json(PACKAGE/'manifest.json',manifest)
 fields=['image_id','filename','image_path','source_image_path','label_path','source_label_path','split','class_id','class_name','instance_id','source_line','image_width','image_height','overlap_pair_count','topology_flags','review_categories','category_reasons_json','annotation_status','human_decision','reviewer_id','reviewed_at']
 with (PACKAGE/'review_manifest.csv').open('w',newline='',encoding='utf-8') as f:
  wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(review_manifest_rows)
 pair_fields=['pair_id','image_a','split_a','asset_path_a','width_a','height_a','image_a_sha256','image_b','split_b','asset_path_b','width_b','height_b','image_b_sha256','phash_distance','review_status','reviewer_decision','reviewer_notes','split_recommendation']
 with (PACKAGE/'leakage_review_manifest.csv').open('w',newline='',encoding='utf-8') as f:
  wr=csv.DictWriter(f,fieldnames=pair_fields);wr.writeheader();
  for x in pair_records:wr.writerow({k:x.get(k,'') for k in pair_fields})
 (PACKAGE/'review_history.jsonl').write_text('',encoding='utf-8')
 atomic_json(PACKAGE/'review_decisions.json',{'instances':{},'images':{},'negative_images':{},'leakage_pairs':{}})
 # Topology counts are polygon-level flags; count the subset source audit candidates.
 topology_count=sum(bool(x['polygon_topology_flags']) for im in report_images for x in im['instances'])
 split_counts={s:sum(im['split']==s for im in report_images) for s in ['train','val','test']}
 category_counts={k:sum(k in im['categories'] for im in report_images) for k in CAT_NAMES}
 progress={'schema_version':1,'generated_at':None,'source_decisions_used':0,'images_total':len(report_images),'instances_total':instance_total,'negative_images_total':sum(x['negative'] for x in report_images),'images_by_split':split_counts,'instances_by_class':dict(Counter(x['class_name'] for im in report_images for x in im['instances'])),'images_by_category':category_counts,'instance_decisions':{'accept':0,'needs_edit':0,'reject':0,'uncertain':0,'pending':instance_total},'image_decisions':{'reviewed':0,'pending':len(report_images),'needs_expert_review':0},'negative_image_decisions':{'confirmed_empty':0,'fruit_present':0,'uncertain':0,'pending':sum(x['negative'] for x in report_images)},'leakage_pairs_total':len(pair_records),'leakage_decisions':{'same_source':0,'different_source':0,'uncertain':0,'pending':len(pair_records)},'topology_candidate_polygons':topology_count,'no_default_accept':True,'counts_basis':'Derived from current saved review_decisions.json; absent human decisions count as pending.'}
 atomic_json(PACKAGE/'review_progress.json',progress)
 errors={'generation_errors':generation_errors,'source_pairs_verified':review_copy_matches,'source_hash_checks':source_ref_hash_checks,'leakage_images_copied_or_reused':len(set(p['asset_id_a'] for p in pair_records)|set(p['asset_id_b'] for p in pair_records)),'annotation_images':len(report_images),'instance_visualizations':instance_total,'leakage_pairs':len(pair_records)}
 atomic_json(PACKAGE/'generation_validation.json',errors)
 print(json.dumps(errors,indent=2))

if __name__=='__main__':main()
