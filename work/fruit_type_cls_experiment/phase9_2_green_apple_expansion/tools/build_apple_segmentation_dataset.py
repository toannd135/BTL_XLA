#!/usr/bin/env python3
"""Build a fresh, duplicate-filtered 3-class segmentation candidate from V5 sources."""
import csv, hashlib, json, math, shutil
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[5]
BASE=ROOT/'fruit/work/obb/seg_green_v5_candidate/data'
PHASE=ROOT/'fruit/work/fruit_type_cls_experiment/phase9_2_green_apple_expansion'
P9=ROOT/'fruit/work/fruit_type_cls_experiment/phase9_green_apple_rebuild'
MBASE=ROOT/'fruit/work/fruit_type_cls_experiment/phase8c_collection/green_apple_phase8d/staging/mendeley_v1'
MMAN=ROOT/'fruit/work/obb/seg_green_v5_candidate/mendeley_green_apple_train_manifest.csv'
OUT=ROOT/'fruit/work/obb/apple_segmentation_ai_v1'
DATA=OUT/'dataset'
NAMES={0:'banana',1:'apple',2:'orange'}

def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def orient(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def proper(a,b,c,d):return orient(a,b,c)*orient(a,b,d)<0 and orient(c,d,a)*orient(c,d,b)<0
def strict_crossing(pts):
 p=[tuple(map(float,x)) for x in pts]
 if len(p)>1 and p[0]==p[-1]:p=p[:-1]
 n=len(p)
 for i in range(n):
  a,b=p[i],p[(i+1)%n]
  for j in range(i+1,n):
   if j==(i+1)%n or (j+1)%n==i:continue
   if proper(a,b,p[j],p[(j+1)%n]):return True
 return False
def valid_yolo_line(line):
 try:v=[float(x) for x in line.split()]
 except:return False,'non_numeric'
 if len(v)<7 or (len(v)-1)%2:return False,'malformed_polygon'
 if not math.isfinite(v[0]) or int(v[0])!=v[0] or int(v[0]) not in NAMES:return False,'invalid_class'
 pts=list(zip(v[1::2],v[2::2]))
 if len(set(pts))<3:return False,'too_few_unique_points'
 if any(not math.isfinite(x) or x<0 or x>1 for p in pts for x in p):return False,'coords_out_of_bounds'
 if strict_crossing(pts):return False,'strict_self_intersection'
 area=abs(sum(pts[i][0]*pts[(i+1)%len(pts)][1]-pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts)))/2)
 if area<1e-8:return False,'zero_area'
 return True,''
def mendeley_line(points,w,h):
 if w<=0 or h<=0:return None,'bad_source_dimensions'
 pts=[(float(x)/w,float(y)/h) for x,y in points]
 if len(set(pts))<3 or any(not (0<=x<=1 and 0<=y<=1) for x,y in pts):return None,'invalid_coordinates'
 if strict_crossing(pts):return None,'strict_self_intersection'
 area=abs(sum(pts[i][0]*pts[(i+1)%len(pts)][1]-pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts)))/2)
 if area<1e-8:return None,'zero_area'
 return '1 '+' '.join(f'{z:.15f}' for p in pts for z in p),''
def copy_pair(src_img,src_lbl,split,name,lines):
 di=DATA/split/'images'/name;dl=DATA/split/'labels'/(Path(name).stem+'.txt')
 di.parent.mkdir(parents=True,exist_ok=True);dl.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(src_img,di);dl.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
 return di,dl

def main():
 if OUT.exists():raise SystemExit(f'Refusing to overwrite existing dataset output {OUT}')
 for s in ('train','val'):
  if not (BASE/s/'images').is_dir() or not (BASE/s/'labels').is_dir():raise SystemExit(f'Missing source split {s}')
 near=readcsv(P9/'near_duplicate_candidates_full_dataset.csv');exact=readcsv(P9/'exact_duplicate_pairs.csv')
 bad={(r['split_a'],r['image_a']) for r in near}|{(r['split_b'],r['image_b']) for r in near}
 bad|={(r['split_a'],r['image_a']) for r in exact}|{(r['split_b'],r['image_b']) for r in exact}
 pending_sha=set();pending_rows=readcsv(ROOT/'fruit/work/audit/phase1_review_manifest.csv')
 missing_review_paths=0
 for r in pending_rows:
  p=Path(r['source_image_path'])
  if p.is_file():pending_sha.add(sha(p))
  else:missing_review_paths+=1
 mrows=readcsv(MMAN); mqueue=readcsv(P9/'review_queue.csv'); ai={r['polygon_id']:r for r in readcsv(P9/'ai_review/ai_review_manifest.csv')}
 mby=defaultdict(list)
 for r in mqueue:mby[(r['source_split_original'],r['original_filename'])].append(r)
 ai_status_by_image=defaultdict(set)
 for r in mqueue:ai_status_by_image[(r['source_split_original'],r['original_filename'])].add(ai.get(r['polygon_id'],{}).get('decision',''))
 manifest=[]; included_images=Counter();included_instances=Counter();excluded_images=Counter();excluded_instances=Counter();reason_counts=Counter();geom_counts=Counter();source_class_counts=defaultdict(Counter)
 # Existing V5 base data except Mendeley (rebuilt from source), test, duplicate candidate, QG pending hash and geometry-invalid rows.
 for split in ('train','val'):
  for src_img in sorted((BASE/split/'images').iterdir()):
   if not src_img.is_file():continue
   name=src_img.name
   if name.startswith('mendeley_v1_'):continue
   src_lbl=BASE/split/'labels'/(src_img.stem+'.txt')
   reason=[]
   if (split,name) in bad:reason.append('exact_or_unresolved_phash_candidate')
   try:img_hash=sha(src_img)
   except Exception:img_hash='';reason.append('image_unreadable')
   if img_hash and img_hash in pending_sha:reason.append('phase1_annotation_review_pending')
   try:
    with Image.open(src_img) as im:im.verify()
   except Exception:reason.append('image_unreadable')
   if not src_lbl.is_file():reason.append('missing_label_file');raw=[]
   else:raw=[x for x in src_lbl.read_text(encoding='utf-8').splitlines() if x.strip()]
   valid=[];polys=[]
   for li,line in enumerate(raw,1):
    ok,why=valid_yolo_line(line)
    if ok:
     valid.append(line);cid=int(float(line.split()[0]));source_class_counts[split][NAMES[cid]]+=1
     polys.append((li,line,cid,'VALID',why))
    else:
     geom_counts[why]+=1;reason_counts['polygon_'+why]+=1;polys.append((li,line,'', 'INVALID',why))
   # Empty source labels are not retained as negatives in this run; negative status was not separately AI-reviewed.
   if not raw:reason.append('empty_or_negative_label_not_ai_verified')
   if reason:
    excluded_images[split]+=1;reason_counts.update(set(reason));
    for li,line,cid,status,why in polys:
     manifest.append({'source_image':str(src_img),'source_dataset':'existing_v5_candidate_non_mendeley','original_label':NAMES.get(cid,'') if cid!='' else line.split()[0],'mapped_label':NAMES.get(cid,'') if cid!='' else '', 'polygon_count_image':len(raw),'instance_line':li,'geometry_status':status if not why else f'{status}:{why}','duplicate_status':'KNOWN_CANDIDATE_EXCLUDED' if (split,name) in bad else 'NO_KNOWN_MATCH','source_split':split,'training_eligibility':'EXCLUDED','exclusion_reason':';'.join(reason),'image_sha256':img_hash,'source_family_id':'','human_verified':'false','annotation_basis':'existing source annotation; AI geometry/leakage filter'})
    if not polys and not raw:manifest.append({'source_image':str(src_img),'source_dataset':'existing_v5_candidate_non_mendeley','original_label':'','mapped_label':'','polygon_count_image':0,'instance_line':'','geometry_status':'EMPTY_IMAGE','duplicate_status':'KNOWN_CANDIDATE_EXCLUDED' if (split,name) in bad else 'NO_KNOWN_MATCH','source_split':split,'training_eligibility':'EXCLUDED','exclusion_reason':';'.join(reason),'image_sha256':img_hash,'source_family_id':'','human_verified':'false','annotation_basis':'existing source annotation; AI geometry/leakage filter'})
    excluded_instances[split]+=len(raw);continue
   if not valid:
    excluded_images[split]+=1;reason_counts['no_valid_instances_after_geometry_filter']+=1;excluded_instances[split]+=len(raw)
    for li,line,cid,status,why in polys:manifest.append({'source_image':str(src_img),'source_dataset':'existing_v5_candidate_non_mendeley','original_label':NAMES.get(cid,'') if cid!='' else line.split()[0],'mapped_label':'','polygon_count_image':len(raw),'instance_line':li,'geometry_status':f'{status}:{why}','duplicate_status':'NO_KNOWN_MATCH','source_split':split,'training_eligibility':'EXCLUDED','exclusion_reason':'no valid instances after geometry filter','image_sha256':img_hash,'source_family_id':'','human_verified':'false','annotation_basis':'existing source annotation; AI geometry/leakage filter'})
    continue
   try:copy_pair(src_img,src_lbl,split,name,valid)
   except Exception as e:raise SystemExit(f'Failed to copy/label {src_img}: {e}')
   included_images[split]+=1;included_instances[split]+=len(valid)
   # add valid rows; invalid rows remain visible in manifest as excluded.
   for li,line,cid,status,why in polys:
    use=bool(why=='')
    manifest.append({'source_image':str(src_img),'source_dataset':'existing_v5_candidate_non_mendeley','original_label':NAMES[cid] if use else line.split()[0],'mapped_label':NAMES[cid] if use else '', 'polygon_count_image':len(raw),'instance_line':li,'geometry_status':status if not why else f'{status}:{why}','duplicate_status':'NO_KNOWN_MATCH','source_split':split,'training_eligibility':f'{split.upper()}_INCLUDED' if use else 'EXCLUDED_INSTANCE','exclusion_reason':'' if use else why,'image_sha256':img_hash,'source_family_id':'','human_verified':'false','annotation_basis':'existing source annotation; AI geometry/leakage filter'})
 # Rebuild Mendeley only from archive-source TRAIN and valid nonduplicate polygons. All maturity names map only to apple.
 m_excluded_img=set()
 for mr in mrows:
  original=mr['original_filename'];srcsplit=mr['source_split_in_archive'];imagekey=f'mendeley_v1_{srcsplit}_{original}'
  if srcsplit=='test':m_excluded_img.add((srcsplit,original))
  if ('train',imagekey) in bad:m_excluded_img.add((srcsplit,original))
  if ai_status_by_image[(srcsplit,original)] & {'DUPLICATE_HOLD'}:m_excluded_img.add((srcsplit,original))
 for mr in mrows:
  original=mr['original_filename'];srcsplit=mr['source_split_in_archive'];imagekey=f'mendeley_v1_{srcsplit}_{original}'
  imgsrc=MBASE/mr['image_relative_path'];img_hash=mr['image_sha256']
  polyrows=mby.get((srcsplit,original),[]);eligible=[];polys=[];exclude_image=(srcsplit,original) in m_excluded_img
  image_reasons=[]
  if srcsplit=='test':image_reasons.append('original_mendeley_source_test_forbidden')
  if ('train',imagekey) in bad:image_reasons.append('exact_or_unresolved_phash_candidate')
  if 'DUPLICATE_HOLD' in ai_status_by_image[(srcsplit,original)]:image_reasons.append('prior_duplicate_or_provenance_hold')
  if not imgsrc.is_file():image_reasons.append('source_image_missing')
  elif sha(imgsrc)!=img_hash:image_reasons.append('source_sha_mismatch')
  # No validation/test Mendeley images are imported; source family metadata is unknown.
  for q in polyrows:
   ar=ai.get(q['polygon_id'],{});label=ar.get('source_label',q.get('source_label',''))
   reason=''
   if ar.get('decision')=='REJECTED':reason='source_polygon_geometry_or_integrity_flag'
   elif ar.get('decision')=='DUPLICATE_HOLD':reason='duplicate_or_provenance_hold'
   else:
    pts=json.loads(q['polygon_points_json']);line,reason=mendeley_line(pts,int(q['image_width']),int(q['image_height']))
   if exclude_image:reason=';'.join(image_reasons)
   elif not reason:eligible.append(line)
   polys.append((q,label,reason))
  if not exclude_image and eligible:
   outname=imagekey
   copy_pair(imgsrc,Path('/dev/null'), 'train',outname,eligible)
   included_images['train']+=1;included_instances['train']+=len(eligible)
  else:
   excluded_images['train']+=1
   if not exclude_image and not eligible:reason_counts['mendeley_no_valid_polygons_after_filter']+=1
  for q,label,reason in polys:
   status='VALID' if not reason else 'EXCLUDED'
   trainok=(not reason and not exclude_image)
   if reason:excluded_instances['train']+=1;reason_counts[reason.split(';')[0]]+=1
   manifest.append({'source_image':str(imgsrc),'source_dataset':'Mendeley Data V1 DOI 10.17632/gfcmdbvw65.1 CC BY 4.0','original_label':label,'mapped_label':'apple' if trainok else '', 'polygon_count_image':len(polyrows),'instance_line':q['region_index'],'geometry_status':status if not reason else reason,'duplicate_status':'KNOWN_CANDIDATE_EXCLUDED' if ('duplicate' in reason or 'phash' in reason) else 'NO_KNOWN_MATCH','source_split':srcsplit,'training_eligibility':'TRAIN_INCLUDED' if trainok else 'EXCLUDED','exclusion_reason':reason,'image_sha256':img_hash,'source_family_id':'','human_verified':'false','annotation_basis':'Mendeley VIA source stage annotation; AI geometry audit; maturity is not green label'})
 # YAML: no test key; evaluation is validation only.
 data_yaml=DATA/'data.yaml';data_yaml.write_text(f"path: {DATA.resolve()}\ntrain: train/images\nval: val/images\nnames:\n  0: banana\n  1: apple\n  2: orange\n",encoding='utf-8')
 fields=list(manifest[0])
 with (OUT/'FINAL_DATASET_MANIFEST.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(manifest)
 # Verify output loader structure and content hashes.
 integrity={'dataset_yaml':str(data_yaml),'classes':NAMES,'split_counts':{},'label_errors':[],'image_hash_mismatches':[],'test_split_present':False}
 for split in ('train','val'):
  ims=sorted((DATA/split/'images').glob('*'));labs=(DATA/split/'labels')
  integrity['split_counts'][split]={'images':len(ims),'labels':len(list(labs.glob('*.txt'))),'instances':sum(len([x for x in f.read_text().splitlines() if x.strip()]) for f in labs.glob('*.txt'))}
  for im in ims:
   lab=labs/(im.stem+'.txt')
   if not lab.is_file():integrity['label_errors'].append(f'missing label {split}/{im.name}');continue
   try:
    with Image.open(im) as x:x.verify()
   except Exception as e:integrity['label_errors'].append(f'unreadable {split}/{im.name}: {e}')
   for n,line in enumerate(lab.read_text().splitlines(),1):
    ok,why=valid_yolo_line(line)
    if not ok:integrity['label_errors'].append(f'{split}/{lab.name}:{n}:{why}')
 integrity['class_instance_counts']={s:dict(source_class_counts[s]) for s in source_class_counts}
 integrity['included_image_counts']={k:v for k,v in included_images.items()}
 integrity['excluded_image_counts']={k:v for k,v in excluded_images.items()}
 integrity['excluded_instance_counts']={k:v for k,v in excluded_instances.items()}
 integrity['exclusion_reason_counts']=dict(reason_counts)
 integrity['geometry_flag_counts']=dict(geom_counts)
 integrity['phase1_review_pending_hash_count']=len(pending_sha)
 integrity['phase1_missing_review_source_paths']=missing_review_paths
 integrity['near_duplicate_pair_count']=len(near)
 integrity['pHash_endpoint_counts']={s:len({name for sp,name in bad if sp==s}) for s in ('train','val','test')}
 integrity['all_known_hash_candidate_endpoints_excluded']=True
 integrity['Mendeley_source_test_images_excluded']=sum(1 for mr in mrows if mr['source_split_in_archive']=='test')
 integrity['Mendeley_source_archive_sha256']=mrows[0]['source_archive_sha256']
 integrity['source_families_assigned']=0
 integrity['independent_test_created']=False
 integrity['loader_integrity_passed']=not integrity['label_errors'] and not integrity['image_hash_mismatches']
 (OUT/'dataset_integrity.json').write_text(json.dumps(integrity,indent=2))
 print(json.dumps(integrity,indent=2))
if __name__=='__main__':main()
