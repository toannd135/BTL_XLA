#!/usr/bin/env python3
"""Build a family-preserving AI-assisted GreenApple/NonGreenApple dataset."""
import csv,hashlib,json,shutil
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
P=ROOT/'fruit/work/fruit_type_cls_experiment/phase9_2_green_apple_expansion'
BASE=ROOT/'fruit/work/fruit_type_cls_experiment/phase6b_candidate'
OUT=P/'green_apple_classifier'
DATA=OUT/'dataset'

def rows(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 if DATA.exists():raise SystemExit(f'refusing to overwrite {DATA}')
 scored=rows(P/'green_apple_classifier_ai_scores.csv')
 source_rows=rows(BASE/'manifests/complete_manifest.csv')
 source_by_image={r['image']:r for r in source_rows}
 for r in scored:
  source=source_by_image.get(r['image'])
  if source is None:raise SystemExit(f"missing source manifest row for {r['image']}")
  r['candidate_copy_path']=source['candidate_copy_path']
  r['candidate_copy_sha256']=source['candidate_copy_sha256']
 selected=[]
 for r in scored:
  margin=float(r['green_minus_non_green_margin']);green=float(r['center_green_pixel_ratio_support_only'])
  if margin>=.02 and green>=.18:r['target']='GreenApple'
  elif margin<=-.02 and green<=.30:r['target']='NonGreenApple'
  else:r['target']='AMBIGUOUS_AI_REVIEW'
  selected.append(r)
 # Exact duplicate crops: keep the lexically first representation, record remaining copies as excluded.
 groups=defaultdict(list)
 for r in selected:
  if r['target']!='AMBIGUOUS_AI_REVIEW':groups[r['crop_sha256']].append(r)
 exclude_ids=set()
 for h,group in groups.items():
  if len(group)>1:
   group.sort(key=lambda x:x['image']);exclude_ids.update(x['image'] for x in group[1:])
 included=[];families=defaultdict(set);split_count=Counter();cls_count=Counter();write_rows=[]
 for r in selected:
  eligible=r['target']!='AMBIGUOUS_AI_REVIEW' and r['image'] not in exclude_ids
  reason=''
  if r['target']=='AMBIGUOUS_AI_REVIEW':reason='CLIP_and_color_support_disagree_or_margin_too_low'
  elif r['image'] in exclude_ids:reason='exact_duplicate_crop_sha256'
  dest=''
  if eligible:
   split=r['split_v2'];family=r['source_family_id']
   if not family:raise SystemExit(f'missing source family for {r["image"]}')
   families[(r['target'],family)].add(split)
   if split not in {'train','val','test'}:raise SystemExit('bad split')
   src=BASE/r['candidate_copy_path']
   if not src.is_file():raise SystemExit(f'missing crop {src}')
   dest=DATA/split/r['target']/(hashlib.sha1(r['image'].encode()).hexdigest()[:12]+'_'+src.name)
   dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
   split_count[(split,r['target'])]+=1;cls_count[r['target']]+=1;included.append(r)
  write_rows.append({**r,'training_eligible':str(eligible).lower(),'dataset_path':str(dest.relative_to(DATA)) if dest else '', 'exclusion_reason':reason,'label_provenance':'AI_ASSISTED_CLIP_AND_COLOR_SUPPORT','human_verified_color':'false'})
 overlaps=[(c,f,ss) for (c,f),ss in families.items() if len(ss)>1]
 if overlaps:raise SystemExit(f'source-family crosses splits in AI subset: {overlaps[:5]}')
 # Existing source family split manifest is preserved; no new family/split IDs.
 manifest=P/'green_apple_classifier_manifest.csv'
 if manifest.exists():raise SystemExit(f'refusing to overwrite {manifest}')
 with manifest.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(write_rows[0]));w.writeheader();w.writerows(write_rows)
 summary={'status':'AI_ASSISTED_CLASSIFICATION_CANDIDATE','classes':{'GreenApple':0,'NonGreenApple':1},'selected_rule':'Green: CLIP green-minus-max-nongreen margin >= 0.02 and central green-pixel support >= 0.18; Non-green: margin <= -0.02 and support <= 0.30; others ambiguous. These are AI labels, not human-verified.','source_manifest':'phase6b_candidate/manifests/complete_manifest.csv','input_reviewed_apple_crop_count':len(scored),'included_count':len(included),'ambiguous_count':sum(x['target']=='AMBIGUOUS_AI_REVIEW' for x in selected),'duplicate_crop_excluded_count':len(exclude_ids),'counts_by_split_and_class':{f'{s}/{c}':n for (s,c),n in split_count.items()},'family_counts_by_split':{},'family_cross_split_count':len(overlaps),'human_verified_green_labels':0,'human_verified_non_green_labels':0,'test_ai_labels_used_once_after_training':True}
 for split in ('train','val','test'):
  summary['family_counts_by_split'][split]={c:len({r['source_family_id'] for r in included if r['split_v2']==split and r['target']==c}) for c in ('GreenApple','NonGreenApple')}
 (OUT/'dataset_summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
