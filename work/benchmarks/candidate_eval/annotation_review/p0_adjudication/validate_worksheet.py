#!/usr/bin/env python3
"""Read-only validator for P0 worksheet, append-only decisions, and frozen inputs."""
import hashlib, json, sys, datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
BASE=ROOT/'work/benchmarks/candidate_eval'; AR=BASE/'annotation_review'; RUN=BASE/'full_runs/fresh_clip_swap_only_345'
sys.path.insert(0,str(BASE));import evaluator as ev
STATUSES={'pending','confirmed','needs_correction','uncertain'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dec(r):
 h,w=r['size'];a=ev.np.zeros(h*w,dtype=ev.np.uint8);o=v=0
 for n in r['counts']:a[o:o+n]=v;o+=n;v=1-v
 if o!=h*w:raise ValueError('bad RLE')
 return a.reshape(h,w).astype(bool)
def image_status(states):
 if 'pending' in states:return 'pending'
 if 'needs_correction' in states:return 'needs_correction'
 if 'uncertain' in states:return 'uncertain'
 return 'confirmed'
def main():
 config,ch=ev.load_and_verify_config(BASE/'config.json');m=read(ROOT/config['dataset']['manifest_path']);me={x['image']:x for x in m['entries']};w=read(AR/'p0_adjudication_worksheet.json');s=read(AR/'p0_adjudication_summary.json');events_path=AR/'p0_decision_events.jsonl'
 if len(w['images'])!=5 or len({x['image_id'] for x in w['images']})!=5:raise ValueError('must have exactly five unique image records')
 if ch!=config.get('config_sha256'):raise ValueError('canonical config mismatch')
 if len(m['entries'])!=345:raise ValueError('frozen manifest entry count mismatch')
 if not events_path.exists():raise ValueError('decision event log missing')
 if w['allowed_review_statuses']!=['pending','confirmed','needs_correction','uncertain']:raise ValueError('status enum mismatch')
 event_rows=[json.loads(x) for x in events_path.read_text(encoding='utf-8').splitlines() if x.strip()]
 items={(x['image_id'],x['instance_id']):x for x in w['instance_review_items']}
 if len(items)!=len(w['instance_review_items']):raise ValueError('duplicate image/instance review item')
 allowed_by_type={k:set(v) for k,v in w['decision_options_by_instance_type'].items()}
 latest={};event_ids=set()
 for e in event_rows:
  required={'event_id','image_id','instance_id','original_annotation_sha256','previous_decision','new_decision','explanation','reviewer','reviewer_type','decision_source','authorization_basis','timestamp','review_status'}
  if not required<=set(e):raise ValueError('decision event misses required audit fields')
  if e['event_id'] in event_ids:raise ValueError('duplicate event_id')
  event_ids.add(e['event_id']);key=(e['image_id'],e['instance_id'])
  if key not in items:raise ValueError(f'event references unknown image/instance: {key}')
  item=items[key]
  if e['original_annotation_sha256']!=item['original_annotation_sha256']:raise ValueError('event annotation hash differs from worksheet')
  if e['previous_decision']!=latest.get(key):raise ValueError(f'broken previous_decision chain for {key}')
  if e['new_decision'] not in item['allowed_decisions'] or e['new_decision'] not in allowed_by_type[item['instance_type']]:raise ValueError('invalid decision enum')
  if not e['explanation'].strip() or not e['reviewer'].strip():raise ValueError('event explanation/actor is empty')
  if e['review_status'] not in STATUSES-{'pending'}:raise ValueError('invalid event status')
  if e['reviewer_type']=='ai_assistant':
   if e['decision_source']!='user_authorized_ai_assisted' or not (e['authorization_basis'] or '').strip():raise ValueError('AI-assisted decision attribution incomplete')
  elif e['reviewer_type']=='human':
   if e['decision_source']!='human_review':raise ValueError('human decision source mismatch')
  else:raise ValueError('unknown reviewer_type')
  try:dt=datetime.datetime.fromisoformat(e['timestamp'].replace('Z','+00:00'))
  except ValueError as ex:raise ValueError('invalid event timestamp') from ex
  if dt.tzinfo is None:raise ValueError('event timestamp has no timezone')
  latest[key]=e
 item_status_counts={x:0 for x in STATUSES}; image_ids=[]
 for image in w['images']:
  rel=image['image_id'];image_ids.append(rel)
  if rel not in me:raise ValueError(f'image missing from frozen manifest: {rel}')
  ent=me[rel];label=ROOT/config['dataset']['split_root']/ent['label'];img=ROOT/config['dataset']['split_root']/rel
  if not img.is_file() or not label.is_file():raise ValueError(f'missing source image/annotation for {rel}')
  if sha(img)!=ent['image_sha256'] or sha(label)!=ent['label_sha256'] or sha(label)!=image['original_annotation_sha256']:raise ValueError(f'image/annotation hash mismatch: {rel}')
  keyhash=hashlib.sha256(rel.encode()).hexdigest();b=read(RUN/'records/baseline'/f'{keyhash}.json');c=read(RUN/'records/candidate'/f'{keyhash}.json');g=read(RUN/'records/ground_truth'/f'{keyhash}.json')
  if b['image_id']!=rel or c['image_id']!=rel or g['image_id']!=rel:raise ValueError('record image id mismatch')
  gt=g['instances'];bp=b['instances'];cp=c['instances'];ww,hh=b['width'],b['height'];gm=[dec(x['mask_rle']) for x in gt];bm=[ev.polygon_to_mask(x['polygon'],ww,hh) for x in bp];cm=[ev.polygon_to_mask(x['polygon'],ww,hh) for x in cp]
  match=ev.class_agnostic_match(gm,bm,.50);cmatch=ev.class_agnostic_match(gm,cm,.50)
  if match['matches']!=cmatch['matches'] or match['unmatched_gt_ids']!=cmatch['unmatched_gt_ids'] or match['unmatched_prediction_ids']!=cmatch['unmatched_prediction_ids']:raise ValueError('candidate geometry/matching changed')
  rows=[x for x in w['instance_review_items'] if x['image_id']==rel]
  for x in rows:
   pair=(rel,x['instance_id']); status=x['review_status']
   if status not in STATUSES:raise ValueError('invalid item status')
   if x['original_annotation_sha256']!=ent['label_sha256']:raise ValueError('item annotation hash mismatch')
   valid=allowed_by_type[x['instance_type']]
   if not set(x['allowed_decisions'])<=valid:raise ValueError(f'invalid allowed-decision enum: {pair}')
   evrow=latest.get(pair)
   if evrow is None:
    if status!='pending' or any(x.get(f) not in (None,'') for f in ('reviewer_decision','reviewer_explanation','reviewer','reviewed_at','reviewer_type','decision_source','authorization_basis')):raise ValueError(f'item has decision state without event: {pair}')
   else:
    expected={'reviewer_decision':evrow['new_decision'],'reviewer_explanation':evrow['explanation'],'reviewer':evrow['reviewer'],'reviewed_at':evrow['timestamp'],'review_status':evrow['review_status'],'reviewer_type':evrow['reviewer_type'],'decision_source':evrow['decision_source'],'authorization_basis':evrow['authorization_basis']}
    for field,value in expected.items():
     if x.get(field)!=value:raise ValueError(f'worksheet/event mismatch at {pair} field {field}')
   item_status_counts[status]+=1
  gt_rows=[x for x in rows if x['instance_type']=='gt'];un_rows=[x for x in rows if x['instance_type']=='unmatched_prediction' and x.get('prediction_arm')=='baseline'];pair_rows=[x for x in rows if x['instance_type']=='matched_pair'];cand_rows=[x for x in rows if x['instance_type']=='unmatched_prediction' and x.get('prediction_arm')=='candidate']
  if {x['instance_id'] for x in gt_rows}!={x['instance_id'] for x in gt}:raise ValueError(f'GT coverage mismatch: {rel}')
  if {x['instance_id'] for x in un_rows}!={bp[i]['instance_id'] for i in match['unmatched_prediction_ids']}:raise ValueError(f'baseline unmatched coverage mismatch: {rel}')
  expected_pairs={f"pair:{gt[x['gt_id']]['instance_id']}:{bp[x['prediction_id']]['instance_id']}" for x in match['matches']}
  if {x['instance_id'] for x in pair_rows}!=expected_pairs:raise ValueError(f'matched-pair coverage mismatch: {rel}')
  expected_cand={cp[i]['instance_id'] for i in cmatch['unmatched_prediction_ids'] if i not in match['unmatched_prediction_ids'] or cp[i]['class_id']!=bp[i]['class_id']}
  if {x['instance_id'] for x in cand_rows}!=expected_cand:raise ValueError(f'candidate-only unmatched coverage mismatch: {rel}')
  if len(rows)!=len(gt)+len(match['unmatched_prediction_ids'])+len(match['matches'])+len(expected_cand):raise ValueError('review item count mismatch')
  states=[x['review_status'] for x in rows]
  if image['original_gt_count']!=len(gt) or image['image_review_status']!=image_status(states):raise ValueError(f'image-level status/count mismatch: {rel}')
  # Corrected GT and adjudicated totals stay unset; this workflow does not edit annotations.
  for field in ['reviewer_confirmed_gt_count','confirmed_missing_instances','confirmed_false_positives','confirmed_missed_instances','confirmed_split_errors','confirmed_merge_errors','confirmed_mask_errors','uncertain_instance_count','reviewer_summary']:
   if image[field] is not None:raise ValueError(f'image-level aggregate should remain unset: {rel}:{field}')
 if len(set(image_ids))!=5 or len(items)!=len(w['instance_review_items']):raise ValueError('duplicate image/item IDs')
 # Validate frozen inputs against the hash snapshot captured before human decisions.
 for name,want in s['integrity_snapshot']['sha256'].items():
  if name=='config_canonical_sha256':
   if want!=ch:raise ValueError('canonical config hash changed')
   continue
  if name.startswith('annotation:'):
   rel=name.split(':',1)[1];p=ROOT/config['dataset']['split_root']/me[rel]['label']
  elif name.startswith(('baseline_record:','candidate_record:','ground_truth_record:')):
   prefix,rel=name.split(':',1);kh=hashlib.sha256(rel.encode()).hexdigest();role={'baseline_record':'baseline','candidate_record':'candidate','ground_truth_record':'ground_truth'}[prefix];p=RUN/'records'/role/f'{kh}.json'
  else:
   loc={'config':BASE/'config.json','manifest':ROOT/config['dataset']['manifest_path'],'detector':ROOT/config['artifacts']['detector']['path'],'clip':ROOT/config['artifacts']['clip_weights']['path'],'evaluator':BASE/'evaluator.py','baseline_jsonl':RUN/'predictions_baseline.jsonl','candidate_jsonl':RUN/'predictions_candidate.jsonl','progress':RUN/'progress.json','events':RUN/'inference_events.jsonl','execution_summary':RUN/'execution_summary.json','metrics_report':RUN/'report.json','per_image_metrics':RUN/'candidate-test-per-image.csv','confusion_csv':RUN/'candidate-test-confusion-matrix.csv','negative_csv':RUN/'candidate-test-negative-images.csv','p0_review':AR/'p0_review.json','p0_validation':AR/'p0_manifest_validation.json'}
   if name not in loc:raise ValueError('unknown frozen artifact in hash snapshot: '+name)
   p=loc[name]
  if sha(p)!=want:raise ValueError('frozen artifact changed: '+name)
 gtroot=AR/'GT-reviewed';actual_gt={str(p.relative_to(ROOT)):sha(p) for p in gtroot.rglob('*') if p.is_file()}
 if actual_gt!=s['integrity_snapshot']['gt_reviewed_files'] or s['integrity_snapshot']['gt_reviewed_annotation_files']:raise ValueError('GT-reviewed changed')
 computed_event_count=len(event_rows);computed_decisions=sum(x.get('reviewer_decision') not in (None,'') for x in w['instance_review_items'])
 computed_img={status:sum(x['image_review_status']==status for x in w['images']) for status in ('pending','confirmed','needs_correction','uncertain')}
 computed_attr={source:sum(x.get('decision_source')==source for x in w['instance_review_items']) for source in ('human_review','user_authorized_ai_assisted')}
 if s['image_count']!=5 or s['decision_event_count']!=computed_event_count or s['reviewer_decision_count']!=computed_decisions:raise ValueError('summary decision counts mismatch')
 if s['review_status_counts']!=item_status_counts or s['image_review_status_counts']!=computed_img:raise ValueError('summary status counts mismatch')
 if s.get('decision_attribution_counts',{'human_review':0,'user_authorized_ai_assisted':0})!=computed_attr:raise ValueError('summary attribution counts mismatch')
 print(json.dumps({'status':'pass','images':5,'review_items':len(w['instance_review_items']),'instance_type_counts':s['instance_review_item_counts'],'decision_events':computed_event_count,'reviewer_decisions':computed_decisions,'item_review_status_counts':item_status_counts,'image_review_status_counts':computed_img,'decision_attribution_counts':computed_attr,'original_hashes_match':True,'frozen_artifacts_unchanged':True,'GT_reviewed_untouched':True,'inference_or_training_executed':False},indent=2))
if __name__=='__main__':main()
