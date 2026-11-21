#!/usr/bin/env python3
"""Append one human decision event and update the worksheet snapshot (never GT)."""
import argparse, datetime, hashlib, json, os, tempfile, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
AR=ROOT/'work/benchmarks/candidate_eval/annotation_review'; WPATH=AR/'p0_adjudication_worksheet.json'; SPATH=AR/'p0_adjudication_summary.json'; EPATH=AR/'p0_decision_events.jsonl'
STATUSES={'confirmed','needs_correction','uncertain'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic_json(path,obj):
 fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(name,path)
  dfd=os.open(path.parent,os.O_DIRECTORY);os.fsync(dfd);os.close(dfd)
 except BaseException:
  try:os.unlink(name)
  except FileNotFoundError:pass
  raise
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--image-id',required=True);ap.add_argument('--instance-id',required=True);ap.add_argument('--decision',required=True);ap.add_argument('--explanation',required=True);ap.add_argument('--reviewer',required=True);ap.add_argument('--review-status',required=True,choices=sorted(STATUSES));ap.add_argument('--reviewer-type',choices=['human','ai_assistant'],default='human');ap.add_argument('--decision-source',choices=['human_review','user_authorized_ai_assisted'],default='human_review');ap.add_argument('--authorization-basis');ap.add_argument('--timestamp',help='ISO-8601 timestamp with timezone; default current UTC')
 a=ap.parse_args()
 if not a.explanation.strip() or not a.reviewer.strip():raise SystemExit('explanation and actor label are required')
 if a.reviewer_type=='ai_assistant' and (a.decision_source!='user_authorized_ai_assisted' or not (a.authorization_basis or '').strip()):raise SystemExit('AI-assisted decisions require user-authorized source and authorization basis')
 if a.reviewer_type=='human' and a.decision_source!='human_review':raise SystemExit('human reviewer must use human_review decision source')
 timestamp=a.timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:parsed=datetime.datetime.fromisoformat(timestamp.replace('Z','+00:00'))
 except ValueError as e:raise SystemExit(f'invalid timestamp: {e}')
 if parsed.tzinfo is None:raise SystemExit('timestamp must include a timezone')
 w=json.loads(WPATH.read_text(encoding='utf-8'));s=json.loads(SPATH.read_text(encoding='utf-8'))
 matches=[x for x in w['instance_review_items'] if x['image_id']==a.image_id and x['instance_id']==a.instance_id]
 if len(matches)!=1:raise SystemExit('unknown or ambiguous image_id/instance_id; decision rejected')
 item=matches[0]
 if a.decision not in item['allowed_decisions']:raise SystemExit('decision not allowed for this instance_type')
 cfg=json.loads((ROOT/'work/benchmarks/candidate_eval/config.json').read_text());manifest=json.loads((ROOT/cfg['dataset']['manifest_path']).read_text());rows={x['image']:x for x in manifest['entries']}
 if a.image_id not in rows:raise SystemExit('image_id not in frozen manifest')
 ent=rows[a.image_id];label=ROOT/cfg['dataset']['split_root']/ent['label'];actual=sha(label)
 if actual!=ent['label_sha256'] or actual!=item['original_annotation_sha256']:raise SystemExit('original GT hash mismatch; decision rejected')
 existing=[]
 if EPATH.exists():
  for line in EPATH.read_text(encoding='utf-8').splitlines():
   if line.strip():existing.append(json.loads(line))
 key=(a.image_id,a.instance_id);latest=None;event_ids=set()
 for e in existing:
  if e['event_id'] in event_ids:raise SystemExit('duplicate historical event_id; log needs audit')
  event_ids.add(e['event_id'])
  if (e['image_id'],e['instance_id'])==key:latest=e['new_decision']
 current=item.get('reviewer_decision')
 if current!=latest:raise SystemExit('worksheet state disagrees with append-only event log; resolve before appending')
 event={'event_id':str(uuid.uuid4()),'image_id':a.image_id,'instance_id':a.instance_id,'original_annotation_sha256':actual,'previous_decision':latest,'new_decision':a.decision,'explanation':a.explanation.strip(),'reviewer':a.reviewer.strip(),'reviewer_type':a.reviewer_type,'decision_source':a.decision_source,'authorization_basis':(a.authorization_basis or '').strip() or None,'timestamp':timestamp,'review_status':a.review_status}
 payload=(json.dumps(event,ensure_ascii=False,separators=(',',':'))+'\n').encode()
 fd=os.open(EPATH,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o644)
 try:
  pos=0
  while pos<len(payload):pos+=os.write(fd,payload[pos:])
  os.fsync(fd)
 finally:os.close(fd)
 item['reviewer_decision']=a.decision;item['reviewer_explanation']=a.explanation.strip();item['reviewer']=a.reviewer.strip();item['reviewed_at']=timestamp;item['review_status']=a.review_status;item['reviewer_type']=a.reviewer_type;item['decision_source']=a.decision_source;item['authorization_basis']=(a.authorization_basis or '').strip() or None
 image=next(x for x in w['images'] if x['image_id']==a.image_id)
 states=[x['review_status'] for x in w['instance_review_items'] if x['image_id']==a.image_id]
 image['image_review_status']='pending' if 'pending' in states else ('needs_correction' if 'needs_correction' in states else ('uncertain' if 'uncertain' in states else 'confirmed'))
 atomic_json(WPATH,w)
 if 'initial_state_snapshot' not in s:s['initial_state_snapshot']={'review_status_counts':dict(s['review_status_counts']),'image_review_status_counts':dict(s['image_review_status_counts']),'decision_event_count':s['decision_event_count'],'reviewer_decision_count':s['reviewer_decision_count']}
 s['decision_attribution_counts']={source:sum(x.get('decision_source')==source for x in w['instance_review_items']) for source in ('human_review','user_authorized_ai_assisted')}
 s['decision_event_count']=len(existing)+1;s['reviewer_decision_count']=sum(x.get('reviewer_decision') not in (None,'') for x in w['instance_review_items'])
 s['review_status_counts']={status:sum(x['review_status']==status for x in w['instance_review_items']) for status in ('pending','confirmed','needs_correction','uncertain')}
 s['image_review_status_counts']={status:sum(x['image_review_status']==status for x in w['images']) for status in ('pending','confirmed','needs_correction','uncertain')}
 s['per_image']=[{k:v for k,v in x.items() if k not in ('worksheet_item_ids',)} for x in w['images']]
 atomic_json(SPATH,s)
 print(json.dumps({'event_id':event['event_id'],'image_id':a.image_id,'instance_id':a.instance_id,'previous_decision':latest,'new_decision':a.decision,'log_is_append_only':True,'GT_modified':False},indent=2))
if __name__=='__main__':main()
