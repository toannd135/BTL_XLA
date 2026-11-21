#!/usr/bin/env python3
"""Local, decision-auditable review UI. Listens only on loopback."""
from __future__ import annotations
import argparse, csv, json, threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PACKAGE=Path(__file__).resolve().parent
INSTANCE_DECISIONS={'accept','needs_edit','reject','uncertain'}
IMAGE_DECISIONS={'reviewed','pending','needs_expert_review'}
NEGATIVE_DECISIONS={'confirm_empty','fruit_present','uncertain'}
LEAKAGE_DECISIONS={'same_source','different_source','uncertain'}
CATEGORIES={'A':'Banana clusters','B':'Orange occlusion','C':'Apple overlap','D':'Mixed fruit overlap','E':'Annotation topology issues','F':'Negative images'}

class ReviewStore:
 def __init__(self):
  self.lock=threading.RLock(); self.manifest=json.loads((PACKAGE/'manifest.json').read_text());self.state=self.replay()
  self.persist()
 def replay(self):
  state={'instances':{},'images':{},'negative_images':{},'leakage_pairs':{}}
  path=PACKAGE/'review_history.jsonl'
  if path.exists():
   for n,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
    if not line.strip():continue
    ev=json.loads(line);bucket=ev['entity'];key=ev['entity_id']
    if bucket not in state:raise ValueError(f'Bad event entity in history line {n}')
    state[bucket][key]=ev
  return state
 def persist(self):
  with self.lock:
   tmp=PACKAGE/'review_decisions.json.tmp';tmp.write_text(json.dumps(self.state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(PACKAGE/'review_decisions.json')
   self.write_manifest_csv();self.write_leak_csv();self.write_progress()
 def write_manifest_csv(self):
  path=PACKAGE/'review_manifest.csv'; rows=list(csv.DictReader(path.open(newline='',encoding='utf-8-sig')))
  tmp=path.with_suffix('.csv.tmp')
  with tmp.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
   for r in rows:
    image_id=r['image_id'];inst=r['instance_id']
    e=self.state['instances'].get(f'{image_id}::{inst}') if inst else None
    ie=self.state['images'].get(image_id);ne=self.state['negative_images'].get(image_id)
    r['human_decision']=e['decision'] if e else '';r['reviewer_id']=e.get('reviewer','') if e else '';r['reviewed_at']=e.get('timestamp','') if e else ''
    r['image_decision']=ie['decision'] if ie else '';r['image_reviewer_id']=ie.get('reviewer','') if ie else '';r['image_reviewed_at']=ie.get('timestamp','') if ie else ''
    r['negative_image_decision']=ne['decision'] if ne else '';r['negative_reviewer_id']=ne.get('reviewer','') if ne else '';r['negative_reviewed_at']=ne.get('timestamp','') if ne else ''
    w.writerow(r)
  tmp.replace(path)
 def write_leak_csv(self):
  path=PACKAGE/'leakage_review_manifest.csv';rows=list(csv.DictReader(path.open(newline='',encoding='utf-8-sig')));tmp=path.with_suffix('.csv.tmp')
  fields=list(rows[0])
  for extra in ('reviewer_id','reviewed_at'):
   if extra not in fields:fields.append(extra)
  with tmp.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for r in rows:
    ev=self.state['leakage_pairs'].get(r['pair_id']);r['reviewer_decision']=ev['decision'] if ev else '';r['review_status']=ev['decision'] if ev else 'pending';r['reviewer_notes']=ev.get('notes','') if ev else '';r['reviewer_id']=ev.get('reviewer','') if ev else '';r['reviewed_at']=ev.get('timestamp','') if ev else '';w.writerow(r)
  tmp.replace(path)
 def progress(self):
  imgs=self.manifest['annotation_images'];pairs=self.manifest['leakage_pairs'];total_inst=sum(x['instance_count'] for x in imgs)
  ic={d:sum(ev['decision']==d for ev in self.state['instances'].values()) for d in INSTANCE_DECISIONS}
  ic['pending']=total_inst-sum(ic.values())
  image_counts={d:sum(ev['decision']==d for ev in self.state['images'].values()) for d in IMAGE_DECISIONS if d!='pending'}
  image_counts['pending']=len(imgs)-sum(image_counts.values())
  negative_total=sum(x['negative'] for x in imgs)
  nc={d:sum(ev['decision']==d for ev in self.state['negative_images'].values()) for d in NEGATIVE_DECISIONS};nc['pending']=negative_total-sum(nc.values())
  lc={d:sum(ev['decision']==d for ev in self.state['leakage_pairs'].values()) for d in LEAKAGE_DECISIONS};lc['pending']=len(pairs)-sum(lc.values())
  bysplit={s:sum(x['split']==s for x in imgs) for s in ['train','val','test']}
  cat={k:sum(k in x['categories'] for x in imgs) for k in CATEGORIES}
  class_counts={c:sum(x['instances_by_class'].get(c,0) for x in imgs) for c in ['banana','apple','orange']}
  return {'schema_version':1,'generated_at':datetime.now(timezone.utc).isoformat(),'images_total':len(imgs),'instances_total':total_inst,'negative_images_total':negative_total,'images_by_split':bysplit,'instances_by_class':class_counts,'images_by_category':cat,'instance_decisions':ic,'image_decisions':image_counts,'negative_image_decisions':nc,'leakage_pairs_total':len(pairs),'leakage_decisions':lc,'topology_candidate_polygons':sum(len(i['polygon_topology_flags']) for x in imgs for i in x['instances']),'saved_instance_decisions':len(self.state['instances']),'saved_image_decisions':len(self.state['images']),'saved_negative_image_decisions':len(self.state['negative_images']),'saved_leakage_decisions':len(self.state['leakage_pairs']),'history_events':sum(1 for x in (PACKAGE/'review_history.jsonl').read_text().splitlines() if x.strip()),'no_default_accept':True,'accept_decisions_are_event_backed':all(x.get('entity')=='instances' for x in self.state['instances'].values() if x.get('decision')=='accept'),'counts_basis':'Current decisions are the latest human-supplied events in review_history.jsonl; absent decisions remain pending.'}
 def write_progress(self):
  p=self.progress();tmp=PACKAGE/'review_progress.json.tmp';tmp.write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(PACKAGE/'review_progress.json')
 def update(self,body):
  entity=body.get('entity');eid=str(body.get('entity_id',''));decision=body.get('decision');notes=str(body.get('notes',''))[:4000];reviewer=str(body.get('reviewer',''))[:200]
  allowed={'instance':INSTANCE_DECISIONS,'image':IMAGE_DECISIONS,'negative_image':NEGATIVE_DECISIONS,'leakage_pair':LEAKAGE_DECISIONS}
  if entity not in allowed or decision not in allowed[entity]:raise ValueError('Unknown entity or decision. Choose an explicit allowed decision.')
  if not reviewer and body.get('reviewer') is not None:reviewer=''
  valid_ids=set()
  if entity=='instance':
   for im in self.manifest['annotation_images']:
    valid_ids.update(f"{im['image_id']}::{x['instance_id']}" for x in im['instances'])
  elif entity=='image':valid_ids={x['image_id'] for x in self.manifest['annotation_images']}
  elif entity=='negative_image':valid_ids={x['image_id'] for x in self.manifest['annotation_images'] if x['negative']}
  else:valid_ids={x['pair_id'] for x in self.manifest['leakage_pairs']}
  if eid not in valid_ids:raise ValueError('Unknown review item id.')
  with self.lock:
   history=PACKAGE/'review_history.jsonl';event={'entity':{'instance':'instances','image':'images','negative_image':'negative_images','leakage_pair':'leakage_pairs'}[entity],'entity_id':eid,'decision':decision,'reviewer':reviewer,'notes':notes,'timestamp':datetime.now(timezone.utc).isoformat()}
   with history.open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n');f.flush()
   self.state[event['entity']][eid]=event;self.persist()
  return event

STORE=ReviewStore()
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(PACKAGE),**kwargs)
 def log_message(self,fmt,*args):print('%s - %s'%(self.log_date_time_string(),fmt%args))
 def send_json(self,obj,status=200):
  raw=json.dumps(obj,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
 def do_GET(self):
  path=urlparse(self.path).path
  if path=='/':self.path='/index.html';return super().do_GET()
  if path=='/api/manifest':return self.send_json(STORE.manifest)
  if path=='/api/state':return self.send_json(STORE.state)
  if path=='/api/progress':return self.send_json(STORE.progress())
  return super().do_GET()
 def do_POST(self):
  if urlparse(self.path).path!='/api/decision':return self.send_json({'error':'not found'},404)
  try:
   n=int(self.headers.get('Content-Length','0'))
   if n<1 or n>20000:raise ValueError('Invalid request body size.')
   body=json.loads(self.rfile.read(n));event=STORE.update(body);return self.send_json({'saved':True,'event':event,'progress':STORE.progress()})
  except Exception as e:return self.send_json({'saved':False,'error':str(e)},400)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765);a=p.parse_args()
 if a.host not in ('127.0.0.1','localhost','::1'):raise SystemExit('For safety, this review server binds to loopback only.')
 server=ThreadingHTTPServer((a.host,a.port),Handler);print(f'Human review UI: http://127.0.0.1:{server.server_port}');print('Press Ctrl+C to stop. Decisions append to review_history.jsonl.');server.serve_forever()
if __name__=='__main__':main()
