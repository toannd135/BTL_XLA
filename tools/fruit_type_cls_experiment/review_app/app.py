#!/usr/bin/env python3
"""Local-only human annotation review UI; originals are read-only."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, tempfile, threading
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'work/fruit_type_cls_experiment'
APP=Path(__file__).resolve().parent
QUEUE=EXP/'review_candidates.csv'
PROGRESS=EXP/'review_progress.csv'
EVENTS=EXP/'review_events.jsonl'
EXPORT=EXP/'review_export.csv'
REPORT_JSON=EXP/'review_progress_report.json'
REPORT_MD=EXP/'review_progress_report.md'
STATUSES={'pending','verified','incorrect','ambiguous'}
CATEGORIES={'none','incorrect_class_label','missing_fruit_annotation','incorrect_bounding_box','multiple_fruits_in_crop','excessive_background','foreign_class_in_crop','duplicate_content','ambiguous'}
CLASSES={'banana','apple','orange'}

def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def atomic_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f'.{path.name}.',suffix='.tmp',dir=path.parent);os.close(fd)
    try:
        with open(tmp,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        dfd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

class ReviewStore:
    def __init__(self):
        if not QUEUE.is_file():raise FileNotFoundError(f'Missing review queue: {QUEUE}')
        self.rows=read_csv(QUEUE);self.fields=list(self.rows[0]);self.by_id={r['review_item_id']:r for r in self.rows}
        self.lock=threading.RLock()
        if len(self.by_id)!=len(self.rows):raise ValueError('Duplicate review_item_id in review_candidates.csv')
        self.progress_fields=['review_item_id','review_status','suggested_label','error_category','reviewer_notes','reviewer_id','reviewed_at_utc']
        self.decisions={}
        if PROGRESS.exists():
            for r in read_csv(PROGRESS):
                if r.get('review_item_id') in self.by_id and r.get('review_status') in STATUSES and r.get('review_status')!='pending':self.decisions[r['review_item_id']]=r
        self.image_info={hashlib.sha256(r['image'].encode()).hexdigest()[:16]:r for r in read_csv(EXP/'image_manifest.csv')}
        merged=read_csv(ROOT/'work/merged_det/manifest.csv');self.source_paths={r['merged_name']:r['orig_image'] for r in merged}
        self.annotations_by_image={}
        for r in self.rows:
            if r['review_item_type']=='whole_image_missing_annotation_check':
                try:self.annotations_by_image[r['image_id']]=json.loads(r['original_label']) if r['original_label'].startswith('[') else []
                except json.JSONDecodeError:self.annotations_by_image[r['image_id']]=[]
        self.save_all()

    def materialize(self,r):
        out=dict(r);decision=self.decisions.get(r['review_item_id'])
        if decision:
            out.update({'review_status':decision['review_status'],'suggested_label':decision['suggested_label'],'reviewer_notes':decision['reviewer_notes'],
                'reviewer_id':decision['reviewer_id'],'reviewed_at_utc':decision['reviewed_at_utc'],'decision_category':decision['error_category']})
        else:out['decision_category']=''
        return out

    def row_fruits(self,r):
        if r['review_item_type']=='single_annotation_class_and_box_review':return {r['original_label']} if r['original_label'] in CLASSES else set()
        try:return {x.get('label') for x in json.loads(r['original_label']) if x.get('label') in CLASSES}
        except Exception:return set()

    def cues(self,r):return set((r.get('error_category') or '').split('|'))
    def priority(self,r):
        c=self.cues(r);score=0
        score=max(score,100 if 'foreign_class_box_intersects_expanded_crop' in c or 'one_or_more_expanded_crops_intersect_foreign_class_box' in c else 0)
        score=max(score,98 if 'same_class_other_instance_bbox_intersects_crop' in c or 'other_fruit_instance_bbox_intersects_crop' in c or 'one_or_more_crops_contain_multiple_same_class_fruit' in c else 0)
        score=max(score,90 if 'multiple_annotated_fruit_instances_in_source_image' in c else 0)
        score=max(score,85 if 'inspect_empty_label_for_possible_missing_fruit' in c else 0)
        score=max(score,75 if 'exact_duplicate_content' in c or 'exact_duplicate_content_review' in c else 0)
        score=max(score,65 if 'polygon_annotation_review' in c else 0)
        score=max(score,60 if 'split_manifest_count_disagrees_with_label_file' in c else 0)
        score=max(score,58 if 'ambiguous' in c else 0)
        return score

    def filtered(self,q):
        cls=q.get('class',['all'])[0].lower();split=q.get('split',['all'])[0].lower();status=q.get('status',['pending'])[0].lower();category=q.get('category',['all'])[0]
        out=[]
        for base in self.rows:
            r=self.materialize(base)
            if status!='all' and r.get('review_status','pending')!=status:continue
            if split!='all' and r['split'].lower()!=split:continue
            if cls!='all' and cls not in self.row_fruits(r):continue
            cues=self.cues(r)
            if category!='all' and category not in cues and category!=r.get('decision_category'):continue
            r['priority']=self.priority(r);r['fruits']=sorted(self.row_fruits(r));r['all_annotations']=self.annotations_by_image.get(r['image_id'],[])
            out.append(r)
        split_rank={'train':0,'val':1,'test':2}
        out.sort(key=lambda r:(-r['priority'],r.get('review_status')!='pending',split_rank.get(r['split'],9),r['image'],r['review_item_id']))
        return out

    def save_decision(self,item_id,status,suggested,category,notes,reviewer):
        if item_id not in self.by_id:raise KeyError('Unknown review item')
        if status not in STATUSES-{ 'pending' }:raise ValueError('Invalid status')
        if category not in CATEGORIES:raise ValueError('Invalid error category')
        if suggested and suggested not in CLASSES:raise ValueError('Suggested label must be banana, apple, orange or blank')
        if not reviewer.strip():raise ValueError('Reviewer ID is required')
        now=datetime.now(timezone.utc).isoformat(timespec='seconds')
        d={'review_item_id':item_id,'review_status':status,'suggested_label':suggested,'error_category':category,'reviewer_notes':notes,'reviewer_id':reviewer.strip(),'reviewed_at_utc':now}
        with self.lock:
            self.decisions[item_id]=d
            self.save_all()
        event={'event':'decision_saved','timestamp_utc':now,**d}
        with EVENTS.open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        self.write_reports()
        return d

    def save_all(self):
        rows=[self.decisions[k] for k in sorted(self.decisions)]
        atomic_csv(PROGRESS,rows,self.progress_fields)
        export_fields=self.fields+['decision_category','reviewer_id','reviewed_at_utc']
        exported=[]
        for r in self.rows:
            d=self.decisions.get(r['review_item_id'])
            if d:
                item=dict(r);item.update({'review_status':d['review_status'],'suggested_label':d['suggested_label'],'reviewer_notes':d['reviewer_notes'],
                    'decision_category':d['error_category'],'reviewer_id':d['reviewer_id'],'reviewed_at_utc':d['reviewed_at_utc']});exported.append(item)
        atomic_csv(EXPORT,exported,export_fields)
        self.write_reports()

    def write_reports(self):
        status=Counter(d['review_status'] for d in self.decisions.values());category=Counter(d['error_category'] for d in self.decisions.values());reviewers=Counter(d['reviewer_id'] for d in self.decisions.values())
        status_counts={s:status.get(s,0) for s in STATUSES};status_counts['pending']=len(self.rows)-len(self.decisions)
        data={'queue_items':len(self.rows),'reviewed_items':len(self.decisions),'pending_items':len(self.rows)-len(self.decisions),'status_counts':{s:status_counts[s] for s in sorted(STATUSES)},'decision_category_counts':dict(category),'reviewer_counts':dict(reviewers),'review_progress_percent':round(100*len(self.decisions)/max(len(self.rows),1),2),'updated_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'original_queue_modified':False,'dataset_labels_modified':False,'test_split_changed':False,'model_training_performed':False}
        REPORT_JSON.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        REPORT_MD.write_text('# Review progress\n\n'+f"- Queue items: **{data['queue_items']}**\n- Reviewed: **{data['reviewed_items']}** ({data['review_progress_percent']}%)\n- Pending: **{data['pending_items']}**\n- Status counts: `{json.dumps(data['status_counts'],ensure_ascii=False)}`\n- Finding categories: `{json.dumps(data['decision_category_counts'],ensure_ascii=False)}`\n- Reviewers: `{json.dumps(data['reviewer_counts'],ensure_ascii=False)}`\n- Last update (UTC): {data['updated_at_utc']}\n\nDecisions are stored in `review_progress.csv` and `review_events.jsonl`. `review_export.csv` contains reviewed items only. Original queue, source annotations, dataset splits and model checkpoints are not modified.\n",encoding='utf-8')

STORE=None
class Handler(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args):print(f'[review-ui] {self.address_string()} {fmt%args}')
    def respond(self,code,payload,content='application/json; charset=utf-8'):
        b=payload if isinstance(payload,bytes) else json.dumps(payload,ensure_ascii=False).encode()
        self.send_response(code);self.send_header('Content-Type',content);self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(b)
    def do_GET(self):
        u=urlparse(self.path)
        if u.path=='/' or u.path=='/index.html':
            p=APP/'index.html';return self.respond(200,p.read_bytes(),'text/html; charset=utf-8')
        if u.path=='/api/options':
            categories=sorted({c for r in STORE.rows for c in STORE.cues(r) if c and c!='none'}|{d['error_category'] for d in STORE.decisions.values() if d['error_category']!='none'})
            counts=Counter(r.get('review_status','pending') for r in [STORE.materialize(x) for x in STORE.rows])
            return self.respond(200,{'categories':categories,'counts':{s:counts.get(s,0) for s in sorted(STATUSES)}})
        if u.path=='/api/rows':
            q=parse_qs(u.query);allrows=STORE.filtered(q);offset=max(0,int(q.get('offset',['0'])[0]));limit=min(100,max(1,int(q.get('limit',['40'])[0])))
            return self.respond(200,{'total':len(allrows),'offset':offset,'limit':limit,'rows':allrows[offset:offset+limit]})
        if u.path=='/api/progress':
            return self.respond(200,json.loads(REPORT_JSON.read_text()))
        if u.path=='/asset/source':
            item_id=parse_qs(u.query).get('id',[''])[0];r=STORE.by_id.get(item_id)
            if not r:return self.respond(404,{'error':'unknown item'})
            info=STORE.image_info.get(r['image_id']);source=STORE.source_paths.get(r['image'])
            if not info or not source:return self.respond(404,{'error':'source image not found'})
            p=Path(source);p=p if p.is_absolute() else ROOT/p
            return self.image_response(p)
        if u.path=='/asset/crop':
            item_id=parse_qs(u.query).get('id',[''])[0];r=STORE.by_id.get(item_id)
            if not r or not r.get('crop_path'):return self.respond(404,{'error':'crop not available for image-level item'})
            return self.image_response(EXP/r['crop_path'])
        if u.path=='/review_export.csv':return self.csv_response(EXPORT)
        if u.path=='/review_progress.csv':return self.csv_response(PROGRESS)
        if u.path=='/review_progress_report.md':return self.csv_response(REPORT_MD)
        return self.respond(404,{'error':'not found'})
    def image_response(self,p,content=None):
        try:
            p=p.resolve();p.relative_to(ROOT.resolve())
            data=p.read_bytes()
        except Exception:return self.respond(404,{'error':'asset not available'})
        mime=content or ('image/jpeg' if p.suffix.lower() in ('.jpg','.jpeg') else 'image/png' if p.suffix.lower()=='.png' else 'application/octet-stream')
        return self.respond(200,data,mime)
    def csv_response(self,p):
        try:data=p.read_bytes()
        except Exception:return self.respond(404,{'error':'CSV not available'})
        self.send_response(200);self.send_header('Content-Type','text/csv; charset=utf-8');self.send_header('Content-Disposition',f'attachment; filename="{p.name}"');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_POST(self):
        if urlparse(self.path).path!='/api/decision':return self.respond(404,{'error':'not found'})
        try:
            payload=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
            d=STORE.save_decision(payload['review_item_id'],payload['review_status'],payload.get('suggested_label',''),payload['error_category'],payload.get('reviewer_notes',''),payload.get('reviewer_id',''))
            return self.respond(200,{'ok':True,'decision':d,'progress':json.loads(REPORT_JSON.read_text())})
        except (KeyError,ValueError,json.JSONDecodeError) as e:return self.respond(400,{'error':str(e)})
        except Exception as e:return self.respond(500,{'error':str(e)})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=8765);args=ap.parse_args()
    if args.host not in ('127.0.0.1','localhost','::1'):raise SystemExit('This review app only binds to localhost.')
    global STORE;STORE=ReviewStore();server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'Fruit dataset review: http://{args.host}:{args.port}/');print(f'Queue: {QUEUE}');print('Press Ctrl-C to stop; saved decisions will resume on next start.')
    try:server.serve_forever()
    except KeyboardInterrupt:print('\nStopping; decisions are already saved.')
    finally:server.server_close()
if __name__=='__main__':main()
