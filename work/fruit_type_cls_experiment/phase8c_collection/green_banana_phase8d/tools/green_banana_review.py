#!/usr/bin/env python3
"""Human-only Green Banana source/crop review UI and provenance event CLI."""
import argparse
import json
import mimetypes
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from progress_report import read_csv, read_jsonl, latest_by_asset

PHASE8D=Path(__file__).resolve().parents[1]
COLLECTION=PHASE8D.parent
INCOMING=(COLLECTION/'incoming').resolve()
STATUSES={'VERIFIED_CORRECT','INCORRECT_LABEL','AMBIGUOUS','INVALID_IMAGE','NEEDS_PROVENANCE'}
REASONS={'INCORRECT_LABEL','AMBIGUOUS','INVALID_IMAGE','NEEDS_PROVENANCE'}
HTML=r'''<!doctype html><html><head><meta charset="utf-8"><title>Green Fruit Review</title><style>
body{font:15px system-ui;margin:20px;color:#17251b;background:#f4f8f3}header,.box{background:white;border:1px solid #ccd8ce;border-radius:10px;padding:14px;margin-bottom:14px}main{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(320px,1fr);gap:14px}.controls{display:flex;gap:8px;align-items:center}canvas{max-width:100%;height:auto;border:1px solid #9bad9f;cursor:crosshair;background:#ddd}#crop{max-height:330px;cursor:default}label{display:block;margin:7px 0}input,select,textarea,button{font:inherit;padding:7px}textarea{width:98%;height:65px}button{cursor:pointer}small{color:#55645a}.warn{color:#8b4a00}#status{font-weight:700}
@media(max-width:850px){main{grid-template-columns:1fr}}
</style></head><body><header><h2>Green Banana — human review</h2><p>Inspect the complete original, draw a target fruit region, inspect the derived crop, then enter the decision. No labels are auto-approved. The ROI is a browser preview and is recorded in original pixel coordinates.</p><div class="controls"><button id="prev">Previous</button><select id="asset"></select><button id="next">Next</button><select id="review-filter"><option value="ALL">All statuses</option><option>PENDING</option><option>VERIFIED_CORRECT</option><option>INCORRECT_LABEL</option><option>AMBIGUOUS</option><option>INVALID_IMAGE</option><option>NEEDS_PROVENANCE</option></select><input id="search" placeholder="Search filename, asset or family"><span id="status"></span></div></header><main><section class="box"><h3>Original image — draw the target fruit region</h3><canvas id="source"></canvas><p id="assetmeta"></p><small>For clusters, select the visible fruit group; if fruit identity or peel color is uncertain, choose AMBIGUOUS.</small></section><section class="box"><h3>Target crop preview</h3><canvas id="crop"></canvas><form id="form">
<label>Reviewer ID <input name="reviewer_id" required></label>
<label>Decision <select name="decision"><option>PENDING</option><option>VERIFIED_CORRECT</option><option>INCORRECT_LABEL</option><option>AMBIGUOUS</option><option>INVALID_IMAGE</option><option>NEEDS_PROVENANCE</option></select></label>
<label>Fruit identity <select name="fruit_identity"><option>Banana</option><option>Apple</option><option>Other</option><option>Unclear</option></select></label>
<label>Peel genuinely green? <select name="green"><option value="yes">Yes</option><option value="no">No</option><option value="unknown">Cannot determine</option></select></label>
<label>Green confidence <select name="peel_color_confidence"><option>CLEAR_GREEN</option><option>UNCERTAIN</option><option>NOT_GREEN</option><option>UNASSESSABLE</option></select></label>
<label>Image suitable? <select name="suitable"><option value="yes">Yes</option><option value="no">No</option><option value="unknown">Cannot determine</option></select></label>
<label>Visibility <select name="fruit_visibility"><option>whole</option><option>partial_identifiable</option><option>poor</option><option>not_visible</option><option>unclear</option></select></label>
<label>Occlusion <select name="occlusion"><option>none</option><option>minor</option><option>major</option><option>unclear</option></select></label>
<label>Background dominance <select name="background_dominance"><option>low</option><option>moderate</option><option>high</option><option>unclear</option></select></label>
<label>Reason (required unless verified) <textarea name="reason"></textarea></label><label>Notes <textarea name="notes"></textarea></label>
<button type="submit">Append review decision</button><span id="result"></span></form></section></main>
<script>
const $=s=>document.querySelector(s), source=$('#source'),ctx=source.getContext('2d'),crop=$('#crop'),cctx=crop.getContext('2d');let items=[],filtered=[],index=0,img=new Image(),start=null,box=null,current=null;
function renderList(preferred){let status=$('#review-filter').value,query=$('#search').value.trim().toLowerCase();filtered=items.filter(a=>(status==='ALL'||a.review_status===status)&&(!query||[a.asset_id,a.incoming_relative_path,a.source_family_id,a.source_image_id].some(v=>(v||'').toLowerCase().includes(query))));let sel=$('#asset');sel.innerHTML='';filtered.forEach((a,i)=>{let o=document.createElement('option');o.value=i;o.textContent=a.asset_id+' | '+a.review_status+' | '+(a.source_family_id||'family unknown')+' | '+a.incoming_relative_path;sel.append(o)});$('#status').textContent=filtered.length+' / '+items.length+' assets';if(filtered.length){let target=preferred?filtered.findIndex(a=>a.asset_id===preferred):-1;show(target>=0?target:Math.min(index,filtered.length-1))}else{$('#status').textContent='No assets match filter';source.width=0;source.height=0;crop.width=0;crop.height=0;$('#assetmeta').textContent='';current=null}}
async function loadList(preferred){let r=await fetch('/api/assets');items=await r.json();renderList(preferred)}
function draw(){if(!img.complete||!img.naturalWidth)return;ctx.clearRect(0,0,source.width,source.height);ctx.drawImage(img,0,0);if(box){ctx.strokeStyle='#ff2e37';ctx.lineWidth=Math.max(3,source.width/300);ctx.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);}}
function makeCrop(){if(!box)return;let [x1,y1,x2,y2]=box;crop.width=x2-x1;crop.height=y2-y1;cctx.clearRect(0,0,crop.width,crop.height);cctx.drawImage(img,x1,y1,crop.width,crop.height,0,0,crop.width,crop.height);}
function show(i){if(!filtered.length)return;index=(i+filtered.length)%filtered.length;$('#asset').value=index;current=filtered[index];box=null;crop.width=0;crop.height=0;$('#form [name=fruit_identity]').value=current.class_label||'Unclear';$('#assetmeta').textContent=`${current.class_label||'class unknown'} · ${current.image_width}×${current.image_height} · family ${current.source_family_id||'unknown'} · source image ${current.source_image_id||'unknown'} · ${current.source_provider||'provider unknown'} · ${current.source_license||'license unknown'} · SHA ${current.image_sha256}`;img=new Image();img.onload=()=>{source.width=img.naturalWidth;source.height=img.naturalHeight;draw()};img.src='/image/'+encodeURIComponent(current.asset_id);}
source.addEventListener('pointerdown',e=>{if(!img.naturalWidth)return;let r=source.getBoundingClientRect();let x=(e.clientX-r.left)*source.width/r.width,y=(e.clientY-r.top)*source.height/r.height;start=[x,y];box=null});source.addEventListener('pointerup',e=>{if(!start)return;let r=source.getBoundingClientRect();let x=(e.clientX-r.left)*source.width/r.width,y=(e.clientY-r.top)*source.height/r.height;box=[Math.max(0,Math.floor(Math.min(start[0],x))),Math.max(0,Math.floor(Math.min(start[1],y))),Math.min(source.width,Math.ceil(Math.max(start[0],x))),Math.min(source.height,Math.ceil(Math.max(start[1],y)))];start=null;if(box[2]-box[0]<2||box[3]-box[1]<2){box=null;return}draw();makeCrop()});source.addEventListener('pointermove',e=>{if(!start)return;let r=source.getBoundingClientRect();let x=(e.clientX-r.left)*source.width/r.width,y=(e.clientY-r.top)*source.height/r.height;box=[Math.min(start[0],x),Math.min(start[1],y),Math.max(start[0],x),Math.max(start[1],y)];draw()});
$('#asset').onchange=e=>show(+e.target.value);$('#prev').onclick=()=>show(index-1);$('#next').onclick=()=>show(index+1);$('#review-filter').onchange=()=>renderList(current&&current.asset_id);$('#search').oninput=()=>renderList(current&&current.asset_id);
$('#form').onsubmit=async e=>{e.preventDefault();if(!current||!box){alert('Draw a target fruit region and inspect the crop preview first.');return}let f=new FormData(e.target),green=f.get('green'),suitable=f.get('suitable');let payload={asset_id:current.asset_id,reviewer_id:f.get('reviewer_id'),decision:f.get('decision'),source_image_viewed:true,target_crop_viewed:true,target_region_xyxy_px:box,fruit_identity:f.get('fruit_identity'),peel_genuinely_green:green==='unknown'?null:green==='yes',peel_color_confidence:f.get('peel_color_confidence'),image_suitable:suitable==='unknown'?null:suitable==='yes',fruit_visibility:f.get('fruit_visibility'),occlusion:f.get('occlusion'),background_dominance:f.get('background_dominance'),reason:f.get('reason')||null,notes:f.get('notes')||null};let res=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});let msg=await res.json();$('#result').textContent=res.ok?'Saved append-only review '+msg.review_id:('Error: '+msg.error);if(res.ok){let previous=current.asset_id;await loadList(previous)}};
loadList();
</script></body></html>'''

PAIR_HTML=r'''<!doctype html><html><head><meta charset="utf-8"><title>Banana near-duplicate review</title><style>
body{font:15px system-ui;margin:20px;background:#f4f8f3;color:#17251b}header,section{background:#fff;border:1px solid #ccd8ce;border-radius:10px;padding:14px;margin-bottom:14px}.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{max-width:100%;max-height:420px;object-fit:contain;border:1px solid #9bad9f}label{display:block;margin:8px 0}textarea{width:98%;height:70px}input,select,button{font:inherit;padding:7px}@media(max-width:850px){main{grid-template-columns:1fr}}
</style></head><body><header><h2>Green Banana — near-duplicate pair review</h2><p>Compare both complete originals. pHash is only a candidate signal. Record SAME_SOURCE, SAME_FAMILY, DISTINCT, or NEEDS_PROVENANCE; decisions append to the audit history and never auto-merge families.</p><div class="bar"><button id="prev">Previous</button><select id="pair"></select><button id="next">Next</button><select id="filter"><option value="PENDING">Pending pairs</option><option value="ALL">All pairs</option><option value="REVIEWED">Reviewed pairs</option></select><span id="count"></span></div></header><main><section><h3 id="lefttitle"></h3><img id="left"><p id="leftmeta"></p></section><section><h3 id="righttitle"></h3><img id="right"><p id="rightmeta"></p></section></main><section><form id="form"><label>Reviewer ID <input name="reviewer_id" required></label><label>Decision <select name="decision"><option>DISTINCT</option><option>SAME_SOURCE</option><option>SAME_FAMILY</option><option>NEEDS_PROVENANCE</option></select></label><label>Reason (required) <textarea name="reason" required></textarea></label><label>Notes <textarea name="notes"></textarea></label><button type="submit">Append pair decision</button> <span id="result"></span></form></section>
<script>
const $=s=>document.querySelector(s);let all=[],items=[],i=0,current=null;
function render(){let f=$('#filter').value,q=$('#search')?.value?.toLowerCase()||'';items=all.filter(p=>(f==='ALL'||(f==='PENDING'&&!p.reviewed)||(f==='REVIEWED'&&p.reviewed))&&(!q||[p.filename_a,p.filename_b,p.family_id_a,p.family_id_b,p.pair_id].some(x=>(x||'').toLowerCase().includes(q))));let sel=$('#pair');sel.innerHTML='';items.forEach((p,n)=>{let o=document.createElement('option');o.value=n;o.textContent=`${p.review_status} | ${p.filename_a} ↔ ${p.filename_b} | ${p.phash_distance}`;sel.append(o)});$('#count').textContent=`${items.length} shown / ${all.length} total`;if(items.length)show(Math.min(i,items.length-1));else{current=null;$('#left').removeAttribute('src');$('#right').removeAttribute('src');$('#count').textContent='No pairs match filter'}}
function show(n){if(!items.length)return;i=(n+items.length)%items.length;$('#pair').value=i;current=items[i];$('#lefttitle').textContent=current.filename_a;$('#righttitle').textContent=current.filename_b;$('#left').src='/image/'+encodeURIComponent(current.asset_id_a);$('#right').src='/image/'+encodeURIComponent(current.asset_id_b);$('#leftmeta').textContent=`Family: ${current.family_id_a||'unknown'} · SHA-256: ${current.sha256_a}`;$('#rightmeta').textContent=`Family: ${current.family_id_b||'unknown'} · SHA-256: ${current.sha256_b}`;$('#result').textContent=`pHash distance ${current.phash_distance}; current decision ${current.review_status}`}
async function load(){all=await (await fetch('/api/pairs')).json();render()}$('#prev').onclick=()=>show(i-1);$('#next').onclick=()=>show(i+1);$('#pair').onchange=e=>show(+e.target.value);$('#filter').onchange=render;
$('#form').onsubmit=async e=>{e.preventDefault();if(!current)return;let f=new FormData(e.target),body={asset_id_a:current.asset_id_a,asset_id_b:current.asset_id_b,decision:f.get('decision'),reviewer_id:f.get('reviewer_id'),reason:f.get('reason'),notes:f.get('notes')||null,both_images_viewed:true};let r=await fetch('/api/pair-review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),v=await r.json();$('#result').textContent=r.ok?'Saved append-only pair decision '+v.pair_id:'Error: '+v.error;if(r.ok)await load()};load();
</script></body></html>'''


def now():return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
def manifest():return read_csv(COLLECTION/'manifests/intake_manifest.csv')
def latest_history(path):return latest_by_asset(read_jsonl(path))
def append_jsonl(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')

def review_assets():
    """Return real Green Apple and Green Banana files for label review."""
    reviews=latest_history(PHASE8D/'review/review_history.jsonl')
    permissions=latest_history(PHASE8D/'metadata/permission_history.jsonl')
    result=[]
    for row in manifest():
        rel=Path(row.get('incoming_relative_path') or '')
        expected={'green_banana':'Banana','green_apple':'Apple'}.get(rel.parts[0] if rel.parts else '')
        if not expected or row.get('class_label')!=expected:continue
        path=(INCOMING/rel).resolve()
        try:path.relative_to(INCOMING)
        except ValueError:continue
        if not path.is_file():continue
        result.append({'asset_id':row['asset_id'],'class_label':row.get('class_label'),'incoming_relative_path':row['incoming_relative_path'],
                       'image_sha256':row.get('image_sha256'),'image_width':row.get('image_width'),'image_height':row.get('image_height'),
                       'source_image_id':row.get('source_image_id'),'source_family_id':row.get('source_family_id'),
                       'source_provider':row.get('source_provider'),'source_license':row.get('source_license'),
                       'permission_status':permissions.get(row['asset_id'],{}).get('permission_status','PENDING'),
                       'review_status':reviews.get(row['asset_id'],{}).get('decision','PENDING')})
    return result

def is_green_banana_asset(row):
    rel=Path(row.get('incoming_relative_path') or '')
    return row.get('class_label')=='Banana' and bool(rel.parts) and rel.parts[0]=='green_banana'

def is_reviewable_green_fruit_asset(row):
    rel=Path(row.get('incoming_relative_path') or '')
    expected={'green_banana':'Banana','green_apple':'Apple'}.get(rel.parts[0] if rel.parts else '')
    return expected is not None and row.get('class_label')==expected

def record_permission(payload):
    aid=payload.get('asset_id');status=payload.get('permission_status')
    if aid not in {r.get('asset_id') for r in manifest() if is_green_banana_asset(r)}:raise ValueError('asset_id is not a Green Banana in the current intake manifest; run Phase 8C scan first')
    if status not in {'VERIFIED','REJECTED','NEEDS_PROVENANCE','PENDING'}:raise ValueError('invalid permission_status')
    if not payload.get('verified_by'):raise ValueError('verified_by is required to record the human permission decision')
    if status=='VERIFIED' and not payload.get('evidence_reference'):raise ValueError('VERIFIED permission requires an evidence_reference')
    row={'asset_id':aid,'permission_status':status,'evidence_reference':payload.get('evidence_reference'),
         'verified_by':payload['verified_by'],'verified_at':payload.get('verified_at') or now(),'notes':payload.get('notes')}
    append_jsonl(PHASE8D/'metadata/permission_history.jsonl',row);return row

def record_review(payload):
    aid=payload.get('asset_id');status=payload.get('decision')
    assets={r.get('asset_id'):r for r in manifest() if is_reviewable_green_fruit_asset(r)}
    if aid not in assets:raise ValueError('asset_id is not a Green Apple or Green Banana in the current intake manifest; run Phase 8C scan first')
    if status not in STATUSES:raise ValueError(f'decision must be one of {sorted(STATUSES)}')
    if not payload.get('reviewer_id'):raise ValueError('reviewer_id is required')
    if status in REASONS and not str(payload.get('reason') or '').strip():raise ValueError(f'reason is required for {status}')
    if payload.get('source_image_viewed') is not True or payload.get('target_crop_viewed') is not True:
        raise ValueError('review must confirm viewing both source image and target crop')
    roi=payload.get('target_region_xyxy_px')
    if not isinstance(roi,list) or len(roi)!=4 or not all(isinstance(v,int) for v in roi):raise ValueError('target_region_xyxy_px must contain four integer pixel coordinates')
    w=int(assets[aid].get('image_width') or 0);h=int(assets[aid].get('image_height') or 0)
    x1,y1,x2,y2=roi
    if not(w and h and 0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('target ROI is outside image dimensions or image dimensions are missing')
    if status=='VERIFIED_CORRECT':
        if payload.get('fruit_identity')!=assets[aid].get('class_label') or payload.get('peel_genuinely_green') is not True or payload.get('peel_color_confidence')!='CLEAR_GREEN':raise ValueError('VERIFIED_CORRECT requires the manifest fruit identity and clearly green peel')
        if payload.get('image_suitable') is not True or payload.get('fruit_visibility') not in {'whole','partial_identifiable'}:raise ValueError('VERIFIED_CORRECT requires a suitable image and identifiable fruit visibility')
    row={k:payload.get(k) for k in ['asset_id','reviewer_id','decision','source_image_viewed','target_crop_viewed','target_region_xyxy_px','fruit_identity','peel_genuinely_green','peel_color_confidence','image_suitable','fruit_visibility','occlusion','background_dominance','reason','notes','review_method','owner_authorized']}
    row['review_id']=payload.get('review_id') or 'gb_review_'+__import__('uuid').uuid4().hex
    row['reviewed_at']=payload.get('reviewed_at') or now()
    append_jsonl(PHASE8D/'review/review_history.jsonl',row);return row

def record_pair_review(payload):
    a,b=payload.get('asset_id_a'),payload.get('asset_id_b')
    asset_ids={r.get('asset_id') for r in manifest() if is_green_banana_asset(r)}
    if not a or not b or a==b or a not in asset_ids or b not in asset_ids:raise ValueError('pair must contain two distinct Green Banana assets in current intake manifest')
    decision=payload.get('decision')
    if decision not in {'SAME_SOURCE','SAME_FAMILY','DISTINCT','NEEDS_PROVENANCE'}:raise ValueError('invalid pair decision')
    if not payload.get('reviewer_id') or not str(payload.get('reason') or '').strip():raise ValueError('reviewer_id and reason are required')
    if payload.get('both_images_viewed') is not True:raise ValueError('pair review must confirm both complete images were viewed')
    pair_key='|'.join(sorted([a,b]));pair_id='pair_'+__import__('hashlib').sha256(pair_key.encode()).hexdigest()[:24]
    row={'pair_id':pair_id,'asset_id_a':a,'asset_id_b':b,'decision':decision,
         'reviewer_id':payload['reviewer_id'],'reviewed_at':payload.get('reviewed_at') or now(),'reason':payload['reason'],'notes':payload.get('notes'),'both_images_viewed':True,
         'review_method':payload.get('review_method'),'owner_authorized':payload.get('owner_authorized') is True}
    append_jsonl(PHASE8D/'review/duplicate_pair_history.jsonl',row);return row

def pair_assets():
    """Return current Green Banana duplicate candidates with latest human event."""
    import hashlib
    manifest_rows={r['asset_id']:r for r in manifest() if is_green_banana_asset(r)}
    latest={}
    for event in read_jsonl(PHASE8D/'review/duplicate_pair_history.jsonl'):
        latest[event.get('pair_id')]=event
    out=[]
    for pair in read_csv(COLLECTION/'review/near_duplicate_review_queue.csv'):
        a=manifest_rows.get(pair.get('asset_id_a'));b=manifest_rows.get(pair.get('asset_id_b'))
        if not a or not b:continue
        key='|'.join(sorted([a['asset_id'],b['asset_id']]));pair_id='pair_'+hashlib.sha256(key.encode()).hexdigest()[:24]
        event=latest.get(pair_id,{})
        reviewer=str(event.get('reviewer_id','')).lower()
        human=bool(reviewer) and not reviewer.startswith(('codex_','ai_','bot_'))
        out.append({'pair_id':pair_id,'asset_id_a':a['asset_id'],'asset_id_b':b['asset_id'],
          'filename_a':a.get('original_filename'),'filename_b':b.get('original_filename'),
          'family_id_a':a.get('source_family_id'),'family_id_b':b.get('source_family_id'),
          'sha256_a':a.get('image_sha256'),'sha256_b':b.get('image_sha256'),
          'phash_distance':pair.get('phash_distance'),'decision':event.get('decision'),
          'review_status':event.get('decision','PENDING'),'reviewed':human and event.get('decision') in {'SAME_SOURCE','SAME_FAMILY','DISTINCT','NEEDS_PROVENANCE'}})
    return out

class Handler(BaseHTTPRequestHandler):
    def _send(self,status,body,ctype='application/json'):
        data=body if isinstance(body,bytes) else body.encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        parsed=urlparse(self.path)
        if parsed.path=='/':return self._send(200,HTML,'text/html; charset=utf-8')
        if parsed.path=='/pair-review':return self._send(200,PAIR_HTML,'text/html; charset=utf-8')
        if parsed.path=='/api/assets':
            return self._send(200,json.dumps(review_assets()))
        if parsed.path=='/api/pairs':return self._send(200,json.dumps(pair_assets()))
        m=re.fullmatch(r'/image/(asset_[a-f0-9]{24})',unquote(parsed.path))
        if m:
            aid=m.group(1)
            for r in manifest():
                if r.get('asset_id')==aid:
                    p=(INCOMING/Path(r['incoming_relative_path'])).resolve()
                    try:p.relative_to(INCOMING)
                    except ValueError:break
                    if p.is_file():return self._send(200,p.read_bytes(),mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
            return self._send(404,json.dumps({'error':'image not found'}))
        return self._send(404,json.dumps({'error':'not found'}))
    def do_POST(self):
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size<=0 or size>100_000:raise ValueError('request body size invalid')
            payload=json.loads(self.rfile.read(size))
            if urlparse(self.path).path=='/api/pair-review':result=record_pair_review(payload)
            elif urlparse(self.path).path=='/api/review':result=record_review(payload)
            else:raise ValueError('not found')
            return self._send(201,json.dumps(result,ensure_ascii=False))
        except Exception as exc:return self._send(400,json.dumps({'error':str(exc)},ensure_ascii=False))
    def log_message(self,fmt,*args):print('review-ui:',fmt%args)

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='cmd',required=True)
    serve=sub.add_parser('serve');serve.add_argument('--host',default='127.0.0.1');serve.add_argument('--port',type=int,default=8768)
    permission=sub.add_parser('record-permission');permission.add_argument('--json',type=Path,required=True)
    pair=sub.add_parser('record-duplicate-review');pair.add_argument('--json',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.cmd=='serve':
            server=ThreadingHTTPServer((args.host,args.port),Handler);print(f'Green Banana review UI: http://{args.host}:{args.port}/');server.serve_forever()
        elif args.cmd=='record-permission':print(json.dumps(record_permission(json.loads(args.json.read_text(encoding='utf-8'))),ensure_ascii=False,indent=2))
        else:print(json.dumps(record_pair_review(json.loads(args.json.read_text(encoding='utf-8'))),ensure_ascii=False,indent=2))
    except KeyboardInterrupt:return 0
    except Exception as exc:print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main() or 0)
