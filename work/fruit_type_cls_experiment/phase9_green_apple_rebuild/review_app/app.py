#!/usr/bin/env python3
"""Local append-only reviewer for the 1,017 Mendeley VIA apple polygons."""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[1]
QUEUE = BASE / "review_queue.csv"
HISTORY = BASE / "review_history.jsonl"
COLLECTION = BASE.parent / "phase8c_collection/green_apple_phase8d/staging/mendeley_v1"
VALID_DECISIONS = {"GREEN_APPLE", "NON_GREEN_APPLE", "UNCERTAIN", "INVALID_ANNOTATION"}
LOCK = threading.Lock()


def load_queue():
    priority = BASE / "review_priority_queue.csv"
    path = priority if priority.exists() else QUEUE
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latest_events():
    latest = {}
    if HISTORY.exists():
        with HISTORY.open(encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                    latest[event["polygon_id"]] = event
                except (json.JSONDecodeError, KeyError):
                    continue
    return latest


def render_image(row, crop=False):
    path = COLLECTION / row["image_relative_path"]
    with Image.open(path) as im:
        base = im.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    target = int(row["region_index"])
    for r in load_queue():
        if (r["source_split_original"], r["original_filename"]) != (row["source_split_original"], row["original_filename"]):
            continue
        pts = json.loads(r["polygon_points_json"])
        if len(pts) < 3:
            continue
        color = (255, 185, 0, 85) if int(r["region_index"]) == target else (40, 150, 255, 28)
        outline = (230, 40, 20, 255) if int(r["region_index"]) == target else (25, 100, 230, 210)
        draw.polygon([tuple(p) for p in pts], fill=color, outline=outline, width=4 if int(r["region_index"]) == target else 2)
    im = Image.alpha_composite(base, overlay).convert("RGB")
    if crop:
        target_row = next(r for r in load_queue() if r["polygon_id"] == row["polygon_id"])
        pts = json.loads(target_row["polygon_points_json"])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = max(8, int(max(max(xs)-min(xs), max(ys)-min(ys)) * .18))
        im = im.crop((max(0, min(xs)-pad), max(0, min(ys)-pad), min(im.width, max(xs)+pad), min(im.height, max(ys)+pad)))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, min(im.width, 680), 32), fill=(0, 0, 0))
    draw.text((8, 8), f"TARGET #{target} — {row['source_label']}", fill=(255, 255, 255))
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=92)
    return out.getvalue()


def current_items(params):
    queue = load_queue()
    latest = latest_events()
    status_filter = params.get("status", ["PENDING"])[0]
    split_filter = params.get("split", ["ALL"])[0]
    label_filter = params.get("label", ["ALL"])[0]
    priority_filter = params.get("priority", ["EARLY_CANDIDATE_PRIORITY"])[0]
    rows = []
    for row in queue:
        decision = latest.get(row["polygon_id"], {}).get("decision", "PENDING")
        if status_filter != "ALL" and decision != status_filter:
            continue
        if split_filter != "ALL" and row["source_split_original"] != split_filter:
            continue
        if label_filter != "ALL" and row["source_label"] != label_filter:
            continue
        if priority_filter != "ALL" and row.get("priority_tier", "EARLY_CANDIDATE_PRIORITY") != priority_filter:
            continue
        rows.append({**row, "decision": decision, "review_event": latest.get(row["polygon_id"])})
    counts = {name: 0 for name in ("PENDING", *sorted(VALID_DECISIONS))}
    for row in queue:
        decision = latest.get(row["polygon_id"], {}).get("decision", "PENDING")
        counts[decision] = counts.get(decision, 0) + 1
    return rows, counts, len(queue)


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Green Apple polygon review</title>
<style>
body{font:15px system-ui,sans-serif;margin:0;background:#f3f5f1;color:#172016}header{background:#163a2c;color:white;padding:14px 22px}main{max-width:1500px;margin:16px auto;padding:0 14px}.warning{background:#fff4ce;border:1px solid #e9c75b;padding:12px;border-radius:8px;margin-bottom:12px}.bar,.filters,.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:white;padding:12px;border:1px solid #ccd8ce;border-radius:8px;margin-bottom:12px}.bar strong{margin-right:12px}.images{display:grid;grid-template-columns:2fr 1fr;gap:12px}.panel{background:white;border:1px solid #ccd8ce;border-radius:8px;padding:10px}.panel img{width:100%;height:auto;max-height:75vh;object-fit:contain;background:#222}.meta{display:grid;grid-template-columns:170px 1fr;gap:6px;padding:10px 0}.meta b{color:#355744}button,select,input,textarea{font:inherit;padding:7px 10px}button{cursor:pointer;border:1px solid #8da798;background:#f8fbf9;border-radius:5px}.decision{display:flex;gap:8px;flex-wrap:wrap}.decision button{background:#e5f2e9;border-color:#80a98c}.decision button[data-d="NON_GREEN_APPLE"]{background:#e9eef8}.decision button[data-d="UNCERTAIN"],.decision button[data-d="INVALID_ANNOTATION"]{background:#fff0dc}.decision button:disabled{opacity:.5;cursor:wait}textarea{width:min(100%,900px);min-height:72px}.small{font-size:13px;color:#59685d}.badge{padding:3px 8px;border-radius:12px;background:#e9f1eb}.right{margin-left:auto}a{color:#0d6543}
</style></head><body><header><b>Green Apple — VIA polygon review</b></header><main>
<div class="warning"><b>Source label is not the review decision.</b> “immature apple” does not automatically mean Green Apple. Inspect the complete source image and highlighted polygon. Nothing is auto-approved; each save appends an immutable event.</div>
<div class="bar"><strong id="counts">Loading review progress…</strong><a href="/api/export.csv">Export latest decisions CSV</a><span class="right small">Append-only history: review_history.jsonl</span></div>
<div class="filters"><label>Decision <select id="fstatus"><option>PENDING</option><option>ALL</option><option>GREEN_APPLE</option><option>NON_GREEN_APPLE</option><option>UNCERTAIN</option><option>INVALID_ANNOTATION</option></select></label><label>Priority <select id="fpriority"><option value="EARLY_CANDIDATE_PRIORITY">Early candidate first</option><option value="ALL">All, including held</option><option value="REVIEW_MASK_CAREFULLY">Mask review</option><option value="HOLD_PROVENANCE_OR_DUPLICATE">Duplicate/provenance hold</option></select></label><label>Original split <select id="fsplit"><option>ALL</option><option>train</option><option>test</option></select></label><label>Source stage <select id="flabel"><option>ALL</option><option>immature apple</option><option>semi-mature apple</option><option>mature apple</option><option>UNKNOWN</option></select></label><button id="apply">Apply filters</button><span id="position" class="small"></span></div>
<div class="images"><section class="panel"><h3>Complete original — all masks; target highlighted</h3><img id="image" alt="Original source with target polygon highlighted"><div id="sourceinfo" class="small"></div><h3>Target detail crop</h3><img id="crop" alt="Enlarged target polygon crop"></section><section class="panel"><h3>Polygon review</h3><div id="meta" class="meta"></div><label>Reviewer ID<br><input id="reviewer" required placeholder="reviewer identifier"></label><p><label>Decision<br><select id="decision"><option value="">Choose a decision…</option><option value="GREEN_APPLE">Green Apple</option><option value="NON_GREEN_APPLE">Non-green Apple</option><option value="UNCERTAIN">Uncertain</option><option value="INVALID_ANNOTATION">Invalid annotation</option></select></label></p><p><label>Mask correctly covers target? <select id="maskvalid"><option value="">Choose…</option><option value="yes">Yes</option><option value="no">No</option></select></label> <label>Image quality sufficient? <select id="quality"><option value="">Choose…</option><option value="yes">Yes</option><option value="no">No</option></select></label></p><label>Reason / evidence (required)<br><textarea id="reason" placeholder="Describe peel color, fruit identity, polygon quality, visibility, or why uncertain/invalid"></textarea></label><p><label>Additional notes<br><textarea id="notes"></textarea></label></p><div class="decision"><button id="save">Save decision (S)</button><button id="prev">Previous (←)</button><button id="next">Next (→)</button><button id="sameprev">Same image ( [ )</button><button id="samenext">Same image ( ] )</button></div><p class="small">Shortcuts: 1 Green · 2 Non-green · 3 Uncertain · 4 Invalid · S save · arrows next/previous · [ / ] same image. Green requires explicit mask and quality confirmation.</p><p id="saved" class="small"></p></section></div>
</main><script>
let rows=[],counts={},pos=0;const $=x=>document.getElementById(x);function params(){return new URLSearchParams({status:$('fstatus').value,split:$('fsplit').value,label:$('flabel').value,priority:$('fpriority').value})}function esc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function load(){let r=await fetch('/api/items?'+params());let d=await r.json();rows=d.items;counts=d.counts;pos=Math.min(pos,Math.max(0,rows.length-1));$('counts').textContent=`${d.total} polygons · pending ${counts.PENDING||0} · Green ${counts.GREEN_APPLE||0} · Non-green ${counts.NON_GREEN_APPLE||0} · uncertain ${counts.UNCERTAIN||0} · invalid ${counts.INVALID_ANNOTATION||0}`;$('position').textContent=rows.length?`${pos+1} / ${rows.length} filtered`:'No items match this filter';show()}
function show(){if(!rows.length){$('image').removeAttribute('src');$('crop').removeAttribute('src');$('meta').innerHTML='No polygons in queue.';$('sourceinfo').textContent='';return}let x=rows[pos];$('image').src='/image/'+encodeURIComponent(x.polygon_id)+'?v='+Date.now();$('crop').src='/crop/'+encodeURIComponent(x.polygon_id)+'?v='+Date.now();$('meta').innerHTML=`<b>Polygon ID</b><span>${esc(x.polygon_id)}</span><b>Image / asset ID</b><span>${esc(x.original_filename)} / ${esc(x.asset_id)}</span><b>Source label</b><span>${esc(x.source_label)} <span class="badge">clue only; not a decision</span></span><b>Original split</b><span>${esc(x.source_split_original)} (source test images cannot enter training)</span><b>Priority</b><span>${esc(x.priority_tier||'not ranked')} · ${esc(x.priority_reason||'')}</span><b>SHA-256</b><span>${esc(x.image_sha256)}</span><b>Dimensions</b><span>${esc(x.image_width)} × ${esc(x.image_height)}</span><b>Current decision</b><span>${esc(x.decision)}</span>`;$('sourceinfo').textContent=`DOI ${x.source_doi} · CC BY 4.0 · target region ${x.region_index} highlighted in orange/red; sibling masks blue.`;$('decision').value='';$('maskvalid').value='';$('quality').value='';$('reason').value='';$('notes').value='';$('saved').textContent=x.review_event?`Latest event by ${x.review_event.reviewer_id} at ${x.review_event.reviewed_at}; prior event retained.`:''}
function move(delta,same=false){if(!rows.length)return;let curr=rows[pos];let candidates=rows.map((r,i)=>({r,i})).filter(x=>!same||x.r.asset_id===curr.asset_id);if(!candidates.length)return;let k=candidates.findIndex(x=>x.i===pos);pos=candidates[(k+delta+candidates.length)%candidates.length].i;show()}$('apply').onclick=()=>{pos=0;load()};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('sameprev').onclick=()=>move(-1,true);$('samenext').onclick=()=>move(1,true);$('save').onclick=async()=>{if(!rows.length)return;const x=rows[pos],decision=$('decision').value,reviewer_id=$('reviewer').value.trim(),reason=$('reason').value.trim(),mask_valid=$('maskvalid').value, image_quality_ok=$('quality').value;if(!decision||!reviewer_id||!reason){alert('Decision, reviewer ID, and reason/evidence are required.');return}if(decision==='GREEN_APPLE'&&(mask_valid!=='yes'||image_quality_ok!=='yes')){alert('Green Apple requires explicit YES for a valid mask and sufficient image quality.');return}let b=$('save');b.disabled=true;try{let r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({polygon_id:x.polygon_id,decision,reviewer_id,reason,notes:$('notes').value,mask_valid,image_quality_ok})});let d=await r.json();if(!r.ok)throw Error(d.error||'Save failed');$('saved').textContent=`Saved ${d.event.decision} at ${d.event.reviewed_at}. History appended.`;await load()}catch(e){alert(e.message)}finally{b.disabled=false}};document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(e.key==='[')move(-1,true);else if(e.key===']')move(1,true);else if(['1','2','3','4'].includes(e.key)){$('decision').value=({'1':'GREEN_APPLE','2':'NON_GREEN_APPLE','3':'UNCERTAIN','4':'INVALID_ANNOTATION'})[e.key];$('decision').focus()}else if(e.key.toLowerCase()==='s')$('save').click()});load();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("review-app:", fmt % args)

    def send_bytes(self, data, content_type, status=200):
        self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.send_bytes(PAGE.encode(), "text/html; charset=utf-8")
        if parsed.path == "/api/items":
            rows, counts, total = current_items(parse_qs(parsed.query))
            return self.send_bytes(json.dumps({"items": rows, "counts": counts, "total": total}).encode(), "application/json")
        if parsed.path.startswith("/image/") or parsed.path.startswith("/crop/"):
            pid = parsed.path.rsplit("/", 1)[-1]
            row = next((x for x in load_queue() if x["polygon_id"] == pid), None)
            if row is None:
                return self.send_bytes(b"not found", "text/plain", 404)
            try:
                return self.send_bytes(render_image(row, crop=parsed.path.startswith("/crop/")), "image/jpeg")
            except Exception as e:
                return self.send_bytes(str(e).encode(), "text/plain", 500)
        if parsed.path == "/api/export.csv":
            rows = load_queue(); latest = latest_events(); out = io.StringIO(newline="")
            fields = list(rows[0]) + ["current_decision", "reviewer_id", "reviewed_at", "reason", "notes", "mask_valid", "image_quality_ok"]
            w = csv.DictWriter(out, fieldnames=fields); w.writeheader()
            for row in rows:
                e = latest.get(row["polygon_id"], {})
                w.writerow({**row, "current_decision": e.get("decision", "PENDING"), "reviewer_id": e.get("reviewer_id", ""), "reviewed_at": e.get("reviewed_at", ""), "reason": e.get("reason", ""), "notes": e.get("notes", ""), "mask_valid": e.get("mask_valid", ""), "image_quality_ok": e.get("image_quality_ok", "")})
            return self.send_bytes(out.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8")
        return self.send_bytes(b"not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/review":
            return self.send_bytes(b"not found", "text/plain", 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if body.get("decision") not in VALID_DECISIONS:
                raise ValueError("Invalid review decision")
            if not str(body.get("reviewer_id", "")).strip() or not str(body.get("reason", "")).strip():
                raise ValueError("Reviewer ID and reason/evidence are required")
            if body["decision"] == "GREEN_APPLE" and (body.get("mask_valid") != "yes" or body.get("image_quality_ok") != "yes"):
                raise ValueError("GREEN_APPLE requires explicit mask_valid=yes and image_quality_ok=yes")
            pid = str(body.get("polygon_id", ""))
            if not any(x["polygon_id"] == pid for x in load_queue()):
                raise ValueError("Unknown polygon_id")
            event = {"event_id": str(uuid.uuid4()), "polygon_id": pid, "decision": body["decision"],
                     "reviewer_id": str(body["reviewer_id"]).strip(), "reviewed_at": datetime.now(timezone.utc).isoformat(),
                     "reason": str(body["reason"]).strip(), "notes": str(body.get("notes", "")).strip(),
                     "mask_valid": body.get("mask_valid", ""), "image_quality_ok": body.get("image_quality_ok", ""),
                     "source": "local_green_apple_review_app", "previous_event_preserved": True}
            with LOCK, HISTORY.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
            return self.send_bytes(json.dumps({"event": event}).encode(), "application/json")
        except Exception as e:
            return self.send_bytes(json.dumps({"error": str(e)}).encode(), "application/json", 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("This review app is local-only; host must be loopback")
    if not QUEUE.is_file():
        raise SystemExit(f"Review queue missing: {QUEUE}; run tools/build_review_queue.py first")
    print(f"Green Apple polygon review: http://127.0.0.1:{args.port}/")
    print(f"Queue: {QUEUE} ({len(load_queue())} polygon records)")
    print(f"History: {HISTORY}; all decisions append and resume after restart")
    try:
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    except OSError as e:
        raise SystemExit(f"Could not bind {args.host}:{args.port}: {e}")


if __name__ == "__main__":
    main()
