"""Headless browser interaction checks with a disposable synthetic fixture."""
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import uuid
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/"out/quick-review-browser-test"
folder.mkdir(exist_ok=True)
for name in ("instance_geometry.js",):
    shutil.copy2(ROOT/"tools"/name,folder/name)
items=[]
for key,color in (("a","red"),("b","green"),("c","blue")):
    Image.new("RGB",(200,160),color).save(folder/(key+".png"))
    items.append({"id":key,"role":"diagnostic","source_image":key+".png","image":key+".png",
                  "boxes":[],"warnings":[],"proposal":{"boxes":[[1,.5,.5,.4,.4]]} if key!="c" else {"boxes":[]}})
manifest={"dataset_id":"isolated-browser-interaction-fixture","items":items,"quick_review_ids":["a","b","c"]}
(folder/"manifest.js").write_text("window.INSTANCE_DATA="+json.dumps(manifest)+";",encoding="utf-8")
page=(ROOT/"tools/instance_review.html").read_text(encoding="utf-8")
prelude='''<script>localStorage.setItem('instance-review:isolated-browser-interaction-fixture',JSON.stringify({a:{id:'a',status:'pending',expected_count:null,boxes:[[0,.4,.5,.3,.4]]}}));</script>'''
page=page.replace('<script src="manifest.js">',prelude+'<script src="manifest.js">')
harness=r'''
<script>
(async()=>{
 const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name)};
 const wait=()=>new Promise(r=>setTimeout(r,150));
 try{
  await wait();
  check(filtered.length===3 && $('quick').checked,'three-image queue');
  check(current().boxes[0][0]===0 && current().boxes[0][1]===.4,'existing manual draft preserved');
  $('proposal').click();check(current().boxes[0][0]===1 && current().status==='pending' && current().expected_count===null,'explicit proposal remains unverified');
  $('undo').click();check(current().boxes[0][0]===0,'undo restores user annotations');
  position=1;load();await wait();
  check(current().boxes.length===1 && current().status==='pending' && current().expected_count===null,'new image starts with proposal, no approval');
  const canvas=$('canvas');canvas.setPointerCapture=()=>{};
  const event=(x,y)=>{const r=canvas.getBoundingClientRect();return{pointerId:1,clientX:r.left+x*r.width,clientY:r.top+y*r.height}};
  canvas.onpointerdown(event(.5,.5));canvas.onpointermove(event(.6,.5));canvas.onpointerup(event(.6,.5));
  check(Math.abs(current().boxes[0][1]-.6)<1e-6,'drag moves selected box');
  $('undo').click();check(Math.abs(current().boxes[0][1]-.5)<1e-6,'one undo restores whole drag');
  canvas.onpointerdown(event(.5,.5));canvas.onpointerup(event(.5,.5));
  canvas.onpointerdown(event(.7,.7));canvas.onpointermove(event(.9,.8));canvas.onpointerup(event(.9,.8));
  check(Math.abs(current().boxes[0][3]-.6)<1e-6 && Math.abs(current().boxes[0][4]-.5)<1e-6,'corner resize works');
  $('undo').click();
  selected=0;
  document.onkeydown({key:'1',ctrlKey:true,metaKey:false,target:document.body,preventDefault(){}});
  check(current().boxes[0][0]===1,'browser Ctrl+1 does not relabel fruit');
  $('expected').value='1';
  document.onkeydown({key:'Enter',ctrlKey:true,metaKey:false,target:$('expected'),preventDefault(){}});
  check(drafts.b.status==='approved' && drafts.b.expected_count===1,'Ctrl+Enter reads current count input');
  check(item().id==='c' && current().status==='pending','approval advances without approving next image');
  const pre=document.createElement('pre');pre.id='browser-results';pre.textContent=JSON.stringify({ok:true,checks});document.body.append(pre);
 }catch(error){const pre=document.createElement('pre');pre.id='browser-results';pre.textContent=JSON.stringify({ok:false,error:error.message,checks});document.body.append(pre)}
})();
</script>
'''
(folder/"review.html").write_text(page+harness,encoding="utf-8")
chrome=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
command=[str(chrome),"--headless=new","--disable-gpu","--no-first-run","--no-default-browser-check",
         "--disable-background-networking","--disable-component-update","--disable-sync",
         "--allow-file-access-from-files","--virtual-time-budget=5000",
         "--user-data-dir="+str(folder/("profile-"+uuid.uuid4().hex[:8])),"--dump-dom",(folder/"review.html").as_uri()]
result=subprocess.run(command,capture_output=True,timeout=35)
(folder/"dom.html").write_bytes(result.stdout)
(folder/"stderr.txt").write_bytes(result.stderr)
match=re.search(r'<pre id="browser-results">(.*?)</pre>',result.stdout.decode("utf-8",errors="replace"),re.S)
if not match:
    raise RuntimeError("No browser test result. See out/quick-review-browser-test/stderr.txt")
report=json.loads(html.unescape(match.group(1)))
(folder/"result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report))
assert report["ok"],report
