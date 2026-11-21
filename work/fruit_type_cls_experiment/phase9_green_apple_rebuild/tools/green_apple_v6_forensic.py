import csv, json, math, hashlib, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT=Path('/mnt/01DC7E1E309A75E0/CODE/BTL_XLA')
BATCH=ROOT/'fruit/work/fruit_type_cls_experiment/phase9_green_apple_rebuild/candidate_batches/green_apple_smoke_001'
RUN=ROOT/'fruit/runs/seg/seg_green_v6_smoke_cpu_20261121'
CKPT=RUN/'weights/best.pt'
OUT=ROOT/'fruit/runs/seg/seg_green_v6_smoke_forensic_20261121'
OUT.mkdir(parents=True,exist_ok=False)
VIS=OUT/'visualizations'; VIS.mkdir()
rows=list(csv.DictReader((BATCH/'candidate_manifest.csv').open()))

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read_gt(label_path,w,h):
 masks=[]
 for line in Path(label_path).read_text().splitlines():
  vals=list(map(float,line.split())); cls=int(vals[0]); pts=np.array(vals[1:],dtype=np.float32).reshape(-1,2)
  pts[:,0]*=w; pts[:,1]*=h
  mask=np.zeros((h,w),np.uint8); cv2.fillPoly(mask,[np.rint(pts).astype(np.int32)],1)
  masks.append({'class_id':cls,'mask':mask.astype(bool),'points':pts})
 return masks
def iou(a,b):
 inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
 return float(inter/union) if union else 0.
def box_iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
 inter=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
 return inter/(aa+bb-inter) if aa+bb-inter else 0.
def color_overlay(img, masks, color, alpha):
 out=img.copy(); layer=img.copy()
 for m in masks:
  layer[m]=color
 return cv2.addWeighted(layer,alpha,out,1-alpha,0)
def draw_panel(img, title, gt, preds, class_names):
 out=img.copy()
 # GT mask green; prediction cyan (overlap blends).
 if gt:
  out=color_overlay(out,[g['mask'] for g in gt],(0,220,0),0.38)
  for g in gt:
   ys,xs=np.where(g['mask'])
   if len(xs):
    x1,y1,x2,y2=int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())
    cv2.rectangle(out,(x1,y1),(x2,y2),(0,200,0),2)
    cv2.putText(out,'GT green_apple',(x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(out,'GT green_apple',(x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
 for p in preds:
  mask=p['mask']; out=color_overlay(out,[mask],(255,180,0),0.32)
  x1,y1,x2,y2=map(int,p['xyxy']); cv2.rectangle(out,(x1,y1),(x2,y2),(255,120,0),2)
  name=class_names.get(str(p['class_id']),class_names.get(p['class_id'],str(p['class_id'])))
  text=f"{name} {p['conf']:.2f}"
  cv2.putText(out,text,(x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3,cv2.LINE_AA)
  cv2.putText(out,text,(x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
 cv2.rectangle(out,(0,0),(out.shape[1],35),(245,245,245),-1)
 cv2.putText(out,title,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.62,(20,20,20),1,cv2.LINE_AA)
 return out

# Current saved inference config, as run previously: conf=.05, IoU=.7, imgsz=640, max_det=300.
# Evaluate all requested confidence thresholds at IoU=.7 plus NMS sweep at conf=.5.
settings=[('baseline_conf005_iou070',.05,.70),('conf025_iou070',.25,.70),('conf040_iou070',.40,.70),('conf050_iou070',.50,.70),('conf060_iou070',.60,.70),('conf075_iou070',.75,.70),('conf050_iou050',.50,.50),('conf050_iou090',.50,.90),('conf005_iou050',.05,.50),('conf005_iou090',.05,.90)]
model=YOLO(str(CKPT))
names=model.names
if isinstance(names,dict): names={str(k):v for k,v in names.items()}
else: names={str(i):v for i,v in enumerate(names)}
print('names',names,'task',model.task)
# Verify image sources and labels, evaluate outputs.
summary=[]; per_image=[]; detail=[]; base_results={}; all_results={}
for setting,conf,iou_nms in settings:
 print('SETTING',setting,flush=True)
 paths=[str(BATCH/r['image_path']) for r in rows]
 results=model.predict(source=paths,imgsz=640,device='cpu',conf=conf,iou=iou_nms,max_det=300,retina_masks=False,verbose=False,save=False)
 all_results[setting]=results
 total_det=0; tp=fp=fn=dups=bg=0; matched_ious=[]
 for row,path,result in zip(rows,paths,results):
  image=cv2.imread(path); h,w=image.shape[:2]
  gt=read_gt(BATCH/row['label_path'],w,h)
  pred=[]
  if result.boxes is not None and result.masks is not None:
   boxes=result.boxes.xyxy.cpu().numpy(); confs=result.boxes.conf.cpu().numpy(); clsids=result.boxes.cls.cpu().numpy().astype(int)
   for i,(box,score,cid) in enumerate(zip(boxes,confs,clsids)):
    poly=result.masks.xy[i]
    pm=np.zeros((h,w),np.uint8)
    if len(poly)>=3: cv2.fillPoly(pm,[np.rint(poly).astype(np.int32)],1)
    pred.append({'xyxy':box.tolist(),'conf':float(score),'class_id':int(cid),'mask':pm.astype(bool)})
  total_det+=len(pred)
  # Greedy one-to-one mask IoU matching at 0.5, class aware.
  pairs=[]
  for pi,p in enumerate(pred):
   for gi,g in enumerate(gt):
    if p['class_id']==g['class_id']:
     pairs.append((iou(p['mask'],g['mask']),pi,gi))
  pairs.sort(reverse=True)
  mp=set();mg=set();tp_img=0
  for ov,pi,gi in pairs:
   if ov>=.5 and pi not in mp and gi not in mg:
    mp.add(pi);mg.add(gi);tp_img+=1;matched_ious.append(ov)
  fp_img=len(pred)-len(mp);fn_img=len(gt)-len(mg);dup_img=0;bg_img=0
  for pi,p in enumerate(pred):
   if pi in mp: continue
   max_iou=max([iou(p['mask'],g['mask']) if p['class_id']==g['class_id'] else 0 for g in gt] or [0])
   if max_iou>=.5: dup_img+=1
   elif max_iou<.05: bg_img+=1
  tp+=tp_img;fp+=fp_img;fn+=fn_img;dups+=dup_img;bg+=bg_img
  per_image.append({'setting':setting,'asset_id':row['asset_id'],'original_filename':row['original_filename'],'detections':len(pred),'gt_instances':len(gt),'tp_mask_iou50':tp_img,'fp':fp_img,'fn':fn_img,'duplicate_fp':dup_img,'background_fp_iou_lt_005':bg_img})
  for pi,p in enumerate(pred):
   gt_ious=[iou(p['mask'],g['mask']) if p['class_id']==g['class_id'] else 0 for g in gt]
   gt_box_ious=[]
   for g in gt:
    if p['class_id']!=g['class_id']: gt_box_ious.append(0); continue
    pts=g['points']; gb=[float(pts[:,0].min()),float(pts[:,1].min()),float(pts[:,0].max()),float(pts[:,1].max())]
    gt_box_ious.append(box_iou(p['xyxy'],gb))
   maxov=max(gt_ious or [0]); target=int(np.argmax(gt_ious)) if gt_ious else -1
   category='TP' if pi in mp else ('DUPLICATE' if maxov>=.5 else ('BACKGROUND_FP' if maxov<.05 else 'FP'))
   detail.append({'setting':setting,'asset_id':row['asset_id'],'original_filename':row['original_filename'],'prediction_index':pi,'class_id':p['class_id'],'class_name':names.get(str(p['class_id']),'unknown'),'confidence':p['conf'],'xyxy':json.dumps([round(x,2) for x in p['xyxy']]),'best_gt_index':target,'best_mask_iou':maxov,'best_box_iou':max(gt_box_ious or [0]),'category':category})
  if setting=='baseline_conf005_iou070': base_results[row['asset_id']]=(image,gt,pred,row)
 summary.append({'setting':setting,'conf':conf,'nms_iou':iou_nms,'imgsz':640,'max_det':300,'images':len(results),'detections':total_det,'gt_instances':sum(len(read_gt(BATCH/r['label_path'],cv2.imread(str(BATCH/r['image_path'])).shape[1],cv2.imread(str(BATCH/r['image_path'])).shape[0])) for r in rows),'tp_mask_iou50':tp,'fp':fp,'fn':fn,'duplicate_fp':dups,'background_fp_iou_lt_005':bg,'mean_matched_mask_iou':float(np.mean(matched_ious)) if matched_ious else None,'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0})

# Save visual grids using outputs from the exact list/batch inference, not single-image reruns.
def results_to_preds(result,h,w):
 preds=[]
 if result.boxes is not None and result.masks is not None:
  for i,(box,score,cid) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy().astype(int))):
   m=np.zeros((h,w),np.uint8); poly=result.masks.xy[i]
   if len(poly)>=3: cv2.fillPoly(m,[np.rint(poly).astype(np.int32)],1)
   preds.append({'xyxy':box.tolist(),'conf':float(score),'class_id':int(cid),'mask':m.astype(bool)})
 return preds
for idx,row in enumerate(rows):
 image,gt,pred,_=base_results[row['asset_id']]
 h,w=image.shape[:2]
 panels=[draw_panel(image,'Original',[],[],names),draw_panel(image,'GT masks (AI-assisted labels)',gt,[],names)]
 for setting,conf,iou_nms in settings[:6]:
  preds=results_to_preds(all_results[setting][idx],h,w)
  panels.append(draw_panel(image,f'{setting}: {len(preds)} det',gt,preds,names))
 target_w=600;target_h=450
 panels=[cv2.resize(p,(target_w,target_h)) for p in panels]
 canvas=np.vstack([np.hstack(panels[:4]),np.hstack(panels[4:])])
 cv2.imwrite(str(VIS/f"{Path(row['original_filename']).stem}_threshold_grid.jpg"),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
 cv2.imwrite(str(VIS/f"{Path(row['original_filename']).stem}_baseline_conf005_iou070.jpg"),draw_panel(image,'Batch-list baseline conf=0.05 IoU=0.70',gt,pred,names),[cv2.IMWRITE_JPEG_QUALITY,94])
 nms_panels=[]
 for setting,conf,iou_nms in [s for s in settings if s[0] in ('conf050_iou050','conf050_iou070','conf050_iou090')]:
  preds=results_to_preds(all_results[setting][idx],h,w)
  nms_panels.append(draw_panel(image,f'{setting}: {len(preds)} det',gt,preds,names))
 nms_panels=[cv2.resize(p,(target_w,target_h)) for p in nms_panels]
 cv2.imwrite(str(VIS/f"{Path(row['original_filename']).stem}_nms_grid.jpg"),np.hstack(nms_panels),[cv2.IMWRITE_JPEG_QUALITY,92])

# Compare batch/list preprocessing with single-image inference at the exact original settings.
single_model=YOLO(str(CKPT)); single_counts=[]
for row in rows:
 r=single_model.predict(source=str(BATCH/row['image_path']),imgsz=640,device='cpu',conf=.05,iou=.7,max_det=300,retina_masks=False,verbose=False)[0]
 single_counts.append({'asset_id':row['asset_id'],'original_filename':row['original_filename'],'list_batch_detections':len(base_results[row['asset_id']][2]),'single_image_detections':len(r.boxes) if r.boxes is not None else 0})
with (OUT/'batch_vs_single_preprocessing.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(single_counts[0]));wr.writeheader();wr.writerows(single_counts)

for name,data in [('threshold_nms_summary.csv',summary),('per_image_counts.csv',per_image),('prediction_iou_matches.csv',detail)]:
 with (OUT/name).open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(data[0]));wr.writeheader();wr.writerows(data)
meta={'checkpoint':str(CKPT),'checkpoint_sha256':sha(CKPT),'dataset_yaml':str(BATCH/'green_apple.yaml'),'source':'candidate batch image copies; same 5 smoke training images','class_names':names,'task':model.task,'imgsz':640,'device':'cpu','max_det':300,'inference_mode':'all five paths passed together as a source list; retina_masks=False for baseline reproduction','mask_matching':'greedy one-to-one class-aware mask IoU >= 0.50','false_positive_subtypes':'duplicate if best same-class GT mask IoU >=.50 after GT already matched; background if max IoU <.05; otherwise unmatched FP','settings':settings,'output':str(OUT)}
(OUT/'forensic_metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps({'out':str(OUT),'summary':summary},indent=2))
