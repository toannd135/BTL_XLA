#!/usr/bin/env python3
"""Audit fruit-type box labels and prepare immutable-per-run crops without touching sources.

Source split assignments come from work/split_images.csv and are never recomputed.
Only newly written crops/manifests under work/fruit_type_cls_experiment are created.
"""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/fruit_type_cls_experiment"
FRUITS = {0: "banana", 1: "apple", 2: "orange"}
PAD = 0.10
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def read_labels(path: Path, own: bool):
    rows=[]; errors=[]
    if not path.is_file(): return rows,["missing_label_file"]
    for n,line in enumerate(path.read_text(errors="replace").splitlines(),1):
        t=line.split()
        if not t: continue
        try:
            cls_value=float(t[0]); cls=int(cls_value); v=np.asarray([float(x) for x in t[1:]],dtype=float)
        except Exception:
            errors.append(f"line_{n}_not_numeric"); continue
        valid_shape=(len(v)==4) if own else (len(v)==4 or (len(v)>=6 and len(v)%2==0))
        if cls_value!=cls or not valid_shape or not np.isfinite(v).all() or not 0<=cls<(6 if own else 3) or not np.all((v>=0)&(v<=1)):
            errors.append(f"line_{n}_invalid_yolo_box"); continue
        fruit_id=cls//2 if own else cls
        if len(v)==4:
            box=v; geom='box'
        else:
            points=v.reshape(-1,2); lo=points.min(axis=0); hi=points.max(axis=0)
            box=np.asarray([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,hi[0]-lo[0],hi[1]-lo[1]])
            geom='polygon_bbox'
        if box[2]<=0 or box[3]<=0:
            errors.append(f"line_{n}_degenerate_box"); continue
        rows.append((FRUITS[fruit_id],box,geom))
    return rows,errors

def padded_box(box,w,h):
    x,y,bw,bh=box
    x1=int(max(0,min(w-1,(x-bw/2)*w-bw*w*PAD)))
    y1=int(max(0,min(h-1,(y-bh/2)*h-bh*h*PAD)))
    x2=int(min(w,max(x1+1,(x+bw/2)*w+bw*w*PAD)))
    y2=int(min(h,max(y1+1,(y+bh/2)*h+bh*h*PAD)))
    return x1,y1,x2,y2

def main():
    splits=pd.read_csv(ROOT/'work/split_images.csv').fillna("")
    merged=pd.read_csv(ROOT/'work/merged_det/manifest.csv').fillna("")
    table=splits.merge(merged,left_on='image',right_on='merged_name',suffixes=('_split','_source'),validate='one_to_one')
    split_group=table.groupby('group')['split'].nunique()
    cross_groups=split_group[split_group>1].index.tolist()
    if cross_groups: raise RuntimeError(f"Image groups leak across splits: {cross_groups[:10]}")
    rows=[]; issues=Counter(); per_image=[]; exact_hash_splits=defaultdict(set); exact_hash_counts=Counter(); class_counts=Counter(); crop_contamination=Counter(); split_mismatches=[]
    crop_root=OUT/'crops'
    for r in table.sort_values('image').itertuples(index=False):
        split=r.split; own=(r.origin_source=='own')
        source=Path(r.orig_image)
        if not source.is_file():
            issues['missing_source_image']+=1; per_image.append({'image':r.image,'split':split,'issue':'missing_source_image'}); continue
        label=(ROOT/'own/labels'/f"{source.stem}.txt") if own else (ROOT/'local_dataset/det'/split/'labels'/f"{Path(r.image).stem}.txt")
        image_hash=sha(source); exact_hash_splits[image_hash].add(split); exact_hash_counts[image_hash]+=1
        img=cv2.imread(str(source),cv2.IMREAD_COLOR)
        if img is None:
            issues['corrupt_or_undecodable_image']+=1; per_image.append({'image':r.image,'split':split,'issue':'corrupt_or_undecodable_image'}); continue
        h,w=img.shape[:2]; objs,bad=read_labels(label,own)
        for b in bad: issues[b]+=1
        if not objs: issues['image_without_valid_fruit_box']+=1
        actual=Counter(x[0] for x in objs)
        expected={name:int(getattr(r,name)) for name in FRUITS.values()}
        if any(actual[name]!=expected[name] for name in FRUITS.values()):
            delta={'image':r.image,'split':split,'counts_in_split_manifest':expected,'counts_in_label_file':{name:actual[name] for name in FRUITS.values()}}
            split_mismatches.append(delta); issues['split_manifest_annotation_count_mismatch']+=1
        image_rec={'image':r.image,'split':split,'source':r.source_source,'origin':r.origin_source,'group':r.group,'image_sha256':image_hash,'label_path':str(label.relative_to(ROOT)) if label.is_relative_to(ROOT) else str(label),'label_sha256':sha(label) if label.is_file() else None,'width':w,'height':h,'valid_boxes':len(objs),'invalid_label_reasons':bad}
        per_image.append(image_rec)
        for j,(fruit,box,geom) in enumerate(objs):
            x1,y1,x2,y2=padded_box(box,w,h); crop=img[y1:y2,x1:x2]
            if crop.size==0:
                issues['empty_crop']+=1; continue
            # Other annotated objects in the expanded box are a measurable crop-noise cue.
            foreign=[]
            for k,(other,ob,_) in enumerate(objs):
                if k==j or other==fruit: continue
                ox1,oy1,ox2,oy2=padded_box(ob,w,h)
                inter=max(0,min(x2,ox2)-max(x1,ox1))*max(0,min(y2,oy2)-max(y1,oy1))
                if inter: foreign.append({'fruit':other,'crop_area_fraction':round(inter/((x2-x1)*(y2-y1)),5)})
            if foreign: crop_contamination[split]+=1
            stem=Path(r.image).stem
            dst=crop_root/split/fruit/f"{stem}__{j:03d}.jpg"
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(dst),crop,[cv2.IMWRITE_JPEG_QUALITY,95]): raise RuntimeError(f"failed to write {dst}")
            class_counts[(split,fruit)]+=1
            rows.append({'image':r.image,'source_image_path':str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),'source_image_sha256':image_hash,'label_path':image_rec['label_path'],'label_sha256':image_rec['label_sha256'],'group':r.group,'split':split,'origin':r.origin_source,'source':r.source_source,'instance_index':j,'fruit_class':fruit,'source_geometry':geom,'box_xywh_normalized':json.dumps(box.tolist()),'crop_xyxy_pixels':json.dumps([x1,y1,x2,y2]),'crop_path':str(dst.relative_to(OUT)),'crop_sha256':sha(dst),'crop_width':crop.shape[1],'crop_height':crop.shape[0],'overlaps_other_class_box':bool(foreign),'overlapping_other_classes':json.dumps(foreign,separators=(',',':')),'annotation_source':'YOLO box or polygon; own images use fruit component of six-state YOLO class'})
    leaks={h:sorted(v) for h,v in exact_hash_splits.items() if len(v)>1}
    repeated_hashes={h:n for h,n in exact_hash_counts.items() if n>1}
    # The split manifest itself is used unchanged; any exact byte duplicate spanning roles is a hard stop.
    if leaks: issues['exact_image_hash_cross_split']+=len(leaks)
    OUT.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else []
    with (OUT/'dataset_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    image_fields=sorted({k for x in per_image for k in x})
    with (OUT/'image_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=image_fields); wr.writeheader(); wr.writerows(per_image)
    by_hash=defaultdict(list)
    for image in per_image:
        if image.get('image_sha256'): by_hash[image['image_sha256']].append(image)
    duplicate_rows=[]
    for hash_value,items in by_hash.items():
        if len(items)>1:
            for item in items: duplicate_rows.append({'image_sha256':hash_value,'duplicate_count':len(items),'image':item['image'],'split':item['split'],'group':item['group']})
    with (OUT/'duplicate_report.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=['image_sha256','duplicate_count','image','split','group']); wr.writeheader(); wr.writerows(sorted(duplicate_rows,key=lambda x:(x['image_sha256'],x['image'])))
    mismatch_images={x['image'] for x in split_mismatches}
    dup_images={x['image'] for x in duplicate_rows}
    crop_risk_images={x['image'] for x in rows if x['overlaps_other_class_box']}
    review_images=mismatch_images|dup_images|crop_risk_images
    image_by_name={x['image']:x for x in per_image}
    with (OUT/'label_review_candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['image','split','source','group','image_sha256','label_path','label_sha256','review_reasons']
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader()
        for name in sorted(review_images):
            item=image_by_name[name]; reasons=[]
            if name in mismatch_images: reasons.append('split_manifest_vs_label_count_mismatch')
            if name in dup_images: reasons.append('exact_duplicate_image_content')
            if name in crop_risk_images: reasons.append('expanded_crop_intersects_other_class_box')
            wr.writerow({**{k:item.get(k,'') for k in fields if k!='review_reasons'},'review_reasons':';'.join(reasons)})
    geometry_counts=Counter((r['split'],r['source_geometry']) for r in rows)
    result={'experiment':'dedicated_fruit_type_classifier','generated_utc_date':'2026-11-19','source_split_manifest':'work/split_images.csv','source_merged_manifest':'work/merged_det/manifest.csv','source_images':len(per_image),'instance_crops':len(rows),'crop_pad_each_side':PAD,'source_split_assignments_preserved':True,'source_test_used_for_training':False,'class_crop_counts':{f'{s}/{c}':n for (s,c),n in sorted(class_counts.items())},'split_image_counts':dict(Counter(str(x.get('split')) for x in per_image if x.get('image_sha256'))),'source_geometry_counts':{f'{s}/{g}':n for (s,g),n in sorted(geometry_counts.items())},'invalid_or_risk_counts':dict(issues),'split_manifest_annotation_count_mismatches':split_mismatches,'exact_duplicate_image_hash_count':len(repeated_hashes),'exact_duplicate_image_instances_beyond_first':sum(n-1 for n in repeated_hashes.values()),'exact_hash_cross_split_count':len(leaks),'overlapping_other_class_crop_count_by_split':dict(crop_contamination),'label_review_candidate_images':len(review_images),'classes':list(FRUITS.values()),'labels':'Fruit classes mapped from original YOLO object labels; folder names alone are not used as evidence. Own source class IDs are reduced to fruit type from banana/apple/orange freshness class pairs.','limitations':['No human adjudication of all source fruit boxes has been found; inherited labels must be treated as weak labels.','The YOLO label files use box and polygon formats. Polygon-only rows are converted to their enclosing boxes for this first crop set; segmentation-mask crops are not compared.','Cross-class objects inside expanded crops are flagged; same-class neighboring fruit and background context remain.','Existing test split is preserved and not included in train/val crop preparation.']}
    (OUT/'dataset_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
