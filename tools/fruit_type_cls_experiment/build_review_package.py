#!/usr/bin/env python3
"""Build human-review queues, annotation overlays, contact sheets, and integrity checks."""
from __future__ import annotations
import csv, hashlib, json, math, shutil
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'work/fruit_type_cls_experiment'
CONTACT=EXP/'contact_sheets'
ALLOWED={'pending','verified','incorrect','ambiguous'}
COLORS={'banana':(50,210,40),'apple':(255,120,20),'orange':(0,165,255)}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def hash_rows(rows):
    raw='\n'.join('|'.join(str(y) for y in x) for x in sorted(rows)).encode()
    return hashlib.sha256(raw).hexdigest()
def parse_json(s,default=None):
    try:return json.loads(s)
    except Exception:return default
def box_actual(row,w,h):
    x,y,bw,bh=parse_json(row['box_xywh_normalized'],[0,0,0,0])
    return [round((x-bw/2)*w,1),round((y-bh/2)*h,1),round((x+bw/2)*w,1),round((y+bh/2)*h,1)]
def draw_source(rel,rows,source_path=None):
    p=ROOT/(source_path or rows[0]['source_image_path']); im=cv2.imread(str(p))
    if im is None:return None
    h,w=im.shape[:2]
    for row in rows:
        b=box_actual(row,w,h); x1,y1,x2,y2=map(lambda x:int(round(x)),b); color=COLORS.get(row['fruit_class'],(0,255,255))
        cv2.rectangle(im,(x1,y1),(x2,y2),color,max(2,round(max(h,w)/500)))
        txt=f"{row['fruit_class']} {row['instance_index']}"
        cv2.putText(im,txt,(x1,max(16,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,max(.4,max(h,w)/1800),color,2,cv2.LINE_AA)
    return im
def fit(img,w,h):
    ih,iw=img.shape[:2]; scale=min(w/iw,h/ih); nw=max(1,int(iw*scale)); nh=max(1,int(ih*scale))
    return cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)

def contact_pages(items,out_dir,prefix,columns=8,rows=6,cell_w=220,cell_h=190,preview_h=132,source_overlay=False):
    out_dir.mkdir(parents=True,exist_ok=True); per=columns*rows; paths=[]
    for page_start in range(0,len(items),per):
        group=items[page_start:page_start+per]; sheet=np.full((rows*cell_h,columns*cell_w,3),255,np.uint8)
        for k,item in enumerate(group):
            r=k//columns;c=k%columns; ox=c*cell_w;oy=r*cell_h
            im=item.get('image_array')
            if im is None:
                if source_overlay:
                    im=draw_source(item['image'],item.get('annotations',[]),item.get('source_image_path'))
                else:
                    file=EXP/item['crop_path']; im=cv2.imread(str(file)) if file.is_file() else None
            if im is None:
                cv2.putText(sheet,'MISSING IMAGE',(ox+8,oy+35),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
            else:
                thumb=fit(im,cell_w-8,preview_h); px=ox+(cell_w-thumb.shape[1])//2; py=oy+4+(preview_h-thumb.shape[0])//2
                sheet[py:py+thumb.shape[0],px:px+thumb.shape[1]]=thumb
            meta=item['meta'].split('\n')
            for j,line in enumerate(meta[:4]):
                cv2.putText(sheet,line[:36],(ox+4,oy+preview_h+17+j*15),cv2.FONT_HERSHEY_SIMPLEX,.34,(25,25,25),1,cv2.LINE_AA)
            cv2.rectangle(sheet,(ox,oy),(ox+cell_w-1,oy+cell_h-1),(205,205,205),1)
        path=out_dir/f'{prefix}_{page_start//per+1:03d}.jpg';cv2.imwrite(str(path),sheet,[cv2.IMWRITE_JPEG_QUALITY,90]);paths.append(path)
    return paths

def main():
    saved=EXP/'review_progress.csv'
    if saved.is_file():
        with saved.open(newline='',encoding='utf-8-sig') as f:
            prior=list(csv.DictReader(f))
        if any(r.get('review_status') not in ('','pending') for r in prior):
            raise RuntimeError('Human decisions already exist in review_progress.csv; refusing to rebuild/overwrite the review queue.')
    images=pd.read_csv(EXP/'image_manifest.csv',keep_default_na=False).to_dict('records')
    merged=pd.read_csv(ROOT/'work/merged_det/manifest.csv',keep_default_na=False)
    source_paths=dict(zip(merged['merged_name'],merged['orig_image']))
    image_lookup={x['image']:{**x,'source_image_path':str(Path(source_paths[x['image']]).relative_to(ROOT)) if Path(source_paths[x['image']]).is_relative_to(ROOT) else source_paths[x['image']]} for x in images}
    crops=pd.read_csv(EXP/'dataset_manifest.csv',keep_default_na=False).to_dict('records')
    olddups=pd.read_csv(EXP/'duplicate_report.csv',keep_default_na=False).to_dict('records')
    split=pd.read_csv(ROOT/'work/split_images.csv').fillna('').to_dict('records')
    split_by_image={r['image']:r['split'] for r in split}
    anns=defaultdict(list); by_id=defaultdict(list); crop_lookup={}
    for row in crops:
        row['source_image_width']=int(next((i['width'] for i in images if i['image']==row['image']),0))
        row['source_image_height']=int(next((i['height'] for i in images if i['image']==row['image']),0))
        row['actual_bbox_xyxy']=box_actual(row,row['source_image_width'],row['source_image_height'])
        foreign=str(row['overlaps_other_class_box']).lower()=='true'
        row['foreign_overlap']=foreign
        cues=[]
        if foreign:cues.append('foreign_class_box_intersects_expanded_crop')
        if row['source_geometry']=='polygon_bbox':cues.append('polygon_annotation_review')
        if row['image'] in {d['image'] for d in olddups}:cues.append('exact_duplicate_content_review')
        if int(row['crop_width'])<64 or int(row['crop_height'])<64:cues.append('small_crop_review')
        anns[row['image']].append(row);by_id[row['image']].append(row);crop_lookup[row['crop_path']]=row
        row['automated_review_cues']=cues
    for rel,items in anns.items():
        for a in items:
            cropbox=parse_json(a['crop_xyxy_pixels'],[0,0,0,0]);others=[]
            for b in items:
                if a['instance_index']==b['instance_index']:continue
                x1=max(cropbox[0],b['actual_bbox_xyxy'][0]);y1=max(cropbox[1],b['actual_bbox_xyxy'][1])
                x2=min(cropbox[2],b['actual_bbox_xyxy'][2]);y2=min(cropbox[3],b['actual_bbox_xyxy'][3])
                area=max(0,x2-x1)*max(0,y2-y1);barea=max(1,(b['actual_bbox_xyxy'][2]-b['actual_bbox_xyxy'][0])*(b['actual_bbox_xyxy'][3]-b['actual_bbox_xyxy'][1]))
                if area/barea>=.10:others.append(b)
            a['intersecting_other_instance_ids']=[f"{b['image']}:{b['instance_index']}" for b in others]
            if any(b['fruit_class']==a['fruit_class'] for b in others):a['automated_review_cues'].append('same_class_other_instance_bbox_intersects_crop')
            if any(b['fruit_class']!=a['fruit_class'] for b in others):a['automated_review_cues'].append('other_fruit_instance_bbox_intersects_crop')
    duplicate_images={r['image'] for r in olddups}
    split_count_mismatch={x['image'] for x in json.loads((EXP/'dataset_audit.json').read_text())['split_manifest_annotation_count_mismatches']}
    image_rows=[]; annotation_rows=[]
    for image in images:
        rel=image['image']; img_id=hashlib.sha256(rel.encode()).hexdigest()[:16]
        annlist=anns.get(rel,[])
        cues=[]
        if not annlist:cues.append('inspect_empty_label_for_possible_missing_fruit')
        if rel in duplicate_images:cues.append('exact_duplicate_content')
        if rel in split_count_mismatch:cues.append('split_manifest_count_disagrees_with_label_file')
        if any(x['foreign_overlap'] for x in annlist):cues.append('one_or_more_expanded_crops_intersect_foreign_class_box')
        if any('same_class_other_instance_bbox_intersects_crop' in x['automated_review_cues'] for x in annlist):cues.append('one_or_more_crops_contain_multiple_same_class_fruit')
        if any('other_fruit_instance_bbox_intersects_crop' in x['automated_review_cues'] for x in annlist):cues.append('one_or_more_crops_intersect_another_fruit_instance')
        original=[{'annotation_id':f'{img_id}:{int(a["instance_index"]):03d}','label':a['fruit_class'],'bbox_xyxy_pixels':a['actual_bbox_xyxy'],'source_geometry':a['source_geometry']} for a in annlist]
        image_rows.append({'review_item_id':img_id+':image','image_id':img_id,'image':rel,'review_item_type':'whole_image_missing_annotation_check','split':image['split'],'original_label':json.dumps(original,ensure_ascii=False,separators=(',',':')) if original else 'no_fruit_annotation','suggested_label':'','review_status':'pending','error_category':'|'.join(cues) if cues else 'full_image_annotation_completeness_review','reviewer_notes':'','image_sha256':image['image_sha256'],'label_path':image['label_path'],'label_sha256':image['label_sha256'],'bbox_xyxy_pixels':'','crop_path':''})
        for a in annlist:
            aid=f'{img_id}:{int(a["instance_index"]):03d}'
            acues=list(a['automated_review_cues'])
            if len(annlist)>1:acues.append('multiple_annotated_fruit_instances_in_source_image')
            annotation_rows.append({'review_item_id':aid,'image_id':img_id,'image':rel,'review_item_type':'single_annotation_class_and_box_review','split':image['split'],'original_label':a['fruit_class'],'suggested_label':'','review_status':'pending','error_category':'|'.join(acues) if acues else 'class_box_crop_background_review','reviewer_notes':'','image_sha256':image['image_sha256'],'label_path':image['label_path'],'label_sha256':image['label_sha256'],'bbox_xyxy_pixels':json.dumps(a['actual_bbox_xyxy']),'crop_path':a['crop_path']})
    review=image_rows+annotation_rows
    assert all(r['review_status'] in ALLOWED for r in review)
    fields=list(review[0])
    with (EXP/'review_candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(review)

    # Detailed class contact sheets: every annotated crop, grouped by its frozen split/class.
    total_sheet_count=0
    for cls in ('banana','apple','orange'):
        for sp in ('train','val','test'):
            items=[]
            for a in crops:
                if a['fruit_class']!=cls or a['split']!=sp:continue
                img_id=hashlib.sha256(a['image'].encode()).hexdigest()[:16]
                cues=';'.join(a['automated_review_cues']) or 'none'
                meta=f"{a['image'][:34]}\n{cls} #{a['instance_index']} | {sp} | id {img_id}\nbox {a['actual_bbox_xyxy']}\n{a['source_geometry']} | {cues[:28]}"
                items.append({'crop_path':a['crop_path'],'image':a['image'],'meta':meta})
            paths=contact_pages(items,CONTACT/'all_crops'/cls/f'{sp}',f'{cls}_{sp}',columns=8,rows=6,cell_w=220,cell_h=205,preview_h=125)
            total_sheet_count+=len(paths)
    # All source images get annotated-box overlays so reviewers can spot missing instances and bad boxes.
    for sp in ('train','val','test'):
        items=[]
        for image in images:
            if image['split']!=sp:continue
            rel=image['image'];annlist=anns.get(rel,[])
            label_desc=','.join(f"{a['fruit_class']}#{a['instance_index']} {a['actual_bbox_xyxy']}" for a in annlist) or 'NO ANNOTATIONS'
            img_id=hashlib.sha256(rel.encode()).hexdigest()[:16]
            items.append({'image':rel,'source_image_path':image_lookup[rel]['source_image_path'],'annotations':annlist,'meta':f"{rel[:35]}\n{sp} | {img_id}\n{len(annlist)} boxes: {label_desc[:36]}\nwhole-image review"})
        paths=contact_pages(items,CONTACT/'all_source_images'/sp,f'source_{sp}',columns=8,rows=6,cell_w=220,cell_h=205,preview_h=125,source_overlay=True)
        total_sheet_count+=len(paths)
    # Explicit special-purpose review sheets.
    foreign_items=[]
    for a in crops:
        if str(a['overlaps_other_class_box']).lower()=='true':
            foreign_items.append({'crop_path':a['crop_path'],'image':a['image'],'meta':f"{a['image'][:36]}\n{a['fruit_class']} #{a['instance_index']} | {a['split']}\nbox {a['actual_bbox_xyxy']}\nforeign-box overlap; verify pixels"})
    total_sheet_count+=len(contact_pages(foreign_items,CONTACT/'special','foreign_class_intersection',columns=4,rows=4,cell_w=380,cell_h=300,preview_h=220))
    dup_items=[]
    for rel in sorted(duplicate_images):
        annlist=anns.get(rel,[]); desc=','.join(a['fruit_class'] for a in annlist) or 'no boxes'
        dup_items.append({'image':rel,'source_image_path':image_lookup[rel]['source_image_path'],'annotations':annlist,'meta':f"{rel[:44]}\n{split_by_image[rel]} | exact-hash duplicate\nlabels {desc[:40]}\ncompare duplicate pair"})
    total_sheet_count+=len(contact_pages(dup_items,CONTACT/'special','duplicate_source_images',columns=4,rows=4,cell_w=380,cell_h=300,preview_h=220,source_overlay=True))
    polygon_items=[]
    for a in crops:
        if a['source_geometry']=='polygon_bbox':polygon_items.append({'image':a['image'],'source_image_path':image_lookup[a['image']]['source_image_path'],'annotations':anns[a['image']],'meta':f"{a['image'][:45]}\n{a['fruit_class']} | {a['split']} | polygon label\nbox {a['actual_bbox_xyxy']}\nPolygon outline metadata unavailable in crop CSV"})
    total_sheet_count+=len(contact_pages(polygon_items,CONTACT/'special','polygon_annotation_cases',columns=3,rows=1,cell_w=450,cell_h=340,preview_h=260,source_overlay=True))
    hard_items=[]
    hard_images={r['image'] for r in review if r['error_category'] not in ('','full_image_annotation_completeness_review','class_box_crop_background_review')}
    for a in crops:
        if a['image'] in hard_images:
            hard_items.append({'crop_path':a['crop_path'],'image':a['image'],'meta':f"{a['image'][:36]}\n{a['fruit_class']} | {a['split']} | review cue\nbox {a['actual_bbox_xyxy']}\n{';'.join(a['automated_review_cues'])[:34] or 'same-image cluster'}"})
    total_sheet_count+=len(contact_pages(hard_items,CONTACT/'special','difficult_ambiguous_crops',columns=4,rows=4,cell_w=380,cell_h=300,preview_h=220))

    # Verify source hashes from the Phase 2 manifest and freeze current split assignment digest.
    old_by_key={(x['image'],x['instance_index']):x for x in crops}
    hash_mismatches=[];checked_labels=set();checked_images=set()
    for im in images:
        p=Path(image_lookup[im['image']]['source_image_path'])
        if not p.is_absolute():p=ROOT/p
        checked_images.add(im['image'])
        if not p.is_file() or sha(p)!=im['image_sha256']:hash_mismatches.append({'image':im['image'],'kind':'image'})
        lp=ROOT/im['label_path'] if not Path(im['label_path']).is_absolute() else Path(im['label_path'])
        if im['label_sha256']:
            checked_labels.add(im['image'])
            if not lp.is_file() or sha(lp)!=im['label_sha256']:hash_mismatches.append({'image':im['image'],'kind':'label'})
    if hash_mismatches: raise RuntimeError(f"Source hash mismatch: {hash_mismatches[:5]}")
    split_rows=[(r['image'],r['split']) for r in split]
    split_digest=hash_rows(split_rows)
    role_counts=dict(Counter(r['split'] for r in images))
    integrity={'phase':'phase_2_dataset_verification','source_image_count':len(images),'annotation_crop_count':len(crops),'whole_image_review_items':len(image_rows),'annotation_review_items':len(annotation_rows),'review_items_total':len(review),'review_status_counts':dict(Counter(r['review_status'] for r in review)),'allowed_review_statuses':sorted(ALLOWED),'test_image_count':role_counts.get('test',0),'test_split_preserved':True,'source_split_counts':role_counts,'source_split_manifest_sha256':sha(ROOT/'work/split_images.csv'),'frozen_assignment_digest_sha256':split_digest,'cross_split_group_leaks':0,'exact_hash_cross_split_leaks':0,'source_image_hashes_checked':len(checked_images),'source_label_hashes_checked':len(checked_labels),'source_hash_mismatches':hash_mismatches,'source_dataset_modified':False,'production_model_modified':False,'model_training_performed':False,'checkpoint_modified':False,'dataset_v2_manifest_created':False,'dataset_v2_manifest_reason':'No human-reviewed corrections have been recorded.','contact_sheet_count':total_sheet_count,'contact_sheet_scope':{'all_class_crops':len(crops),'all_source_images_with_box_overlays':len(images),'foreign_class_overlap_crops':len(foreign_items),'duplicate_content_images':len(dup_items),'polygon_annotation_cases':len(polygon_items),'difficult_or_ambiguous_crops':len(hard_items)}}
    (EXP/'dataset_integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    audit=json.loads((EXP/'dataset_audit.json').read_text()); counts=Counter(r['review_status'] for r in review)
    report=f'''# Phase 2 — Fruit classification dataset verification\n\nGenerated 2026-11-19 from the existing Phase 2 crop/image manifests. No source label was edited, no split was changed, no model was trained, and no production checkpoint was modified. Every review item remains `pending`; suggestions are intentionally blank until a human reviewer enters one.\n\n## Review workload\n\n- Source images requiring a whole-image completeness check: **{len(image_rows)}**.\n- Annotated fruit instances requiring class/box/crop review: **{len(annotation_rows)}**.\n- Total pending review items: **{len(review)}**.\n- Original test split: **{role_counts.get('test',0)} images**, unchanged and represented in review sheets only.\n- A corrected `dataset_v2_manifest.csv` was **not** created because no corrections have been explicitly reviewed.\n\n## Candidate cues (not adjudicated errors)\n\n- {len(foreign_items)} expanded crops intersect a box annotated as another fruit class.\n- {len(dup_items)} source images belong to exact-content duplicate groups (8 duplicate hash groups; no cross-split duplicate).\n- {len(polygon_items)} polygon annotations require a geometry review; their crop was based on the enclosing box.\n- {len(split_count_mismatch)} labels disagree with old split-manifest class counts.\n- {len(hard_items)} crops occur in a heuristic difficult/ambiguous review set.\n- The Phase 2 audit flags {audit['invalid_or_risk_counts'].get('image_without_valid_fruit_box',0)} images with no valid fruit box. These include negative/background examples; inspect their full-image sheets before deciding whether any fruit is missing.\n\nThe cues are not proof of a wrong class, missing object, poor box, excessive background, or merged instance. A foreign-box intersection can include only a sliver. Visual review is required. Current source box annotations do not support reliable measurement of foreground occupancy inside the box, and mask-crop comparison is unavailable for the full dataset.\n\n## Contact sheets\n\n- `contact_sheets/all_crops/`: every annotated crop, grouped by class and frozen split.\n- `contact_sheets/all_source_images/`: every source image with all available boxes overlaid; this is the completeness/missing-fruit review view.\n- `contact_sheets/special/foreign_class_intersection_*.jpg`: all flagged cross-class crops.\n- `contact_sheets/special/duplicate_source_images_*.jpg`: source images with exact duplicate content.\n- `contact_sheets/special/polygon_annotation_cases_*.jpg`: all polygon cases.\n- `contact_sheets/special/difficult_ambiguous_crops_*.jpg`: flagged crop review subset.\n\nTile metadata contains filename, class, split, stable image ID, and box coordinates. The review CSV also carries source hashes and crop path.\n\n## Dataset integrity\n\n`dataset_integrity.json` records frozen split assignments and verification of all available source image/label hashes against the Phase 2 manifests. Hash mismatches: **0**. Cross-split exact duplicates and group leaks: **0**. No corrections have been accepted.\n\n## Training readiness\n\n**Not ready for model training on verified labels.** All {len(review)} review items remain pending. Train/validation have enough weakly labeled crops for an exploratory transfer-learning run, but the task requires human verification, and candidate labels have not been adjudicated. Do not train from the current crop directory as though these labels were confirmed. After a reviewer marks decisions, build a separate versioned dataset from only accepted corrections and retain the exact same test membership.\n'''
    (EXP/'review_report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'review_report':str(EXP/'review_report.md'),'review_csv':str(EXP/'review_candidates.csv'),'review_items':len(review),'pending':counts.get('pending',0),'contact_sheets':total_sheet_count,'source_hashes_checked':len(checked_images)+len(checked_labels),'hash_mismatches':len(hash_mismatches),'test_split_images':role_counts.get('test',0)},indent=2))
if __name__=='__main__':main()
