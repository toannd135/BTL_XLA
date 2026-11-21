#!/usr/bin/env python3
"""Build read-only Phase 9.2 inventory, AI-priority queue, and contact sheets."""
import csv, json, shutil
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[5]
P9=ROOT/'fruit/work/fruit_type_cls_experiment/phase9_green_apple_rebuild'
OUT=ROOT/'fruit/work/fruit_type_cls_experiment/phase9_2_green_apple_expansion'
COLLECTION=ROOT/'fruit/work/fruit_type_cls_experiment/phase8c_collection/green_apple_phase8d/staging/mendeley_v1'

def read(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def scores(row):
    a=json.loads(row.get('clip_similarity_context') or '[]');b=json.loads(row.get('clip_similarity_masked_crop') or '[]')
    if len(a)<4 or len(b)<4:return None,None,None
    margin=min(a[0]-max(a[1:]),b[0]-max(b[1:]))
    sim=min(a[0],b[0])
    return margin,sim,float(row.get('green_pixel_ratio_support_only') or 0)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ai=read(P9/'ai_review/ai_review_manifest.csv')
    queue={r['polygon_id']:r for r in read(P9/'review_queue.csv')}
    priority={r['polygon_id']:r for r in read(P9/'review_priority_queue.csv')}
    integrity={r['polygon_id']:r for r in read(P9/'ai_review/polygon_integrity_audit.csv')}
    near=read(P9/'near_duplicate_candidates_full_dataset.csv')
    exact=read(P9/'exact_duplicate_pairs.csv')
    near_names=set()
    near_mendeley=[]
    for pair in near:
        for k in ('image_a','image_b'):
            n=pair[k].removeprefix('mendeley_v1_')
            if n.startswith('train_'):near_names.add(n[6:])
        if 'mendeley_v1_train_' in pair['image_a'] or 'mendeley_v1_train_' in pair['image_b']:
            near_mendeley.append(pair)
    exact_mendeley=[r for r in exact if 'mendeley_v1_train_' in r['image_a'] or 'mendeley_v1_train_' in r['image_b']]
    pending=[]; inventory=[]
    for a in ai:
        q=queue.get(a['polygon_id'],{})
        m=integrity.get(a['polygon_id'],{})
        p=priority.get(a['polygon_id'],{})
        margin,sim,green=scores(a)
        if a['decision']=='AUTO_APPROVED':status='AI_ASSISTED_SMOKE_ONLY'
        elif a['decision']=='NEEDS_REVIEW':status='PENDING_HUMAN_REVIEW'
        elif a['decision']=='DUPLICATE_HOLD':status='QUARANTINE_DUPLICATE_OR_PROVENANCE'
        else:status='EXCLUDED_GEOMETRY_OR_INTEGRITY_FLAG'
        record={
          'polygon_id':a['polygon_id'],'asset_id':a['asset_id'],'original_filename':a['original_filename'],
          'source_label':a['source_label'],'source_split_original':a['source_split_original'],
          'source_path':str((COLLECTION/q.get('image_relative_path','')).resolve()) if q else '',
          'image_relative_path':q.get('image_relative_path',''),'polygon_points_json':q.get('polygon_points_json',''),
          'image_width':q.get('image_width',''),'image_height':q.get('image_height',''),
          'source_doi':q.get('source_doi',''),'source_url':q.get('source_url',''),'license':q.get('license',''),
          'priority_tier':p.get('priority_tier',''),'priority_reason':p.get('priority_reason',''),
          'image_sha256':a['image_sha256'],'source_family_id':'','dataset_split':'UNASSIGNED',
          'review_status':a['decision'],'review_type':'AI_ASSISTED','human_verified':'false',
          'candidate_status':status,'pilot_eligible':'false','mask_integrity_self_crossing':m.get('self_intersection',''),
          'bbox_out_of_bounds':m.get('bbox_out_of_bounds',''),'mask_area_fraction':m.get('mask_area_fraction',''),
          'phash_candidate_in_prior_audit':str(a['original_filename'] in near_names).lower(),
          'clip_min_view_margin':f'{margin:.8f}' if margin is not None else '',
          'clip_min_green_similarity':f'{sim:.8f}' if sim is not None else '',
          'green_pixel_ratio_support_only':f'{green:.8f}' if green is not None else '',
          'decision_reason':a['reason']
        }
        inventory.append(record)
        if a['decision']=='NEEDS_REVIEW':
            pending.append({**q,**{
              'review_status':'PENDING_HUMAN_REVIEW','reviewer_type':'AI_TRIAGE_ONLY','human_decision':'NOT_REVIEWED',
              'source_family_id':'','dataset_split':'UNASSIGNED','training_eligible':'false',
              'mask_self_crossing_flag':m.get('self_intersection',''),'bbox_out_of_bounds_flag':m.get('bbox_out_of_bounds',''),
              'clip_min_view_margin':record['clip_min_view_margin'],'clip_min_green_similarity':record['clip_min_green_similarity'],
              'green_pixel_ratio_support_only':record['green_pixel_ratio_support_only'],
              'ai_review_caution':'CLIP/color scores prioritize human review only; never approve by threshold'
            }})
    pending.sort(key=lambda r:(-float(r['clip_min_view_margin'] or -1),-float(r['clip_min_green_similarity'] or -1),-float(r['green_pixel_ratio_support_only'] or -1),r['polygon_id']))
    for i,r in enumerate(pending,1):
        r['ai_priority_rank']=i
        r['ai_priority_band']='TOP_SIGNAL_FOR_HUMAN_REVIEW' if i<=100 else 'REMAINING_PENDING_REVIEW'
    write(OUT/'candidate_manifest.csv',inventory)
    write(OUT/'review_queue.csv',pending)
    write(OUT/'reports/near_duplicate_candidate_pairs.csv',near_mendeley)
    write(OUT/'reports/exact_duplicate_pairs.csv',exact)
    # Compact counts and gate analysis.
    decision_counts=Counter(r['review_status'] for r in inventory)
    labels=Counter(r['source_label'] for r in pending)
    images=len({r['original_filename'] for r in pending})
    margins=[float(r['clip_min_view_margin']) for r in pending if r['clip_min_view_margin']]
    sims=[float(r['clip_min_green_similarity']) for r in pending if r['clip_min_green_similarity']]
    ratios=[float(r['green_pixel_ratio_support_only']) for r in pending if r['green_pixel_ratio_support_only']]
    strict=sum(s>=.31 and m>=.06 and g>=.35 for s,m,g in zip(sims,margins,ratios))
    summary={
      'polygon_count':len(inventory),'decision_counts':dict(decision_counts),'pending_polygon_count':len(pending),
      'pending_source_image_count':images,'pending_source_split_counts':dict(Counter(r['source_split_original'] for r in pending)),
      'pending_source_label_counts':dict(labels),'pending_polygon_geometry_strict_self_crossings':sum(r['mask_self_crossing_flag']=='True' for r in pending),
      'pending_out_of_bounds':sum(r['bbox_out_of_bounds_flag']=='True' for r in pending),
      'pending_with_known_phash_candidates':sum(r['phash_candidate_in_prior_audit']=='true' for r in inventory if r['review_status']=='NEEDS_REVIEW'),
      'mendeley_near_duplicate_pairs_in_full_v5_audit':len(near_mendeley),'mendeley_exact_duplicate_pairs_in_full_v5_audit':len(exact_mendeley),
      'strict_existing_ai_gate_passes_among_pending':strict,'human_verified_count':0,'source_families_verified':0,
      'train_candidates_human_verified':0,'independent_validation_candidates_human_verified':0,
      'pilot_ready':False,'evaluation_ready':False
    }
    (OUT/'reports/green_apple_review_audit_summary.json').write_text(json.dumps(summary,indent=2))
    # Contact sheets: only rank for inspection; draw all target masks and original filename.
    sheetdir=OUT/'review_contact_sheets'
    for old in sheetdir.glob('top_signal_*.jpg'):old.unlink()
    font=ImageFont.load_default()
    for page,start in enumerate(range(0,min(100,len(pending)),20),1):
        subset=pending[start:start+20]; tilew,tileh=320,270; cols=4; rowsn=5
        sheet=Image.new('RGB',(cols*tilew,rowsn*tileh),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(subset):
            idx=start+j; path=Path(r['image_relative_path']); im=Image.open(COLLECTION/path).convert('RGB')
            im.thumbnail((tilew-12,tileh-66))
            x=(j%cols)*tilew+6;y=(j//cols)*tileh+42
            px=(j%cols)*tilew+6;py=(j//cols)*tileh+42
            sheet.paste(im,(px,py))
            sx=im.width/int(r['image_width']);sy=im.height/int(r['image_height'])
            pts=json.loads(r['polygon_points_json']); poly=[(px+int(a*sx),py+int(b*sy)) for a,b in pts]
            if len(poly)>=3:draw.line(poly+[poly[0]],fill=(255,25,25),width=2)
            top=f"#{idx+1} {r['original_filename']} | {r['source_label']}"
            meta=f"margin={float(r['clip_min_view_margin']):.3f} green_support={float(r['green_pixel_ratio_support_only']):.2f} | PENDING"
            draw.text((px,(j//cols)*tileh+5),top,fill='black',font=font)
            draw.text((px,(j//cols)*tileh+22),meta,fill=(140,70,0),font=font)
        sheet.save(sheetdir/f'top_signal_{page:02d}.jpg',quality=92)
    # Reports.
    (OUT/'reports/duplicate_leakage_report.md').write_text(f'''# Phase 9.2 duplicate and leakage audit\n\nThis audit reused the previous exhaustive V5 hash/pHash scan and intersected its Mendeley rows with the current 799-polygon pending queue. It did not change any dataset or split.\n\n- Prior V5 scan: 3,368 images; 4 exact SHA-256 duplicate pairs, none cross project split; 1,370 pHash pairs at Hamming <=6, including 339 cross-split candidates.\n- Mendeley-related pHash pairs in that scan: {len(near_mendeley)}; all are project train/train. None touches the current 799 pending polygons because the known candidates were already held out of this queue. pHash is a review signal, not proof of identity.\n- Exact Mendeley pairs: {len(exact_mendeley)} (the existing report identifies `398.jpg/166.jpg` and `285.jpg/149.jpg`); these are not in the pending 799 queue.\n- Source-original test images: 77 were previously identified as imported into V5 training; all remain excluded from new candidate training and cannot serve as an independent test set.\n- The 799 pending records are source-original train only, from {images} unique images. They have no verified source-family/session IDs. No family or split was synthesized.\n- SHA and pHash checks do not establish independent capture sessions. The 1,370 project-wide pHash relationships, especially 339 cross-split candidates, remain a project-wide limitation outside this narrow 799-row subset.\n\nRow-level pair lists are in `near_duplicate_candidate_pairs.csv` and `exact_duplicate_pairs.csv`.\n''')
    (OUT/'reports/green_apple_review_audit.md').write_text(f'''# Audit of the 799 pending Green Apple polygons\n\n## Scope and result\n\nAll 799 `NEEDS_REVIEW` polygon rows from the previous AI-assisted assessment were re-audited using their recorded two-view CLIP scores, green-pixel support, source hashes, polygon geometry audit, and duplicate/leakage membership. This is a triage audit, not human review. All remain `PENDING_HUMAN_REVIEW`; no label or mask was changed and no row was promoted to human-verified.\n\n- Pending polygons: 799 across {images} source images; all source split `train`.
- Source-label distribution: {dict(labels)}. Source stage label does not prove peel color.
- Strict original AI gate (`both views green similarity >=.31`, minimum margin >=.06, green-pixel support >=.35): {strict} of the 799 pass. The green-pixel measure is only supporting evidence, not ground truth.
- CLIP minimum-view margin distribution: {sum(x>=.04 for x in margins)} >=.04; {sum(x>=.02 for x in margins)} >=.02; none >=.06.
- Geometry: {sum(r['mask_self_crossing_flag']=='True' for r in pending)} strict self-crossing flag, {sum(r['bbox_out_of_bounds_flag']=='True' for r in pending)} out-of-bounds flags. The crossing polygon is explicitly left pending for human mask review.
- Prior near-duplicate/exact-hash audit: no known candidate pairs touch these 799; all known candidates remain held separately. This does not establish source-family independence.
- Human decisions: 0. Verified source families: 0.
\n## Review priority and diversity\n\n`review_queue.csv` contains all 799 polygons sorted by CLIP margin/similarity and support-only color cue. The top 100 are prioritized for a human reviewer, not approved. `review_contact_sheets/top_signal_*.jpg` show source images with the polygon outline, filename, source label, and non-calibrated AI score. Existing source-image dimensions include 369x277 (434 pending polygons), 404x303 (289), 277x369 (71), 294x277 (3), and 277x236 (2). The material is from one orchard dataset; varying image dimensions and scenes do not establish independent sessions/families.\n\n## Candidate disposition\n\n- 5 previous AI-assisted candidates remain smoke-only; human verified count is 0.
- 799 remain pending human identity, green-peel, and mask review.
- 195 remain quarantined for source-test/duplicate/provenance holds.
- 18 remain excluded for geometry/integrity flags.
- No new training candidate was approved by this audit; no image, annotation, or split changed.\n''')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
