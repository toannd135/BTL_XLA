#!/usr/bin/env python3
"""Build corrected AI-assisted classifier dataset using exact crop identity."""
import csv, hashlib, json, shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
P=ROOT/'fruit/work/fruit_type_cls_experiment/phase9_2_green_apple_expansion'
BASE=ROOT/'fruit/work/fruit_type_cls_experiment/phase6b_candidate'
OUT=P/'green_apple_classifier'
DATA=OUT/'dataset_v2'

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    if DATA.exists(): raise SystemExit(f'refusing to overwrite {DATA}')
    scored=rows(P/'green_apple_classifier_ai_scores.csv')
    src_rows=rows(BASE/'manifests/complete_manifest.csv')
    by_key=defaultdict(list)
    for r in src_rows: by_key[(r['image'],r['candidate_copy_sha256'])].append(r)
    enriched=[]
    for r in scored:
        matches=by_key.get((r['image'],r['crop_sha256']),[])
        if len(matches)!=1: raise SystemExit(f"expected one source crop for {r['image']} sha={r['crop_sha256']}; found {len(matches)}")
        src=matches[0]
        r={**r,'candidate_copy_path':src['candidate_copy_path'],'candidate_copy_sha256':src['candidate_copy_sha256'],'source_family_id':src['source_family_id'],'split_v2':src['split_v2']}
        margin=float(r['green_minus_non_green_margin']); support=float(r['center_green_pixel_ratio_support_only'])
        if margin>=.02 and support>=.18:r['target']='GreenApple'
        elif margin<=-.02 and support<=.30:r['target']='NonGreenApple'
        else:r['target']='AMBIGUOUS_AI_REVIEW'
        enriched.append(r)
    by_hash=defaultdict(list)
    for r in enriched:
        if r['target'] in ('GreenApple','NonGreenApple'):by_hash[r['crop_sha256']].append(r)
    duplicate_hashes={h for h,rr in by_hash.items() if len(rr)>1}
    # If duplicate bytes cross labels or splits, quarantine all copies. Otherwise retain only a stable one.
    excluded=set()
    for h in duplicate_hashes:
        rr=by_hash[h]
        if len({(x['target'],x['split_v2'],x['source_family_id']) for x in rr})>1:
            excluded.update(id(x) for x in rr)
        else:
            excluded.update(id(x) for x in sorted(rr,key=lambda x:(x['image'],x['candidate_copy_path']))[1:])
    fields=list(enriched[0])+['training_eligible','dataset_path','exclusion_reason','label_provenance','human_verified_color']
    manifest_rows=[]; count=Counter(); families=defaultdict(set)
    for r in enriched:
        eligible=r['target'] in ('GreenApple','NonGreenApple') and id(r) not in excluded
        reason=''
        if r['target']=='AMBIGUOUS_AI_REVIEW':reason='AI_color_margin_or_green_support_uncertain'
        elif id(r) in excluded:reason='duplicate_crop_hash_quarantined'
        dst=''
        if eligible:
            source=BASE/r['candidate_copy_path']
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=r['crop_sha256']:
                raise SystemExit(f"crop integrity mismatch: {source}")
            fam=r['source_family_id']; split=r['split_v2']
            if not fam or split not in {'train','val','test'}:raise SystemExit(f'missing family/split metadata for {r["image"]}')
            dst=DATA/split/r['target']/(r['crop_sha256'][:16]+'_'+source.name)
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
            count[(split,r['target'])]+=1;families[(split,r['target'])].add(fam)
        manifest_rows.append({**r,'training_eligible':str(eligible).lower(),'dataset_path':str(dst.relative_to(DATA)) if dst else '', 'exclusion_reason':reason,'label_provenance':'AI_ASSISTED_CLIP_AND_COLOR_SUPPORT','human_verified_color':'false'})
    # Reject any exact crop hash crossing splits, and family intersections are inherited from phase6b.
    included_hashes=[r['crop_sha256'] for r in manifest_rows if r['training_eligible']=='true']
    if len(set(included_hashes)) != len(included_hashes):
        raise SystemExit('duplicate crop hash remains in included rows')
    mp=P/'green_apple_classifier_manifest_v2.csv'
    if mp.exists():raise SystemExit(f'refusing to overwrite {mp}')
    with mp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(manifest_rows)
    summary={'status':'AI_ASSISTED_CLASSIFICATION_CANDIDATE','classes':{'GreenApple':0,'NonGreenApple':1},'scored_crop_rows':len(enriched),'included_unique_crop_count':sum(count.values()),'ambiguous_crop_count':sum(r['target']=='AMBIGUOUS_AI_REVIEW' for r in enriched),'duplicate_crop_hashes_quarantined':len(duplicate_hashes),'counts_by_split_and_class':{f'{s}/{c}':n for (s,c),n in count.items()},'source_families_by_split':{s:{c:len(families[(s,c)]) for c in ('GreenApple','NonGreenApple')} for s in ('train','val','test')},'family_cross_split_count':0,'human_verified_color_labels':0,'test_label_provenance':'AI-assisted CLIP/color-support decisions; test agreement is exploratory only'}
    (OUT/'dataset_v2_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
