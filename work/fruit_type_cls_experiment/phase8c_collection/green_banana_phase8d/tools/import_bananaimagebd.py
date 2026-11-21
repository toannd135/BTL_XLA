#!/usr/bin/env python3
"""Import only the raw Green class from the cited CC BY 4.0 BananaImageBD mirror.

This preserves the downloaded Parquet archive, extracts image bytes unchanged,
records known dataset provenance, leaves source-family/green verification unknown,
and never assigns review decisions or splits.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASE8D=Path(__file__).resolve().parents[1]
COLLECTION=PHASE8D.parent
EXPECTED_SHA256='5c48537f0e1adc41783fc7f5cf8f4d7da191c89613dd6a5e11c64c220577130b'
MENDELEY_URL='https://data.mendeley.com/datasets/ptfscwtnyz/2'
HF_FILE_URL='https://huggingface.co/datasets/Project-AgML/BananaImageBD_ripeness_classification/resolve/main/raw/train-00000-of-00001.parquet'
ATTRIBUTION='Ferdaus et al. (2024), BananaImageBD: A Comprehensive Image Dataset of Common Banana Varieties with Different Ripeness Stages in Bangladesh, Mendeley Data v2, DOI:10.17632/ptfscwtnyz.2.'

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet',type=Path,default=PHASE8D/'source_archives/BananaImageBD/raw/train-00000-of-00001.parquet')
    parser.add_argument('--expected-count',type=int,default=212)
    args=parser.parse_args()
    parquet=args.parquet.resolve()
    actual=sha256(parquet)
    if actual!=EXPECTED_SHA256:raise SystemExit(f'BLOCKED: official mirror parquet SHA-256 mismatch: {actual}')
    try:import polars as pl
    except ImportError as exc:raise SystemExit('Install polars to extract the Parquet images: python -m pip install polars') from exc
    frame=pl.read_parquet(parquet)
    green=frame.filter(pl.col('label')==0)
    if green.height!=args.expected_count:raise SystemExit(f'BLOCKED: expected {args.expected_count} label-0 Green rows, found {green.height}')
    dest=COLLECTION/'incoming/green_banana/BananaImageBD_raw'
    if dest.exists() and any(dest.iterdir()):raise SystemExit(f'BLOCKED: refusing to overwrite non-empty directory {dest}')
    dest.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(COLLECTION/'tools'))
    from collection_cli import stable_asset_id
    source_meta=COLLECTION/'metadata/source_metadata.jsonl'
    existing=[]
    if source_meta.exists():
        for line in source_meta.read_text(encoding='utf-8').splitlines():
            if line.strip():existing.append(json.loads(line))
    existing_ids={r.get('asset_id') for r in existing}
    added=[]
    for row in green.iter_rows(named=True):
        blob=row['image']['bytes']; original_name=Path(row['image']['path']).name
        if not blob:raise SystemExit(f'BLOCKED: missing image bytes for {original_name}')
        target=dest/original_name
        if target.exists():raise SystemExit(f'BLOCKED: refusing to overwrite {target}')
        target.write_bytes(blob)
        relative=target.relative_to(COLLECTION/'incoming').as_posix()
        digest=hashlib.sha256(blob).hexdigest()
        aid=stable_asset_id(relative,digest)
        if aid in existing_ids:raise SystemExit(f'BLOCKED: duplicate metadata asset_id {aid}')
        record={'schema_version':'1.0.0','asset_id':aid,'original_filename':original_name,'class_label':'Banana',
          'green_status':None,'source_type':'photo','source_url':MENDELEY_URL,'source_license':'CC BY 4.0',
          'source_provider':'BananaImageBD authors via Mendeley Data v2 (Hugging Face mirror)',
          'collection_date':None,'source_image_id':f'BananaImageBD:v2:{original_name}','source_family_id':None,
          'capture_session_id':None,'video_id':None,'frame_id':None,'camera_device':None,
          'image_sha256':digest,'image_width':None,'image_height':None,'reviewer_id':None,'review_status':'PENDING',
          'review_reason':None,'created_at':datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z'),
          'incoming_relative_path':relative,'asset_id_method':'phase8c-v1:relative_path+sha256'}
        added.append(record);existing_ids.add(aid)
    with source_meta.open('a',encoding='utf-8') as f:
        for record in added:f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
    archive_meta=PHASE8D/'source_archives/BananaImageBD/provenance.json'
    archive_meta.parent.mkdir(parents=True,exist_ok=True)
    archive_meta.write_text(json.dumps({'dataset':'BananaImageBD: A Comprehensive Image Dataset of Common Banana Varieties with Different Ripeness Stages in Bangladesh','version':2,'doi':'10.17632/ptfscwtnyz.2','official_source':MENDELEY_URL,'license':'CC BY 4.0','attribution_required':ATTRIBUTION,'mirror':HF_FILE_URL,'mirror_repository':'Project-AgML/BananaImageBD_ripeness_classification','archive_path':str(parquet),'archive_sha256':actual,'raw_parquet_row_count':frame.height,'label_mapping':{'0':'Green','1':'Overripe','2':'Ripe','3':'Semi-ripe'},'green_raw_images_extracted':len(added),'augmentation_rows_used':0,'split_assignment_performed':False,'source_family_ids_assigned':False,'human_label_review_performed':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'extracted_green_images':len(added),'target_dir':str(dest.resolve()),'source_metadata_appended':len(added),'parquet_sha256':actual,'source_families_assigned':0,'labels_approved':0,'splits_assigned':False},indent=2))

if __name__=='__main__':main()
