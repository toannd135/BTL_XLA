#!/usr/bin/env python3
"""Generate a read-only Green Banana intake/provenance progress report."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PHASE8D = Path(__file__).resolve().parents[1]
COLLECTION = PHASE8D.parent
EXPERIMENT = COLLECTION.parent
PROJECT = EXPERIMENT.parent.parent
TARGETS = {'train': {'images': 200, 'families': 40}, 'validation': {'images': 50, 'families': 10}, 'test': {'images': 50, 'families': 10}}
STATUSES = ['PENDING', 'VERIFIED_CORRECT', 'INCORRECT_LABEL', 'AMBIGUOUS', 'INVALID_IMAGE', 'NEEDS_PROVENANCE']


def read_csv(path):
    if not path.exists(): return []
    with path.open(encoding='utf-8-sig', newline='') as stream: return list(csv.DictReader(stream))


def read_jsonl(path):
    if not path.exists(): return []
    records=[]
    for n,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if line.strip():
            try: records.append(json.loads(line))
            except json.JSONDecodeError as exc: raise ValueError(f'{path}:{n}: invalid JSON: {exc}') from exc
    return records


def latest_by_asset(records):
    latest={}
    for row in records: latest[row['asset_id']]=row
    return latest


def make_report():
    scanned_assets=read_csv(COLLECTION/'manifests/intake_manifest.csv')
    assets=[r for r in scanned_assets if r.get('class_label')=='Banana' and (r.get('incoming_relative_path') or '').startswith('green_banana/')]
    reviews=latest_by_asset(read_jsonl(PHASE8D/'review/review_history.jsonl'))
    permissions=latest_by_asset(read_jsonl(PHASE8D/'metadata/permission_history.jsonl'))
    near=[r for r in read_csv(COLLECTION/'review/near_duplicate_review_queue.csv')
          if (r.get('incoming_path_a') or '').startswith('green_banana/') and (r.get('incoming_path_b') or '').startswith('green_banana/')]
    pair_events=read_jsonl(PHASE8D/'review/duplicate_pair_history.jsonl')
    ai_pair_screen=read_csv(PHASE8D/'reports/bananaid_cross_family_pair_ai_screen.csv')
    latest_pairs={}
    for event in pair_events:
        latest_pairs[event['pair_id']]=event
    metadata_errors=[]
    metadata_path=COLLECTION/'metadata/source_metadata.jsonl'
    try: source_metadata=latest_by_asset(read_jsonl(metadata_path))
    except ValueError as exc: source_metadata={};metadata_errors.append(str(exc))
    actual_assets=[]
    for row in assets:
        aid=row.get('asset_id','')
        event=reviews.get(aid,{})
        permission=permissions.get(aid,{})
        supplied=source_metadata.get(aid,{})
        decision=event.get('decision','PENDING')
        actual_assets.append({'asset_id':aid,'row':row,'review':event,'decision':decision,
                              'permission':permission,'source_metadata':supplied})
    status_counts=Counter(x['decision'] for x in actual_assets)
    for status in STATUSES: status_counts.setdefault(status,0)
    permission_counts=Counter(x['permission'].get('permission_status','PENDING') for x in actual_assets)
    readable=sum(str(x['row'].get('image_readable')).lower()=='true' for x in actual_assets)
    unique_sha={x['row'].get('image_sha256') for x in actual_assets if x['row'].get('image_sha256')}
    unique_source_ids={x['source_metadata'].get('source_image_id') or x['row'].get('source_image_id') for x in actual_assets if x['source_metadata'].get('source_image_id') or x['row'].get('source_image_id')}
    known_families={x['source_metadata'].get('source_family_id') or x['row'].get('source_family_id') for x in actual_assets if x['source_metadata'].get('source_family_id') or x['row'].get('source_family_id')}
    verified_visual=[x for x in actual_assets if x['decision']=='VERIFIED_CORRECT' and x['review'].get('fruit_identity')=='Banana' and x['review'].get('peel_genuinely_green') is True and x['review'].get('peel_color_confidence')=='CLEAR_GREEN' and x['review'].get('image_suitable') is True and x['review'].get('source_image_viewed') is True and x['review'].get('target_crop_viewed') is True]
    fully_cleared=[x for x in verified_visual if x['permission'].get('permission_status')=='VERIFIED' and x['permission'].get('evidence_reference') and (x['source_metadata'].get('source_family_id') or x['row'].get('source_family_id'))]
    families_by_split=Counter()  # no split is assigned in Phase 8D
    family_by_source=defaultdict(set)
    session_counts=Counter()
    source_counts=Counter()
    missing_fields=Counter()
    for x in actual_assets:
        row=x['row']; meta=x['source_metadata']
        source=meta.get('source_provider') or row.get('source_provider') or 'UNKNOWN'
        source_counts[source]+=1
        session=meta.get('capture_session_id') or row.get('capture_session_id') or meta.get('video_id') or row.get('video_id') or 'UNKNOWN'
        session_counts[session]+=1
        fam=meta.get('source_family_id') or row.get('source_family_id')
        if fam: family_by_source[source].add(fam)
        required=['source_image_id','source_family_id','source_type','source_license']
        if not (meta.get('source_url') or row.get('source_url') or meta.get('source_provider') or row.get('source_provider')):
            required.append('source_url_or_provider')
        for field in required:
            if not (meta.get(field) or row.get(field)): missing_fields[field]+=1
    permission_unverified=sum(x['permission'].get('permission_status')!='VERIFIED' or not x['permission'].get('evidence_reference') for x in actual_assets)
    unresolved=[x for x in actual_assets if x['decision'] in {'AMBIGUOUS','NEEDS_PROVENANCE'}]
    pair_decisions=Counter()
    unresolved_duplicate_candidates=[]
    for candidate in near:
        a=candidate.get('asset_id_a') or candidate.get('asset_id_1') or candidate.get('asset_id')
        b=candidate.get('asset_id_b') or candidate.get('asset_id_2')
        pair_id='pair_'+__import__('hashlib').sha256('|'.join(sorted([a or '',b or ''])).encode()).hexdigest()[:24] if a and b else None
        event=latest_pairs.get(pair_id) if pair_id else None
        reviewer=str(event.get('reviewer_id','')).lower() if event else ''
        human_event=bool(reviewer) and not reviewer.startswith(('codex_','ai_','bot_')) and event.get('both_images_viewed') is True
        decision=event.get('decision') if human_event else None
        if not decision:
            status=candidate.get('review_status','PENDING')
            if status in {'RESOLVED_DISTINCT','RESOLVED_SAME_SOURCE','RESOLVED_SAME_FAMILY'}:
                decision=status.removeprefix('RESOLVED_')
        if decision: pair_decisions[decision]+=1
        if decision not in {'DISTINCT','SAME_SOURCE','SAME_FAMILY'}:
            unresolved_duplicate_candidates.append(candidate)
    exact_file_groups=Counter(x['row'].get('image_sha256') for x in actual_assets if x['row'].get('image_sha256'))
    exact_duplicate_extra=sum(n-1 for n in exact_file_groups.values() if n>1)

    # Legacy Phase 6B reference is reported separately and never merged into intake counts.
    legacy={'verified_green_crop_count':0,'distinct_source_image_count':0,'source_family_count':0,'split_counts':{},'counted_in_phase8d':False,'permission_status':'NOT_REVALIDATED_FOR_PHASE8D'}
    green_report=EXPERIMENT/'phase6b_candidate/reports/green_fruit_report.csv'
    if green_report.exists():
        records=read_csv(green_report)
        for role in ['train','val','test']:
            matches=[r for r in records if r.get('fruit_class')=='banana' and r.get('split')==role and r.get('green_visual_status')=='verified_green']
            legacy['split_counts'][role]=sum(int(r['included_verified_crops']) for r in matches)
        legacy['verified_green_crop_count']=sum(legacy['split_counts'].values())
        legacy['distinct_source_image_count']=sum(int(r['included_distinct_source_images']) for r in records if r.get('fruit_class')=='banana' and r.get('green_visual_status')=='verified_green')
        legacy['source_family_count']=len({r.get('included_source_families') for r in records if r.get('fruit_class')=='banana' and r.get('green_visual_status')=='verified_green' and int(r.get('included_source_families') or 0)>0})
    ready_train=sum(1 for x in fully_cleared if x['row'].get('split')=='train')
    ready_val=sum(1 for x in fully_cleared if x['row'].get('split') in {'val','validation'})
    ready_test=sum(1 for x in fully_cleared if x['row'].get('split')=='test')
    report={
      'generated_at':datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z'),
      'status':'DATA_COLLECTION_IN_PROGRESS',
      'split_assignment_performed':False,
      'intake':{'file_count':len(actual_assets),'scanned_collection_file_count':len(scanned_assets),'non_green_banana_files_excluded':len(scanned_assets)-len(actual_assets),
                'readable_file_count':readable,'distinct_exact_sha256_content_count':len(unique_sha),'duplicate_file_extra_count':exact_duplicate_extra,
                'distinct_source_image_id_count':len(unique_source_ids),'known_source_family_count':len(known_families),
                'source_family_count_is_lower_bound':any(not (x['source_metadata'].get('source_family_id') or x['row'].get('source_family_id')) for x in actual_assets)},
      'review':{'latest_status_counts':dict(status_counts),'review_event_asset_count':len([x for x in actual_assets if x['decision']!='PENDING']),
                'human_reviewed_asset_count':len([x for x in actual_assets if x['decision']!='PENDING' and not str(x['review'].get('reviewer_id','')).lower().startswith(('codex_','ai_','bot_'))]),
                'ai_visual_assessed_asset_count':len([x for x in actual_assets if x['decision']!='PENDING' and str(x['review'].get('reviewer_id','')).lower().startswith(('codex_','ai_','bot_'))]),
                'owner_authorized_ai_review_count':len([x for x in actual_assets if x['decision']!='PENDING' and x['review'].get('reviewer_id')=='codex_ai_user_authorized' and x['review'].get('owner_authorized') is True]),
                'verified_green_banana_count_visual_and_target_checked':len(verified_visual),'verified_green_banana_count_with_permission_and_family':len(fully_cleared),
                'ambiguous_or_provenance_assets_requiring_re_review':len(unresolved)},
      'sources_and_sessions':{'files_by_source_provider':dict(source_counts),'files_by_session_or_video':dict(session_counts),
                              'known_families_by_provider':{k:len(v) for k,v in family_by_source.items()}},
      'duplicates':{'exact_duplicate_extra_files':exact_duplicate_extra,'exact_and_near_duplicate_review_candidates':len(near),
                    'pending_near_duplicate_candidates':len(unresolved_duplicate_candidates),
                    'ai_visual_pair_suggestions_not_human_decisions':len(ai_pair_screen),
                    'latest_human_pair_decision_counts':dict(pair_decisions),
                    'reviewed_pair_event_count':len(pair_events)},
      'metadata':{'assets_missing_required_values':sum(1 for x in actual_assets if any(not x['source_metadata'].get(f) and not x['row'].get(f) for f in ['source_image_id','source_family_id','source_type','source_license'])),
                  'missing_field_counts':dict(missing_fields),'source_metadata_parse_errors':metadata_errors},
      'license_permission':{'assets_without_verified_permission_evidence':permission_unverified,'permission_status_counts':dict(permission_counts)},
      're_review':{'ambiguous_or_needs_provenance_count':len(unresolved),'unresolved_duplicate_candidate_count':len(unresolved_duplicate_candidates)},
      'targets':TARGETS,
      'current_phase8d_ready_counts':{'train':ready_train,'validation':ready_val,'test':ready_test,'assigned_splits':False},
      'legacy_phase6b_reference_not_counted_in_phase8d':legacy,
      'limitations':['BananaID family groups are an AI visual inference from four matching view images per variety/index group; the official BananaID page does not itself publish those family links.',
                     'The 212 BananaImageBD images and five user-provided images still have unknown source-family IDs and are not eligible for family-level split assignment.',
                     'The 408 BananaID label and pair decisions were made by AI under explicit project-owner authorization; they are not human reviews.',
                     '175 cross-family BananaID pHash candidates have AI-authorized DISTINCT decisions; other source groups still need provenance work.',
                     'Legacy Phase 6B visually verified green-banana crops are one time-lapse family and license/permission has not been revalidated for Phase 8D.',
                     'No train/validation/test split has been assigned; split allocation is deferred to Phase 8E.']
    }
    return report


def render_md(r):
    i=r['intake'];v=r['review'];l=r['legacy_phase6b_reference_not_counted_in_phase8d'];t=r['targets']
    lines=['# Phase 8D — Green Banana Collection Progress','',f"**Status: `{r['status']}`**",'',f"Generated: {r['generated_at']}",'',
      '## Current Phase 8D intake','',
      f"- Green Banana candidate files: **{i['file_count']}**; readable: **{i['readable_file_count']}**. Phase 8C scanned {i['scanned_collection_file_count']} total files; {i['non_green_banana_files_excluded']} non-Banana files are excluded from this queue.",
      f"- Distinct SHA-256 contents: **{i['distinct_exact_sha256_content_count']}**; duplicate extra files: **{i['duplicate_file_extra_count']}**.",
      f"- Source image IDs: **{i['distinct_source_image_id_count']}**; known source families: **{i['known_source_family_count']}**.",
      f"- Review events: **{v['review_event_asset_count']}/{i['file_count']}**; human reviewed: **{v['human_reviewed_asset_count']}**; AI visual triage: **{v['ai_visual_assessed_asset_count']}**.",
      f"- Verified Green Banana with source+target crop reviewed: **{v['verified_green_banana_count_visual_and_target_checked']}**.",
      f"- Verified with acceptable permission evidence and source family: **{v['verified_green_banana_count_with_permission_and_family']}**.",
      '- Source/session distribution: see the JSON report. BananaID supplies **408 views / 102 family IDs** (four named views per banana); BananaImageBD supplies 212 images with family IDs unknown; the five user-supplied images also have unknown families.',
      '- Exact/near-duplicate review candidates: **'+str(r['duplicates']['exact_and_near_duplicate_review_candidates'])+'**.',
      '- AI side-by-side pair-screen suggestions (not decisions): **'+str(r['duplicates']['ai_visual_pair_suggestions_not_human_decisions'])+'**; the human pair-review history remains authoritative.',
      '- Missing required source fields: **'+str(r['metadata']['assets_missing_required_values'])+' Green Banana assets** (unknown source families remain null).',
      '- Permission evidence not verified: **'+str(r['license_permission']['assets_without_verified_permission_evidence'])+' Green Banana assets**.',
      '- Split assignment: **none**. Phase 8D does not assign roles.','',
      '## Review status counts','', '| Status | Assets |','|---|---:|']
    for k in ['PENDING','VERIFIED_CORRECT','INCORRECT_LABEL','AMBIGUOUS','INVALID_IMAGE','NEEDS_PROVENANCE']:
        lines.append(f"| {k} | {v['latest_status_counts'].get(k,0)} |")
    lines += ['', '## Legacy Phase 6B reference (not counted as Phase 8D intake)','',
      f"Phase 6B contains {l['verified_green_crop_count']} visually verified green-banana crop rows from {l['distinct_source_image_count']} source images and {l['source_family_count']} source family/families (train {l['split_counts'].get('train',0)}, val {l['split_counts'].get('val',0)}, test {l['split_counts'].get('test',0)}). These are repeated frames from one time-lapse family and permission was not revalidated for this collection. They are reference-only and are not copied, promoted or counted toward Phase 8D targets.",'',
      '## Targets and remaining gap','',
      '| Future role | Target images | Minimum independent families | Phase 8D ready images |', '|---|---:|---:|---:|']
    for role in ['train','validation','test']:
        lines.append(f"| {role} | {t[role]['images']} | {t[role]['families']} | {r['current_phase8d_ready_counts'][role]} |")
    lines += ['', 'Targets are not satisfied by file count alone. Distinct source families, source image hashes and human-approved provenance/permission are required. Split assignment is deferred to Phase 8E.','',
      '## Remaining risks','',
      '- Only the 408 BananaID files have source-family IDs (102 groups, four views each); BananaImageBD and five user-provided images lack family IDs. The 102 family count is enough for a numerical 40/10/10 family target only if human green-label review and near-duplicate checks pass.',
      '- Both public datasets are studio-style, single-banana photographs on controlled backgrounds; this limits real-world background diversity.',
      f"- {r['duplicates']['pending_near_duplicate_candidates']} pHash near-duplicate candidates remain unresolved; pHash similarity alone does not prove duplicate identity or shared source family.",
      '- Five user-provided assets have AI triage events only; all 625 assets still require human visual verification.',
      '- Existing Phase 6B green-banana evidence is one fixed time-lapse family with overlapping bunches; it provides no independent validation/test family.',
      '- The legacy source video permission/license is not recorded as verified for Phase 8D and cannot be assumed.',
      '- No collection frames, crops, labels or split assignments were added in this phase.', '',
      '## Final status','',f"`{r['status']}`",'']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PHASE8D/'reports')
    args=parser.parse_args()
    r=make_report(); out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'green_banana_progress.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'green_banana_progress.md').write_text(render_md(r),encoding='utf-8')
    print(json.dumps({'status':r['status'],'intake_files':r['intake']['file_count'],'verified_green_banana':r['review']['verified_green_banana_count_with_permission_and_family'],'report_dir':str(out)},indent=2))

if __name__=='__main__':main()
