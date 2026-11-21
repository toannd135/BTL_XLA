#!/usr/bin/env python3
"""Read-only data gate for a future Green Banana training run; never trains or splits."""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PHASE8D = Path(__file__).resolve().parents[1]
COLLECTION = PHASE8D.parent
DATASET_PREFIX = 'green_banana/BananaID_ripeness_original/'


def read_csv(path):
    if not path.exists(): return []
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def read_jsonl(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def latest(records, key):
    out = {}
    for row in records: out[row[key]] = row
    return out


def preflight():
    rows = [r for r in read_csv(COLLECTION / 'manifests/intake_manifest.csv')
            if (r.get('incoming_relative_path') or '').startswith(DATASET_PREFIX)]
    reviews = latest(read_jsonl(PHASE8D / 'review/review_history.jsonl'), 'asset_id')
    permissions = latest(read_jsonl(PHASE8D / 'metadata/permission_history.jsonl'), 'asset_id')
    families = Counter(r.get('source_family_id') for r in rows if r.get('source_family_id'))
    cleared = []
    reasons = Counter()
    for row in rows:
        aid = row['asset_id']; review = reviews.get(aid, {}); perm = permissions.get(aid, {})
        is_human = bool(review.get('reviewer_id')) and not str(review.get('reviewer_id')).lower().startswith(('codex_', 'ai_', 'bot_'))
        owner_authorized_ai = (
            review.get('reviewer_id') == 'codex_ai_user_authorized'
            and review.get('review_method') == 'AI visual review; explicitly authorized by project owner'
            and review.get('owner_authorized') is True
        )
        gates = {
            'green_review': (is_human or owner_authorized_ai) and review.get('decision') == 'VERIFIED_CORRECT' and review.get('fruit_identity') == 'Banana' and review.get('peel_genuinely_green') is True and review.get('peel_color_confidence') == 'CLEAR_GREEN' and review.get('image_suitable') is True and review.get('source_image_viewed') is True and review.get('target_crop_viewed') is True,
            'permission': perm.get('permission_status') == 'VERIFIED' and bool(perm.get('evidence_reference')),
            'source_family': bool(row.get('source_family_id')),
            'readable': str(row.get('image_readable')).lower() == 'true',
        }
        if all(gates.values()): cleared.append(row)
        for name, ok in gates.items():
            if not ok: reasons[name] += 1
    cleared_families = {r.get('source_family_id') for r in cleared if r.get('source_family_id')}
    human_cleared = [r for r in cleared if reviews.get(r['asset_id'], {}).get('reviewer_id') and not str(reviews[r['asset_id']]['reviewer_id']).lower().startswith(('codex_', 'ai_', 'bot_'))]
    ai_cleared = [r for r in cleared if reviews.get(r['asset_id'], {}).get('reviewer_id') == 'codex_ai_user_authorized']

    byid = {r['asset_id']: r for r in rows}
    pair_events = read_jsonl(PHASE8D / 'review/duplicate_pair_history.jsonl')
    pair_latest = {}
    for event in pair_events: pair_latest[event.get('pair_id')] = event
    unresolved_cross_family = []
    unresolved_same_family = []
    for p in read_csv(COLLECTION / 'review/near_duplicate_review_queue.csv'):
        a, b = byid.get(p.get('asset_id_a')), byid.get(p.get('asset_id_b'))
        if not a or not b: continue
        # Missing family IDs are unknown provenance, not evidence that two
        # assets belong to one family. Only a shared, non-empty ID can make a
        # pair a within-family candidate.
        family_a, family_b = a.get('source_family_id'), b.get('source_family_id')
        if family_a and family_b and family_a == family_b:
            unresolved_same_family.append(p)
            continue
        pairkey = '|'.join(sorted([p['asset_id_a'], p['asset_id_b']]))
        pairid = 'pair_' + __import__('hashlib').sha256(pairkey.encode()).hexdigest()[:24]
        event = pair_latest.get(pairid)
        reviewer = str(event.get('reviewer_id', '')).lower() if event else ''
        is_human_pair_review = bool(reviewer) and not reviewer.startswith(('codex_', 'ai_', 'bot_'))
        is_owner_authorized_ai_pair_review = (
            event is not None
            and event.get('reviewer_id') == 'codex_ai_user_authorized'
            and event.get('review_method') == 'AI visual review; explicitly authorized by project owner'
            and event.get('owner_authorized') is True
        )
        if not event or not (is_human_pair_review or is_owner_authorized_ai_pair_review) or event.get('both_images_viewed') is not True or event.get('decision') not in {'DISTINCT', 'SAME_FAMILY', 'SAME_SOURCE'}:
            unresolved_cross_family.append(p)

    target = {'train': {'images': 200, 'families': 40}, 'validation': {'images': 50, 'families': 10}, 'test': {'images': 50, 'families': 10}}
    family_view_counts = Counter()
    for row in cleared:
        family_view_counts[row.get('source_family_id')] += 1
    requirements = {
        'at_least_300_owner_authorized_verified_green_images': len(cleared) >= 300,
        'at_least_60_owner_authorized_verified_source_families': len(cleared_families) >= 60,
        'all_cross_family_near_duplicates_reviewed': len(unresolved_cross_family) == 0,
        'no_family_has_unreviewed_views_in_candidate': all(family_view_counts[f] == families[f] for f in cleared_families),
    }
    ready = all(requirements.values())
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'scope': 'BananaID original Unripe image cohort only; other intake stays pending and is not silently included',
        'status': 'DATA_GATE_READY_FOR_PHASE8E' if ready else 'TRAINING_BLOCKED',
        'training_started': False, 'splits_created': False,
        'cohort': {'candidate_files': len(rows), 'candidate_source_families': len(families),
                   'family_size_distribution': dict(Counter(families.values())),
                   'owner_authorized_verified_green_images': len(cleared),
                   'owner_authorized_reviewed_families': len(cleared_families)},
        'permission_verified_count': sum(1 for r in rows if (permissions.get(r['asset_id'], {}).get('permission_status') == 'VERIFIED' and permissions.get(r['asset_id'], {}).get('evidence_reference'))),
        'authorized_ai_verified_green_images': sum(1 for r in rows if (reviews.get(r['asset_id'], {}).get('reviewer_id') == 'codex_ai_user_authorized' and reviews.get(r['asset_id'], {}).get('review_method') == 'AI visual review; explicitly authorized by project owner' and reviews.get(r['asset_id'], {}).get('owner_authorized') is True and reviews.get(r['asset_id'], {}).get('decision') == 'VERIFIED_CORRECT')),
        'human_verified_green_images': sum(1 for r in rows if (reviews.get(r['asset_id'], {}).get('reviewer_id') and not str(reviews.get(r['asset_id'], {}).get('reviewer_id')).lower().startswith(('codex_', 'ai_', 'bot_')) and reviews.get(r['asset_id'], {}).get('decision') == 'VERIFIED_CORRECT')),
        'near_duplicate_review': {'unresolved_cross_family_pairs': len(unresolved_cross_family),
                                  'within_family_pairs_same_split_by_construction': len(unresolved_same_family)},
        'missing_gate_counts': dict(reasons), 'requirements': requirements, 'targets': target,
        'limits': ['A data-gate pass permits a controlled split-design phase; it does not itself make a model production-ready.',
                   'The BananaID cohort uses a controlled white-background acquisition setup and is not an external real-world evaluation set.',
                   'Green Apple examples are outside this single-class gate and are needed for a Green Banana vs Green Apple model.'],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=PHASE8D / 'reports/green_banana_training_preflight.json')
    args = p.parse_args(); result = preflight(); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'candidate_files': result['cohort']['candidate_files'],
                      'owner_authorized_verified_green_images': result['cohort']['owner_authorized_verified_green_images'],
                      'candidate_families': result['cohort']['candidate_source_families'],
                      'output': str(args.output.resolve())}, indent=2))


if __name__ == '__main__': main()
