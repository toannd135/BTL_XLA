#!/usr/bin/env python3
"""Reproducible, non-destructive intake tools for green-fruit originals."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageOps, ExifTags
except ImportError as exc:  # pragma: no cover - environment-specific message
    raise SystemExit('Pillow is required: python -m pip install Pillow') from exc

HERE = Path(__file__).resolve().parents[1]
SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.gif'}
CLASS_DIRS = {'green_banana': 'Banana', 'green_apple': 'Apple'}
REVIEW_STATUSES = {'PENDING', 'VERIFIED_CORRECT', 'INCORRECT_LABEL', 'AMBIGUOUS', 'INVALID_IMAGE', 'NEEDS_PROVENANCE'}
REVIEW_REASON_REQUIRED = {'INCORRECT_LABEL', 'AMBIGUOUS', 'INVALID_IMAGE', 'NEEDS_PROVENANCE'}
METADATA_FIELDS = [
    'schema_version', 'asset_id', 'original_filename', 'class_label', 'green_status', 'source_type', 'source_url',
    'source_license', 'source_provider', 'collection_date', 'source_image_id', 'source_family_id',
    'capture_session_id', 'video_id', 'frame_id', 'camera_device', 'image_sha256',
    'perceptual_hash', 'image_width', 'image_height', 'reviewer_id', 'review_status',
    'review_reason', 'created_at', 'incoming_relative_path', 'originals_relative_path',
    'image_readable', 'metadata_missing_fields', 'asset_id_method', 'exif_summary'
]
SOURCE_METADATA_FIELDS = {
    'green_status', 'source_type', 'source_url', 'source_license', 'source_provider', 'collection_date',
    'source_image_id', 'source_family_id', 'capture_session_id', 'video_id', 'frame_id', 'camera_device'
}
REVIEW_FIELDS = [
    'review_id', 'asset_id', 'reviewer_id', 'reviewed_at', 'decision', 'fruit_identity',
    'peel_genuinely_green', 'image_suitable', 'fruit_visibility', 'occlusion',
    'background_dominance', 'reason', 'notes'
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stable_asset_id(relative_path: str, digest: str) -> str:
    """Stable for the same incoming relative path and exact bytes."""
    key = f'phase8c-v1\0{Path(relative_path).as_posix()}\0{digest}'
    return 'asset_' + hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]


def dct_phash(image: Image.Image) -> str:
    """64-bit DCT perceptual hash using Pillow + NumPy (no network/model dependency)."""
    import numpy as np
    gray = ImageOps.exif_transpose(image).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    n = 32
    idx = np.arange(n)
    basis = np.cos((math.pi / n) * (idx[:, None] + 0.5) * idx[None, :])
    basis[:, 0] *= 1 / math.sqrt(2)
    basis *= math.sqrt(2 / n)
    coeff = basis.T @ pixels @ basis
    low = coeff[:8, :8].flatten()
    median = float(np.median(low[1:]))
    bits = low > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f'{value:016x}'


def hamming_hex(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError('perceptual hashes must have equal hexadecimal length')
    return (int(a, 16) ^ int(b, 16)).bit_count()


def load_source_metadata(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    if not path.exists():
        return records, errors
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f'{path}:{line_no}: invalid JSON: {exc}')
            continue
        asset_id = row.get('asset_id') if isinstance(row, dict) else None
        if not asset_id:
            errors.append(f'{path}:{line_no}: asset_id is required')
        elif asset_id in records:
            errors.append(f'{path}:{line_no}: duplicate asset_id {asset_id}')
        else:
            records[asset_id] = row
    return records, errors


def image_metadata(path: Path) -> tuple[bool, int | None, int | None, str | None, str | None, str | None]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            width, height = im.size
            phash = dct_phash(im)
            exif = im.getexif()
            keep = {'DateTime', 'Make', 'Model', 'Orientation'}
            exif_summary = {ExifTags.TAGS.get(k, str(k)): str(v)[:160] for k, v in exif.items() if ExifTags.TAGS.get(k, str(k)) in keep}
        return True, width, height, phash, None, json.dumps(exif_summary, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return False, None, None, None, f'{type(exc).__name__}: {exc}', None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def scan_collection(root: Path, near_threshold: int = 6) -> dict[str, Any]:
    incoming = root / 'incoming'
    metadata_path = root / 'metadata' / 'source_metadata.jsonl'
    supplied, metadata_errors = load_source_metadata(metadata_path)
    assets: list[dict[str, Any]] = []
    unsupported: list[dict[str, str]] = []
    id_counts: dict[str, int] = {}
    for path in sorted(incoming.rglob('*')):
        if not path.is_file():
            continue
        if path.name == '.gitkeep':
            continue
        relative = path.relative_to(incoming).as_posix()
        if path.suffix.lower() not in SUPPORTED:
            unsupported.append({'incoming_relative_path': relative, 'suffix': path.suffix.lower()})
            continue
        try:
            digest = sha256_file(path)
        except OSError as exc:
            unsupported.append({'incoming_relative_path': relative, 'suffix': path.suffix.lower(), 'error': str(exc)})
            continue
        asset_id = stable_asset_id(relative, digest)
        id_counts[asset_id] = id_counts.get(asset_id, 0) + 1
        class_dir = relative.split('/', 1)[0] if '/' in relative else ''
        declared_class = CLASS_DIRS.get(class_dir)
        readable, width, height, phash, read_error, exif_summary = image_metadata(path)
        supplied_row = supplied.get(asset_id, {})
        # Intake directory communicates a claimed label only; human review stays PENDING.
        row: dict[str, Any] = {field: None for field in METADATA_FIELDS}
        row.update(supplied_row)
        row.update({
            'schema_version': '1.0.0',
            'asset_id': asset_id,
            'original_filename': path.name,
            'class_label': supplied_row.get('class_label', declared_class),
            'image_sha256': digest,
            'perceptual_hash': phash,
            'image_width': width,
            'image_height': height,
            'review_status': supplied_row.get('review_status') or 'PENDING',
            'created_at': supplied_row.get('created_at') or utc_now(),
            'incoming_relative_path': relative,
            'originals_relative_path': None,
            'image_readable': readable,
            'asset_id_method': 'sha256(relative_path + exact_file_sha256), phase8c-v1',
            'exif_summary': exif_summary,
            '_read_error': read_error,
            '_absolute_path': str(path.resolve()),
        })
        missing = []
        for field in sorted(SOURCE_METADATA_FIELDS):
            if row.get(field) in (None, ''):
                missing.append(field)
        if not row.get('class_label'):
            missing.append('class_label')
        if not row.get('source_family_id'):
            missing.append('source_family_id (required before any future split)')
        row['metadata_missing_fields'] = ';'.join(missing)
        if row['review_status'] != 'PENDING':
            # Metadata can never approve an image through scanning.
            row['review_status'] = 'PENDING'
        assets.append(row)

    # Duplicate IDs can arise only if a path is repeated through malformed traversal;
    # keep every row visible and report the conflict rather than dropping one.
    duplicate_asset_ids = sorted(asset_id for asset_id, count in id_counts.items() if count > 1)
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for row in assets:
        by_hash.setdefault(str(row['image_sha256']), []).append(row)
    duplicate_groups = [group for group in by_hash.values() if len(group) > 1]
    pairs: list[dict[str, Any]] = []
    for group in duplicate_groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.append({'asset_id_a': group[i]['asset_id'], 'asset_id_b': group[j]['asset_id'],
                              'incoming_path_a': group[i]['incoming_relative_path'], 'incoming_path_b': group[j]['incoming_relative_path'],
                              'sha256_a': group[i]['image_sha256'], 'sha256_b': group[j]['image_sha256'],
                              'phash_distance': 0 if group[i]['perceptual_hash'] == group[j]['perceptual_hash'] else hamming_hex(group[i]['perceptual_hash'], group[j]['perceptual_hash']),
                              'match_type': 'EXACT_SHA256', 'review_status': 'PENDING'})
    for i, a in enumerate(assets):
        if not a.get('perceptual_hash'):
            continue
        for b in assets[i + 1:]:
            if not b.get('perceptual_hash') or a['image_sha256'] == b['image_sha256']:
                continue
            distance = hamming_hex(a['perceptual_hash'], b['perceptual_hash'])
            if 0 < distance <= near_threshold:
                pairs.append({'asset_id_a': a['asset_id'], 'asset_id_b': b['asset_id'],
                              'incoming_path_a': a['incoming_relative_path'], 'incoming_path_b': b['incoming_relative_path'],
                              'sha256_a': a['image_sha256'], 'sha256_b': b['image_sha256'],
                              'phash_distance': distance, 'match_type': 'PHASH_NEAR_DUPLICATE_CANDIDATE',
                              'review_status': 'PENDING'})

    public_assets = [{k: v for k, v in row.items() if not k.startswith('_')} for row in assets]
    write_csv(root / 'manifests' / 'intake_manifest.csv', public_assets, METADATA_FIELDS)
    json_path = root / 'manifests' / 'intake_manifest.json'
    json_path.write_text(json.dumps(public_assets, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_csv(root / 'review' / 'near_duplicate_review_queue.csv', pairs,
              ['asset_id_a', 'asset_id_b', 'incoming_path_a', 'incoming_path_b', 'sha256_a', 'sha256_b', 'phash_distance', 'match_type', 'review_status'])
    write_csv(root / 'reports' / 'unsupported_files.csv', unsupported, ['incoming_relative_path', 'suffix', 'error'])
    missing_rows = [{'asset_id': row['asset_id'], 'incoming_relative_path': row['incoming_relative_path'],
                     'missing_fields': row['metadata_missing_fields']} for row in public_assets if row['metadata_missing_fields']]
    write_csv(root / 'reports' / 'missing_metadata.csv', missing_rows, ['asset_id', 'incoming_relative_path', 'missing_fields'])
    review_rows = [{'asset_id': row['asset_id'], 'class_label': row['class_label'], 'review_status': 'PENDING',
                    'reviewer_id': None, 'reviewed_at': None, 'reason': None} for row in public_assets]
    write_csv(root / 'review' / 'review_template.csv', review_rows,
              ['asset_id', 'class_label', 'review_status', 'reviewer_id', 'reviewed_at', 'reason'])
    report = {
        'generated_at': utc_now(), 'scan_root': str(root.resolve()), 'incoming_asset_count': len(assets),
        'readable_image_count': sum(bool(row['image_readable']) for row in assets),
        'unreadable_image_count': sum(not row['image_readable'] for row in assets),
        'unsupported_file_count': len(unsupported), 'exact_duplicate_pair_count': sum(p['match_type'] == 'EXACT_SHA256' for p in pairs),
        'near_duplicate_candidate_pair_count': sum(p['match_type'] == 'PHASH_NEAR_DUPLICATE_CANDIDATE' for p in pairs),
        'duplicate_asset_ids': duplicate_asset_ids, 'missing_metadata_asset_count': len(missing_rows),
        'source_metadata_errors': metadata_errors, 'near_duplicate_hamming_threshold': near_threshold,
        'all_review_statuses_pending': all(row['review_status'] == 'PENDING' for row in assets),
        'split_assignments_created': False,
    }
    (root / 'reports' / 'latest_scan_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    source_family_issues = check_source_family_consistency(public_assets)
    report['source_family_consistency_issues'] = source_family_issues
    report['source_family_consistency_passed'] = not source_family_issues
    (root / 'reports' / 'latest_scan_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    completion = review_completion_report(root)
    (root / 'reports' / 'review_completion.json').write_text(json.dumps(completion, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def check_source_family_consistency(rows: list[dict[str, Any]]) -> list[str]:
    """Flag conflicting declarations; never infer or merge source families."""
    issues: list[str] = []
    for field in ('video_id', 'source_image_id', 'capture_session_id'):
        mapping: dict[str, set[str]] = {}
        for row in rows:
            value, family = row.get(field), row.get('source_family_id')
            if value and family:
                mapping.setdefault(str(value), set()).add(str(family))
        for value, families in mapping.items():
            if len(families) > 1:
                issues.append(f'{field} {value} maps to multiple source_family_id values: {sorted(families)}')
    return issues


def archive_originals(root: Path) -> dict[str, Any]:
    """Copy readable incoming originals into an immutable archive; never overwrite."""
    report_path = root / 'reports' / 'latest_scan_report.json'
    if not report_path.exists():
        raise RuntimeError('Run scan before archive.')
    manifest_path = root / 'manifests' / 'intake_manifest.json'
    rows = json.loads(manifest_path.read_text(encoding='utf-8'))
    archived_path = root / 'manifests' / 'archived_originals.csv'
    previous = read_csv_rows(archived_path)
    known = {row['asset_id']: row for row in previous}
    added = []
    for row in rows:
        if not row['image_readable']:
            continue
        rel = row['incoming_relative_path']
        source = root / 'incoming' / rel
        class_dir = row['class_label'] or 'unclassified'
        suffix = source.suffix.lower()
        dest_rel = Path(class_dir) / f"{row['asset_id']}{suffix}"
        dest = root / 'originals' / dest_rel
        if row['asset_id'] in known:
            existing = root / 'originals' / known[row['asset_id']]['originals_relative_path']
            if not existing.is_file() or sha256_file(existing) != row['image_sha256']:
                raise RuntimeError(f"Previously archived asset changed or is missing: {row['asset_id']}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with source.open('rb') as src, dest.open('xb') as out:
                shutil.copyfileobj(src, out)
            if sha256_file(dest) != row['image_sha256']:
                raise RuntimeError(f"SHA-256 changed while archiving {row['asset_id']}")
            dest.chmod(0o444)
        except FileExistsError as exc:
            raise RuntimeError(f'Refusing to overwrite existing original: {dest}') from exc
        item = {'asset_id': row['asset_id'], 'incoming_relative_path': rel,
                'originals_relative_path': dest_rel.as_posix(), 'image_sha256': row['image_sha256'],
                'archived_at': utc_now()}
        added.append(item)
    if added:
        write_csv(archived_path, previous + added,
                  ['asset_id', 'incoming_relative_path', 'originals_relative_path', 'image_sha256', 'archived_at'])
    return {'archived_count': len(added), 'archive_manifest': str(archived_path), 'overwrites': 0}


def record_review(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get('decision')
    if status not in REVIEW_STATUSES - {'PENDING'}:
        raise ValueError(f'decision must be a completed review status; got {status!r}')
    if not payload.get('asset_id') or not payload.get('reviewer_id'):
        raise ValueError('asset_id and reviewer_id are required')
    manifest_path = root / 'manifests' / 'intake_manifest.csv'
    if not manifest_path.is_file():
        raise ValueError('Run scan first; intake manifest is missing')
    known_assets = {row.get('asset_id') for row in read_csv_rows(manifest_path)}
    if payload['asset_id'] not in known_assets:
        raise ValueError(f"asset_id is not present in the current intake manifest: {payload['asset_id']}")
    if status in REVIEW_REASON_REQUIRED and not str(payload.get('reason') or '').strip():
        raise ValueError(f'reason is required for {status}')
    record = {field: payload.get(field) for field in REVIEW_FIELDS}
    record['review_id'] = record.get('review_id') or 'review_' + uuid.uuid4().hex
    record['reviewed_at'] = record.get('reviewed_at') or utc_now()
    required_review_fields = {'fruit_identity', 'peel_genuinely_green', 'image_suitable', 'fruit_visibility'}
    if any(field not in payload for field in required_review_fields) or not record.get('fruit_identity') or not record.get('fruit_visibility'):
        raise ValueError('review must include fruit_identity, peel_genuinely_green, image_suitable and fruit_visibility; unknown values may be null')
    path = root / 'review' / 'review_history.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    return record


def review_completion_report(root: Path) -> dict[str, Any]:
    rows = read_csv_rows(root / 'manifests' / 'intake_manifest.csv')
    history_path = root / 'review' / 'review_history.jsonl'
    events = []
    if history_path.exists():
        events = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        latest[event['asset_id']] = event
    counts: dict[str, int] = {}
    for asset in rows:
        decision = latest.get(asset['asset_id'], {}).get('decision', 'PENDING')
        counts[decision] = counts.get(decision, 0) + 1
    return {'asset_count': len(rows), 'review_event_count': len(events), 'reviewed_asset_count': sum(v for k, v in counts.items() if k != 'PENDING'),
            'pending_asset_count': counts.get('PENDING', 0), 'latest_decision_counts': counts,
            'all_assets_human_reviewed': len(rows) > 0 and counts.get('PENDING', 0) == 0}


def verify_archive(root: Path) -> dict[str, Any]:
    path = root / 'manifests' / 'archived_originals.csv'
    rows = read_csv_rows(path)
    failures = []
    for row in rows:
        asset_path = root / 'originals' / row['originals_relative_path']
        if not asset_path.is_file():
            failures.append({'asset_id': row['asset_id'], 'error': 'missing archived original'})
        elif sha256_file(asset_path) != row['image_sha256']:
            failures.append({'asset_id': row['asset_id'], 'error': 'SHA-256 mismatch'})
    return {'archive_record_count': len(rows), 'verified_count': len(rows) - len(failures),
            'failure_count': len(failures), 'failures': failures, 'passed': not failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE, help='phase8c_collection directory (default: this tool parent)')
    sub = parser.add_subparsers(dest='command', required=True)
    scan = sub.add_parser('scan', help='read incoming images and write manifests/reports')
    scan.add_argument('--near-threshold', type=int, default=6)
    sub.add_parser('archive', help='copy scanned readable incoming originals into immutable originals/')
    sub.add_parser('verify-archive', help='read-only SHA-256 integrity check for archived originals')
    review = sub.add_parser('record-review', help='append one human review event from a JSON file')
    review.add_argument('--record', type=Path, required=True)
    sub.add_parser('review-report', help='summarize latest human review status')
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    try:
        if args.command == 'scan':
            report = scan_collection(root, args.near_threshold)
        elif args.command == 'archive':
            report = archive_originals(root)
        elif args.command == 'record-review':
            report = record_review(root, json.loads(args.record.read_text(encoding='utf-8')))
        elif args.command == 'verify-archive':
            report = verify_archive(root)
        else:
            report = review_completion_report(root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
