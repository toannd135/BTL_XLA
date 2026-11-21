#!/usr/bin/env python3
"""Import only original Unripe images from the official BananaID release.

The archive is kept intact. Images remain byte-identical. Filename indices are
grouped across four view directions as source-family IDs because the dataset
paper documents four views of each individual banana. Labels remain pending
human review; "Unripe" is not treated as proof that the peel is green.
"""
import argparse
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PHASE8D = Path(__file__).resolve().parents[1]
COLLECTION = PHASE8D.parent
ARCHIVE = PHASE8D / 'source_archives/BananaID_ripeness_original.zip'
DEST = COLLECTION / 'incoming/green_banana/BananaID_ripeness_original'
SOURCE_URL = 'https://data.mendeley.com/datasets/h6n5srjjyw/1'
ATTRIBUTION = ('Mutrofin et al. (2025), BananaID: An image dataset of banana varieties '
               'and ripeness stages in Indonesia designed for classification, Mendeley Data v1, '
               'DOI: 10.17632/h6n5srjjyw.1.')
FILENAME = re.compile(r'^(Ambon|Cavendish|Raja|Saba|Mas)_Unripe_(Bottom|Left|Right|Top)_(\d{4})\.jpg$')


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def import_images(archive: Path = ARCHIVE) -> dict:
    if not archive.is_file():
        raise RuntimeError(f'Official original-image component is missing: {archive}')
    sys.path.insert(0, str(COLLECTION / 'tools'))
    from collection_cli import stable_asset_id

    with zipfile.ZipFile(archive) as zf:
        selected = []
        for info in zf.infolist():
            m = FILENAME.fullmatch(Path(info.filename).name)
            if '/Unripe/' in info.filename and m:
                selected.append((info, m))
        if len(selected) != 408:
            raise RuntimeError(f'Expected 408 original Unripe views, found {len(selected)}')
        groups: dict[str, set[str]] = {}
        for _, m in selected:
            variety, side, index = m.groups()
            groups.setdefault(f'{variety}_{index}', set()).add(side)
        if len(groups) != 102 or any(sides != {'Bottom', 'Left', 'Right', 'Top'} for sides in groups.values()):
            raise RuntimeError('Unripe source-family pattern failed: expected 102 groups with four unique views each')

        dest_files = {Path(info.filename).name: info for info, _ in selected}
        DEST.mkdir(parents=True, exist_ok=True)
        for name in dest_files:
            target = DEST / name
            if target.exists():
                if target.read_bytes() != zf.read(dest_files[name]):
                    raise RuntimeError(f'Refusing to overwrite conflicting image: {target}')

        metadata_path = COLLECTION / 'metadata/source_metadata.jsonl'
        existing = []
        if metadata_path.exists():
            existing = [json.loads(line) for line in metadata_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        known = {r.get('asset_id') for r in existing}
        new_records = []
        for info, m in selected:
            original_name = Path(info.filename).name
            variety, side, index = m.groups()
            blob = zf.read(info)
            target = DEST / original_name
            if not target.exists():
                target.write_bytes(blob)
            relative = target.relative_to(COLLECTION / 'incoming').as_posix()
            digest = sha256(blob)
            asset_id = stable_asset_id(relative, digest)
            if asset_id in known:
                continue
            rec = {
                'schema_version': '1.0.0', 'asset_id': asset_id,
                'original_filename': original_name, 'class_label': 'Banana',
                'green_status': None, 'source_type': 'photo', 'source_url': SOURCE_URL,
                'source_license': 'CC BY 4.0', 'source_provider': 'BananaID authors via Mendeley Data v1',
                'collection_date': None,
                'source_image_id': f'BananaID:v1:{variety}:Unripe:{side}:{index}',
                'source_family_id': f'BananaID:v1:{variety}:{index}',
                'capture_session_id': None, 'video_id': None, 'frame_id': None,
                'camera_device': None, 'image_sha256': digest, 'image_width': None,
                'image_height': None, 'reviewer_id': None, 'review_status': 'PENDING',
                'review_reason': None, 'created_at': utc_now(),
                'incoming_relative_path': relative,
                'asset_id_method': 'phase8c-v1:relative_path+sha256',
                'exif_summary': None,
            }
            new_records.append(rec)
            known.add(asset_id)
        if new_records:
            with metadata_path.open('a', encoding='utf-8') as stream:
                for rec in new_records:
                    stream.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + '\n')

        provenance = {
            'dataset': 'BananaID: An image dataset of banana varieties and ripeness stages in Indonesia designed for classification',
            'version': 1, 'doi': '10.17632/h6n5srjjyw.1', 'official_source': SOURCE_URL,
            'license': 'CC BY 4.0', 'attribution_required': ATTRIBUTION,
            'archive_component': str(archive.resolve()),
            'archive_component_sha256': sha256(archive.read_bytes()),
            'extracted_class': 'Unripe', 'raw_images_extracted': 408,
            'source_family_rule': 'BananaID:v1:<variety>:<index>; released names share a four-digit index across Bottom/Left/Right/Top. The paper documents four sides photographed per banana. This links views of the same physical banana; it does not verify green peel or human label correctness.',
            'source_family_count': 102, 'views_per_family': 4,
            'family_evidence': 'Filename index + four view directions + published acquisition protocol; see official Mendeley dataset and source paper.',
            'augmentation_images_used': 0, 'green_peel_human_reviews': 0,
            'split_assignment_performed': False,
        }
        prov_path = PHASE8D / 'source_archives/BananaID_provenance.json'
        prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return {'selected_original_images': len(selected), 'source_families': len(groups),
                'views_per_family': 4, 'metadata_records_appended': len(new_records),
                'human_approvals': 0, 'splits_assigned': False,
                'archive_component_sha256': provenance['archive_component_sha256'],
                'destination': str(DEST.resolve())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=ARCHIVE)
    args = p.parse_args()
    try:
        print(json.dumps(import_images(args.archive), ensure_ascii=False, indent=2))
    except Exception as exc:
        raise SystemExit(f'BLOCKED: {type(exc).__name__}: {exc}') from exc


if __name__ == '__main__':
    main()
