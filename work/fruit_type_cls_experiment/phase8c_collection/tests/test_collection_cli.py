"""Infrastructure tests use copies of existing source images, never edit originals."""
import csv
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

CLI_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'collection_cli.py'
SPEC = importlib.util.spec_from_file_location('phase8c_collection_cli', CLI_PATH)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)
REPO = Path(__file__).resolve().parents[5]
IMAGE_FIXTURE = REPO / 'fruit/raw/rf_v7/train/images/o_r069_png.rf.09e46ebd0725c16915dba94640b97920.jpg'
NEAR_FIXTURE = REPO / 'fruit/raw/rf_v7/valid/images/o_r093_png.rf.5f02696874d5f6ce4ce0aa073dcbd1a0.jpg'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CollectionInfrastructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not IMAGE_FIXTURE.is_file() or not NEAR_FIXTURE.is_file():
            raise unittest.SkipTest('existing read-only image fixtures are unavailable')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'phase8c_collection'
        (self.root / 'incoming/green_apple').mkdir(parents=True)
        (self.root / 'incoming/green_banana').mkdir(parents=True)
        for name in ['green_apple','green_banana','originals','metadata','manifests','review','reports','quarantine']:
            (self.root / name).mkdir(exist_ok=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_and_invalid_image_files_and_missing_metadata(self):
        shutil.copyfile(IMAGE_FIXTURE,self.root/'incoming/green_apple/good.jpg')
        (self.root/'incoming/green_apple/broken.jpg').write_bytes(b'not an image')
        (self.root/'incoming/green_apple/readme.txt').write_text('unsupported type')
        result=cli.scan_collection(self.root)
        self.assertEqual(result['incoming_asset_count'],2)
        self.assertEqual(result['readable_image_count'],1)
        self.assertEqual(result['unreadable_image_count'],1)
        self.assertEqual(result['unsupported_file_count'],1)
        rows=cli.read_csv_rows(self.root/'manifests/intake_manifest.csv')
        self.assertTrue(all(r['review_status']=='PENDING' for r in rows))
        self.assertTrue(all(r['source_family_id']=='' for r in rows))
        self.assertTrue(all(r['metadata_missing_fields'] for r in rows))
        self.assertTrue((self.root/'reports/missing_metadata.csv').is_file())
        self.assertEqual(result['split_assignments_created'],False)

    def test_exact_duplicates_are_reported_without_deletion(self):
        a=self.root/'incoming/green_apple/a.jpg';b=self.root/'incoming/green_banana/b.jpg'
        shutil.copyfile(IMAGE_FIXTURE,a);shutil.copyfile(IMAGE_FIXTURE,b)
        before=(digest(a),digest(b))
        result=cli.scan_collection(self.root)
        self.assertEqual(result['exact_duplicate_pair_count'],1)
        queue=cli.read_csv_rows(self.root/'review/near_duplicate_review_queue.csv')
        self.assertEqual(queue[0]['match_type'],'EXACT_SHA256')
        self.assertEqual((digest(a),digest(b)),before)
        self.assertTrue(a.exists() and b.exists())

    def test_near_duplicate_candidates_use_existing_images_and_are_not_auto_approved(self):
        a=self.root/'incoming/green_apple/source_a.jpg';b=self.root/'incoming/green_apple/source_b.jpg'
        shutil.copyfile(IMAGE_FIXTURE,a);shutil.copyfile(NEAR_FIXTURE,b)
        result=cli.scan_collection(self.root,near_threshold=6)
        queue=cli.read_csv_rows(self.root/'review/near_duplicate_review_queue.csv')
        near=[r for r in queue if r['match_type']=='PHASH_NEAR_DUPLICATE_CANDIDATE']
        self.assertGreaterEqual(len(near),1)
        self.assertTrue(all(r['review_status']=='PENDING' for r in near))
        self.assertGreaterEqual(result['near_duplicate_candidate_pair_count'],1)

    def test_stable_id_generation_and_duplicate_metadata_ids(self):
        sha='a'*64
        first=cli.stable_asset_id('green_apple/a.jpg',sha)
        self.assertEqual(first,cli.stable_asset_id('green_apple/a.jpg',sha))
        self.assertNotEqual(first,cli.stable_asset_id('green_apple/b.jpg',sha))
        p=self.root/'metadata/source_metadata.jsonl'
        p.write_text('{"asset_id":"asset_duplicate"}\n{"asset_id":"asset_duplicate"}\n')
        _,errors=cli.load_source_metadata(p)
        self.assertTrue(any('duplicate asset_id' in e for e in errors))

    def test_duplicate_asset_ids_are_reported_without_dropping_rows(self):
        shutil.copyfile(IMAGE_FIXTURE,self.root/'incoming/green_apple/a.jpg')
        shutil.copyfile(NEAR_FIXTURE,self.root/'incoming/green_apple/b.jpg')
        with mock.patch.object(cli,'stable_asset_id',return_value='asset_'+'b'*24):
            result=cli.scan_collection(self.root)
        self.assertEqual(result['duplicate_asset_ids'],['asset_'+'b'*24])
        rows=cli.read_csv_rows(self.root/'manifests/intake_manifest.csv')
        self.assertEqual(len(rows),2)

    def test_scan_never_approves_from_metadata(self):
        source=self.root/'incoming/green_apple/a.jpg';shutil.copyfile(IMAGE_FIXTURE,source)
        aid=cli.stable_asset_id('green_apple/a.jpg',digest(source))
        (self.root/'metadata/source_metadata.jsonl').write_text(json.dumps({'asset_id':aid,'review_status':'VERIFIED_CORRECT','green_status':'VERIFIED_GREEN'})+'\n')
        cli.scan_collection(self.root)
        row=cli.read_csv_rows(self.root/'manifests/intake_manifest.csv')[0]
        self.assertEqual(row['review_status'],'PENDING')
        self.assertEqual(row['green_status'],'VERIFIED_GREEN')

    def test_append_only_review_history_and_reason_requirement(self):
        aid='asset_'+'a'*24
        cli.write_csv(self.root/'manifests/intake_manifest.csv',[{'asset_id':aid}],['asset_id'])
        record={'asset_id':aid,'reviewer_id':'reviewer1','decision':'NEEDS_PROVENANCE','reason':'Source family not documented',
                'fruit_identity':'Unclear','peel_genuinely_green':None,'image_suitable':True,'fruit_visibility':'unclear',
                'occlusion':'unclear','background_dominance':'unclear'}
        with self.assertRaisesRegex(ValueError,'reason is required'):
            cli.record_review(self.root,{**record,'reason':''})
        cli.record_review(self.root,record)
        cli.record_review(self.root,{**record,'decision':'AMBIGUOUS','reason':'Fruit identity needs second review'})
        lines=(self.root/'review/review_history.jsonl').read_text().splitlines()
        self.assertEqual(len(lines),2)
        self.assertEqual(json.loads(lines[0])['decision'],'NEEDS_PROVENANCE')
        self.assertEqual(json.loads(lines[1])['decision'],'AMBIGUOUS')

    def test_archive_is_copy_only_and_sha_verifiable(self):
        incoming=self.root/'incoming/green_apple/archive_me.jpg';shutil.copyfile(IMAGE_FIXTURE,incoming)
        source_hash=digest(incoming)
        cli.scan_collection(self.root)
        result=cli.archive_originals(self.root)
        archive_rel=cli.read_csv_rows(self.root/'manifests/archived_originals.csv')[0]['originals_relative_path']
        archived=self.root/'originals'/archive_rel
        self.assertEqual(result['archived_count'],1)
        self.assertEqual(digest(archived),source_hash)
        self.assertEqual(digest(incoming),source_hash)
        self.assertFalse(archived.stat().st_mode & 0o222)
        self.assertTrue(cli.verify_archive(self.root)['passed'])
        # Repeating archive for the same asset verifies and skips without overwrite.
        self.assertEqual(cli.archive_originals(self.root)['archived_count'],0)
        self.assertEqual(digest(archived),source_hash)

    def test_source_family_consistency_flags_multiple_families_per_video(self):
        rows=[{'video_id':'v1','source_image_id':'frame1','capture_session_id':'s1','source_family_id':'fam_a'},
              {'video_id':'v1','source_image_id':'frame2','capture_session_id':'s1','source_family_id':'fam_b'}]
        issues=cli.check_source_family_consistency(rows)
        self.assertTrue(any('video_id v1' in item for item in issues))
        self.assertTrue(any('capture_session_id s1' in item for item in issues))

    def test_original_project_image_is_unchanged(self):
        before=digest(IMAGE_FIXTURE)
        shutil.copyfile(IMAGE_FIXTURE,self.root/'incoming/green_apple/fixture.jpg')
        cli.scan_collection(self.root)
        cli.archive_originals(self.root)
        self.assertEqual(digest(IMAGE_FIXTURE),before)


if __name__=='__main__':
    unittest.main(verbosity=2)
