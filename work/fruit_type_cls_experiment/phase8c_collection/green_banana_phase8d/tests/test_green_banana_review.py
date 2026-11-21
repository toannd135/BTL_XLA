import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
spec=importlib.util.spec_from_file_location('gbreview',TOOLS/'green_banana_review.py')
gb=importlib.util.module_from_spec(spec);spec.loader.exec_module(gb)
progress_spec=importlib.util.spec_from_file_location('gbprogress',TOOLS/'progress_report.py')
progress=importlib.util.module_from_spec(progress_spec);progress_spec.loader.exec_module(progress)

class GreenBananaReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name)
        self.collection=root/'phase8c_collection';self.phase8d=self.collection/'green_banana_phase8d'
        for d in [self.collection/'manifests',self.collection/'metadata',self.collection/'review',self.collection/'incoming',self.phase8d/'metadata',self.phase8d/'review']:d.mkdir(parents=True,exist_ok=True)
        (self.collection/'incoming/green_banana').mkdir(parents=True)
        (self.collection/'incoming/green_banana/sample.png').write_bytes(b'test')
        self.aid='asset_'+'c'*24
        self.fields=['asset_id','image_width','image_height','source_family_id','source_image_id','image_sha256','image_readable','class_label','source_provider','source_license','source_type','source_url','video_id','capture_session_id','incoming_relative_path']
        with (self.collection/'manifests/intake_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=self.fields);w.writeheader();w.writerow({'asset_id':self.aid,'image_width':640,'image_height':480,'source_family_id':'fam_1','source_image_id':'img_1','image_sha256':'a'*64,'image_readable':'True','class_label':'Banana','source_provider':'self_capture','source_license':'owned','incoming_relative_path':'green_banana/sample.png'})
        gb.COLLECTION=self.collection;gb.INCOMING=(self.collection/'incoming').resolve();gb.PHASE8D=self.phase8d
        progress.COLLECTION=self.collection;progress.PHASE8D=self.phase8d;progress.EXPERIMENT=self.collection.parent
    def tearDown(self):self.tmp.cleanup()
    def review(self,**extra):
        p={'asset_id':self.aid,'reviewer_id':'reviewer_1','decision':'VERIFIED_CORRECT','source_image_viewed':True,'target_crop_viewed':True,'target_region_xyxy_px':[10,10,300,300],'fruit_identity':'Banana','peel_genuinely_green':True,'peel_color_confidence':'CLEAR_GREEN','image_suitable':True,'fruit_visibility':'whole','occlusion':'minor','background_dominance':'low','reason':None,'notes':None};p.update(extra);return p
    def add_second_asset(self):
        other='asset_'+'d'*24
        (self.collection/'incoming/green_banana/other.png').write_bytes(b'test2')
        with (self.collection/'manifests/intake_manifest.csv').open('a',encoding='utf-8',newline='') as f:
            csv.DictWriter(f,fieldnames=self.fields).writerow({'asset_id':other,'image_width':640,'image_height':480,'source_family_id':'fam_1','source_image_id':'img_2','image_sha256':'b'*64,'image_readable':'True','class_label':'Banana','source_provider':'self_capture','source_license':'owned','incoming_relative_path':'green_banana/other.png'})
        return other
    def test_verified_requires_permission_and_valid_roi(self):
        with self.assertRaisesRegex(ValueError,'permission evidence'):gb.record_review(self.review())
        gb.record_permission({'asset_id':self.aid,'permission_status':'VERIFIED','evidence_reference':'consent:record-42','verified_by':'rights-reviewer'})
        result=gb.record_review(self.review());self.assertEqual(result['decision'],'VERIFIED_CORRECT')
        self.assertTrue(result['source_image_viewed'] and result['target_crop_viewed'])
        self.assertEqual(len((self.phase8d/'review/review_history.jsonl').read_text().splitlines()),1)
    def test_invalid_review_evidence_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'viewing both'):gb.record_review(self.review(target_crop_viewed=False))
        with self.assertRaisesRegex(ValueError,'outside image dimensions'):gb.record_review(self.review(target_region_xyxy_px=[0,0,900,100]))
        with self.assertRaisesRegex(ValueError,'clearly green'):gb.record_review(self.review(peel_genuinely_green=None,peel_color_confidence='UNCERTAIN'))
    def test_nonapproved_needs_reason(self):
        with self.assertRaisesRegex(ValueError,'reason is required'):gb.record_review(self.review(decision='AMBIGUOUS',reason=None))
    def test_permission_requires_evidence_when_verified(self):
        with self.assertRaisesRegex(ValueError,'requires an evidence_reference'):gb.record_permission({'asset_id':self.aid,'permission_status':'VERIFIED','verified_by':'reviewer'})
    def test_duplicate_pair_review_requires_reason_and_both_images(self):
        other=self.add_second_asset();payload={'asset_id_a':self.aid,'asset_id_b':other,'decision':'SAME_FAMILY','reviewer_id':'reviewer_1','reason':'Both are views of the same documented fruit.','both_images_viewed':True}
        with self.assertRaisesRegex(ValueError,'reviewer_id and reason'):gb.record_pair_review({**payload,'reason':''})
        with self.assertRaisesRegex(ValueError,'both complete images'):gb.record_pair_review({**payload,'both_images_viewed':False})
        first=gb.record_pair_review(payload);second=gb.record_pair_review({**payload,'decision':'NEEDS_PROVENANCE','reason':'Later evidence was inconclusive.'})
        history=self.phase8d/'review/duplicate_pair_history.jsonl';self.assertEqual(first['pair_id'],second['pair_id']);self.assertEqual(len(history.read_text().splitlines()),2)
    def test_review_queue_excludes_green_apple_assets(self):
        apple='asset_'+'e'*24;(self.collection/'incoming/green_apple').mkdir();(self.collection/'incoming/green_apple/apple.jpg').write_bytes(b'test')
        with (self.collection/'manifests/intake_manifest.csv').open('a',encoding='utf-8',newline='') as f:
            csv.DictWriter(f,fieldnames=self.fields).writerow({'asset_id':apple,'image_width':640,'image_height':480,'source_family_id':'fam_2','source_image_id':'img_apple','image_sha256':'d'*64,'image_readable':'True','class_label':'Apple','source_provider':'self_capture','source_license':'owned','incoming_relative_path':'green_apple/apple.jpg'})
        self.assertEqual([row['asset_id'] for row in gb.review_assets()],[self.aid])
        with self.assertRaisesRegex(ValueError,'not a Green Banana'):gb.record_review({'asset_id':apple})
    def test_pair_api_marks_ai_pair_decisions_unreviewed(self):
        other=self.add_second_asset();pairid='pair_'+__import__('hashlib').sha256('|'.join(sorted([self.aid,other])).encode()).hexdigest()[:24]
        with (self.collection/'review/near_duplicate_review_queue.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['asset_id_a','asset_id_b','incoming_path_a','incoming_path_b','phash_distance']);w.writeheader();w.writerow({'asset_id_a':self.aid,'asset_id_b':other,'incoming_path_a':'green_banana/sample.png','incoming_path_b':'green_banana/other.png','phash_distance':4})
        gb.append_jsonl(self.phase8d/'review/duplicate_pair_history.jsonl',{'asset_id_a':self.aid,'asset_id_b':other,'pair_id':pairid,'decision':'DISTINCT','reviewer_id':'codex_visual_triage_ai','both_images_viewed':True})
        rows=gb.pair_assets();self.assertEqual(len(rows),1);self.assertFalse(rows[0]['reviewed'])
    def test_progress_separates_zero_phase8d_from_legacy(self):
        report=progress.make_report();self.assertEqual(report['status'],'DATA_COLLECTION_IN_PROGRESS')
        self.assertEqual(report['intake']['file_count'],1);self.assertEqual(report['current_phase8d_ready_counts']['assigned_splits'],False)

if __name__=='__main__':unittest.main(verbosity=2)
