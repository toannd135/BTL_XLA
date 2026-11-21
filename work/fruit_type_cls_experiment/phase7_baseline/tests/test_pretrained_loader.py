"""No-epoch tests for the Phase 7 verified local checkpoint loader."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'train_baseline.py'
CHECKPOINT = Path('/home/toannd/models/mobilenet/mobilenet_v3_large-5c1a4163.pth')
BLOCKED_RUN = Path(__file__).resolve().parents[2] / 'phase7_baseline'
SPEC = importlib.util.spec_from_file_location('phase7_train_baseline', SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def tree_digest(path):
    result={}
    for file in sorted(path.rglob('*')):
        if file.is_file():
            result[str(file.relative_to(path))]=hashlib.sha256(file.read_bytes()).hexdigest()
    return result


class VerifiedCheckpointLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT.is_file():
            raise unittest.SkipTest(f'approved checkpoint unavailable: {CHECKPOINT}')

    def test_correct_local_checkpoint_loads_strictly_and_matches_backbone(self):
        import torch
        original=torch.nn.Module.load_state_dict
        calls=[]
        def strict_spy(model,state_dict,*args,**kwargs):
            calls.append(kwargs.get('strict'))
            return original(model,state_dict,*args,**kwargs)
        with mock.patch.object(torch.nn.Module,'load_state_dict',strict_spy):
            model,verification=module.load_verified_pretrained_model(checkpoint_path=CHECKPOINT)
        self.assertEqual(calls,[True])
        self.assertTrue(verification['strict_load'])
        self.assertEqual(verification['missing_keys'],[])
        self.assertEqual(verification['unexpected_keys'],[])
        self.assertEqual(verification['checkpoint_sha256'],module.EXPECTED_PRETRAINED_SHA256)
        self.assertTrue(verification['all_backbone_tensors_match'])
        self.assertGreater(verification['backbone_tensors_compared'],0)
        self.assertTrue(verification['all_floating_parameters_finite'])
        self.assertEqual(model.classifier[3].out_features,3)

    def test_checkpoint_path_can_be_configured_by_environment(self):
        with mock.patch.dict(os.environ,{'PHASE7_PRETRAINED_WEIGHTS':str(CHECKPOINT)}):
            path,digest=module.resolve_pretrained_checkpoint()
        self.assertEqual(path,CHECKPOINT.resolve())
        self.assertEqual(digest,module.EXPECTED_PRETRAINED_SHA256)

    def test_wrong_sha_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong=Path(tmp)/'wrong.pth';wrong.write_bytes(CHECKPOINT.read_bytes()[:4096])
            with self.assertRaisesRegex(RuntimeError,'SHA-256 mismatch'):
                module.load_verified_pretrained_model(checkpoint_path=wrong)

    def test_missing_checkpoint_is_a_hard_failure(self):
        with self.assertRaisesRegex(RuntimeError,'missing or not a regular file'):
            module.load_verified_pretrained_model(checkpoint_path='/tmp/phase7-checkpoint-does-not-exist.pth')

    def test_corrupt_checkpoint_fails_safe_load(self):
        corrupt=b'not a torch checkpoint'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'corrupt.pth';path.write_bytes(corrupt)
            # Simulate the fixed expected digest matching a corrupt artifact so
            # this exercises weights_only deserialization, after hash validation.
            with mock.patch.object(module,'EXPECTED_PRETRAINED_SHA256',hashlib.sha256(corrupt).hexdigest()):
                with self.assertRaisesRegex(RuntimeError,'weights_only=True'):
                    module.load_verified_pretrained_model(checkpoint_path=path)

    def test_missing_state_key_fails_before_non_strict_load_is_possible(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'modified.pth'
            state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
            state.pop(next(iter(state)))
            torch.save(state,path)
            with mock.patch.object(module,'EXPECTED_PRETRAINED_SHA256',module.sha256(path)):
                with self.assertRaisesRegex(RuntimeError,'state_dict structure mismatch'):
                    module.load_verified_pretrained_model(checkpoint_path=path)

    def test_no_automatic_download_function_is_called(self):
        import torch
        import torchvision.models
        with mock.patch.object(torch.hub,'load_state_dict_from_url',side_effect=AssertionError('download attempted')):
            with mock.patch.object(torchvision.models._api,'load_state_dict_from_url',side_effect=AssertionError('torchvision download attempted')):
                with mock.patch.object(torchvision.models,'mobilenet_v3_large',wraps=torchvision.models.mobilenet_v3_large) as constructor:
                    # loader imports constructor from torchvision.models when called
                    model,verification=module.load_verified_pretrained_model(checkpoint_path=CHECKPOINT)
                    self.assertEqual(constructor.call_args.kwargs,{'weights':None})
                    self.assertEqual(model.classifier[3].out_features,3)

    def test_failed_checkpoint_validation_blocks_before_epoch_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'phase7-run'
            env=os.environ.copy();env['PHASE7_RUN_DIR']=str(run);env['PHASE7_PRETRAINED_WEIGHTS']=str(Path(tmp)/'missing.pth')
            completed=subprocess.run([sys.executable,str(SCRIPT)],env=env,text=True,capture_output=True,timeout=180)
            self.assertEqual(completed.returncode,3,completed.stdout+'\n'+completed.stderr)
            self.assertFalse((run/'logs'/'training_log.csv').exists())
            blocked=json.loads((run/'reports'/'pretrained_checkpoint_verification.json').read_text())
            self.assertEqual(blocked['status'],'TRAINING_BLOCKED')
            self.assertFalse(blocked['training_started'])
            self.assertIn('missing or not a regular file',blocked['reason'])

    def test_output_safety_rejects_original_and_non_empty_directories(self):
        with self.assertRaisesRegex(RuntimeError,'original Phase 7'):
            module.validate_run_directory(BLOCKED_RUN)
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'existing';run.mkdir();(run/'artifact.txt').write_text('keep')
            with self.assertRaisesRegex(RuntimeError,'non-empty'):
                module.validate_run_directory(run)

    def test_existing_blocked_artifacts_remain_unchanged(self):
        before=tree_digest(BLOCKED_RUN)
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'blocked-attempt'
            env=os.environ.copy();env['PHASE7_RUN_DIR']=str(run);env['PHASE7_PRETRAINED_WEIGHTS']=str(Path(tmp)/'missing.pth')
            subprocess.run([sys.executable,str(SCRIPT)],env=env,text=True,capture_output=True,timeout=180,check=False)
        self.assertEqual(tree_digest(BLOCKED_RUN),before)


if __name__=='__main__':
    unittest.main(verbosity=2)
