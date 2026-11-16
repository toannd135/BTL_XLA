"""Verify package integrity, split safety, class exposure and runner CPU guard."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
package = json.loads((HERE / "package_summary.json").read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    archive_path = Path(package["zip"])
    assert digest(archive_path.read_bytes()) == package["sha256"]
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        members = archive.namelist()
        assert len(members) == len(set(members))
        assert all(not Path(p).is_absolute() and ".." not in Path(p).parts for p in members)
        assert json.loads(archive.read("manifest.json")) == manifest
        for record in manifest["files"]:
            assert digest(archive.read(record["image"])) == record["image_sha256"]
            assert digest(archive.read(record["label"])) == record["label_sha256"]
        for model in manifest["models"].values():
            assert digest(archive.read(model["path"])) == model["sha256"]
        train = [r for r in manifest["files"] if r["split"] == "train"]
        evaluation = [r for r in manifest["files"] if r["split"] != "train"]
        assert not {r["image_sha256"] for r in train} & {r["image_sha256"] for r in evaluation}
        assert not {r["group"] for r in train} & {r.get("group") for r in evaluation}
        labels = {r["image"]: r for r in manifest["files"]}
        exposures = Counter()
        train_list = [ln.removeprefix("./") for ln in archive.read("train.txt").decode().splitlines() if ln]
        assert len(train_list) == manifest["summary"]["train_entries"]
        for path in train_list:
            assert path in members and labels[path]["split"] == "train"
            exposures.update({int(k): v for k, v in labels[path]["class_counts"].items()})
        assert dict(exposures) == {int(k): v for k, v in manifest["summary"]["weighted_total_boxes"].items()}
        assert max(exposures.values()) / min(exposures.values()) < 1.5
        for split in ("val", "test", "neg_test"):
            expected = {r["image"] for r in manifest["files"] if r["split"] == split}
            actual = {ln.removeprefix("./") for ln in archive.read(split + ".txt").decode().splitlines() if ln}
            assert expected == actual
        for name in ("run_all3.py", "metrics_all3.py"):
            assert archive.read(name) == (HERE / name).read_bytes(), name
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    notebook = json.loads((HERE / "finetune_all3_v2_colab.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            if "EXPECTED_SHA256 =" in source:
                assert package["sha256"] in source
    # Actual inference API integration, with exact truth predictions plus a duplicate.
    # No image dataset inference, no model loading, no training.
    import numpy as np
    import torch
    from types import SimpleNamespace
    from metrics_all3 import evaluate_counts
    from ultralytics.utils.ops import xyxyxyxy2xywhr
    from PIL import Image
    import tempfile
    with tempfile.TemporaryDirectory(dir=HERE, prefix="metric_check_") as temp:
        root = Path(temp)
        Image.new("RGB", (100, 100)).save(root / "sample.png")
        corners = [[10, 10], [30, 10], [30, 30], [10, 30]]
        (root / "sample.txt").write_text("1 .1 .1 .3 .1 .3 .3 .1 .3\n")
        pts = np.asarray([corners, corners], dtype=np.float32)
        pred = SimpleNamespace(cls=torch.tensor([1., 1.]), conf=torch.tensor([.9, .8]),
                               xywhr=torch.from_numpy(xyxyxyxy2xywhr(pts)), xyxyxyxy=torch.from_numpy(pts))
        class FakeModel:
            def predict(self, *args, **kwargs):
                return [SimpleNamespace(obb=pred)]
        records = [{"image": "sample.png", "label": "sample.txt", "source": "synthetic_unit_case"}]
        result = evaluate_counts(FakeModel(), root, records, 1280)
        assert result["rows"][0]["tp"] == [0, 1, 0]
        assert result["rows"][0]["fp"] == [0, 1, 0]
        assert result["summary"]["count_MAE_all_classes"] == 1
    assert not torch.cuda.is_available(), "CPU guard check expects this laptop's CPU-only environment"
    guard_project = HERE / "cpu_guard_must_not_create"
    assert not guard_project.exists()
    run = subprocess.run([sys.executable, str(HERE / "run_all3.py"), "train", "--root", str(HERE),
                          "--project", str(guard_project)], capture_output=True, text=True, timeout=45)
    assert run.returncode != 0 and "Select T4 GPU" in run.stderr
    assert not guard_project.exists()
    assert digest((REPO / "exports/det_obb_s.pt").read_bytes()) == manifest["models"]["baseline"]["sha256"]
    report = {"archive_CRC_and_all_file_hashes": "passed", "no_duplicate_archive_members": "passed",
              "train_eval_exact_hash_and_source_group_checks": "passed", "all_three_class_exposures": dict(exposures),
              "matching_duplicate_and_wrong_class_checks": "passed (check_metrics.py)",
              "float32_OBB_count_integration": "passed", "notebook_code_and_package_hash": "passed",
              "runner_CPU_training_guard": "passed", "app_baseline_unchanged": True,
              "training_executed_locally": False, "GPU_Colab_execution_not_yet_tested": True}
    (HERE / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
