"""Safety and provenance tests for exporting individually reviewed fruit labels."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from instance_review import build, prepare, sha256, validate_boxes


class InstanceReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        for role in ("train", "val", "test"):
            folder = self.source / "local_dataset/det" / role
            (folder / "images").mkdir(parents=True)
            (folder / "labels").mkdir()
            Image.new("RGB", (80, 60), {"train": "red", "val": "green", "test": "blue"}[role]).save(folder / "images" / (role + ".jpg"))
            (folder / "labels" / (role + ".txt")).write_text("0 .2 .5 .2 .4\n1 .5 .5 .2 .4\n2 .8 .5 .2 .4\n")
        self.diagnostic = self.root / "diagnostic"
        self.diagnostic.mkdir()
        Image.new("RGB", (80, 60), "yellow").save(self.diagnostic / "probe.jpg")
        self.bundle = self.root / "review"
        with contextlib.redirect_stdout(io.StringIO()):
            self.manifest = prepare(self.source, self.bundle, self.diagnostic)
        self.reviews = {"schema": 1, "dataset_id": self.manifest["dataset_id"], "reviews": [
            {"id": item["id"], "status": "approved", "expected_count": len(item["boxes"]), "boxes": item["boxes"]}
            for item in self.manifest["items"]]}
        self.review_file = self.root / "reviews.json"

    def compile(self):
        self.review_file.write_text(json.dumps(self.reviews))
        with contextlib.redirect_stdout(io.StringIO()):
            return build(self.bundle, self.review_file, self.root / "clean")

    def test_split_preservation_diagnostic_exclusion_and_original_integrity(self):
        original = {p: sha256(p) for p in self.source.rglob("*") if p.is_file()}
        report = self.compile()
        for role in ("train", "val", "test"):
            self.assertEqual(report["splits"][role]["images"], 1)
            self.assertEqual(len(list((self.root / "clean" / role / "labels").glob("*.txt"))), 1)
        self.assertNotIn("diagnostic", [p["role"] for p in report["provenance"]])
        self.assertEqual(original, {p: sha256(p) for p in original})
        self.assertNotIn("path:", (self.root / "clean/data.yaml").read_text())

    def test_pending_and_excluded_images_are_not_exported_as_negatives(self):
        for status in ("pending", "excluded"):
            test_id = next(x["id"] for x in self.manifest["items"] if x["role"] == "test")
            next(r for r in self.reviews["reviews"] if r["id"] == test_id)["status"] = status
            self.review_file.write_text(json.dumps(self.reviews))
            with contextlib.redirect_stdout(io.StringIO()):
                report = build(self.bundle, self.review_file, self.root / ("clean_"+status))
            self.assertEqual(report["splits"]["test"]["images"], 0)

    def test_wrong_count_rejected_before_output_is_created(self):
        train_id = next(x["id"] for x in self.manifest["items"] if x["role"] == "train")
        next(r for r in self.reviews["reviews"] if r["id"] == train_id)["expected_count"] = 999
        with self.assertRaisesRegex(ValueError, "count"):
            self.compile()
        self.assertFalse((self.root / "clean").exists())

    def test_changed_dataset_or_image_is_rejected(self):
        self.reviews["dataset_id"] = "another-dataset"
        with self.assertRaisesRegex(ValueError, "another dataset"):
            self.compile()
        self.reviews["dataset_id"] = self.manifest["dataset_id"]
        item = next(x for x in self.manifest["items"] if x["role"] == "train")
        (self.bundle / item["image"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.compile()

    def test_out_of_bounds_duplicate_and_non_finite_boxes_rejected(self):
        for boxes in [[[0, .1, .1, .5, .5]], [[0, float("nan"), .5, .2, .2]],
                      [[0, .5, .5, 0, .2]], [[True, .5, .5, .2, .2]],
                      [[0, .5, .5, .2, .2], [0, .5, .5, .2, .2]]]:
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                validate_boxes(boxes)

    def test_manual_zero_is_allowed_as_a_reviewed_background(self):
        # Keep the three positive classes elsewhere in train and val.
        test_id = next(x["id"] for x in self.manifest["items"] if x["role"] == "test")
        review = next(r for r in self.reviews["reviews"] if r["id"] == test_id)
        review.update(boxes=[], expected_count=0)
        report = self.compile()
        self.assertEqual(report["splits"]["test"]["backgrounds"], 1)
        label = next((self.root / "clean/test/labels").glob("*.txt"))
        self.assertEqual(label.read_text(), "")

    def test_banana_only_review_cannot_create_a_three_class_training_recipe(self):
        for review in self.reviews["reviews"]:
            review["boxes"] = [box for box in review["boxes"] if box[0] == 0]
            review["expected_count"] = len(review["boxes"])
        with self.assertRaisesRegex(ValueError, "all three classes"):
            self.compile()

    def test_source_group_cannot_cross_train_val(self):
        path = self.bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        for item in manifest["items"]:
            if item["role"] in {"train", "val"}:
                item["source_group"] = "same-scene"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "leakage"):
            self.compile()


if __name__ == "__main__":
    unittest.main()
