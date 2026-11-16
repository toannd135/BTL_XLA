"""Build a reproducible all-class experiment without changing original data."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OLD = REPO / "out/accuracy_trial_20261116"
NAMES = {0: "banana", 1: "apple", 2: "orange"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_group(source):
    stem = Path(source).stem
    if stem.startswith("own-") and "__" in stem:
        return stem.split("__")[0]
    return re.sub(r"(_(?:jpg|jpeg|png|webp))?\.rf\.[a-z0-9]+$", "", stem, flags=re.I)


def label_counts(text):
    count = Counter()
    for line in text.splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 9 or values[0] not in NAMES or not all(math.isfinite(v) and 0 <= v <= 1 for v in values[1:]):
            raise ValueError("Invalid normalized OBB label")
        count[int(values[0])] += 1
    return count


def main():
    destination = HERE / "fruit_all3_v2_20261116.zip"
    if destination.exists():
        raise FileExistsError("Preserve the existing input version")
    old_manifest = json.loads((OLD / "input_manifest.json").read_text(encoding="utf-8"))
    old_records = old_manifest["files"]
    val = [dict(r, group=source_group(r["source"])) for r in old_records if r["split"] == "val"]
    test = []
    for image in sorted((REPO / "work/obb/data/real/test/images").glob("*")):
        if not image.is_file():
            continue
        label = image.parent.parent / "labels" / (image.stem + ".txt")
        text = label.read_text(encoding="utf-8-sig")
        alias = "test_" + hashlib.sha256(image.relative_to(REPO).as_posix().encode()).hexdigest()[:24]
        test.append({"split": "test", "image": f"images/test/{alias}{image.suffix.lower()}",
                     "label": f"labels/test/{alias}.txt", "source": image.relative_to(REPO).as_posix(),
                     "image_sha256": sha(image), "label_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "class_counts": dict(label_counts(text)), "group": source_group(image.name), "origin": "legacy_test"})
    eval_groups = {r["group"] for r in val + test}
    eval_hashes = {r["image_sha256"] for r in val + test}
    positives, negatives, synthetic, excluded = [], [], [], []
    for original in old_records:
        if original["split"] != "train":
            continue
        r = dict(original, group=source_group(original["source"]))
        name = Path(r["source"]).name
        reason = None
        if r["image_sha256"] in eval_hashes or r["group"] in eval_groups:
            reason = "source group or exact image also in legacy validation/test; exclude from this new fine-tune"
        elif "/synth/" in r["source"]:
            synthetic.append(r)
            continue
        elif r["class_counts"]:
            if name.startswith("rf_v7__") and not name.startswith(("rf_v7__a_", "rf_v7__b_", "rf_v7__Screen")):
                reason = "uncertain positive labels from distractor-source groups; quarantine without relabeling"
            else:
                positives.append(r)
                continue
        elif name.startswith("neg_"):
            negatives.append(r)
            continue
        else:
            reason = "empty legacy labels outside curated negative pool; quarantine without declaring background"
        excluded.append({"source": r["source"], "reason": reason})

    real_counts = Counter()
    for r in positives:
        real_counts.update({int(k): v for k, v in r["class_counts"].items()})
    assert set(real_counts) == {0, 1, 2}
    # Deterministic image repetitions capped at 3; all three target classes participate.
    target = max(real_counts.values())
    class_repeat = {k: min(3., target / real_counts[k]) for k in NAMES}
    rng = random.Random(42)
    train_entries, weighted_counts = [], Counter()
    for r in positives:
        weight = max(class_repeat[int(k)] for k in r["class_counts"])
        repeat = math.floor(weight) + int(rng.random() < weight % 1)
        if r["origin"] == "submitted_review_geometry_padded":
            repeat = max(3, repeat)
        r["repetitions"] = repeat
        train_entries.extend(["./" + r["image"]] * repeat)
        weighted_counts.update({int(k): v * repeat for k, v in r["class_counts"].items()})
    # At most one synthetic annotation per two real annotation exposures in EACH class.
    budget = {k: max(1, int(weighted_counts[k] * .5)) for k in NAMES}
    synth_counts, chosen_synth = Counter(), []
    rng.shuffle(synthetic)
    for r in synthetic:
        count = {int(k): v for k, v in r["class_counts"].items()}
        if len(chosen_synth) < 200 and all(synth_counts[k] + count.get(k, 0) <= budget[k] for k in NAMES):
            r["repetitions"] = 1
            chosen_synth.append(r)
            synth_counts.update(count)
            train_entries.append("./" + r["image"])
    rng.shuffle(negatives)
    chosen_neg = negatives[:min(len(negatives), max(40, round(len(train_entries) / 9)))]
    for r in chosen_neg:
        r["repetitions"] = 1
        train_entries.append("./" + r["image"])
    rng.shuffle(train_entries)
    selected = positives + chosen_synth + chosen_neg
    assert not {r["group"] for r in selected} & eval_groups
    assert not {r["image_sha256"] for r in selected} & eval_hashes
    records = selected + val + test

    neg_test = []
    train_hashes = {r["image_sha256"] for r in selected}
    for image in sorted((REPO / "work/exp/neg/test/images").glob("*")):
        if not image.is_file():
            continue
        digest = sha(image)
        assert digest not in train_hashes
        alias = "neg_" + hashlib.sha256(image.relative_to(REPO).as_posix().encode()).hexdigest()[:24]
        neg_test.append({"split": "neg_test", "image": f"images/neg_test/{alias}{image.suffix.lower()}",
                         "label": f"labels/neg_test/{alias}.txt", "source": image.relative_to(REPO).as_posix(),
                         "image_sha256": digest, "label_sha256": hashlib.sha256(b"").hexdigest(),
                         "class_counts": {}, "origin": "legacy_negative_regression"})
    assert len(neg_test) == 71
    records += neg_test
    model_paths = {"baseline": REPO / "exports/det_obb_s.pt", "trial_v1": REPO / "exports/det_obb_trial_v1.pt"}
    assert sha(model_paths["baseline"]) == old_manifest["baseline_model_sha256"]
    assert sha(model_paths["trial_v1"]) == "750cadadaf9b8c80354d7c22aba332794300f75b54d295bc71a02faf79a83095"
    summary = {"unique_real_positive_images": len(positives), "unique_synthetic_images": len(chosen_synth),
               "unique_negative_train_images": len(chosen_neg), "train_entries": len(train_entries),
               "class_names": NAMES, "class_repetitions": class_repeat,
               "weighted_real_boxes": dict(weighted_counts), "synthetic_boxes": dict(synth_counts),
               "weighted_total_boxes": dict(weighted_counts + synth_counts),
               "validation_images": len(val), "legacy_test_images": len(test), "negative_test_images": len(neg_test),
               "excluded_images_by_reason": dict(Counter(r["reason"] for r in excluded))}
    manifest = {"schema": 2, "recipe": "all3_balanced_v2", "seed": 42, "summary": summary,
                "models": {name: {"path": f"models/{name}.pt", "sha256": sha(path)} for name, path in model_paths.items()},
                "files": records, "excluded": excluded,
                "limitations": [
                    "Existing real/synthetic labels are inherited, not newly human-certified; automatic quarantine is conservative.",
                    "Legacy validation/test have historical bunch labels. Supplementary human-reviewed validation is banana-only.",
                    "Own-video source overlaps are excluded from new training, but initialization models previously saw legacy data. This is not an independent final benchmark.",
                    "71 negative images are an existing regression subset and can overlap legacy test backgrounds; do not add sample counts or tune on them.",
                    "No newly verified grape negatives are available in this package; reduced grape confusion is not guaranteed.",
                    "Raw OBB detector evaluation only. CLIP/teacher/freshness and CPU app latency need separate application checks.",
                ]}
    names = "names:\n  0: banana\n  1: apple\n  2: orange\n"
    with zipfile.ZipFile(OLD / "fruit_accuracy_trial_20261116.zip") as old_zip, zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for i, r in enumerate(records):
            if r["split"] in ("train", "val"):
                image_data, label_data = old_zip.read(r["image"]), old_zip.read(r["label"])
            else:
                image_data = (REPO / r["source"]).read_bytes()
                label_data = b"" if r["split"] == "neg_test" else (REPO / r["source"]).parent.parent.joinpath("labels", Path(r["source"]).stem + ".txt").read_text(encoding="utf-8-sig").encode()
            assert hashlib.sha256(image_data).hexdigest() == r["image_sha256"]
            assert hashlib.sha256(label_data).hexdigest() == r["label_sha256"]
            assert dict(label_counts(label_data.decode())) == {int(k): v for k, v in r["class_counts"].items()}
            archive.writestr(r["image"], image_data)
            archive.writestr(r["label"], label_data)
        archive.writestr("data.yaml", "train: train.txt\nval: val.txt\ntest: test.txt\n" + names)
        archive.writestr("train.txt", "\n".join(train_entries) + "\n")
        for split in ("val", "test", "neg_test"):
            archive.writestr(split + ".txt", "\n".join("./" + r["image"] for r in records if r["split"] == split) + "\n")
        for name, path in model_paths.items():
            archive.write(path, f"models/{name}.pt")
        for name in ("run_all3.py", "metrics_all3.py"):
            archive.write(HERE / name, name)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    package = {"zip": str(destination), "bytes": destination.stat().st_size, "sha256": sha(destination), "summary": summary}
    (HERE / "package_summary.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    print(json.dumps(package, indent=2), flush=True)


if __name__ == "__main__":
    main()
