# Controlled human annotation audit protocol

## Purpose and scope

This protocol prepares a human review of the frozen 345-image fruit segmentation benchmark. It is intended to distinguish missing/incomplete or incorrect annotations from detector false positives, missed fruit, instance split/merge, and mask geometry errors.

Preparation is not annotation correction. No inference, training, threshold change, prediction edit, evaluator change, or annotation modification is part of this workflow. Candidate predictions remain NO-GO; review is performed against the raw baseline detector outputs and original GT.

## Frozen inputs and integrity

- Benchmark config canonical SHA256: `65f552c3b25f545ab20643d986dd0777e8f7d30b482e889d4953ae223b0b53b4`.
- Test manifest: 345 entries, SHA256 `9b40bfb089e8697347fa1c34842b58c7046eda2f5c5757777cd275fc758fddc1`.
- Matching is the existing class-agnostic one-to-one Hungarian matching at mask IoU >= 0.50.
- All 345 image hashes and all 345 original annotation hashes were rechecked against the frozen manifest. The exact original GT SHA256 is copied into each review item as `original_gt_sha256`.
- The original baseline/candidate prediction JSONLs, inference event log and execution summary were hashed before preparation and rechecked unchanged afterward. They remain in the original full-run directory.

`GT-original` refers to labels at `work/obb/seg_data_candidate/test/labels/` under the frozen manifest. These labels must remain read-only. `work/benchmarks/candidate_eval/annotation_review/GT-reviewed/` is the reserved separate destination for a future reviewed version; it currently contains no annotation files.

## Queue contents and priority

`work/benchmarks/candidate_eval/annotation_review/review_manifest.json` contains one record for every manifest image, including images with no flagged mismatch. It records image and annotation paths, both source hashes, GT/prediction classes and counts, matched/unmatched totals, unmatched instance details, suspected issues, rationale, priority and human review status.

| Priority | Images | Meaning |
|---|---:|---|
| P0 | 5 | Required focus: `mbk__orange_76.jpg`, `mbk__apple_94.jpg`, `mbk__banana_10.jpg`, and both negative images with false detections. |
| P1 | 16 | Images containing one or more of the 34 bbox-high/mask-IoU-low geometry candidates, except P0 images which remain P0. |
| P2 | 75 | Other images with unmatched predictions or GT instances. |
| P3 | 249 | Context-only images included for exhaustive manifest coverage; no unmatched instance was flagged. |

There are **96 unique images requiring review**, covering all unmatched prediction and GT instances. The 34 mask-geometry candidates comprise 16 unmatched predictions and 18 unmatched GT instances; they are preserved individually in each image record. The automated issue labels are triage signals only. `uncertain` is appropriate when the image and annotation do not support a reliable diagnosis.

## Human workflow

1. Open the source image and original annotation together. Use the image and existing GT as evidence; do not treat a model prediction as proof that a fruit exists.
2. Apply one consistent instance policy for partial occlusion, touching fruit and individual fruits in bunches. Record ambiguity rather than guessing.
3. Inspect each unmatched instance, including its saved mask, box, class and match overlaps. Decide whether the primary issue is missing/incomplete annotation, incorrect annotation, false positive, missed fruit, split, merge, mask geometry or uncertain.
4. Append a decision to `review_decisions.json`. Each record must include reviewer, image ID, original GT hash, primary decision, status, explanation and ISO-8601 timestamp with timezone. `corrected_instance_count` may be filled only for a confirmed correction decision; otherwise leave it null.
5. Use the statuses exactly as follows: `pending` means not reviewed; `confirmed` means the stated diagnosis is supported by visual evidence; `needs_correction` flags an annotation that requires a later, separately authorized edit; `uncertain` preserves unresolved ambiguity. A confirmed diagnosis does not itself modify GT.
6. Do not overwrite earlier decisions to hide disagreements. Add separate reviewer entries and resolve disagreement explicitly.
7. If annotation correction is authorized later, write only to `GT-reviewed/` with the same relative paths as the manifest. Record the original hash, reviewed label hash, reviewer, correction rationale and timestamp in a separate correction ledger. Never replace `GT-original`.

## Separate reviewed-GT evaluation

After a future authorization to produce reviewed labels:

1. Freeze a reviewed manifest that points to the reviewed label paths and identifies the review decision/correction ledger version.
2. Verify the image hashes still match the frozen manifest and calculate SHA256 for every reviewed label.
3. Reuse the persisted baseline and candidate predictions from `work/benchmarks/candidate_eval/full_runs/fresh_clip_swap_only_345/records/`. Do not run inference.
4. Run the existing evaluator with the original config and unchanged class-agnostic matching threshold (mask IoU >= 0.50). Do not change matching, confidence, NMS, postprocessing, or prediction records.
5. Publish reviewed-GT metrics beside, not over, original-GT results. Label both annotation versions and hashes clearly; preserve the original report and records.

## Stop conditions

Pause a review item if its source hash does not match the manifest, image and label dimensions/paths do not correspond, or the reviewer cannot decide instance policy. Do not repair the original files in place. Record the issue as `uncertain` and escalate it for adjudication.

## Validation performed for this preparation

- 345 manifest images and 345 label files represented in the review manifest.
- Every review item maps to a frozen manifest entry; image and label hashes match.
- Required focus images, both negative images and all 34 geometry candidates are present.
- All review statuses are initially `pending`; the decision log is empty.
- Original annotation hashes and frozen prediction/event artifacts were unchanged after preparation.
- No inference, training, or annotation correction was run.
