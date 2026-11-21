# Detection error analysis — fresh 345-image benchmark

**Scope:** Forensic analysis of persisted baseline predictions, candidate records, GT records and source images/labels. No inference, training, threshold changes or annotation edits were performed. Candidate remains NO-GO; analysis uses raw detector baseline.

## Provenance and integrity

- Canonical config SHA256: `65f552c3b25f545ab20643d986dd0777e8f7d30b482e889d4953ae223b0b53b4`; raw JSON SHA256 `6bcad0b2bf331af07623993eaf8bbdb21c0c69eaf10a7cdef545954067b2c487`.
- Detector SHA256: `6da28310bb7627d00dfa574cc36147d215afb48e4d0c3d0705e62ecb05917982`.
- CLIP checkpoint SHA256: `5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b`.
- Manifest SHA256: `9b40bfb089e8697347fa1c34842b58c7046eda2f5c5757777cd275fc758fddc1` (345 entries).
- Recomputed hashes verified for all 345 image files and 345 label files; per-image records matched their source hashes.
- Matching recomputed from persisted masks with pinned class-agnostic one-to-one Hungarian maximum eligible IoU, threshold 0.50.

## Aggregate findings

345 images contain 349 GT instances and 364 raw detector predictions: 268 matched, 96 unmatched predictions and 81 unmatched GT. Baseline total-count MAE is 0.2348; exact count accuracy is 82.90%. Matching ignores class; class is compared only after matching.

### Per-class results

| Class | GT | Pred | Matched by GT class | Unmatched GT | Unmatched pred by predicted class | Precision | Recall | F1 | Count MAE | Exact count accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| banana | 156 | 158 | 109 | 47 | 49 | 0.684 | 0.692 | 0.688 | 0.128 | 90.1% |
| apple | 98 | 103 | 79 | 19 | 25 | 0.738 | 0.776 | 0.756 | 0.067 | 95.1% |
| orange | 95 | 103 | 80 | 15 | 22 | 0.767 | 0.832 | 0.798 | 0.075 | 93.9% |

Precision/recall/F1 are class-aware counts after class-agnostic matching: a wrong-class match contributes FP and FN to the respective classes. Banana contributes the most unmatched instances: 49 predicted banana instances plus 47 banana GT misses (96 of 177 total unmatched).

### Geometric triage of unmatched instances

| Arm | Triage label | Count | Meaning |
|---|---|---:|---|
| FP | `suspiciously_large_mask` | 20 | Unmatched prediction mask covers at least 20% of image; review flag only. |
| FP | `uncertain` | 49 | No cause identifiable from saved geometry alone. |
| FP | `possible_merge` | 5 | Prediction overlaps multiple GT masks; their union covers at least 60% of prediction mask. Candidate merge signature only. |
| FP | `possible_duplicate` | 6 | Unmatched prediction overlaps a matched prediction with mask IoU at least 0.50; may also be adjacent fruit. |
| FP | `poor_mask_geometry_candidate` | 16 | BBox overlap is at least 0.50 but mask IoU is below matching threshold. |
| FN | `uncertain` | 58 | No cause identifiable from saved geometry alone. |
| FN | `poor_mask_geometry_candidate` | 18 | BBox overlap is at least 0.50 but mask IoU is below matching threshold. |
| FN | `possible_split` | 5 | Multiple predictions overlap one GT and union covers at least 60% of GT mask; candidate split signature only. |

These are reproducible geometric triage counts, not confirmed semantic causes. Partial occlusion, small/distant fruit, annotation completeness and fruit visibility cannot be assigned reliably to every error without visual review. Per-instance IoUs, box overlap, mask areas and overlap links are in `error_analysis.json`.

### Image-level error rankings

Rankings below are top 8; JSON contains top 30 for each requested ordering.

#### Most unmatched predictions

| Image | GT | Pred | Matched | FP | FN | Abs count error | Overlap pairs | Large masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `images/mbk__orange_76.jpg` | 2 | 8 | 2 | 6 | 0 | 6 | 0 | 2 |
| `images/negative__14a8727b42ec4e6f.jpg` | 0 | 4 | 0 | 4 | 0 | 4 | 0 | 0 |
| `images/mbk__apple_94.jpg` | 2 | 5 | 2 | 3 | 0 | 3 | 0 | 1 |
| `images/mbk__banana_87.jpg` | 4 | 5 | 2 | 3 | 2 | 1 | 0 | 1 |
| `images/rf_v7__b_f068_png.rf.531d3a5896b7189fddd9b97ecd4c1245.jpg` | 1 | 4 | 1 | 3 | 0 | 3 | 0 | 1 |
| `images/mbk__apple_86.jpg` | 2 | 4 | 2 | 2 | 0 | 2 | 0 | 0 |
| `images/mbk__banana_35.jpg` | 1 | 3 | 1 | 2 | 0 | 2 | 0 | 0 |
| `images/mbk__banana_4.jpg` | 2 | 4 | 2 | 2 | 0 | 2 | 0 | 0 |

#### Most unmatched ground truths

| Image | GT | Pred | Matched | FP | FN | Abs count error | Overlap pairs | Large masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `images/mbk__banana_10.jpg` | 4 | 1 | 0 | 1 | 4 | 3 | 0 | 1 |
| `images/mbk__banana_17.jpg` | 4 | 1 | 1 | 0 | 3 | 3 | 0 | 0 |
| `images/mbk__banana_30.jpg` | 3 | 1 | 0 | 1 | 3 | 2 | 0 | 1 |
| `images/mbk__banana_54.jpg` | 3 | 1 | 0 | 1 | 3 | 2 | 0 | 1 |
| `images/mbk__banana_32.jpg` | 4 | 3 | 2 | 1 | 2 | 1 | 0 | 1 |
| `images/mbk__banana_50.jpg` | 4 | 2 | 2 | 0 | 2 | 2 | 0 | 0 |
| `images/mbk__banana_87.jpg` | 4 | 5 | 2 | 3 | 2 | 1 | 0 | 1 |
| `images/rf_v7__b_r050_png.rf.9ffec6e630bea1e04bd907b2e0360e4b.jpg` | 3 | 2 | 1 | 1 | 2 | 1 | 0 | 0 |

#### Largest absolute count errors

| Image | GT | Pred | Matched | FP | FN | Abs count error | Overlap pairs | Large masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `images/mbk__orange_76.jpg` | 2 | 8 | 2 | 6 | 0 | 6 | 0 | 2 |
| `images/negative__14a8727b42ec4e6f.jpg` | 0 | 4 | 0 | 4 | 0 | 4 | 0 | 0 |
| `images/mbk__apple_94.jpg` | 2 | 5 | 2 | 3 | 0 | 3 | 0 | 1 |
| `images/mbk__banana_10.jpg` | 4 | 1 | 0 | 1 | 4 | 3 | 0 | 1 |
| `images/mbk__banana_17.jpg` | 4 | 1 | 1 | 0 | 3 | 3 | 0 | 0 |
| `images/rf_v7__b_f068_png.rf.531d3a5896b7189fddd9b97ecd4c1245.jpg` | 1 | 4 | 1 | 3 | 0 | 3 | 0 | 1 |
| `images/mbk__apple_86.jpg` | 2 | 4 | 2 | 2 | 0 | 2 | 0 | 0 |
| `images/mbk__banana_30.jpg` | 3 | 1 | 0 | 1 | 3 | 2 | 0 | 1 |

#### Most overlapping predictions, mask IoU >=0.50

| Image | GT | Pred | Matched | FP | FN | Abs count error | Overlap pairs | Large masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `images/mbk__banana_11.jpg` | 1 | 2 | 1 | 1 | 0 | 1 | 1 | 0 |
| `images/mbk__banana_89.jpg` | 3 | 4 | 3 | 1 | 0 | 1 | 1 | 0 |
| `images/mbk__orange_92.jpg` | 1 | 2 | 1 | 1 | 0 | 1 | 1 | 2 |
| `images/rf_v7__a_f052_png.rf.5e6c760f4891ee4c7845ebb60bc8908e.jpg` | 1 | 2 | 1 | 1 | 0 | 1 | 1 | 2 |
| `images/rf_v7__b_f007_png.rf.157b26fbe0929125f41367255eb4073f.jpg` | 1 | 2 | 1 | 1 | 0 | 1 | 1 | 2 |
| `images/rf_v7__b_f007_png.rf.8acaf250d7d6c1435e537154f71f251c.jpg` | 1 | 2 | 0 | 2 | 1 | 1 | 1 | 2 |
| `images/rf_v7__b_r051_png.rf.57d64a1019f8182da991d820e6be7ae2.jpg` | 2 | 2 | 1 | 1 | 1 | 0 | 1 | 1 |
| `images/mbk__apple_1.jpg` | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 1 |

#### Most suspiciously large masks

| Image | GT | Pred | Matched | FP | FN | Abs count error | Overlap pairs | Large masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `images/mbk__orange_18.jpg` | 1 | 2 | 1 | 1 | 0 | 1 | 0 | 2 |
| `images/mbk__orange_33.jpg` | 1 | 2 | 1 | 1 | 0 | 1 | 0 | 2 |
| `images/mbk__orange_63.jpg` | 1 | 2 | 1 | 1 | 0 | 1 | 0 | 2 |
| `images/mbk__orange_76.jpg` | 2 | 8 | 2 | 6 | 0 | 6 | 0 | 2 |
| `images/mbk__orange_92.jpg` | 1 | 2 | 1 | 1 | 0 | 1 | 1 | 2 |
| `images/rf_v7__a_f052_png.rf.5e6c760f4891ee4c7845ebb60bc8908e.jpg` | 1 | 2 | 1 | 1 | 0 | 1 | 1 | 2 |
| `images/rf_v7__b_f007_png.rf.157b26fbe0929125f41367255eb4073f.jpg` | 1 | 2 | 1 | 1 | 0 | 1 | 1 | 2 |
| `images/rf_v7__b_f007_png.rf.8acaf250d7d6c1435e537154f71f251c.jpg` | 1 | 2 | 0 | 2 | 1 | 1 | 1 | 2 |

### Negative images

| Image | FP count | Predicted classes | Confidence | Visual inspection |
|---|---:|---|---|---|
| `images/negative__0522908ff4e4e487.jpg` | 1 | apple | 0.617 |  |
| `images/negative__14a8727b42ec4e6f.jpg` | 4 | apple, apple, apple, orange | 0.553, 0.475, 0.455, 0.404 |  |

Full-resolution inspection found both negative photos show peppers/other produce at predicted regions, not target apple/orange instances. These five predictions are strong false-positive/class-confusion cases.

## Prioritized annotation review

The CSV manifest lists all 96 images with unmatched instances, ranked by unmatched count, count error, overlap pairs and large-mask flags. It includes hashes, counts, suspected issue, rationale and confidence.

| Rank | Image | GT | Pred | Matched | FP | FN | Suspected cause and evidence | Confidence |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | `images/mbk__orange_76.jpg` | 2 | 8 | 2 | 6 | 0 | 6 unmatched prediction(s), 0 unmatched GT; geometric triage: uncertain:5; suspiciously_large_mask:1. | low |
| 2 | `images/mbk__banana_10.jpg` | 4 | 1 | 0 | 1 | 4 | 1 unmatched prediction(s), 4 unmatched GT; geometric triage: poor_mask_geometry_candidate:2; uncertain:2; possible_merge:1. | medium |
| 3 | `images/negative__14a8727b42ec4e6f.jpg` | 0 | 4 | 0 | 4 | 0 | 4 unmatched prediction(s), 0 unmatched GT; geometric triage: uncertain:4. | low |
| 4 | `images/mbk__apple_94.jpg` | 2 | 5 | 2 | 3 | 0 | 3 unmatched prediction(s), 0 unmatched GT; geometric triage: uncertain:3. | low |
| 5 | `images/rf_v7__b_f068_png.rf.531d3a5896b7189fddd9b97ecd4c1245.jpg` | 1 | 4 | 1 | 3 | 0 | 3 unmatched prediction(s), 0 unmatched GT; geometric triage: uncertain:3. | low |
| 6 | `images/mbk__banana_17.jpg` | 4 | 1 | 1 | 0 | 3 | 0 unmatched prediction(s), 3 unmatched GT; geometric triage: uncertain:3. | low |
| 7 | `images/mbk__banana_30.jpg` | 3 | 1 | 0 | 1 | 3 | 1 unmatched prediction(s), 3 unmatched GT; geometric triage: uncertain:4. | low |
| 8 | `images/mbk__banana_54.jpg` | 3 | 1 | 0 | 1 | 3 | 1 unmatched prediction(s), 3 unmatched GT; geometric triage: uncertain:3; possible_merge:1. | medium |
| 9 | `images/mbk__banana_87.jpg` | 4 | 5 | 2 | 3 | 2 | 3 unmatched prediction(s), 2 unmatched GT; geometric triage: possible_merge:2; poor_mask_geometry_candidate:2; uncertain:1. | medium |
| 10 | `images/rf_v7__b_f007_png.rf.8acaf250d7d6c1435e537154f71f251c.jpg` | 1 | 2 | 0 | 2 | 1 | 2 unmatched prediction(s), 1 unmatched GT; geometric triage: suspiciously_large_mask:2; uncertain:1. | low |
| 11 | `images/mbk__apple_86.jpg` | 2 | 4 | 2 | 2 | 0 | 2 unmatched prediction(s), 0 unmatched GT; geometric triage: uncertain:2. | low |
| 12 | `images/mbk__banana_32.jpg` | 4 | 3 | 2 | 1 | 2 | 1 unmatched prediction(s), 2 unmatched GT; geometric triage: uncertain:2; suspiciously_large_mask:1. | low |
| 13 | `images/mbk__banana_35.jpg` | 1 | 3 | 1 | 2 | 0 | 2 unmatched prediction(s), 0 unmatched GT; geometric triage: uncertain:2. | low |
| 14 | `images/mbk__banana_4.jpg` | 2 | 4 | 2 | 2 | 0 | 2 unmatched prediction(s), 0 unmatched GT; geometric triage: uncertain:2. | low |
| 15 | `images/mbk__banana_50.jpg` | 4 | 2 | 2 | 0 | 2 | 0 unmatched prediction(s), 2 unmatched GT; geometric triage: uncertain:2. | low |
| 16 | `images/mbk__orange_92.jpg` | 1 | 2 | 1 | 1 | 0 | 1 unmatched prediction(s), 0 unmatched GT; geometric triage: possible_duplicate:1. | medium |
| 17 | `images/rf_v7__a_f052_png.rf.5e6c760f4891ee4c7845ebb60bc8908e.jpg` | 1 | 2 | 1 | 1 | 0 | 1 unmatched prediction(s), 0 unmatched GT; geometric triage: possible_duplicate:1. | medium |
| 18 | `images/rf_v7__b_f007_png.rf.157b26fbe0929125f41367255eb4073f.jpg` | 1 | 2 | 1 | 1 | 0 | 1 unmatched prediction(s), 0 unmatched GT; geometric triage: possible_duplicate:1. | medium |
| 19 | `images/rf_v7__b_f042_png.rf.0c1b6ac8672ea917aa237a15b252f07c.jpg` | 1 | 2 | 0 | 2 | 1 | 2 unmatched prediction(s), 1 unmatched GT; geometric triage: uncertain:3. | low |
| 20 | `images/rf_v7__b_f063_png.rf.8fb030190023b57789df99f29f678411.jpg` | 1 | 2 | 0 | 2 | 1 | 2 unmatched prediction(s), 1 unmatched GT; geometric triage: poor_mask_geometry_candidate:2; possible_split:1. | medium |

Full-resolution visual spot checks:

- `images/mbk__orange_76.jpg`: GT=2, predictions=8. Numerous oranges are visible; unmatched predictions may be true fruits omitted from GT rather than detector FPs. Manual instance mapping needed.
- `images/mbk__apple_94.jpg`: GT=2, predictions=5. Several apples are visible; unmatched predictions plausibly indicate incomplete GT.
- `images/mbk__banana_10.jpg`: GT=4, prediction=1, no eligible match. The bunch contains multiple distinct banana bodies; under-segmentation/instance separation is plausible.
- Negative `images/negative__14a8727b42ec4e6f.jpg`: 4 predictions land on bell peppers/other produce.
- Negative `images/negative__0522908ff4e4e487.jpg`: one apple prediction lands on a bell pepper.

These five full-resolution images were inspected; the remaining image-level diagnoses are geometric and uncertain.

## Root-cause assessment

- **Annotation quality:** significant concern. The two inspected high-overcount fruit images appear to have fewer GT annotations than visible target fruit. Thus the aggregate 96 unmatched predictions are not equivalent to 96 confirmed false detections.
- **Detector recall / instance separation:** also material: 81 unmatched GT instances, with banana contributing 47. Banana_10 is a clear review candidate for merging/under-segmentation.
- **False positives:** five strong examples on two negative produce images, where peppers are classified as apple/orange.
- **Segmentation geometry:** 34 unmatched instances have high bbox overlap but sub-threshold mask IoU. Twenty unmatched prediction masks exceed 20% of image area. These need review; they do not prove whether GT or predicted masks are wrong.
- **Instance separation:** geometric triage flags 6 duplicate, 5 merge and 5 split candidates. Do not treat those as confirmed without visual inspection.
- **Class confusion:** visible in negative examples; baseline class-aware F1 is lowest for banana (0.688), then apple (0.756), then orange (0.798).

**Conclusion:** evidence supports a combination of likely GT incompleteness, detector recall/instance separation issues, and fruit-like-vegetable false positives. It cannot apportion all unmatched objects among causes. Annotation quality is a major confound; detector misses are independently evidenced by unmatched GT.

## Recommended next experiment

Before training or changing detector settings, conduct a blinded manual review from the prioritized manifest. First map visible objects in high-count images and inspect both negative images; set a consistent policy for partially occluded fruit and individual bananas in bunches. Keep predictions fixed, record GT omissions/invalid labels separately, then recompute metrics under the frozen evaluator. This separates annotation error from detector FN/FP and geometry errors.

## Traceability

- Per-image matches and all 177 unmatched instance records: `work/benchmarks/candidate_eval/error_analysis.json`.
- Prioritized manual-review manifest: `work/benchmarks/candidate_eval/manual_review_manifest.csv`.
- Source prediction and GT records are retained under `work/benchmarks/candidate_eval/full_runs/fresh_clip_swap_only_345/records/`.
- No inference/training was run and no annotations were changed.
