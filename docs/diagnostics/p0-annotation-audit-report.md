# P0 annotation audit — frozen fruit benchmark

## Executive summary

Reviewed the five P0 images using original images/annotations and the persisted baseline and candidate prediction records. Generated native-resolution comparison images with separate original, GT, baseline, candidate and combined overlay panels. Matching was recomputed class-agnostically with the frozen Hungarian mask IoU >= 0.50 method.

| Image | GT | Baseline pred | Candidate pred | Matched | Unmatched GT | Unmatched pred | Human status |
|---|---:|---:|---:|---:|---:|---:|---|
| `mbk__orange_76.jpg` | 2 | 8 | 8 | 2 | 0 | 6 | `pending` |
| `mbk__apple_94.jpg` | 2 | 5 | 5 | 2 | 0 | 3 | `pending` |
| `mbk__banana_10.jpg` | 4 | 1 | 1 | 0 | 4 | 1 | `pending` |
| `negative__0522908ff4e4e487.jpg` | 0 | 1 | 1 | 0 | 0 | 1 | `pending` |
| `negative__14a8727b42ec4e6f.jpg` | 0 | 4 | 4 | 0 | 0 | 4 | `pending` |

**Triage totals:** 9 unmatched predictions are plausible missing-annotation candidates across the orange and apple images; 5 detections on the two negative images are strong suspected false positives; 4 GT instances in the banana image remain unmatched, alongside one broad unmatched prediction. Those banana items are consistent with under-segmentation/merge or mask geometry error, but need a human call. The remaining **14 unmatched cases in positive images** are unresolved and are preserved as uncertain. No reviewer decision is populated.

CLIP did not swap the class for any P0 prediction. Baseline and candidate counts and geometry are identical in these examples. This review does not establish a corrected GT or automatically classify unmatched fruit.

## Visual evidence and interpretation

Each visualization preserves the source panel dimensions and adds titles/labels. Blue is GT; orange is prediction. `M` means matched; `U` means unmatched. Baseline and candidate panels are separate. The combined panel shows GT and candidate mask boundaries together. Instance labels include IDs and classes. Paths are relative to the repository.

## mbk__orange_76.jpg

- Image ID/path: `images/mbk__orange_76.jpg`
- Annotation path: `labels/mbk__orange_76.txt`
- Original annotation SHA256: `b8f9723c0a4783758f3fbf51bfe45af5741a7934058832be6e4698da5004e883`
- Counts: GT 2; baseline 8; candidate 8; matched 2; unmatched GT 0; unmatched predictions 6.
- Suspected issues: possible incomplete GT annotation; unmatched predictions may correspond to visible oranges; partial overlap/occlusion may make object count ambiguous.
- Visual evidence: [`mbk__orange_76_comparison.png`](/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/work/benchmarks/candidate_eval/annotation_review/p0_visualizations/mbk__orange_76_comparison.png)
- Dimensions: source panels 1620×1080 px, not resized; comparison canvas 8172×1176 px.
- Recommended human action: Compare source image and original annotation; inspect unmatched instance boundaries and instance-count policy. Record a human decision only after review.
- Review status: `pending`; reviewer decision and explanation are intentionally empty.

### Per-instance findings

The two GT oranges are matched at mask IoU **0.942** and **0.883**. Six unmatched predictions are on visible orange-colored fruit surfaces, including edge-clipped/partly occluded fruit. They are plausible annotation-omission candidates, not established false positives. The image contains touching/overlapping oranges, so the human reviewer should determine the instance policy and whether all visible fruit count. No unmatched prediction pair was flagged as a duplicate by the saved overlap check.

- GT `gt_0000` (orange): matched to `det_0001`; class-agnostic mask IoU 0.942. Candidate class: orange.
- GT `gt_0001` (orange): matched to `det_0007`; class-agnostic mask IoU 0.883. Candidate class: orange.
- Prediction `det_0000`: baseline **orange**, candidate **orange**, confidence 0.958; bbox `[x=0.1–317.3, y=4.7–779.5]`; unmatched prediction; best GT IoU 0.000. CLIP swapped: False.
- Prediction `det_0001`: baseline **orange**, candidate **orange**, confidence 0.912; bbox `[x=630.4–1434.4, y=0.0–732.9]`; matched; best GT IoU 0.942. CLIP swapped: False.
- Prediction `det_0002`: baseline **orange**, candidate **orange**, confidence 0.877; bbox `[x=1336.6–1619.9, y=97.2–832.9]`; unmatched prediction; best GT IoU 0.000. CLIP swapped: False.
- Prediction `det_0003`: baseline **orange**, candidate **orange**, confidence 0.818; bbox `[x=157.7–986.8, y=311.8–1074.7]`; unmatched prediction; best GT IoU 0.000. CLIP swapped: False.
- Prediction `det_0004`: baseline **orange**, candidate **orange**, confidence 0.607; bbox `[x=0.0–63.2, y=883.8–1075.1]`; unmatched prediction; best GT IoU 0.000. CLIP swapped: False.
- Prediction `det_0005`: baseline **orange**, candidate **orange**, confidence 0.607; bbox `[x=1222.3–1617.1, y=0.0–177.1]`; unmatched prediction; best GT IoU 0.006. CLIP swapped: False.
- Prediction `det_0006`: baseline **orange**, candidate **orange**, confidence 0.586; bbox `[x=55.9–823.5, y=0.0–192.2]`; unmatched prediction; best GT IoU 0.005. CLIP swapped: False.
- Prediction `det_0007`: baseline **orange**, candidate **orange**, confidence 0.550; bbox `[x=825.9–1620.0, y=654.7–1080.0]`; matched; best GT IoU 0.883. CLIP swapped: False.

## mbk__apple_94.jpg

- Image ID/path: `images/mbk__apple_94.jpg`
- Annotation path: `labels/mbk__apple_94.txt`
- Original annotation SHA256: `4615a2d2891f643bd7f7c6586cb023277f11c4146af0f60e47b16c6f86018a8f`
- Counts: GT 2; baseline 5; candidate 5; matched 2; unmatched GT 0; unmatched predictions 3.
- Suspected issues: possible incomplete GT annotation; touching/partially occluded apples; unmatched predictions may be visible apples.
- Visual evidence: [`mbk__apple_94_comparison.png`](/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/work/benchmarks/candidate_eval/annotation_review/p0_visualizations/mbk__apple_94_comparison.png)
- Dimensions: source panels 673×448 px, not resized; comparison canvas 3437×544 px.
- Recommended human action: Compare source image and original annotation; inspect unmatched instance boundaries and instance-count policy. Record a human decision only after review.
- Review status: `pending`; reviewer decision and explanation are intentionally empty.

### Per-instance findings

The two GT apples are matched at mask IoU **0.948** and **0.936**. Three unmatched predictions align with additional partially visible apples near image edges/top. GT omission is plausible. These are not labeled false positives without reviewer confirmation. Touching/occlusion and whether partial edge fruit are included should be decided consistently.

- GT `gt_0000` (apple): matched to `det_0000`; class-agnostic mask IoU 0.948. Candidate class: apple.
- GT `gt_0001` (apple): matched to `det_0001`; class-agnostic mask IoU 0.936. Candidate class: apple.
- Prediction `det_0000`: baseline **apple**, candidate **apple**, confidence 0.983; bbox `[x=123.4–417.6, y=112.9–423.5]`; matched; best GT IoU 0.948. CLIP swapped: False.
- Prediction `det_0001`: baseline **apple**, candidate **apple**, confidence 0.980; bbox `[x=352.3–621.2, y=0.0–235.2]`; matched; best GT IoU 0.936. CLIP swapped: False.
- Prediction `det_0002`: baseline **apple**, candidate **apple**, confidence 0.962; bbox `[x=0.0–100.0, y=34.2–322.9]`; unmatched prediction; best GT IoU 0.000. CLIP swapped: False.
- Prediction `det_0003`: baseline **apple**, candidate **apple**, confidence 0.803; bbox `[x=1.4–202.9, y=0.0–183.6]`; unmatched prediction; best GT IoU 0.000. CLIP swapped: False.
- Prediction `det_0004`: baseline **apple**, candidate **apple**, confidence 0.691; bbox `[x=214.8–406.4, y=0.0–120.1]`; unmatched prediction; best GT IoU 0.010. CLIP swapped: False.

## mbk__banana_10.jpg

- Image ID/path: `images/mbk__banana_10.jpg`
- Annotation path: `labels/mbk__banana_10.txt`
- Original annotation SHA256: `8a32fe8caacd00df66383a5a66b0d08536cc96b075b028af0dc2f56b6d5fd3a0`
- Counts: GT 4; baseline 1; candidate 1; matched 0; unmatched GT 4; unmatched predictions 1.
- Suspected issues: possible missed fruit/instance under-segmentation; one unmatched broad prediction may overlap multiple banana instances; mask geometry requires visual inspection.
- Visual evidence: [`mbk__banana_10_comparison.png`](/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/work/benchmarks/candidate_eval/annotation_review/p0_visualizations/mbk__banana_10_comparison.png)
- Dimensions: source panels 720×480 px, not resized; comparison canvas 3672×576 px.
- Recommended human action: Compare source image and original annotation; inspect unmatched instance boundaries and instance-count policy. Record a human decision only after review.
- Review status: `pending`; reviewer decision and explanation are intentionally empty.

### Per-instance findings

All four GT banana instances are unmatched. The single baseline prediction is also unmatched; its mask IoU with the four GT instances is respectively **0.341**, **0.247**, **0.011**, and **0.002**. The broad prediction spans multiple banana bodies but does not match any one GT mask at 0.50. This supports a possible merge/under-segmentation or poor mask geometry; it does not show that four fruits were wholly missed. No predicted pair exists to label as a split. Review GT boundaries and the individual-fruit policy.

- GT `gt_0000` (banana): **unmatched GT**; best prediction `det_0000` has mask IoU 0.341.
- GT `gt_0001` (banana): **unmatched GT**; best prediction `det_0000` has mask IoU 0.247.
- GT `gt_0002` (banana): **unmatched GT**; best prediction `det_0000` has mask IoU 0.011.
- GT `gt_0003` (banana): **unmatched GT**; best prediction `det_0000` has mask IoU 0.002.
- Prediction `det_0000`: baseline **banana**, candidate **banana**, confidence 0.777; bbox `[x=13.4–674.0, y=23.6–444.1]`; unmatched prediction; best GT IoU 0.341. CLIP swapped: False.

## negative__0522908ff4e4e487.jpg

- Image ID/path: `images/negative__0522908ff4e4e487.jpg`
- Annotation path: `labels/negative__0522908ff4e4e487.txt`
- Original annotation SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Counts: GT 0; baseline 1; candidate 1; matched 0; unmatched GT 0; unmatched predictions 1.
- Suspected issues: likely false positive on bell pepper due to visual similarity; mask extent should be compared to pepper boundary.
- Visual evidence: [`negative__0522908ff4e4e487_comparison.png`](/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/work/benchmarks/candidate_eval/annotation_review/p0_visualizations/negative__0522908ff4e4e487_comparison.png)
- Dimensions: source panels 960×746 px, not resized; comparison canvas 4872×842 px.
- Recommended human action: Compare source image and original annotation; inspect unmatched instance boundaries and instance-count policy. Record a human decision only after review.
- Review status: `pending`; reviewer decision and explanation are intentionally empty.

### Per-instance findings

The single apple prediction is unmatched and covers a red bell pepper. Its mask follows the pepper-like object reasonably closely; this looks more like visual similarity/class confusion than a mask geometry failure.

- Prediction `det_0000`: baseline **apple**, candidate **apple**, confidence 0.617; bbox `[x=131.8–487.3, y=100.8–341.3]`; unmatched prediction; best GT IoU 0.000. CLIP swapped: False.

## negative__14a8727b42ec4e6f.jpg

- Image ID/path: `images/negative__14a8727b42ec4e6f.jpg`
- Annotation path: `labels/negative__14a8727b42ec4e6f.txt`
- Original annotation SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Counts: GT 0; baseline 4; candidate 4; matched 0; unmatched GT 0; unmatched predictions 4.
- Suspected issues: multiple likely fruit-like vegetable false positives; per-detection target-class status should be confirmed from image.
- Visual evidence: [`negative__14a8727b42ec4e6f_comparison.png`](/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/work/benchmarks/candidate_eval/annotation_review/p0_visualizations/negative__14a8727b42ec4e6f_comparison.png)
- Dimensions: source panels 960×640 px, not resized; comparison canvas 4872×736 px.
- Recommended human action: Compare source image and original annotation; inspect unmatched instance boundaries and instance-count policy. Record a human decision only after review.
- Review status: `pending`; reviewer decision and explanation are intentionally empty.

### Per-instance findings

All four detections are unmatched. Approximate locations use source-image pixel coordinates from each prediction bounding box. Manual visual inspection shows three apple predictions over red bell peppers; the orange prediction covers a pale/yellow produce item consistent with a melon or pepper rather than an orange. The masks broadly follow those produce objects, so visual similarity/class confusion is the stronger explanation than poor mask extent, but the reviewer should confirm each.

- Prediction `det_0000`: baseline **apple**, candidate **apple**, confidence 0.553; bbox `[x=626.1–749.7, y=178.6–295.9]`; unmatched prediction; best GT IoU 0.000. CLIP swapped: False.
- Prediction `det_0001`: baseline **apple**, candidate **apple**, confidence 0.475; bbox `[x=576.4–655.7, y=111.5–194.3]`; unmatched prediction; best GT IoU 0.000. CLIP swapped: False.
- Prediction `det_0002`: baseline **apple**, candidate **apple**, confidence 0.455; bbox `[x=207.8–334.9, y=123.2–265.6]`; unmatched prediction; best GT IoU 0.000. CLIP swapped: False.
- Prediction `det_0003`: baseline **orange**, candidate **orange**, confidence 0.404; bbox `[x=817.3–952.7, y=34.4–153.8]`; unmatched prediction; best GT IoU 0.000. CLIP swapped: False.

## P0 interpretation summary

| Possible cause | Suspected cases | Evidence and limit |
|---|---:|---|
| Missing/incomplete annotation | 9 candidate instances (6 orange, 3 apple) | Visibly align with additional fruit in source photos; exact count and inclusion of partially visible fruit require reviewer decision. |
| Incorrect GT class/geometry | 0 confirmed | The two matched orange GT masks and two matched apple masks align well by IoU. No annotation error is confirmed. Banana boundaries still need review. |
| False-positive detection | 5 likely (1 + 4 on negatives) | Predictions land on peppers/other produce; masks broadly follow these objects. Class similarity is more likely than poor geometry. |
| Missed fruit | 4 unmatched GT instances in banana_10 | All four GT masks have no eligible match; one prediction partially overlaps two masks, so a merged/bad mask may explain some or all. |
| Split/merge | Possible merge/under-segmentation in banana_10; no split evidence in these five | One broad mask overlaps several GTs below threshold; no pair of predictions forms a split pattern. |
| Mask geometry error | Possible in banana_10; no conclusive geometry-only failure elsewhere | Orange/apple matched masks have high IoU. Human inspection is needed to separate annotation boundary errors from prediction boundary errors. |
| Uncertain | 14 positive-image unmatched cases remain unresolved; all five image review statuses pending | Records and overlays provide evidence but do not authorize a GT decision. |

The counts in this table are audit hypotheses, not adjudicated annotation changes. In particular, the 9 fruit omission candidates and 4 unmatched banana GTs need reviewer confirmation.

## Hash and safety validation

- Config canonical SHA256 and manifest SHA256 match the frozen experiment values. Detector and CLIP checkpoint hashes match the pinned values.
- All five source image and annotation paths exist and their hashes match the frozen manifest. The exact original GT SHA256 is recorded per image.
- Baseline and candidate records have matching image IDs and counts. Frozen baseline/candidate JSONL, event log, execution summary, evaluator and model hashes were identical before and after rendering.
- All five visualizations were generated from persisted records at source resolution; no inference was run.
- All five review records remain `pending`; `reviewer_decision` and `reviewer_explanation` remain null.
- `GT-reviewed` has not been populated with annotations. No original annotation, prediction, model, config, manifest or evaluator was modified.

## Files

- Structured evidence: `work/benchmarks/candidate_eval/annotation_review/p0_review.json`.
- Integrity validation: `work/benchmarks/candidate_eval/annotation_review/p0_manifest_validation.json`.
- Visualizations: `work/benchmarks/candidate_eval/annotation_review/p0_visualizations/`.
- Review protocol and status definitions: `docs/diagnostics/annotation-review-protocol.md`.

Human decisions are required before any annotation correction.
