# Fruit Instance Segmentation Dataset & Model Audit v1

**Scope:** read-only audit for the three-class YOLO11-Seg system. The only project artifact created for this task is this report. No dataset, labels, checkpoint, configuration, or split membership was changed; no training was run.

## Executive findings

- The user-provided 194-image review-subset instance totals are correct: train 407 banana / 59 apple / 64 orange; val 42 / 4 / 5; test 43 / 9 / 6. These are **unreviewed SAM pseudo-labels**, not human-accepted ground truth.
- The current checkpoint was trained against the **full `seg_data_candidate`** dataset (2,530 images), not the 194-image review subset.
- Full dataset association/structural checks pass, but the separate polygon-topology audit flags **1,041 self-intersection candidates**. The 194-image subset has 151 such candidates.
- Cross-split pHash screening has 362 near-duplicate candidates (332 train–val, 30 train–test), and the human-review manifest remains pending. No exact cross-split SHA256 duplicates were found.
- On the fixed Phase 0.5 raw-mask benchmark, the custom model has better aggregate metrics than the COCO YOLO11s-Seg baseline, but banana cluster exact-count accuracy is only 46.15% on test; 14 merge predictions and 3 duplicate predictions are flagged. These are comparisons to pseudo-labels and do not establish production accuracy.
- Standard per-class precision/AP and independent test mAP were not saved. The three reported image-specific predictions are not present in a saved inference record, and all three source images are in train.

## 1. Dataset provenance

### Recorded lineage

1. `work/obb/seg_data_clean/seg.yaml` describes the source labels as “original polygons plus independently masked synthetic bananas” and maps IDs 0/1/2 to banana/apple/orange (artifact: lines 1–9).
2. `work/obb/prepare_seg_candidate.py` copies `work/obb/seg_data_clean` to `work/obb/seg_data_candidate`, then adds 358 empty-label background images sourced from `own_neg/` (lines 9–12, 25–64). The recorded deterministic rule assigns 287 to train and 71 to test.
3. `work/obb/seg_data_candidate/candidate_manifest.json` records `source_dataset: work/obb/seg_data_clean`, the negative-image sources, and the warning that the fruit polygons are SAM pseudo-labels. The complete audit JSON also labels the annotation version `source-sam-pseudolabel-unreviewed` (`work/audit/phase1_dataset_audit.json`, `$.annotation_version`).
4. `work/obb/seg_data_review_v1/dataset_manifest.json` identifies the review package as a 194-image subset, with annotation version `source-sam-pseudolabel-unreviewed`, status `pending`, and source image/label hashes (`$.dataset_version`, `$.review_items`). The current `phase1_review_manifest.csv` still has every row pending.

### Provenance limits

The candidate manifest keeps source paths for the added negatives, but it does not provide source-file lineage for each positive fruit image. Prefixes such as `mbk__`, `rf_v7__`, and `own-` suggest different origins but are not sufficient evidence of exact source, capture session, license, or preprocessing history. No complete positive-image provenance manifest or human-reviewed CVAT export was found. The model's `args.yaml` names the full candidate `seg.yaml` as its training data.

## 2. Class imbalance and image-level distribution

Image-presence counts are non-exclusive because one image can contain multiple fruit classes.

### Full candidate dataset used for training

| Split | Images | Banana instances / images | Apple instances / images | Orange instances / images | Empty-label negatives |
|---|---:|---:|---:|---:|---:|
| Train | 1,884 | 2,764 / 907 | 473 / 395 | 438 / 328 | 287 |
| Val | 301 | 152 / 110 | 108 / 105 | 102 / 88 | 0 |
| Test | 345 | 156 / 106 | 98 / 88 | 95 / 84 | 71 |
| **Total** | **2,530** | **3,072 / 1,123** | **679 / 588** | **635 / 500** | **358** |

Bananas make up 70.0% of all fruit instances; apples 15.5%; oranges 14.5%. There are no negative images in validation, so validation cannot measure background-only false positives. Test has 71 negatives.

### 194-image review subset (still pending)

| Split | Images | Banana instances / images | Apple instances / images | Orange instances / images |
|---|---:|---:|---:|---:|
| Train | 143 | 407 / 99 | 59 / 24 | 64 / 27 |
| Val | 27 | 42 / 22 | 4 / 3 | 5 / 4 |
| Test | 24 | 43 / 17 | 9 / 5 | 6 / 4 |
| **Total** | **194** | **492 / 138** | **72 / 32** | **75 / 35** |

The review subset is banana-heavy (77.0% of its instance rows) and is not the training input recorded for the current checkpoint. Review status counts in `work/audit/phase1_review_manifest.csv` are pending for all 664 rows, including 639 instances and 25 negative-image checks.

**Evidence:** `work/audit/phase1_dataset_audit.json`, `$.splits`; `work/obb/seg_data_review_v1/dataset_manifest.json`, `$.review_items`; `work/audit/phase1_review_manifest.csv`; `work/obb/seg_data_candidate/seg.yaml`, lines 1–8.

## 3. Annotation quality

### Integrity checks

The Phase 1 source audit associates 2,530 images and labels by unique same-stem split path. It reports zero missing labels, orphan labels, missing images, empty raster masks, and exact duplicate annotation pairs. The structural YOLO-seg validator reports zero structural problems: class IDs, numeric normalized polygon coordinates, and nonempty rows parse. The class order is consistently `0 banana`, `1 apple`, `2 orange`.

Structural validity does not prove that a polygon follows the visible fruit. The separate topology audit flags **1,041 self-intersection/touch candidates**: train 956, val 41, test 44 (banana 874, apple 108, orange 59). The review subset has 151 candidates. These remain automated candidates, not adjudicated human errors. The older `seg_dataset_candidate_audit.json` value `invalid_polygons: 0` is from a structural check that does not count these topology findings; see `docs/phase1-dataset-audit.md`, lines 36–48 and 65–69.

There are 358 empty labels, consistent with the intentionally added background set. This verifies file representation, not that every negative image is visually fruit-free. No trusted reviewed ground truth is available to establish missing-object annotation rates across the dataset.

### Merged-instance risk

No same-class duplicate-mask candidates at raster IoU >= 0.98 were found by the Phase 1 audit. This does **not** establish that every mask covers one physical fruit: a broad polygon can still include multiple fruit, and geometry alone cannot adjudicate that. In the three cited examples, the source labels have separate rows (3 apples, 3 bananas, and 7 oranges respectively); there is no evidence that those whole groups were encoded as one label row. The banana case has severe intersection between two separate masks, and its third polygon is self-intersection flagged. Review is still required to decide whether those boundaries are correct.

## 4. Occlusion and overlap analysis

The Phase 1 geometric grouping found 350 images with intersecting raster masks, containing 483 mask-overlap pairs. By split: train 320 images / 433 pairs; val 11 / 22; test 19 / 28. The audit also has 661 images with at least three annotated bananas (train 631, val 13, test 17). Group rules are geometric triage only, not confirmed occlusion truth (`tools/phase1_dataset_audit.py`, lines 139–157 and 160–177; `docs/phase1-dataset-audit.md`, lines 71–100).

| Candidate pattern in full dataset | Images | Notes |
|---|---:|---|
| Banana clusters / partial-overlap group C | 672 | 2,586 rows; includes 661 images with >=3 annotated bananas and 11 additional banana overlap candidates |
| Severe banana overlap group G | 25 | 94 rows; at least one intersection covers >=50% of the smaller raster mask |
| Orange overlap group E | 25 | 83 rows, including 76 orange rows |
| Apple overlap group D | 13 | 51 apple rows |
| Mixed-class overlap group F | 6 | 26 rows |

The review selector includes 67 partial-overlap banana cases, 25 severe-occlusion candidates, 12 orange-overlap cases, 12 apple-overlap cases, and 7 mixed-fruit overlap cases, in addition to single-fruit, multi-fruit, and negative groups. In the subset, 106/194 images have at least one intersecting mask (88 train, 7 val, 11 test); 92 have at least three annotated bananas (75/7/10 train/val/test). Selection and overlap counts are not a human review result.

## 5. Duplicate and data-leakage analysis

The full audit found **zero exact image SHA256 duplicates across splits**. Its 64-bit DCT pHash screen at Hamming distance <=4 found **362 cross-split candidate pairs**: 332 train–val and 30 train–test. Pair distance counts are 99 at 0, 170 at 2, and 93 at 4. A zero pHash distance is not proof of identical source pixels; exact SHA256 is a separate check. The candidate pair list digest recorded in the Phase 1.2 report is `d5e2e7043a9ea63c0db01312e034a036d81ffabea70799efe9b01e8bfb7191ba`.

All 362 decisions remain pending in `work/audit/phase1.2_leakage_review_manifest.csv`. Therefore same-capture leakage is neither confirmed nor ruled out for these candidates. Preserve test independence; never use test images in training. Any future correction needs a separately versioned dataset and human decisions, not an automatic pHash-based move.

## 6. Model training and checkpoint audit

### Recorded configuration

`runs/seg/seg_boundary_candidate_gpu/args.yaml` records:

- Task/model/data: segment, initialized from `yolo11s-seg.pt`, trained on `work/obb/seg_data_candidate/seg.yaml` (lines 1–4).
- 40 epochs, patience 12, batch 8, image size 640, AMP on, cache off, workers 2 (lines 5–14, 28–29).
- Seed 42, nondeterministic, resume off, save period `-1` (lines 11, 22–29).
- `optimizer: auto` in the run args; checkpoint `train_args` records resolved `AdamW`.
- Loss weights: box 7.5, classification 0.5, DFL 1.5; segmentation uses the model's `train/seg_loss` objective (lines 81–92; `results.csv`, header line 1).
- Augmentation: HSV H/S/V 0.015/0.7/0.4, rotation 15°, translate 0.1, scale 0.5, horizontal flip 0.5, mosaic 1.0 (closed for final 10 epochs), mixup/copy-paste 0 (lines 94–121).

The current environment's earlier hardware audit reported `torch.cuda.is_available() == false`; the checkpoint name containing “gpu” and `device: ''` do not prove which hardware trained it. GPU model, VRAM peak, and training device cannot be verified from the saved run artifacts.

### Checkpoint identity

`runs/seg/seg_boundary_candidate_gpu/weights/best.pt` exists, size 20,513,629 bytes, SHA256 `6da28310bb7627d00dfa574cc36147d215afb48e4d0c3d0705e62ecb05917982`. It is a segment checkpoint with names `{0: banana, 1: apple, 2: orange}`; internal Ultralytics version is 8.4.180. Its `train_args` point to the full candidate dataset. Checkpoint fields report `epoch: -1` and `best_fitness: null`, so the precise selected checkpoint epoch cannot be read from those fields. Its `train_metrics` agree with the last row of `results.csv`.

### Standard validation metrics saved by training

These are aggregate validation metrics at the final saved epoch (row 41, epoch 40 in `runs/seg/seg_boundary_candidate_gpu/results.csv`). They are trained/validated against unreviewed candidate annotations.

| Head | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Box | 0.8611 | 0.7822 | 0.8632 | 0.7539 |
| Mask | 0.8335 | 0.7440 | 0.8111 | 0.7193 |

No per-class precision/AP table was saved. There is also no standard Ultralytics per-class mAP evaluation on the independent test split. These aggregate values must not be read as reviewed-ground-truth accuracy.

## 7. Error analysis and fixed benchmark results

### Reproducible comparison already on disk

`work/audit/phase05_benchmark_raw_masks.json` compares the COCO YOLO11s-Seg baseline and current checkpoint on identical full-source val/test manifests. Frozen settings: 640 px, confidence 0.40, NMS IoU 0.60, CPU FP32, no distance-transform refinement. Manifest hashes are `9fc1e01f...be6f` (val) and `94a849d8...6698` (test). The benchmark's mask recall uses greedy one-to-one same-class raster-mask matching at IoU >=0.50 (`tools/benchmark_seg_reproducible.py`, lines 74–112 and 247–275).

| Split / measure | COCO YOLO11s-Seg baseline | Current `best.pt` |
|---|---:|---:|
| Val count exact / MAE | 62.46% / 0.475 | 87.04% / 0.233 |
| Val mask recall, banana / apple / orange | 38.16% / 51.85% / 67.65% | 67.76% / 81.48% / 87.25% |
| Val banana-cluster exact count | 14.29% (3/21) | 42.86% (9/21) |
| Val merge / duplicate prediction counts | 18 / 0 | 12 / 1 |
| Test count exact / MAE | 63.19% / 0.493 | 83.77% / 0.226 |
| Test mask recall, banana / apple / orange | 41.67% / 42.86% / 52.63% | 69.23% / 77.55% / 83.16% |
| Test banana-cluster exact count | 15.38% (4/26) | 46.15% (12/26) |
| Test merge / duplicate prediction counts | 19 / 0 | 14 / 3 |
| Test negative images with >=1 prediction | 2/71 (2.82%) | 2/71 (2.82%) |

For current `best.pt`, the test match counts are banana 108/156, apple 76/98, orange 79/95; under this matching rule, 48 banana, 22 apple, and 16 orange GT masks are unmatched. On val, unmatched GT counts are 49 banana, 20 apple, and 13 orange. Unmatched GT does not distinguish a missed object from an otherwise present but poorly localized mask. Similarly, unmatched prediction masks are not automatically false positives. The benchmark reports 361 test predictions and 349 GT instances.

The benchmark's merge flag means one predicted mask covers at least 25% of two same-class GT masks; duplicate flag means an extra same-class prediction pair with IoU >=0.70 (`tools/benchmark_seg_reproducible.py`, lines 91–112 and 336–340). These are defined diagnostics, not standard COCO metrics.

### Boundary-refinement ablation

The separate older `work/audit/phase05_benchmark.json` is explicitly documented as the distance-refinement ablation, not the canonical raw-mask result. With the helper, current-model test exact count falls to 72.17%, MAE rises to 0.791, banana-cluster exact falls to 23.08%, and merge/duplicate counts are 10/2. Do not treat this as evidence that watershed/distance refinement improves the current model. See `docs/phase-0.5-results.md`, “Refinement ablation.”

### Three reported image errors

All three images are in **train**, not the val/test benchmark. The existing benchmark JSON contains no corresponding per-image predictions, so the user-reported inference outputs were not independently reproduced from a saved prediction artifact.

| Image | Source-label evidence | Audit finding |
|---|---|---|
| `mbk__apple_24.jpg` | `train/labels/mbk__apple_24.txt`: 3 apple rows, no banana; manifest CSV lines 413–415 | The reported banana prediction (confidence 0.491) is a user-provided finding. The source/label overlay shows apples and no banana. There is no saved prediction row to verify the class/confidence. |
| `mbk__banana_66.jpg` | `train/labels/mbk__banana_66.txt`: 3 banana rows; manifest CSV lines 369–371 | User reports 3 fruit, 2 predicted. Source audit puts it in banana-cluster and severe-overlap groups. Label masks 1 and 3 have raster IoU 0.744 and intersection 87.1% of the smaller mask; polygon row 3 is self-intersection flagged. This is both a model failure candidate and a GT boundary ambiguity. |
| `mbk__orange_1.jpg` | `train/labels/mbk__orange_1.txt`: 7 orange rows; manifest CSV lines 421–427 | User reports a missed partially visible orange. The annotation is in the orange-overlap review group, but its recorded pairwise mask intersections are tiny. The visual source contains cut orange cross-sections not represented as separately labeled masks; whether a cut half counts as a separate instance is not defined by the current annotation policy, so this remains a review question rather than a confirmed omission. |

The missing banana/apple/orange metrics by standard class-specific AP and the lack of a saved prediction file for these training examples are material limits. Their training membership also means they are not independent evidence of generalization.

## 8. Dataset v2 proposal (no changes applied)

Build a human-reviewed, immutable version only after review decisions are recorded. Use one visible fruit as one instance; trace visible pixels for partial occlusion; do not infer fully hidden fruit; explicitly decide how cut fruit halves are counted and apply that rule consistently. Resolve all 151 subset topology candidates and prioritize the broader 1,041 full-source candidates before using affected labels.

Collect or select additional examples specifically for:

- **Banana clusters:** 2–6 fruit, varied orientations, shared stems, touching and high occlusion; store one mask per visible banana and keep source/capture groups together.
- **Orange occlusion:** stacked oranges and partially visible edge fruit, including cases with only a small visible arc.
- **Partially visible fruit across all classes:** image-edge truncation, fruit behind bowls/other fruit, and varying visible-area fractions.
- **Hard negatives:** apple/bowl scenes that trigger banana predictions, curved yellow objects, citrus-colored round objects, leaves/highlights, and other non-fruit objects from the intended deployment domain. Include genuine negative images with reviewed empty labels.

Acquire examples through the project owner’s approved sources; do not fetch or synthesize imagery in this audit. Do not auto-label or rebalance by duplicating images. Track counts by class presence, instance count, visible fraction, occlusion type, source session, and negative subtype. Keep test independent; first resolve the 362 candidate pairs manually and version any future split decision without moving test data into train.

## 9. Controlled experiment plan

1. **Freeze a reviewed dataset v2.** Record source and label hashes; preserve train/val/test roles. Keep pending/needs-review objects out of training and evaluation ground truth.
2. **Lock a baseline.** Evaluate current `best.pt` and a lightweight YOLO11n-Seg baseline on identical manifests and frozen settings. Keep this checkpoint immutable.
3. **Respect low VRAM.** The earlier runtime audit could not see CUDA or measure VRAM, so inspect the actual target GPU first. On the known 2–3 GB target, start only with an explicitly memory-tested configuration such as batch 1 and 320 px, AMP only if supported; then benchmark 416 and larger sizes after successful memory tests. Report CUDA OOM rather than silently switching to CPU. Measure peak allocated/reserved memory and CPU/GPU latency.
4. **Run one-factor ablations.** Compare (a) reviewed data additions, (b) model size/resolution, and (c) optional postprocessing separately. Do not tune confidence/NMS against test data. The existing distance-refinement ablation regressed and should not be reintroduced without a targeted validation study.
5. **Measure the failure modes directly.** Report standard mask AP50/AP50–95 and per-class precision/recall, plus exact-count accuracy/MAE, banana-cluster exact count, per-class unmatched GT, merges, duplicates, negative-image false-positive rate, boundary quality, latency, and GPU memory. Use val for selection; run test once after choices are frozen.

## 10. Unverified items and evidence index

### Not verifiable from current artifacts

- Human correctness of any of the 4,386 fruit polygons; all available annotation review decisions remain pending.
- Whether the 362 pHash pairs are same-capture leakage; none has a human decision.
- Exact source/license/session lineage for every positive fruit image.
- Per-class standard precision, AP50/AP50–95, and standard test-set mAP for the current checkpoint.
- The actual prediction records for the three reported train-image failures; only the user-supplied description is available.
- The exact epoch represented by `best.pt` and the actual GPU/VRAM used during its training.
- Whether cut orange halves in `mbk__orange_1.jpg` should count as separate instances under a consistent annotation policy.

### Primary evidence

- Dataset counts, per-image class counts, overlap geometry, polygon flags, negative images, hashes, and pHash candidates: `work/audit/phase1_dataset_audit.json` (`$.splits`, `$.images`, `$.leakage.near_duplicate_image_candidates_phash_le4`).
- Dataset generation and negative provenance: `work/obb/prepare_seg_candidate.py`, lines 9–64; `work/obb/seg_data_clean/seg.yaml`, lines 1–9; `work/obb/seg_data_candidate/candidate_manifest.json` keys `source_dataset`, `backgrounds`, `warning`.
- Review status and ID-to-source references: `work/audit/phase1_review_manifest.csv` (all 664 `review_status` entries are pending); `work/audit/phase1.2_leakage_review_manifest.csv` (362 pending decisions); `work/audit/phase1.4_annotation_review_progress.json`.
- Training arguments: `runs/seg/seg_boundary_candidate_gpu/args.yaml`, lines 1–43, 73–121; final metrics `runs/seg/seg_boundary_candidate_gpu/results.csv`, header line 1 and final row line 41; checkpoint metadata and SHA256 as recorded above.
- Frozen model comparison: `work/audit/phase05_benchmark_raw_masks.json` (`$.config`, `$.models[*].splits.val.summary`, `$.models[*].splits.test.summary`); metric definitions and matching implementation: `tools/benchmark_seg_reproducible.py`, lines 74–112, 190–207, 247–339.
- Refinement ablation: `work/audit/phase05_benchmark.json`; interpretation: `docs/phase-0.5-results.md`, “Refinement ablation.”
- Reconciled structural/topology audit: `docs/phase1-dataset-audit.md`, lines 24–69; human review still pending: `docs/phase1.4-annotation-review-progress.md`, lines 9–67.

---

This is an audit and proposal only. No implementation, relabeling, data acquisition, split change, or training action was performed.
