# Training dataset preparation report

## Outcome

Created an isolated dataset copy at [`work/obb/seg_data_training_prep_v1`](/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/work/obb/seg_data_training_prep_v1). No training or inference was run. The source dataset remains unchanged.

- Source tree SHA256: `231a786fab3bd669f280eef60864344ca5f3556979cbd304205ef0ccbd5cb1e6`
- Prepared dataset tree SHA256: `2efda8f9908eb1e0a249e73aadae9d6b4097ce4e678ec2097bc9dafa17563d1e`
- Prepared dataset size: 297,995,720 bytes
- Dataset YAML: [`seg.yaml`](/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/work/obb/seg_data_training_prep_v1/seg.yaml) (SHA256 `3031b817a04ae8fae6618217be57734467a28be6f57cf009d0a62802e4981669`)
- Preparation provenance: [`dataset_preparation_manifest.json`](/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/work/obb/seg_data_training_prep_v1/dataset_preparation_manifest.json)
- Optional annotations: [`annotation_proposals/`](/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/work/obb/seg_data_training_prep_v1/annotation_proposals)

Tree hash definition: sorted files contribute relative POSIX path, NUL byte, and raw SHA256 digest of file content. The source hash includes its original YAML; the new hash includes the clone-specific YAML, proposal files and provenance manifest.

## Adjudication handling

The worksheet/event log show 19 confirmed, 3 uncertain and 5 pending items. Six user-authorized AI-assisted `true_target_object_missing_from_gt` decisions are represented as **optional test-only proposals**: four orange instances from `mbk__orange_76.jpg` and two apple instances from `mbk__apple_94.jpg`. Each proposal includes source event ID, prediction-record hashes, class/confidence, mask geometry and a YOLO-seg polygon row.

These proposal files are separate from active `train/labels`, `val/labels` and `test/labels`; the default YAML does not use them. Thus the 4,386 active source annotation rows remain unchanged. The three uncertain detections are excluded. `banana_10`'s confirmed `merge_error` remains one broad mask; no split polygons were invented. The two negative images remain pending with empty labels.

| Split | Images | Active instances | Banana | Apple | Orange | Empty labels |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,884 | 3,675 | 2,764 | 473 | 438 | 287 |
| Val | 301 | 362 | 152 | 108 | 102 | 0 |
| Test | 345 | 349 | 156 | 98 | 95 | 71 |
| **Total** | **2,530** | **4,386** | **3,072** | **679** | **635** | **358** |

If all optional proposals were later accepted for evaluation labels, test counts would be 355 instances (banana 156, apple 100, orange 99) and totals would be banana 3,072, apple 681, orange 639. This is not the active training distribution; all proposals are in test and excluded from training.

## Validation

- **Split integrity:** 1,884/301/345 image-label pairs; no missing labels or orphan labels. Exact image SHA duplicates across splits: **0**. The existing perceptual-hash screen still has **362 near-duplicate candidates** (Hamming distance ≤4) requiring review before relying on split independence.
- **Structural polygon validation:** passed with 0 structural errors; known class IDs, finite normalized coordinates and image/label pairings are valid. The background manifest has 358 entries (287 train, 71 test); all referenced images exist and labels are empty.
- **Topology:** the existing dataset contains **1,041 self-intersection findings** in source polygons (train 956, val 41, test 44). Structural validation passes, but the separate topology validator reports those findings. Source rows were preserved without repair.
- **Overlap/duplicates:** 483 existing annotation-mask overlap pairs in 350 images (train 433 pairs, val 22, test 28); same-class duplicate-mask candidates at IoU ≥0.98: **0**. These overlaps remain review leads, not automatic errors.
- **Proposal geometry:** all 6 polygons have valid class IDs, finite normalized coordinates, ≥3 vertices and no self-intersection. No proposal duplicates an original same-class mask at IoU ≥0.98; maximum proposal/original same-class IoU is below 0.006. Proposal-to-proposal overlap is also negligible (maximum IoU 0.0038).
- The source dataset audit identifies annotations as SAM-derived pseudo-labels; structural validity does not guarantee visual correctness.

## Remaining work and risks

- Three P0 prediction items remain uncertain; five false detections across two negative P0 images remain pending. No label changes were proposed for them.
- The banana merge decision has no verified per-fruit polygons; it must not be split into training labels until those boundaries are separately annotated and checked.
- The 1,041 self-intersecting source polygons and 362 near-duplicate split candidates remain unresolved. These should be reviewed before training or claiming clean split separation.
- The validation split contains no negative images, unlike train and test.
- The six additions are test-only annotation proposals and do not increase training examples for touching/occluded fruit.

No source dataset, original GT, `GT-reviewed`, predictions, evaluator, manifest, model/checkpoint, backend or benchmark configuration was modified. Candidate model remains YOLO11s-Seg; no training was started.
