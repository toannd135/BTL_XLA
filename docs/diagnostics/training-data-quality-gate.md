# Training data quality gate — YOLO11s-Seg

**Decision: conditional GO for a controlled first training run after applying the proposed train-only near-duplicate exclusion list.** No training or inference was run; the dataset, labels, splits, model and backend were not changed.

## What was checked

- Dataset: `work/obb/seg_data_training_prep_v1`; 2,530 images and 4,386 active annotation rows across train/val/test.
- Read the YOLO-seg parser/rasterizer in `tools/validate_yolo_seg_export.py` and topology checker in `tools/phase1_dataset_audit.py`. Structural validation passed: **0 errors**.
- YOLO rows contain a class ID and normalized polygon coordinates; vertices are rounded to image pixels and rasterized by OpenCV `fillPoly`.
- Compared all **362 pHash candidate edges** (Hamming distance ≤4) using grayscale SSIM and ORB/RANSAC; visually inspected all six ambiguous pairs.

## Polygon findings

| Finding | Rows | Label files | Interpretation |
|---|---:|---:|---|
| Proper non-adjacent edge crossings | 250 | 231 | Genuine topology anomalies. All rasterize to non-empty, single-component masks; usable by the loader but should be prioritized for annotation review. |
| Non-adjacent touching/contact only | 791 | 542 | Conservative checker flags. No proper crossing; all rasterize. Do not mass-exclude based on this flag. |
| Structurally invalid / unusable | 0 | 0 | No polygon requires exclusion on rasterization grounds. |

The checker treats non-adjacent touching and collinear contact as self-intersection, so 1,041 does not mean 1,041 unusable polygons. Representative images are in `training_data_quality_gate_visuals/`; no unusable sample exists. Files with true crossings are listed in JSON as review leads, not repaired labels.

## Near-duplicate findings and split fairness

- **0 exact SHA-256 duplicate images** across splits.
- Metrics categorized 330 edges as same/near-identical time-lapse frames and 26 as related frames from the same scene. Visual review of the remaining six: **4 are different fruit photos** (pHash false positives for capture leakage); **2 negative images share a generic stripe motif but are distinct source images**.
- Ten high-confidence temporal groups cross split boundaries. Four include test images; six are train/val only. These are correlated frames, not byte-identical images.
- Minimal fairness action: keep val/test intact and exclude the **62 train-side images** connected to these high-confidence groups. Proposed image list and YAML are outside the dataset at `work/audit/seg_data_training_prep_v1_qg_train.txt` and `work/audit/seg_data_training_prep_v1_qg.yaml`. This leaves **1,822 train images** (3613 instances: banana 2754, apple 448, orange 411).
- Validation has **0 negative images** while test has **71**. Validation cannot estimate negative-image false positives. Test retains temporal correlation in four source groups; group-level uncertainty or per-sequence reporting is preferable.

## Training decision table

| Category | Gate result | Action |
|---|---|---|
| Safe to train | Conditional | Use the proposed filtered train list; active rows are structurally valid and rasterizable. Track 250 true-crossing rows as label-quality risk. |
| Needs exclusion | 62 train images | Exclude train-side temporal neighbors to val/test via external list. No val/test image removed. |
| Needs annotation review | 231 files with 250 proper-crossing rows | Targeted review. Touch-only flags are low priority. |
| Safe for validation/test | Conditional | No exact duplicate leakage; group correlation remains, and validation has no negative images. |
| Split leakage risk | High before filtering | Reduced by proposed exclusion; future split by capture/source sequence is preferable. |

## Proposed command (not run)

For an RTX 3060 Laptop GPU with 6 GB VRAM, begin conservatively at 640 px, batch 1, AMP enabled, no image cache:

```bash
yolo segment train model=yolo11s-seg.pt data=work/audit/seg_data_training_prep_v1_qg.yaml epochs=40 imgsz=640 batch=1 device=0 workers=2 cache=False amp=True seed=42 project=runs/seg name=seg_training_prep_v1_qg exist_ok=False
```

If the actual runtime is not CUDA or runs out of memory, stop and adjust in a separately approved task; do not change thresholds or annotations as part of this gate.

## Integrity

- Prepared dataset tree SHA-256 before and after: `2efda8f9908eb1e0a249e73aadae9d6b4097ce4e678ec2097bc9dafa17563d1e` (identical).
- Source dataset tree SHA-256 recorded/current: `231a786fab3bd669f280eef60864344ca5f3556979cbd304205ef0ccbd5cb1e6` (identical).
- Per-file dataset hashes match the forensic baseline. See [`training-data-quality-gate-hashes.json`](training-data-quality-gate-hashes.json).
- Machine-readable lists, all 362 pair metrics, polygon rows, proposed exclusions and group memberships: [`training-data-quality-gate.json`](training-data-quality-gate.json).
- No original dataset, active annotation, proposal, model/backend or split file was edited.
