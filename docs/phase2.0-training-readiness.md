# Phase 2.0 — Training Readiness Audit

**Decision: BLOCKED**

This is an audit only. No model training was started. No source images, labels, review manifests, checkpoints, or train/val/test assignments were modified. No proposed immutable training dataset version was created because review evidence is incomplete.

## Review progress

The annotation manifest has 664 rows covering 194 unique images. Every row remains `pending`.

| Split | Accepted | Corrected | Needs review | Pending |
|---|---:|---:|---:|---:|
| Train | 0 | 0 | 0 | 553 |
| Val | 0 | 0 | 0 | 51 |
| Test | 0 | 0 | 0 | 60 |
| **Total** | **0** | **0** | **0** | **664** |

There are no adjudicated annotation decisions. In particular, a valid Phase 1.3 workflow-validator result means the review manifests obeyed the review schema; it does not mean annotations have been accepted or corrected.

| Leakage decision | Count |
|---|---:|
| `same_capture` | 0 |
| `near_duplicate` | 0 |
| `similar_scene` | 0 |
| `unrelated` | 0 |
| `needs_review` | 0 |
| `pending` | 362 |

All 362 reviewer decisions are blank. These include 332 train–val pairs and 30 train–test pairs. No same-capture pair is confirmed. The 30 train–test candidates remain unresolved; no test image was moved.

The available `cvat_import_review_v1.zip` is an import archive, not a human-reviewed export. It contains 194 images and 194 label files; its expanded files match the extracted import package. No reviewed CVAT export was found in the repository.

## Dataset integrity

The source is `work/obb/seg_data_candidate`:

| Split | Images | Instances | Empty-label negative images |
|---|---:|---:|---:|
| Train | 1,884 | 3,675 | 287 |
| Val | 301 | 362 | 0 |
| Test | 345 | 349 | 71 |

Class IDs are consistent in the dataset YAML: `0 banana`, `1 apple`, `2 orange`. Structural YOLO segmentation validation passed with zero structural problems, zero missing/orphan image-label associations, and zero empty/malformed segmentation instances. The review subset's 194 images all map to source images; all copied image and label hashes match the source manifest, with zero split mismatches. Empty-label negatives remain present (358 in source, 25 in the review subset), but their semantic correctness is not human-confirmed.

Structural validity does not establish polygon topology. The separate topology check found **1,041 self-intersection candidates** in the full source dataset: 956 train, 41 val, and 44 test. The 194-image review subset and the CVAT import copy each have 151 such candidates. These are geometric findings for review, not automatically adjudicated annotation errors. They remain unresolved and prevent approval of the current labels as training-ready.

The Phase 1.3 review validator reports `valid`, with zero workflow/schema issues, while separately reporting all 664 annotation rows and all 362 leakage pairs pending. It does not override the polygon-topology audit.

## Leakage assessment and split preservation

- Confirmed same-capture train–test pairs: **0**.
- Confirmed same-capture train–val pairs: **0**.
- Unresolved train–test candidates: **30**.
- Unresolved train–val candidates: **332**.
- Test-to-train violations: **0**.
- Split changes applied: **0**.

No leakage decision was inferred from pHash distance. Keep test independent; never move test images into train. Reassess split assignments only after human review, in a separately versioned dataset.

## Proposed training dataset

No immutable dataset proposal is created. All annotation rows are pending, topology candidates remain unreviewed, and leakage decisions remain pending. Pending or `needs_review` labels must not be silently included. Keep the intended roles as train for fitting, val for model selection, and test for final independent evaluation when a reviewed dataset version is eventually approved.

## Hardware and feasibility

Measured in this runtime:

| Item | Observed |
|---|---|
| Python | 3.11.8 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA build | 12.1 |
| CUDA available / device count | False / 0 |
| GPU model and VRAM | Not observable; `nvidia-smi` cannot communicate with the NVIDIA driver |
| Ultralytics / OpenCV / NumPy | 8.4.180 / 5.0.0 / 2.4.6 |
| Disk | 69 GB total, 66 GB used, 2.8 GB available (96% used) |

Because no GPU is visible, this audit cannot verify the deployment GPU model, its available VRAM, or compatibility of a CUDA training configuration. It therefore does not claim that any CUDA setup fits. The 2.8 GB available disk is also a significant constraint. No training configuration is approved in this phase.

The existing checkpoint `runs/seg/seg_boundary_candidate_gpu/weights/best.pt` exists (20,513,629 bytes; SHA256 `6da28310bb7627d00dfa574cc36147d215afb48e4d0c3d0705e62ecb05917982`). Its metadata indicates a segmentation task with class IDs banana/apple/orange and historical arguments `yolo11s-seg`, batch 8, image size 640, AMP enabled, cache disabled, and two workers. Those historical settings are recorded for context only; they do not prove fit on the current or target hardware. The checkpoint was not changed.

## Readiness blockers

1. No annotation review is complete: 664 pending rows across 194 images.
2. The full source has 1,041 unadjudicated polygon self-intersection candidates.
3. All 362 cross-split leakage candidates remain pending, including 30 train–test candidates.
4. No human-reviewed CVAT export is available.
5. CUDA is unavailable in this audit runtime, so actual GPU/VRAM compatibility cannot be verified.
6. Only 2.8 GB of disk space is available.

## Evidence and outputs

- Review counts and decisions: `work/audit/phase1_review_manifest.csv`, `work/audit/phase1.2_leakage_review_manifest.csv`, `work/audit/phase1.3_review_validation.json`.
- Source inventory and class/split metadata: `work/audit/phase1_dataset_audit.json` and `work/obb/seg_data_candidate/seg.yaml`.
- Structural and topology validation were run separately on the full source, review subset, and CVAT import folder. Their machine-readable console outputs are in `/tmp/phase20_*_*.txt` for this runtime.
- Machine-readable Phase 2.0 summary: `work/audit/phase2.0_training_readiness.json`.

**Stop condition:** Phase 2.0 ends here. Resolve human annotation and leakage reviews, adjudicate topology findings, provision/inspect the actual training GPU, and address disk capacity before reconsidering training readiness. No training was started.
