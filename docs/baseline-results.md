# Baseline results

Audit date: 2026-11-18. This file separates pre-existing benchmark artifacts
from checks run during Phase 0. No benchmark report records GPU memory use.

## Current runtime attempt

- `python -m unittest test_detection.py` in `backend/`: **26 tests passed** in
  0.048 s. These tests cover legacy coordinate/box utilities, type refinement,
  teacher fusion and the older OBB banana splitter; they do not exercise the
  current API end-to-end or the current distance-transform helper.
- Hardware probe: PyTorch 2.5.1+cu121, CUDA build 12.1, `torch.cuda.is_available()`
  is false. `nvidia-smi` cannot communicate with a driver. No VRAM value is
  available.
- Uvicorn completed FastAPI startup but could not bind localhost ports in this
  environment. A TestClient API inference attempt on CPU did not produce a
  response within the observation window and was stopped. Therefore API
  end-to-end CPU success and latency are **not established**.
- With default `CLIP_CHECK=1`, startup attempted to contact Hugging Face for
  OpenCLIP weights; DNS/network access is unavailable here. The code catches
  this and proceeds without CLIP, but that path was not confirmed through an
  API response. The startup attempt with `CLIP_CHECK=0` also did not yield an
  API response in the available environment.

## Existing stored count benchmarks

These JSON files were already present under `work/audit/`. They do not include a
run timestamp, full CLI arguments or GPU memory readings. Treat them as useful
historical evidence, not as a reproducible audit run. Count metrics below are
not box/mask mAP.

| Checkpoint/report | Split | Images | Exact count | Count MAE | Banana cluster exact | Cluster MAE | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `yolo11s-seg.pt`, `seg_coco_baseline_count_val_iou060_conf040.json` | val | 301 | 62.13% | 0.4784 | 14.29% (3/21) | 1.8571 | Generic COCO segmentation baseline. |
| `yolo11s-seg.pt`, `seg_coco_baseline_count_test.json` | test | 345 | 63.19% | 0.4928 | 15.38% (4/26) | 1.6154 | Negative images: 2/71 had false positives. |
| custom `best.pt`, `seg_candidate_count_val_iou060_conf040.json` | val | 301 | 87.04% | 0.2326 | 42.86% (9/21) | 1.0952 | Over-count 6.98%, under-count 5.98%. |
| custom `best.pt`, `seg_candidate_count_test.json` | test | 345 | 83.77% | 0.2261 | 46.15% (12/26) | 0.8462 | Over-count 8.70%, under-count 7.54%; negative false positives 2/71. |
| custom `best.pt`, `seg_candidate_watershed_count_val.json` | val | 301 | 83.39% | 0.2824 | 28.57% (6/21) | 1.3333 | Stored legacy watershed/OBB experiment; no demonstrated gain. |

The custom candidate's aggregate exact-count result is better than the generic
COCO checkpoint in these saved reports, but its crowded-banana exact count is
still low. The stored watershed experiment is worse than the unsplit candidate
report on validation. The current distance-transform watershed helper is a
different code path and has **no measured result**.

## Demo backend selection

The backend default uses `runs/seg/seg_boundary_candidate_gpu/weights/best.pt`
(SHA-256 `6da28310bb7627d00dfa574cc36147d215afb48e4d0c3d0705e62ecb05917982`),
the custom checkpoint with the strongest saved test count result above. Its
demo defaults are confidence `0.40`, IoU `0.60`, and image size `640`, matching
the recorded benchmark settings. The test result covers 345 images; exact count
was 83.77%, MAE 0.2261, and crowded-banana exact count was 46.15% (12/26).
These figures describe the saved benchmark, not a guarantee for new demo
images.

Additional stored experiment `banana_count_base_fast.json` covers 106 images
from a separate counting set: detector `det_finetune_best.pt`, 640 px,
confidence 0.40, IoU 0.60; exact count 63.2%, MAE 0.491. Its split and label
source differ from the segmentation benchmarks, so it should not be compared as
if it were the same test set.

## Reproducible baseline plan

Before accepting a model or post-processing change:

1. Freeze dataset manifests and run configuration (checkpoint hash, split,
   confidence, IoU, image size, device, precision, software versions).
2. Run the unchanged segmentation baseline and candidate on the same independent
   test split. Do not tune thresholds on test.
3. Report per-class precision/recall and mask mAP50-95, plus visible-instance
   recall, exact count, count MAE, over/under-count, merge and duplicate rates.
4. Break out banana cluster, other overlapping fruits, negatives and near-total
   occlusion. Do not count hidden/inferred estimates as visible ground truth.
5. Capture per-image latency and peak allocated/reserved CUDA memory only when a
   working GPU is available. On this host those values remain unmeasured.
6. Run the 15 scenario cases from the task, including rectangular/rotated input
   and small/large fruit, with reviewed ground truth.

The current dataset labels are polygon counts; there are no official
visible-vs-inferred ground-truth counts or occlusion severity annotations, so
those metrics cannot yet be calculated reliably.
