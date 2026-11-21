# Phase 0.5 — Baseline validation and runtime corrections

## Checkpoint verification

The custom checkpoint exists at
`runs/seg/seg_boundary_candidate_gpu/weights/best.pt` (resolved path in the
benchmark JSON: `/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/runs/seg/seg_boundary_candidate_gpu/weights/best.pt`).

- Task: `segment`.
- Class order: `0 banana`, `1 apple`, `2 orange`; all three required fruit
  classes are present. The API validates task and class membership at startup.
- Training metadata: YOLO11s-Seg initialization, `imgsz=640`, batch 8,
  40 epochs, AMP enabled, dataset cache disabled.
- SHA256: `6da28310bb7627d00dfa574cc36147d215afb48e4d0c3d0705e62ecb05917982`.
- The checkpoint loads through the same Ultralytics segmentation inference path
  used by the current API. Its class indices map to the expected fruit names;
  API objects retain the existing label, confidence, state, polygon and mask
  metric fields. Transform/runtime data are additive. No bounding-box response
  field is introduced.

Baseline checkpoint `yolo11s-seg.pt` SHA256 is
`1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3`; it is
also task `segment`, but uses COCO class IDs 46 banana, 47 apple, 49 orange.

## Frozen evaluation

Both checkpoints ran on exactly the same 301 validation and 345 test images,
with the same manifest hashes, image size 640, confidence 0.40, NMS IoU 0.60,
CPU FP32, and no distance-transform refinement. There was no threshold tuning.
Count exact accuracy and MAE are per image. Per-class recall is one-to-one
same-class mask matching at IoU >= 0.50. Cluster exact accuracy is computed
on the 21 validation and 26 test images identified as banana clusters.

### Validation split

| Checkpoint | Exact count accuracy | Count MAE | Banana / apple / orange recall | Banana cluster exact | Merge rate | Duplicate rate | Negative FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11s COCO baseline | 62.46% | 0.4751 | 38.16% / 51.85% / 67.65% | 14.29% (3/21) | 6.41% | 0.00% | N/A (no negative images) |
| Custom best.pt | 87.04% | 0.2326 | 67.76% / 81.48% / 87.25% | 42.86% (9/21) | 3.17% | 0.26% | N/A (no negative images) |

### Test split

| Checkpoint | Exact count accuracy | Count MAE | Banana / apple / orange recall | Banana cluster exact | Merge rate | Duplicate rate | Negative FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11s COCO baseline | 63.19% | 0.4928 | 41.67% / 42.86% / 52.63% | 15.38% (4/26) | 6.48% | 0.00% | 2/71 images (2.82%) |
| Custom best.pt | 83.77% | 0.2261 | 69.23% / 77.55% / 83.16% | 46.15% (12/26) | 3.88% | 0.83% | 2/71 images (2.82%) |

For additional context, exact count accuracy on ordinary single-fruit images
was 67.92% baseline and 91.32% custom on validation (265 images), and 59.39%
baseline and 85.15% custom on test (229 images).

### Refinement ablation

The first full run with the distance-transform split helper enabled produced
substantially worse counting than raw model masks. On test, custom exact count
accuracy changed from 83.77% to 72.17% and banana-cluster exact accuracy from
46.15% to 23.08%; baseline test exact accuracy changed from 63.19% to 55.94%.
Accordingly, production inference no longer applies that unvalidated helper.
It remains an explicit opt-in benchmark ablation (`--distance-refinement`) for
future evaluation; this does not alter confidence/NMS thresholds.

## Artifacts and reproducibility

- Canonical raw-mask comparison: `work/audit/phase05_benchmark_raw_masks.json`.
- Earlier distance-refinement ablation: `work/audit/phase05_benchmark.json`.
- The JSON records checkpoint hashes, task and class names, exact dataset
  manifests and hashes, split, image size, confidence, NMS IoU, device,
  precision, software versions, timing, and CUDA memory status.
- Frozen settings: `imgsz=640`, `conf=0.40`, `iou=0.60`, CPU FP32. Both model
  runs share the validation manifest hash
  `9fc1e01f59842df03909623445942a17884c3d94606aa83941b4c927c080be6f` and test
  manifest hash `94a849d857800128dbb9a8c8c1348d3b2f4f58134eed675ea042876f698d6698`.

Re-run with:

```bash
python tools/benchmark_seg_reproducible.py --device cpu --precision fp32 --imgsz 640 --conf 0.40 --iou 0.60 --output work/audit/phase05_benchmark_raw_masks.json
```

## Limits and manual verification

The available host has PyTorch 2.5.1+cu121, but `torch.cuda.is_available()` is
false and NVML cannot initialize. GPU model, VRAM capacity, allocations and
CUDA latency therefore remain unmeasured. The target 2–3 GB CUDA host must be
benchmarked before claiming CUDA compatibility. API listener binding was
unavailable in this environment, so checkpoint inference was validated via
the benchmark runner, but the HTTP serialization/rendering path still needs a
deployment smoke test. No model training or Phase 1 work was started.
