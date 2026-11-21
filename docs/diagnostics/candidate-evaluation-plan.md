# Fresh CLIP swap-only evaluation plan

**Status:** approved design, implemented in `work/benchmarks/candidate_eval/`; no inference performed. This is a new experiment. It does not recover, resume, or claim any part of the historical 200-image run.

## Comparison to run

- **Dataset:** `work/audit/phase05_benchmark_raw_masks_manifests/test-94a849d857800128dbb9a8c8c1348d3b2f4f58134eed675ea042876f698d6698.json`, 345 entries. The manifest file SHA256 is `9b40bfb089e8697347fa1c34842b58c7046eda2f5c5757777cd275fc758fddc1`; its manifest ID is `94a849d857800128dbb9a8c8c1348d3b2f4f58134eed675ea042876f698d6698`.
- **Detector for both arms:** `runs/seg/seg_boundary_candidate_gpu/weights/best.pt`, SHA256 `6da28310bb7627d00dfa574cc36147d215afb48e4d0c3d0705e62ecb05917982`.
- **Detector settings:** `imgsz=640`, `conf=0.40`, detector NMS IoU `0.70`, CPU, FP32. Keep the detector implementation and postprocessing identical between arms. The existing benchmark helper is not reused for its matching logic because it requires equal classes during matching.
- **Baseline:** detector outputs without any CLIP operation.
- **Candidate:** those same in-memory detector outputs followed by the swap-only rule below. The pipeline should run detector inference once per image, serialize the baseline instances, and derive candidate instances from those objects, avoiding a second detector pass.

## Proposed CLIP pin

These are pins for the **fresh** experiment, not claims about the historical run.

| Component | Proposed pin |
|---|---|
| OpenCLIP model | `ViT-L-14`, pretrained tag `laion2b_s32b_b82k` |
| Model repository revision | Hugging Face snapshot `1627032197142fbe2a7cfec626f4ced3ae60d07a` |
| Weight file | `work/weights/hf/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/snapshots/1627032197142fbe2a7cfec626f4ced3ae60d07a/open_clip_pytorch_model.bin` (resolves to the cached blob; size 1,710,631,365 bytes) |
| Weight SHA256 | `5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b` |
| Runtime | `open_clip_torch 2.30.0`, `torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`, CPU inference, FP32 |
| Prompts | Current `backend/clip_check.py` prompt strings and template, pinned by source SHA256 `0adb44e382cb24bae93399b0c4d6f6891b3a1586ed973f69def9eb716fee941c` and canonical prompt payload SHA256 `1504ef1f4b753b26c4aa77a5ad9f1f83f6726e1084a86e52934caeedd8900eb1` |
| Prompt groups | banana, apple, orange, other; group IDs 0, 1, 2, 3 respectively |
| Preprocessing | Exact `open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")` validation transform from the pinned OpenCLIP package; BGR crops converted to RGB PIL images before transform |
| Crop source | Each detector-predicted bounding box, from the original decoded image; no GT data used |
| Crop generation | Pin current `ClipChecker.probs_boxes` implementation by the source hash above: expanded first crop using `pad=0.10`, integer truncation/clamping as implemented; if grouped `other` probability is >= `0.80`, compute inner crop with `inner=-0.10`; use inner-view scores only when its maximum fruit probability is >= `0.90` |
| Class mapping | Detector fruit names and CLIP groups map `banana=0`, `apple=1`, `orange=2`; `other=3` is never a candidate output class |
| Swap threshold and condition | `t_swap=0.80`: if the final top CLIP group is one of banana/apple/orange, differs from detector class, and its grouped probability is >= 0.80, replace only the class ID/name. Otherwise retain the detector class. A top group of `other` never deletes or changes an instance. No banana protection, rescue, filtering, or score thresholding is applied in the swap-only adapter. |

The cache also contains an empty `.no_exist` marker for `open_clip_model.safetensors`; it is not the selected model file. The `.bin` symlink resolves to the non-empty blob above, whose content hash is recorded. Before running, implementation must verify that symlink target and SHA256 still match; otherwise abort before processing any image.

This deliberately pins the current checker’s crop scoring behavior but uses a separate pure swap adapter rather than production `refine_types`, which can filter instances and has additional protections/rescue behavior. The evaluator must not modify production code. Code hashes and package versions must be placed in the experiment config before evaluation begins.

## Storage and resume design

Create `work/benchmarks/candidate_eval/` with:

- `config.json`: immutable experiment parameters, manifest and detector/CLIP/code/prompt hashes, package versions, matching version and threshold.
- `predictions_baseline.jsonl` and `predictions_candidate.jsonl`: one record per image, including path, image and label hashes, dimensions, config/checkpoint hashes, inference status, instance IDs/counts, and full instance records (class, confidence, bbox, polygon/mask, mask area). Candidate records additionally store original class, CLIP grouped probabilities, selected view, resulting class and swap decision.
- `progress.json`: manifest ID, config hash, and state for every image (`pending`, `running`, `success`, or `error`).
- `inference.jsonl`: append-only per-image start/completion/error events, with exception text and traceback on failure.
- `report.json`: evaluation output only after all image predictions validate; otherwise status remains incomplete and metrics are not presented as final.

Write each image result to a temporary file, flush and `fsync`, atomically rename it into the per-image result store, validate all instances, then update `progress.json` atomically to `success`. The JSONL files are deterministic exports/materializations from those per-image atomic records. A resume skips an image only when both arm records exist, hashes/config match, status is successful, and full schema/invariant validation passes. Never infer success from line counts or aggregate metrics. Per-image failure is logged and remains an error; the run cannot report 345/345 success until resolved.

## Swap-only invariants

For each image, compare baseline and candidate by stable `instance_id` (detector output order within that image). Assert exact equality of:

- instance ID set and instance count;
- detector class confidence values;
- bounding box coordinates;
- polygon coordinates or persisted binary masks and their dimensions;
- mask area computed from the persisted segmentation representation.

Only class ID/name may differ. Validate finite numeric values, polygon shape, class range, image dimensions, and record hashes before marking an image successful. Any failed assertion fails that image and blocks a complete report.

## Matching and metrics

Rasterize GT and predicted polygons at original image resolution using one pinned implementation. Build all GT/prediction mask IoUs without consulting classes. Use deterministic maximum-weight one-to-one assignment over eligible pairs with IoU >= 0.50 (Hungarian assignment; tie-break by stable GT/prediction IDs). Only after assignment, compare classes to form baseline and candidate confusion matrices. Unmatched predictions are false positives; unmatched GT are false negatives. Report per-class precision/recall/F1, corrected errors, newly introduced errors, correct-to-wrong and wrong-to-correct transitions, unchanged matched labels, per-image paired detection/classification/count metrics, and negative-image false positives. Bootstrap confidence intervals are computed by resampling images, not instances.

## Implementation sequence

1. Create a standalone evaluator module, with no import of production application startup or mutation of backend behavior.
2. Implement and validate config/hash checks, manifest integrity, deterministic mask rasterization/matching, atomic per-image records, progress resume, and invariants.
3. Add a dry-run validation mode that checks all 345 manifest entries, referenced image/label hashes, model hashes, and environment versions without invoking model inference.
4. Run static/schema/invariant unit checks on synthetic in-memory records only; no test image inference and no dataset writes.
5. Inspect the generated immutable config and storage layout. Evaluation begins only after explicit approval.
6. On approval, run the 345-image fresh evaluation with append-only logs; produce `report.json` only after every image has validated successfully.

## Decisions / gates

Approved for this fresh experiment: current `clip_check.py` prompt/crop scoring path, grouped CLIP swap threshold `0.80`, and deterministic Hungarian maximum-IoU matching. These choices do not imply that the historic run used the same code. The implementation is guarded so inference only runs with the explicit `--run-inference` flag. No inference has been performed and no historical result has been reused.
