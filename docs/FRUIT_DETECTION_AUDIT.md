# Fruit detection and counting audit

Audit date: 2026-11-19. Scope: current source, checked-in project assets, saved benchmark reports, and the existing frontend API contract. This audit does not claim new model performance.

## Current architecture

The application is a FastAPI backend in `backend/main.py` and a React/Vite frontend. The image endpoint accepts one uploaded image, applies `enhance_image`, runs a YOLO segmentation model, optionally validates crop labels with CLIP, classifies each remaining crop as fresh/rotten, computes geometric mask warnings, and returns the instances. The frontend counts the returned `objects` list and draws each returned polygon. Feedback and collection endpoints are separate.

The runtime checkpoint is selected by `MODEL_PATH`; its default is `runs/seg/seg_boundary_candidate_gpu/weights/best.pt`. The checkpoint hash recorded in the existing reports is `6da28310bb7627d00dfa574cc36147d215afb48e4d0c3d0705e62ecb05917982`. This is a custom YOLO segmentation model with banana/apple/orange classes. `yolo11s-seg.pt` is the generic COCO baseline. `det_obb_s.pt` is a separate custom OBB detector, and HBB/OBB/classification checkpoints also exist under `exports/` and `runs/`.

## Actual inference flow

1. Startup resolves a device and eagerly loads the segmentation detector, the optional state classifier, and optionally CLIP. Detector inference is protected by a process-wide lock. The configured detector must be a segmentation model with all three fruit class names.
2. `POST /api/predict` validates the upload, decodes with OpenCV, enhances the image, and tracks the resize transform so returned polygon coordinates are mapped to the uploaded image.
3. `detect_seg` calls Ultralytics once with `agnostic_nms=True`, `max_det=300`, configured confidence/IoU, and the configured image size. It extracts one outer contour per model mask. No independent instance detector or authoritative annotation is consulted.
4. If CLIP is loaded, it can discard or relabel detected crops. It cannot propose a fruit for which YOLO emitted no box. Banana labels/counts are protected in the current CLIP path.
5. The classifier runs on crops and returns fresh/rotten. It has no reject/unknown-fruit class. A missing classifier produces `unknown`.
6. `assess_detection` emits review warnings based on confidence, mask geometry, overlap and occlusion. These are heuristics; they do not alter or validate the number of physical fruits.
7. The API returns `objects`, polygon and state fields, `quality`, `runtime`, and image geometry. The frontend draws polygons, allows edits, and computes totals from the returned list.

The OBB inference and banana splitter functions in `backend/detection.py` are not called by the current API. `collect.py` uses a separate teacher-box/classifier flow to generate collection labels. SAM 2.1 weights are present at `work/weights/sam2.1_b.pt`, but there is no SAM2 provider/configuration in the API. No Roboflow provider or configured credential was found. The current API uses `CONF=0.4`, `NMS_IOU=0.6`, `IMGSZ=640` by default in this working tree.

## Existing evidence and model inventory

| Artifact | Actual role/evidence |
|---|---|
| `runs/seg/seg_boundary_candidate_gpu/weights/best.pt` | Custom segmentation model. Saved count test: 345 images, exact total count 83.77%, MAE 0.2261; banana-cluster exact count 46.15% (12/26). |
| `runs/seg/seg_v3_qg_controlled/weights/best.pt` | Later segmentation model. Forensic report shows worse than V2 on all eight aggregate validation metrics; it is not promoted. |
| `yolo11s-seg.pt` | Generic COCO segmentation baseline. Saved count test: exact total count 63.19%, MAE 0.4928; banana-cluster exact count 15.38% (4/26). |
| `exports/det_obb_s.pt`, `det_obb_s.pt` | Custom OBB model used by offline experiments/legacy app versions. Current API does not load it. Existing OBB-assisted count validation is worse than raw candidate segmentation on exact count and banana clusters. |
| `exports/cls_finetune_best.pt` | Fruit-condition classifier loaded per crop. It predicts fresh/rotten, not fruit-vs-background. The count benchmark does not measure state classification. |
| `work/weights/sam2.1_b.pt` | SAM 2.1 weights exist; installed runtime, config compatibility and API latency/quality have not been established by the saved benchmarks. |
| `work/audit/seg_candidate_count_test.json` | Historical count aggregate and per-image counts; not mask/matching predictions and does not establish API performance. |

Saved test metrics support choosing the custom segmenter over the generic COCO model for the evaluated split. They do not establish reliable performance on arbitrary demo imagery. The candidate training data audit describes SAM-derived pseudo-labels; polygon structural validity does not verify that every physical fruit was separately annotated. V3 is not a better substitute based on its available validation comparison.

## Root causes by reported problem

### A. Banana instance merging

- The API accepts the segmenter's instance list as authoritative. When training labels or inference produce one mask for several touching bananas, the API has no proven second source to establish the missing instance count.
- The saved candidate test has only 46.15% exact count on the 26 banana-cluster images (cluster MAE 0.8462). This directly documents the failure mode.
- OBB-assisted and watershed experiments exist in offline code/reports but are not enabled by the API; the saved OBB-split validation result is below the raw candidate count result (83.39% vs 87.04% exact total count; 28.57% vs 42.86% banana-cluster exact count).
- Banana shape alone is not reliable evidence of multiple fruit. Distance-transform splitting can divide one curved banana or leave multiple touching bananas merged. It must remain opt-in until matched-instance evaluation demonstrates a gain.

### B. Overlapping fruits

- A mask model can return separate masks for overlap only when the learned instance boundaries and labels support it. The current postprocessing cannot recreate invisible boundaries.
- Current quality checks report mask overlap, solidity, and occlusion heuristically. They do not prove a merge, nor estimate a hidden-fruit count.
- Training labels are partly pseudo-label-derived and do not consistently encode visible/amodal geometry, visibility, or annotation uncertainty. This leaves the model without a well-defined target for hard occlusions.

### C. Incorrect counting

- Frontend totals are `objects.length`; the backend does not perform explicit cross-provider instance matching or emit a separate count with uncertainty.
- Current API uses a single segmenter, so its count inherits missed, merged, and duplicate masks. The OBB split helper is offline and combining raw model counts would create duplicate counting rather than a solution.
- Overall exact count on the stored test is 83.77%, but banana-cluster exact count is 46.15%. Overall averages conceal the target failure cases. The report does not contain the per-instance outputs required to retrospectively measure merges/duplicates in that benchmark.
- Hidden fruit cannot be counted reliably from RGB without visible evidence or a separately trained/annotated amodal task. Such counts must not be silently inferred.

### D. Incorrect classification

- Fruit type is first predicted by the segmentation model. Optional CLIP sees only crops produced from those detections; it cannot recover a missed fruit. A wrong crop/merged crop can confuse CLIP, and automatic relabeling risks changing a correct detector result.
- The current classifier only distinguishes fresh/rotten and is run on every fruit crop. It cannot reject a false positive as background. The count benchmarks do not report class confusion or state calibration for the demo examples.
- Saved aggregate counts and class-specific count MAE are not a substitute for per-instance class precision/recall or calibrated confidence.

## Components to retain, disable, or replace

**Retain:** FastAPI upload/health/feedback endpoints; frontend response fields and polygon rendering; coordinate-transform code; device/memory utilities; the custom segmentation checkpoint as the best measured local baseline; `detect_seg` and the existing geometry helpers where covered by tests.

**Keep optional and evidence-gated:** OBB recovery, distance/watershed splitting, CLIP type checks, SAM2 refinement, cloud/Roboflow inference. OBB and watershed do not currently demonstrate a gain on the saved validation count benchmark. SAM2/Roboflow lack a compatible, reproducible end-to-end benchmark/config in this repo.

**Replace incrementally:** implicit model selection in global startup code with validated provider configuration; unstructured tuples with a normalized internal instance record; scattered warning logic with a named quality assessment; list length as the only count result with a count summary tied to one authoritative instance list and explicit uncertainty. Preserve the current JSON contract additively.

## Proposed incremental architecture

1. Introduce `backend/inference/` for configuration, provider selection, and normalized instances. Start with the existing local YOLO segmentation model only; fail fast on invalid task/classes/path. Do not load optional models unless enabled.
2. Introduce `backend/postprocessing/` for same-instance matching, mask checks, class mapping, and count summary. The canonical count remains the length of one deduplicated final instance list. Matching must use class compatibility plus mask/box geometry and scores; a single nested box or mask must not be interpreted as a physical fruit count.
3. Add conservative `needs_review` indicators and provenance/uncertainty diagnostics additively. Never claim a heuristic proves a merge or a hidden count.
4. Add OBB recovery only behind configuration after per-instance proposal recall/duplicate analysis and paired evaluation. Add SAM2 only as a lazy, opt-in refiner for selected prompts after installed dependencies/config, coordinate transforms, latency, and mask quality are validated.
5. Keep CLIP optional and non-authoritative unless a pinned, paired class-validation benchmark establishes thresholds. Keep the state classifier separate from fruit detection and expose uncertain state when calibrated confidence is low.
6. Benchmark fixed manifests and checkpoint hashes. Report per-class metrics, cluster subsets, exact count, MAE, merge/duplicate rates, latency and memory. Do not report mAP or causal improvements unless the required predictions and ground truth are available.

## Risks and limitations

- Saved benchmark JSONs do not include every prediction mask, runtime setting, log or timestamp. Existing aggregates cannot support new mask mAP or per-instance diagnosis.
- Dataset audit identifies pseudo-label and annotation risks; manually reviewed dense-overlap visible-instance masks are still needed for a credible improvement claim.
- The custom candidate is the best stored overall count model but remains weak on banana clusters. It is a demo baseline, not production-validated.
- SAM2 weights being present does not establish the Python package, config, compatible checkpoint, available VRAM, performance or accuracy.
- Roboflow cloud inference has no discovered provider/config/credential. No network call should be made by default.
- Classifier state confidence is not calibrated by the available count reports. State classification quality is outside the cited test metrics.
- Do not delete training data, model weights, annotation/review artifacts, benchmark JSONs, or reports as “junk”: their provenance and future value are documented, and many are untracked user work.

## Cleanup decision

The only objectively disposable project files found during this audit are Python bytecode/cache directories (`__pycache__`, `*.pyc`) and Vite's generated `frontend/node_modules/.vite` cache. Dependency `node_modules`, datasets, model weights, generated review artifacts, run outputs, and untracked reports are retained because they are inputs, reproducibility evidence, or existing user work. Add cache patterns to `.gitignore` so these files do not return as project clutter.
