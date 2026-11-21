# Architecture audit

Audit date: 2026-11-18. This records the repository as found during Phase 0. The
worktree already contained uncommitted application and benchmark changes; they
were preserved. No application source was changed for this audit.

## Repository layout

- `backend/`: FastAPI application, Ultralytics inference and mask post-processing,
  CLIP fruit-type verification, the legacy frame collector, and 26 unit tests.
- `frontend/`: React 18 + Vite + Ant Design. `ImagesView` and `LiveView` call the
  API; `ImageBoxes`/`draw.js` render predictions and `ResultTable` supports
  reviewing individual instances.
- `exports/`, repository root, `runs/`, and `work/weights/`: exported detection,
  segmentation, OBB and classifier checkpoints, plus training outputs.
- `own/`, `own_neg/`, `work/obb/seg_data_clean/`, and
  `work/obb/seg_data_candidate/`: detection/classification sources, negative
  images, SAM-derived instance polygons, and a segmentation candidate dataset.
- `tools/`, `work/`, and `docs/`: training, review, dataset, evaluation scripts,
  experiment reports and handoff documentation.

## Models discovered

Checkpoint tasks and class names were inspected with Ultralytics 8.4.180.

| Checkpoint | Task/classes | Observed purpose/state |
|---|---|---|
| `yolo11s-seg.pt`, `yolo11n-seg.pt` | segment, COCO 80 classes | Generic pretrained segmentation checkpoints; banana/apple/orange are COCO classes 46/47/49. |
| `runs/seg/seg_boundary_candidate_gpu/weights/best.pt` | segment, 3 fruit classes | Fine-tuned candidate; trained from `yolo11s-seg.pt` on `seg_data_candidate`, 640 px, batch 8, 40 epochs. The dataset manifest warns that original fruit polygons are SAM pseudo-labels. |
| `det_obb_s.pt`, `exports/det_obb_s.pt`, `work/weights/yolo11s-obb.pt` | OBB, 3 fruit classes | Auxiliary/experimental rotated-box models. The current API does not load an OBB helper. An older OBB splitter remains in `backend/detection.py` and benchmark tooling. |
| `exports/det_A_baseline_backup.pt`, `det_B_clean_backup.pt`, `det_C_candidate.pt`, `det_C_neg.pt`, `det_baseline_best.pt`, `det_finetune_best.pt` | detect, 3 fruit classes | Axis-aligned detector experiments/backups. |
| `yolo11n.pt`, `yolo11s.pt`, `yolo11l.pt` | detect, COCO 80 classes | Generic pretrained detector checkpoints; `yolo11l.pt` is used by the legacy collector as teacher. |
| `exports/cls_baseline_best.pt`, `cls_finetune_best.pt` | classify, 2 classes | Fresh/rotten state classifier; the fine-tuned checkpoint is the API default. |
| `work/weights/sam2.1_b.pt` | SAM2 checkpoint | Used in offline SAM-assisted annotation tooling. |

The API default detector is `yolo11s-seg.pt` (overridable with `MODEL_PATH`),
not the fine-tuned 3-class candidate. Startup rejects non-segmentation models.
The classifier is loaded during app startup. CLIP defaults on and attempts to
load OpenCLIP ViT-L/14 from its model cache/network. OBB and HBB are not loaded
by the current API. The code still uses Ultralytics detection boxes internally
to associate class/confidence outputs and crop objects for CLIP/state
classification; returned fruit geometry is the segmentation polygon.

## Existing inference and API flow

1. `POST /api/predict` accepts an image and `mode` (`fast`/`careful`).
2. `read_image()` validates type, compressed size and pixel count, decodes with
   OpenCV, then calls `enhance_image()`.
3. `detect_seg()` runs the configured YOLO segmentation checkpoint at `imgsz=640`,
   configured confidence/IoU, `max_det=300`, and agnostic NMS. It reconstructs
   outer polygons from binary masks and computes mask area, solidity and
   pairwise mask overlap/occlusion metrics.
4. The current uncommitted runtime path calls a distance-transform marker and
   watershed helper on banana masks. It retains the original mask when the
   split does not pass its internal area/coverage checks. This path has no
   stored benchmark report yet.
5. If CLIP loaded, it can filter or relabel non-banana detections. Current code
   explicitly preserves detected bananas through this step.
6. The classifier estimates fresh/rotten from per-instance crops. The API returns
   polygons, class/state confidence, and mask metrics. It does not return inferred
   or uncertain hidden fruit counts.
7. `POST /api/feedback` stores reviewed predictions. `POST /api/collect` remains
   a separate legacy auto-label collector based on two box detectors and writes
   YOLO detection labels; it is not an occlusion-aware segmentation annotation
   workflow.

Other endpoints: `/api/health`, `/api/feedback/stats`, `/api/collect/stats`.

## Image sizes, coordinates and device selection

- API detector input size is fixed at 640. Classifier input is 224. Inference
  confidence defaults to 0.40 and NMS IoU to 0.70; `HALF=1` passes quantize 16.
- `enhance_image()` upscales images whose short side is below 480, sharpens, and
  applies CLAHE. The returned API width/height and masks use the enhanced image
  dimensions. The frontend displays the originally uploaded image bytes. There
  is no inverse scale transform in `read_image()`/`predict_api()`, so a small
  input image can have misaligned overlays and coordinates.
- Ultralytics chooses its device automatically; the app does not expose an
  explicit CUDA-memory policy or sequential model unloading. Detector and
  classifier objects are both loaded in startup, and CLIP may also be loaded
  when enabled. This is unsafe to assume fits a 2–3 GB GPU.
- Preprocessing, OpenCV mask contour extraction, mask metrics and current
  watershed post-processing run on CPU. The CUDA/CPU placement of Ultralytics
  weights and CLIP is not currently measured per request.
- The legacy collector and rotation/detection utilities have their own box
  coordinate logic. Standard and rotated inference paths are not equivalent in
  the current API: API startup requires a segmentation model and the OBB path is
  not integrated there.

## Root causes and evidence for overlap/count failures

- The generic default segmentation checkpoint is trained on COCO, not on this
  project's fruit instance annotations. It undercounts multi-banana clusters in
  the stored benchmark (see `baseline-results.md`).
- The custom candidate improves aggregate count metrics, but banana cluster
  exact-count accuracy is still only 42.86% on the saved validation report and
  46.15% on the saved test report. Residual cluster MAE is above 0.8 fruit/image.
- Candidate segmentation labels originate from SAM masks prompted by boxes and
  include synthetic bananas. The repo explicitly labels these as pseudo-labels;
  polygon validity alone does not prove that each banana is consistently
  annotated as one visible instance.
- Candidate data has no visible/amodal mask distinction, occlusion severity,
  visibility ratio, annotation confidence, or explicit inferred-instance labels.
- Existing OBB-assisted/watershed experiment reports do not show an improvement
  over unsplit candidate predictions on the saved validation set. Splitting
  from heuristic markers alone is not demonstrated to solve the crowded cases.
- NMS/confidence settings trade missed and duplicate instances. Existing reports
  show materially worse over-counting at lower confidence / higher input size in
  tested detector experiments. These are count benchmarks, not mask mAP.
- A completely hidden fruit cannot be confirmed from one RGB image without
  evidence; no current model or API provides a trained amodal predictor.

## Dataset status and missing annotation requirements

The saved candidate audit reports 2,530 images and 4,386 instances: train 1,884
images / 3,675 instances, validation 301 / 362, test 345 / 349. Class totals are
banana 3,072, apple 679, orange 635. It reports 661 images with at least three
banana labels and 358 negative images (287 train, 71 test). The source manifest
states that fruit polygons are SAM pseudo-labels and says to review them before
production use. These totals do not establish that all annotations are manual
or independently verified. The existing dataset is under `work/obb/`, not the
requested `datasets/fruit_occlusion/` structure.

Before final fine-tuning, manually verify visible per-fruit polygons in crowded
and occluded scenes; keep images grouped by capture source across splits; add
the requested per-instance occlusion metadata; preserve any amodal annotation
separately; and explicitly identify uncertain examples. No hidden boundary
should be inferred as annotation ground truth.
