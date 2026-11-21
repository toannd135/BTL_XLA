# Inference memory profile

Status: no GPU inference measurement succeeded during Phase 0. The host reports
`torch.cuda.is_available() == false`, and `nvidia-smi` cannot query a driver.
All GPU-memory values below are therefore **not measured**, not zero.

## Runtime loading observed in source

- Current API default: `yolo11s-seg.pt`; custom candidate can be selected with
  `MODEL_PATH`.
- Classifier `exports/cls_finetune_best.pt` loads at startup and is warmed up.
- CLIP is enabled by default and is constructed at startup when its dependency
  and weights are available. In this environment model-weight lookup fails due
  to unavailable network access; startup catches the error and disables CLIP.
- The current API no longer loads OBB/HBB helpers. OBB utilities/checkpoints
  remain in the repository for older experiments.
- There is no explicit sequential load/unload manager, CUDA memory monitor or
  CUDA OOM policy in the application. The current default can keep more than one
  model object resident; it has not been proven safe on 2–3 GB VRAM.
- OpenCV decode/enhancement, polygon extraction, distance-transform watershed,
  mask metrics and JSON work are CPU operations. Ultralytics device placement
  is automatic; the app currently does not expose per-model residency controls.

## Measured configurations

| Configuration inspected/tested | Result | Peak GPU memory |
|---|---|---:|
| Generic YOLO11s-Seg, 640, stored count reports | Historical JSON metrics exist; run host/device/latency not recorded | Not measured |
| Custom YOLO11s-Seg candidate, 640, batch 8 training args | Historical checkpoint/run exists; not run on this audit host | Not measured here |
| Current API startup with default CLIP | Blocked on model fetch/network; API response not established | Not measured |
| Current API startup with CLIP disabled | FastAPI lifespan reached startup in the attempted server path; socket binding was unavailable, TestClient inference timed out and was stopped | Not measured |
| YOLO11n-Seg at 320/416/512, batch 1 | Not run | Not measured |
| OBB checkpoint | Task/classes inspected only; no inference memory test | Not measured |

Stored count reports do not contain peak VRAM or reliable latency data. Their
aggregate metrics are recorded in `baseline-results.md` and should not be used
as memory compatibility evidence.

## Required profile before enabling target GPU inference

1. Query actual GPU name, total/free VRAM, driver, CUDA build and PyTorch runtime.
2. Load one selected segmentation model only; measure load peak and a warm
   single-image forward at candidate sizes 320, 416, 512 and 640.
3. Record PyTorch `max_memory_allocated` and `max_memory_reserved`, plus external
   process memory samples. Reset peak counters before each case.
4. Measure CPU and GPU latency separately, including preprocessing and postprocess.
5. Load/unload the classifier for one crop, then CLIP only if enabled; verify
   that only the required model occupies GPU memory at each stage.
6. Test graceful CUDA OOM handling and explicitly report CPU fallback on
   inference. CPU fallback during training is not allowed.
7. Repeat with the real API and concurrent requests disabled first; then measure
   concurrency only if single-request headroom is adequate.

No supported resolution/model combination is certified for the target GPU yet.
