# Phase 0.5 — Runtime and memory report

## Environment measurement

Measured in this execution environment on 2026-11-18:

- Python 3.11.8; PyTorch 2.5.1+cu121; PyTorch CUDA build 12.1;
  Ultralytics 8.4.180; OpenCV 5.0.0.
- `torch.cuda.is_available()` returned false. `nvidia-smi` could not communicate
  with an NVIDIA driver. GPU model and total/free VRAM are unavailable.
- No CUDA allocation, peak allocation or peak reservation measurement was
  possible. These values are **unavailable**, not zero. The reported 2–3 GB
  target must be re-measured on the deployment computer.

## Runtime changes

- `DEVICE=auto|cpu|cuda[:index]` selects the segmentation device. `auto` logs
  the reason for CPU selection when CUDA is unavailable. An explicit CUDA request
  fails clearly; it does not silently fall back.
- `CLASSIFIER_CHECK` independently controls state-classifier loading/inference.
  Disabled classifier inference preserves the response fields with
  `state="unknown"`, confidence 0.
- `CLIP_CHECK` remains independently configurable. Classifier and CLIP default
  to CPU devices so a CUDA segmentation detector can be the only resident CUDA
  model. Startup rejects configurations assigning multiple enabled models to
  CUDA simultaneously.
- The app keeps models resident; it does not unload/reload them per request.
- Inference memory snapshots report total/free, allocated/reserved and peak
  allocated/reserved MB when CUDA exists. `/api/health` reports selection,
  optional-model state, last inference snapshot and device reason. Predict
  responses add a `runtime` object.
- CUDA OOM returns HTTP 503 with an explicit no-CPU-retry message. `auto` CPU
  selection happens only when CUDA is not detected and is logged/reported.

Configuration examples are in `backend/.env.example`.

## Measurements

| Model/configuration | Device | Latency | Peak allocated/reserved VRAM |
|---|---|---:|---:|
| API inference | No successful API response in this environment | Not measured | Not measurable |
| YOLO11s-Seg, raw masks | CPU FP32, imgsz 640; val 301 / test 345 | 0.1222 / 0.1118 s per image (val/test mean) | Not applicable |
| Custom best.pt, raw masks | CPU FP32, imgsz 640; val 301 / test 345 | 0.1104 / 0.1088 s per image (val/test mean) | Not applicable |

Benchmark details and per-split p95 timings are recorded in
`work/audit/phase05_benchmark_raw_masks.json`. These are CPU timings, not
GPU latency estimates. The separate legacy refinement run is also recorded in
`work/audit/phase05_benchmark.json`.

No compatibility claim is made for a 2–3 GB GPU. Before deployment, run the
benchmark on the target host and inspect `/api/health` plus the `runtime` field
from a representative prediction.
