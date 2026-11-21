# Low-VRAM training plan

Status: planned only. Training was not run during Phase 0 because this host has
no observable CUDA device/VRAM and the candidate masks require manual quality
review.

## Initial recipe for a verified 2–3 GB CUDA device

- Model: `yolo11n-seg.pt`.
- Input size: 320; batch size: 1; workers: 0 initially (1 only if stable).
- AMP: enable only after CUDA/device support is confirmed by a short run.
- Cache: `false`; do not cache images in GPU memory.
- Device: explicitly request CUDA and fail/report if unavailable. Never silently
  fall back to CPU for training.
- Save checkpoints frequently and retain last/best plus full run arguments.
- Start with a short memory smoke test, then a few training steps. Only then run
  a full controlled experiment.
- Increase resolution in separate experiments: 416, then 512, only when the
  preceding configuration completes with measured VRAM headroom.
- Gradient accumulation can increase effective batch size, but does not reduce
  per-image activation memory.

Do not reduce annotation review or mask fidelity to fit the GPU. Use a cloud GPU
for training if batch 1 at 320 does not fit or the local GPU is unavailable.

## Existing run for comparison

The saved run `runs/seg/seg_boundary_candidate_gpu/args.yaml` trained a custom
YOLO11s-Seg candidate for 40 epochs at 640 px, batch 8, two workers, AMP on and
cache off. This is a historical run, not a demonstrated fit for the target
hardware. Its dataset manifest identifies fruit polygons as SAM pseudo-labels.
Do not repeat that configuration on the constrained GPU without measured
headroom.

## Controlled comparison matrix

All runs must use the same reviewed train/validation/test manifests and fixed
seed. Keep one change per run and do not tune on test.

| Run | Model | Size | Batch | AMP | Purpose |
|---|---|---:|---:|---|---|
| N320 | YOLO11n-Seg | 320 | 1 | supported only | First memory/quality baseline |
| N416 | YOLO11n-Seg | 416 | 1 | same policy | Resolution effect after N320 fits |
| N512 | YOLO11n-Seg | 512 | 1 | same policy | Resolution effect after N416 fits |
| S320 optional | YOLO11s-Seg | 320 | 1 | same policy | Compare model capacity if it fits |
| OBB optional | existing OBB | n/a | 1 | same policy | Sequential auxiliary evidence comparison, not an independent count source |

For every run record startup/free VRAM, peak allocated/reserved VRAM, OOM status,
image size, batch, AMP, workers, cache, epoch/time, mask metrics, per-class
instance recall and count metrics. No measurements are available yet.

## Resume and OOM policy

- Save `last.pt` at every epoch and keep `best.pt` separately.
- Resume from `last.pt` only with the same dataset/configuration; record any
  changed setting as a new run.
- On CUDA OOM: stop the run, record the traceback and peak memory, clear the
  process, and retry only as a new configuration (e.g. lower size). Do not catch
  OOM and continue as if the run succeeded.
- Do not load detector, OBB, classifier and CLIP onto the GPU together.

The existing training entry point `work/obb/train_seg.py` is a candidate trainer
with a pseudo-label opt-in guard; it is not yet the requested low-VRAM training
workflow and should be adapted only after Phase 1 validation is accepted.
