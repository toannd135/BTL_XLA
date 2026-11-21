# Implementation plan after Phase 0

This is a proposal only. Phase 0 did not change application source. Stop here
until the audit and plan are reviewed.

## Constraints and priorities

1. Preserve all existing weights and data. Keep the current frontend/API behavior
   until a replacement has passing compatibility checks.
2. First verify and review dataset annotations. The existing SAM-derived masks
   are explicitly pseudo-labels; do not train a final model on them as though
   manually verified.
3. Correct and test image coordinate restoration before evaluating overlay or
   mask-boundary quality. `enhance_image()` can upscale inputs without an inverse
   coordinate transform.
4. Keep visible, inferred and uncertain instance records distinct. Do not infer
   a fully hidden fruit from a single RGB image without validated evidence.
5. Optimize for the detected hardware profile, which currently reports no usable
   CUDA device. On the target 2–3 GB device, start at batch 1 and 320 px, then
   measure before increasing resolution.

## Proposed sequence

### Phase 1 — Dataset and annotation infrastructure

- Add `datasets/fruit_occlusion/` organization and schema for visible mask,
  optional separate amodal mask, source/group ID, class, occlusion category,
  visibility ratio and annotation confidence.
- Add validation, visualization, conversion and statistics tools listed in the
  task. Validate normalized polygon range, topology/area, class IDs, per-image
  instance IDs, image/label pairing, source-group split leakage and metadata.
- Keep pseudo-label provenance explicit. Visually review crowded banana,
  touching apple/orange, mixed-class overlap and negative examples. Produce a
  report before any final training.

### Phase 2 — Model experiments

- Start from the existing YOLO11n-Seg checkpoint and batch 1, 320 px, AMP only
  after a CUDA capability check, workers 0–1 and cache disabled.
- Compare against the current generic `yolo11s-seg.pt` and custom candidate using
  the exact same independent test split. Preserve the OBB checkpoint as optional
  comparison/evidence only; load one GPU model at a time.
- Increase 320 → 416 → 512 only after a short forward/train memory test succeeds.
  Record peak allocated/reserved VRAM, latency, mask/count metrics and exact
  config per run. If CUDA is unavailable or OOM occurs, report it and prepare a
  cloud-training recipe; do not silently switch training to CPU.
- Add augmentation one controlled change at a time after labels are validated.

### Phase 3 — Refinement and counting

- Extract boundary/overlap analysis, conditional watershed and instance
  validation into independent modules. Preserve original masks and log why each
  refinement was or was not accepted.
- Use segmentation masks as the count source. Treat OBB as auxiliary evidence
  only and never sum its detections independently.
- Return visible, inferred and uncertain counts with provenance/confidence.
  Initially report no inferred count unless a separately validated component
  supports it.

### Phase 4 — Integration and UI

- Version the API response while retaining compatibility for current fields.
- Correct resize/coordinate transforms and test original-size overlays.
- Keep inferred/uncertain visual styles distinct from visible instance masks.
- Keep one GPU model resident; place mask refinement, matching, counting and
  serialization on CPU. Load optional classifier/CLIP only when needed and
  measure memory before/after each load/unload.

### Phase 5 — Evaluation

- Add geometry, mask, count, duplicate and coordinate tests; add integration
  coverage for API, CPU fallback and GPU memory/OOM behavior.
- Evaluate all 15 scenarios on a source-separated test set. Report visible
  counting separately from inferred estimates and failure cases separately.
- Do not describe the system as production-ready until held-out mask and count
  benchmarks, latency and target-device memory are measured.

## Approval gate

Review Phase 0 documents and decide whether to proceed with Phase 1 annotation
infrastructure. No final model training should start before the annotation
format and dataset quality report are accepted.
