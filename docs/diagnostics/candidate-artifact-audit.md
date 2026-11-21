# Candidate evaluation artifact forensic audit

Audit scope: repository `fruit/`, its benchmark, diagnostics, classifier and run directories, plus matching files found under `/tmp` and the project CLIP cache. No files were deleted or overwritten. No inference was run during this audit.

## Findings

The repository contains a test summary for the old CLIP swap-only experiment, but not the per-image instance predictions or the code path that produced those reported swaps. A separate reproducible segmentation benchmark contains the 345-image manifest and aggregate per-image counts, but no polygons or masks. The claim that 200 images were processed is not supported by a progress manifest or per-image candidate outputs. Those 200 results cannot be verified or safely resumed.

Phase 2 is therefore **blocked**. The available records identify some settings, but do not determine the exact historical crop generation and class-swap implementation used by the test summary. In particular, the summary gives no CLIP prompt-set/version hash, crop/bbox source, per-instance CLIP scores or decisions, and no code revision or inference log linking the summary to the current `clip_check.py`. No `locked_config.json` is created, because filling in these gaps would require guessing.

## Artifact inventory

Hashes below are SHA256. File sizes are bytes. “Reusable” means reusable for the stated purpose without inferring instance predictions from aggregates.

| Path | Size | SHA256 | Contents and reuse assessment | Missing data |
|---|---:|---|---|---|
| `work/audit/seg_old_clip_swaponly_test_final.json` | 3,084 | `668e70f3de1c1d6ab165e42f494526da5cb94c43106d5899ca29a6085f0f32db` | Old test summary. Records candidate checkpoint, 345 images, `imgsz=640`, `conf=0.4`, detector `iou=0.7`, CLIP ViT-L-14 / `laion2b_s32b_b82k`, CPU, swap threshold 0.8, and two swaps. Reusable only as a historical aggregate claim. | No image IDs, prediction instances, masks, classes before/after, GT refs, manifest hash, config hash, logs, or errors. `raw` is aggregate only. Does not prove 345 successful inferences. |
| `work/audit/seg_old_clip_swaponly_val.json` | 2,153 | `0f9a2121a7b9fcf90c4525ebabdfad58e29b365ae8fee7d0ee56c46fd0f53e19` | Validation summary; records 301 images, threshold 0.8 and five swaps; includes validation manifest hash `9fc1e01f...`. Reusable as historical aggregate/config clue. | No predictions, per-image swap decisions, mask records, or exact code/config hash. |
| `/tmp/clip_swap_only_old_val.json` | 1,908 | `57d4b852cf8afb42514e35ce46e506fe3d4db09e80f791769708e8448982e731` | Duplicate/related validation summary; specifies detector checkpoint path and parameters matching the test summary. Reusable as a clue only. | No instance outputs or transformation implementation. Not a test progress record. |
| `runs/seg/seg_boundary_candidate_gpu/weights/best.pt` | 20,513,629 | `6da28310bb7627d00dfa574cc36147d215afb48e4d0c3d0705e62ecb05917982` | Checkpoint named in the old test summary; segment model. Reusable as detector checkpoint identity. | Does not encode the external CLIP transformation or its exact run configuration. |
| `runs/seg/seg_boundary_candidate_gpu/weights/last.pt` | 20,513,629 | `58c7f4106c1aec0e497a4d5f7a42d9bd4841a93512af21f7e823b4a63127ccba` | Separate last checkpoint from same run. Not the checkpoint named by the old summary; not substituted. | Not evidence of the old evaluation checkpoint. |
| `yolo11s-seg.pt` | 20,669,228 | `1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3` | COCO segmentation checkpoint used as baseline in the separate raw-mask benchmark. Reusable for that benchmark only. | Not identified as the baseline in the old CLIP summary. Different detector from the candidate checkpoint. |
| `work/audit/phase05_benchmark_raw_masks.json` | 565,009 | Not recorded in this pass | Reproducible benchmark summary for COCO and custom checkpoints; configuration is `imgsz=640`, `conf=0.4`, detector IoU 0.6, CPU FP32, no distance refinement. Per-image rows contain counts and aggregate mask-match counts, not geometry. | No polygons/masks or instance-level matches; config differs from old test’s IoU 0.7; not a source for CLIP candidate predictions. |
| `work/audit/phase05_benchmark_raw_masks_manifests/test-94a849d857800128dbb9a8c8c1348d3b2f4f58134eed675ea042876f698d6698.json` | 119,467 | `9b40bfb089e8697347fa1c34842b58c7046eda2f5c5757777cd275fc758fddc1` | Test split manifest: 345 entries with image and label paths/hashes; manifest ID `94a849d857800128dbb9a8c8c1348d3b2f4f58134eed675ea042876f698d6698`. Reusable as the manifest for the separate raw-mask benchmark. | Old CLIP test summary does not cite this manifest, so identity with its evaluated input set is not established. |
| `work/audit/seg_candidate_count_test.json` | 89,014 | `f02a3530ba82f65d247aea86b22518494eb71f829c5065696a3d5b503e74e66a` | Custom detector count benchmark; 345 named per-image rows. Reusable for total/per-class counts present in this file. | No per-instance classes, polygons or masks; no CLIP changes; cannot support requested matching or paired classification analysis. |
| `work/audit/seg_coco_baseline_count_test.json` | 88,422 | `b0cd672d52bd87cad6bf501a8b808b0be054800902ec8b1bc81aa24bdba54bb3` | COCO baseline count benchmark; 345 named per-image rows. Reusable for counts only. | No instance geometry or CLIP outputs; detector parameters are not established as identical to old test experiment. |
| `tools/benchmark_seg_reproducible.py` | 16,975 | `fa38588989cf90e38ff1b64d64b51dd90017d48b7909f6405f96fab023f73c5d` | Reproducible segmentation benchmark implementation. It runs detector and matching but saves summary/image rows rather than instance polygons. Its matcher filters pairs by equal class, so it is not suitable as-is for the required class-agnostic matching. | Not the old CLIP swap-only runner; no resumable per-image prediction store. |
| `backend/clip_check.py` | 5,340 | `0adb44e382cb24bae93399b0c4d6f6891b3a1586ed973f69def9eb716fee941c` | Current CLIP checker code. Defines prompts, template, grouped probabilities, preprocessing via `open_clip`, and `probs_boxes` crop parameters (`pad=0.10`, `inner=-0.10`, `t_other=0.8`, `t_inner=0.9`). Reusable as current implementation evidence only. | No proof this exact file/version generated the old test artifact; `probs_boxes` includes a second-look path unlike the summary’s explicitly swap-only behavior. No historical source hash is attached to the summary. |
| `work/fruitcls/tuneL_test.log` | 2,112 | `e603ccc3f785fbb4a50c124a4e202879eb4d47701cb4dd1fe7783e751b1d9da4` | CLIP prompt tuning report on 349 fruit crops and 1,397 detector “other” crops, with prompt sets and thresholds. | Not the 345-image segmentation test inference log; does not store crops or per-instance outputs and is not tied to `seg_old_clip_swaponly_test_final.json`. |
| `work/fruitcls/eval_gate_clip_test.json` | 3,751 | `54c9216d7cf420263b19ed77b80eb7792330dc9025c7831989fc8071d814091d` | Separate detector/CLIP gate aggregates. | Different detector (`exports/det_obb_s.pt`), settings and purpose; not the old candidate run. |
| `work/weights/hf/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/...` | varies | Not recorded | CLIP cache metadata/lock artifacts were found. | No complete checkpoint file was established in the search; no verified model revision or content hash for the CLIP weights. |
| `work/diag/**/crops/*` | varies | Not recorded | A few manually generated diagnostic crops exist. | They are not keyed to the 345-image manifest and do not contain a complete prediction or class-change record. |

Search also found no candidate-evaluation progress manifest, 200-image inference log, per-image candidate prediction cache, or candidate crop directory tied to the 345-image test. The only `/tmp` match was the validation summary listed above. Ordinary training logs and unrelated evaluation files are not evidence for this run.

## Phase 2 configuration recovery

| Field | Recoverable evidence | Status |
|---|---|---|
| Detector/checkpoint | `seg_boundary_candidate_gpu/weights/best.pt`; hash above | Identified as the detector named by the test summary |
| Detector runtime parameters | `imgsz=640`, confidence `0.4`, NMS IoU `0.7`, CPU, FP32 | Reported by summary; exact software/runtime versions absent |
| Candidate classifier family | OpenCLIP ViT-L-14, pretrained label `laion2b_s32b_b82k`, CPU | Reported by summary, but model artifact/revision hash absent |
| Candidate rule | Swap-only and threshold 0.8; two swaps reported | Broad rule is described in the validation note, but no per-instance scores or executable historical implementation ties it to test output |
| Class mapping | Detector names banana/apple/orange; current CLIP code groups in that order plus `other` | Current code evidence; historical mapping not pinned by test artifact |
| Crop generation/preprocessing | Current `clip_check.py` uses detector boxes, 10% expanded crop, PIL RGB conversion and OpenCLIP preprocess; second-look crop also exists | **Not established as the old swap-only run’s exact path.** No archived runner/crop settings/version reference; cannot select among code paths safely |
| Baseline identity | Separate raw benchmark compares COCO `yolo11s-seg.pt` and the custom checkpoint; old CLIP artifact only reports a raw-vs-candidate block for its custom detector | **Not established unambiguously for the requested baseline comparison** |
| Test manifest | Separate raw benchmark manifest has 345 images and hash `94a849d8...` | Available, but old CLIP summary has no manifest reference |

Because the exact candidate transformation and its connection to the test summary are not recoverable, the requested immutable `work/benchmarks/candidate_eval/locked_config.json` is intentionally not created. No inference or evaluation was started. The final report, prediction stores, resumable runner and metrics are also not created from incomplete evidence.

## Safe next step

Run a new, explicitly authorized controlled evaluation using a newly pinned config and code hash, writing per-image baseline/candidate instances and progress atomically. Treat it as a fresh evaluation; do not represent the reported 200 as reusable. Before that run, resolve the intended baseline checkpoint and freeze the exact detector-to-CLIP crop path, CLIP weight revision, prompts, class mapping and swap condition. This audit does not make those choices on the user's behalf.
