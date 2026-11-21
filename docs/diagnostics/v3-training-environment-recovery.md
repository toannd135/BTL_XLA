# V3 training environment recovery

**Audit date:** 2026-11-19  
**Scope:** Read-only checkpoint and NVIDIA/CUDA diagnostics. No training, benchmark inference, software installation, file deletion, checkpoint replacement, or dataset edit was performed.

## Summary

| Blocker | Finding | Status |
|---|---|---|
| V2 checkpoint | Found outside the repository at `/mnt/01DC7E1E309A75E0/runs/segment/runs/seg/seg_v2_finetune/weights/best.pt`. Its SHA-256 matches both V2 benchmark records. | **GO — verified** |
| NVIDIA host driver/GPU | RTX 3060 Laptop GPU and driver work when checked outside the Codex sandbox. | **GO — host healthy** |
| CUDA in this Codex command sandbox | NVIDIA character devices are absent from the sandbox `/dev`; PyTorch there reports no GPU. The same Python executable sees CUDA when run outside the sandbox. | **NO-GO inside sandbox; GO outside with device access** |
| Dataset configuration | Quality-gate YAML, train-list, validation path/count, and class names match the previously approved values. | **GO — unchanged** |

The earlier GPU failure was caused by device visibility in the command sandbox, not by a missing host GPU, unloaded kernel driver, or CPU-only PyTorch build. The checkpoint path previously supplied relative to the repository was incorrect; the existing checkpoint is in the parent data volume. It was inspected in place and not moved or renamed.

## A. V2 checkpoint recovery

### Located artifact

- **Path:** `/mnt/01DC7E1E309A75E0/runs/segment/runs/seg/seg_v2_finetune/weights/best.pt`
- **Size:** 20,507,549 bytes
- **SHA-256:** `0366c022c85cf6a55c7cd01911444d60adec1d84fc3aa4820699952097783af6`
- **Checkpoint metadata date:** `2026-11-18T16:33:16.196571+07:00`
- **Ultralytics version recorded in checkpoint:** `8.4.180`
- **Task / model class:** `segment`, `ultralytics.nn.tasks.SegmentationModel`
- **Architecture:** YOLO11s-Seg (`yolo11s-seg.yaml`, scale `s`)
- **Class mapping:** `{0: banana, 1: apple, 2: orange}`

### Provenance checks

1. Both `work/audit/seg_v2_phaseA_conf040_iou070_val_cpu.json` and `work/audit/seg_v2_phaseA_conf040_iou060_cpu.json` record this exact absolute checkpoint path and the same SHA-256 above. Both record task `segment`, the same three class names, and input resolution 640.
2. The checkpoint's embedded `train_args` identify the V2 run as `project=runs/seg`, `name=seg_v2_finetune`, `data=work/obb/seg_data_v2/seg.yaml`, 50 epochs, batch 4, image size 640, device `0`, workers 4, AMP enabled, AdamW, and seed 42.
3. The embedded `train_args.model` points to `runs/seg/seg_boundary_candidate_gpu/weights/best.pt`, establishing V2 as a fine-tune from that earlier fruit segmentation checkpoint, rather than the generic COCO `yolo11s-seg.pt`.
4. The run directory contains `args.yaml`, `results.csv` with 50 epoch rows, plots, and both weights files. `args.yaml` records the output directory `/mnt/01DC7E1E309A75E0/runs/segment/runs/seg/seg_v2_finetune`.
5. `weights/last.pt` is a distinct file: 20,507,549 bytes, SHA-256 `f3e912884a088f876f7136cf75ebf3ebbefb1bf5e86aaf2d9630145fd9d7a1e8`.

The serialized `best.pt` has `epoch=-1` and `best_fitness=null`; these fields are not usable for identifying its selected epoch. Provenance is instead established by the exact benchmark-recorded checkpoint hash, embedded V2 training arguments/model metadata, and the run artifacts. The sibling `last.pt` hash differs, so it was not confused with `best.pt`.

### Distinguishing other checkpoints

- Earlier in-repository fruit segmentation candidate `runs/seg/seg_boundary_candidate_gpu/weights/best.pt`: SHA-256 `6da28310bb7627d00dfa574cc36147d215afb48e4d0c3d0705e62ecb05917982`; this is the V2 run's initialization source, not the V2 output.
- Generic `yolo11s-seg.pt`: COCO-pretrained initialization, not the requested V2 checkpoint.
- The V2 `best.pt` is **not** present at the erroneous repository-relative path `runs/segment/runs/seg/seg_v2_finetune/weights/best.pt`; its verified location is the parent-volume absolute path above. No file was copied, moved, renamed, or replaced.

Hashes of V2 sidecar provenance files:

- `args.yaml`: `49a6d52e45aad5bfce3676c18e82e789552d3b064907ca33fe461e6ca41e1e52`
- `results.csv`: `717d9d89e322e7d8097a436b1d2ea67dd51f1f8967bab3b479ad22483eaf718e`

## B. NVIDIA GPU and runtime diagnosis

### Host/device evidence

- PCI enumeration (`lspci`): NVIDIA GA106M / GeForce RTX 3060 Mobile at `0000:01:00.0`; Intel integrated graphics is also present.
- NVIDIA kernel modules `nvidia`, `nvidia_modeset`, `nvidia_uvm`, and `nvidia_drm` are loaded.
- `/proc/driver/nvidia/version`: NVIDIA open kernel module `580.178.04`.
- `/proc/driver/nvidia/gpus/0000:01:00.0/information` identifies an NVIDIA GeForce RTX 3060 Laptop GPU and reports it active.
- In the ordinary sandboxed command, `nvidia-smi` fails with “couldn't communicate with the NVIDIA driver”; `/dev/nvidia*` and `/dev/dri` are absent from that command's `/dev` view. The NVIDIA character-device majors are registered in `/proc/devices`.
- A read-only host-side check outside the Codex sandbox succeeded: `nvidia-smi` reported driver `580.178.04`, CUDA version `13.0`, RTX 3060 Laptop GPU, 6,144 MiB VRAM, 144 MiB occupied, and 0% GPU utilization at the check time. Host `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, and related nodes were present.

### Python and CUDA

- Python executable: `/home/toannd/.pyenv/versions/3.11.8/bin/python` (Python 3.11.8).
- PyTorch: `2.5.1+cu121`; compiled CUDA build: `12.1`.
- Inside the sandbox: `torch.cuda.is_available() == False`, device count 0, and `torch.cuda.init()` reports `No CUDA GPUs are available`.
- Outside the sandbox, using the **same Python executable and packages**: `torch.cuda.is_available() == True`, device count 1, device `NVIDIA GeForce RTX 3060 Laptop GPU`.
- `libcuda.so.1` and `libnvidia-ml.so.1` are installed and discoverable.
- The process is not reporting a Docker/VM virtualization layer (`/.dockerenv` absent and `systemd-detect-virt` reports `none`), but the Codex command sandbox has a restricted `/dev` view. This pinpoints the failure to device-node visibility/access in the sandbox execution context.

**Cause assessment:** Host driver and GPU hardware are operational. The CUDA-enabled PyTorch environment is also operational when it can access host devices. The failure is the sandbox's missing NVIDIA device nodes, not a CUDA build mismatch. No evidence supports installing/upgrading the driver, CUDA, or PyTorch; no such change was made.

## C. Training configuration integrity

- Quality-gate YAML SHA-256: `136642d3cef55160922665f7578051743255c5bd89fc8f9856ff43a75c5e72f9`
- Train-list SHA-256: `72c14ba131fca84730ac7a38e59f2553662cffe883095aa386ebe9aadf37dd96`
- Train-list: 1,822 unique image paths; referenced images and labels exist.
- Validation: original dataset `val/images`, 301 images.
- Class mapping: banana / apple / orange.
- Free disk at audit: approximately 15 GiB (`df` reported 15G available).
- Suggested safe settings read from `docs/diagnostics/training-data-quality-gate.md`: 40 epochs, image size 640, batch 1, device 0, workers 2, cache disabled, AMP enabled, seed 42. The V2 checkpoint may replace the report's generic starting model only by using its verified absolute path in a separately authorized V3 command; this recovery task did not edit the YAML or start training.

## Next action

For a future authorized V3 run, execute the training process in a host execution context that exposes `/dev/nvidia*` to the process. First repeat a read-only preflight there: verify the checkpoint SHA-256 above, YAML/train-list hashes above, `torch.cuda.is_available() == True`, GPU name and available VRAM, and free disk. Use the verified absolute V2 checkpoint path and the already approved quality-gate settings. Do not run training from the current sandbox context unless NVIDIA device access is explicitly available there.

## Safety confirmation

No training or inference benchmark was run. No run directory was created. No checkpoint was modified, moved, renamed, or substituted. No dataset, annotation, YAML, train-list, model/backend code, or installed software was changed.
