# V3 training execution report

**Status: existing V3 run completed and verified; no second run was launched.** The run `seg_v3_qg_controlled` had already completed 40/40 epochs in the preceding authorized execution. This pass rechecked its artifacts and created a post-training hash inventory. It did not start training or load the test split.

## Execution environment

- Execution was on the **host GPU**, not the Codex sandbox: `/home/toannd/.pyenv/versions/3.11.8/bin/python`, Python 3.11.8, PyTorch 2.5.1+cu121, CUDA build 12.1. The host process reported `torch.cuda.is_available() == True` and `CUDA:0 (NVIDIA GeForce RTX 3060 Laptop GPU)`.
- GPU: RTX 3060 Laptop GPU, driver 580.178.04, 6,144 MiB VRAM. No CPU fallback was used.
- Run exit code: 0. Console log includes `Transferred 561/561 items from pretrained weights`.

## Integrity checks

- V2 source checkpoint: `/mnt/01DC7E1E309A75E0/runs/segment/runs/seg/seg_v2_finetune/weights/best.pt`; SHA-256 `0366c022c85cf6a55c7cd01911444d60adec1d84fc3aa4820699952097783af6`. This is the verified V2 checkpoint, not the V1 initialization parent or COCO checkpoint.
- Quality-gate YAML SHA-256: `136642d3cef55160922665f7578051743255c5bd89fc8f9856ff43a75c5e72f9`; train-list SHA-256: `72c14ba131fca84730ac7a38e59f2553662cffe883095aa386ebe9aadf37dd96`.
- Train list: 1822 unique images; validation points to the original `/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/work/obb/seg_data_training_prep_v1/val/images` split with 301 images. Class map is banana / apple / orange.
- Dataset tree SHA-256 before/after: `2efda8f9908eb1e0a249e73aadae9d6b4097ce4e678ec2097bc9dafa17563d1e`; all 5,070 file hashes match the pretraining inventory. The YAML and train-list hashes also match.
- Ultralytics regenerated `train/labels.cache` and `val/labels.cache`; both were restored byte-for-byte to their pretraining hashes from matching source cache artifacts. Image, active label, split, YAML and train-list content is unchanged.

## Run configuration and configuration caveat

- Model: V2 checkpoint above; task segmentation; run directory `runs/seg/seg_v3_qg_controlled`.
- Pinned settings: epochs=40, imgsz=640, batch=1, device=0, workers=2, cache=False, amp=True, seed=42, data=`/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/work/audit/seg_data_training_prep_v1_qg.yaml`, name=`seg_v3_qg_controlled`, exist_ok=False.
- Optimizer recorded as `auto`; log resolved this to **AdamW(lr=0.001429, momentum=0.9)**. Saved `lr0=0.01` was ignored by Ultralytics auto-optimizer.
- Augmentations recorded in `args.yaml`: HSV 0.015/0.7/0.4; degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=10, auto_augment=randaugment, erasing=0.4.
- **Plan limitation:** the quality-gate recipe pinned resource and run settings but did not explicitly specify optimizer or augmentation policy. The older `docs/low-vram-training.md` is marked planned-only and its N320 recipe is not this V2→V3 recipe. Therefore this completed run used Ultralytics defaults for optimizer/augmentation; a new run must not be started until those are explicitly approved. No second run was launched.

## Completion and metrics

- Completed epochs: **40/40**. Best epoch: **38**, using Ultralytics validation fitness (box + mask mAP50–95). Last epoch: **40**.
- Time: `4441.89` seconds from results.csv; host monitor interval `4450.51` seconds.
- Best checkpoint: `runs/seg/seg_v3_qg_controlled/weights/best.pt`, SHA-256 `d6b598df678dd01df05198a9c46f1534e33903b0d4d699b790b052b3b48b7cdc`. Last checkpoint SHA-256 `46793f73a37a95dfcf08e1c58c1ea7e3390522121cfb588c879431d0154c02fb`.

| Validation metric | V2 | V3 best | V3 − V2 |
|---|---:|---:|---:|
| Box precision | 0.7730 | 0.7287 | -0.0443 |
| Box recall | 0.7928 | 0.7246 | -0.0682 |
| Box mAP50 | 0.8610 | 0.8234 | -0.0377 |
| Box mAP50–95 | 0.7537 | 0.7017 | -0.0520 |
| Mask precision | 0.8956 | 0.7373 | -0.1583 |
| Mask recall | 0.6932 | 0.6773 | -0.0159 |
| Mask mAP50 | 0.8332 | 0.7659 | -0.0673 |
| Mask mAP50–95 | 0.7464 | 0.6721 | -0.0743 |

Both checkpoints used the same host-GPU validation protocol: quality-gate validation split only (301 images / 362 instances), imgsz=640, batch=1, conf=0.001, IoU=0.70, AMP enabled. Test was not loaded.

| Class | Box P | Box R | Box mAP50 | Box mAP50–95 | Mask P | Mask R | Mask mAP50 | Mask mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| banana | 0.7583 | 0.5987 | 0.7258 | 0.4845 | 0.7516 | 0.4934 | 0.5940 | 0.4142 |
| apple | 0.8758 | 0.7222 | 0.8753 | 0.7822 | 0.8930 | 0.6954 | 0.8420 | 0.7793 |
| orange | 0.5518 | 0.8529 | 0.8690 | 0.8384 | 0.5674 | 0.8431 | 0.8616 | 0.8227 |

Banana remains the key weakness: V2 → V3 box recall `0.7102 → 0.5987`, mask recall `0.5329 → 0.4934`, mask mAP50–95 `0.5442 → 0.4142`. V3 is below V2 on all eight aggregate metrics above. The checkpoint remains unpromoted.

### Training and validation losses

| Loss | Train epoch 1 | Train epoch 40 | Val at best epoch 38 | Val epoch 40 |
|---|---:|---:|---:|---:|
| Box | 0.46906 | 0.29314 | 0.60740 | 0.61790 |
| Segmentation | 1.30089 | 0.79520 | 1.56359 | 1.71302 |
| Classification | 1.11873 | 0.51839 | 1.10801 | 1.08502 |
| DFL | 1.05657 | 0.84710 | 1.23416 | 1.26036 |

## GPU resources, errors, and warnings

- GPU monitor: 446 samples at 10-second intervals. System-wide VRAM peak **1167/6144 MiB**; Ultralytics trainer-reported peak **0.775 GiB**.
- GPU utilization: max 67%, mean 31.5%; temperature max 72°C.
- CUDA OOM, traceback, or failed-image markers: none. Ultralytics printed an update-available notice; no package was installed or upgraded.

## Artifacts and post-training inventory

- Run artifacts and hashes: [`seg_v3_qg_controlled_posttraining_inventory.json`](../../work/audit/seg_v3_qg_controlled_posttraining_inventory.json).
- Summary and full validation comparison: [`v3-training-summary.md`](v3-training-summary.md) and [`seg_v3_validation_comparison.json`](../../work/audit/seg_v3_validation_comparison.json).
- Run directory: [`seg_v3_qg_controlled`](../../runs/seg/seg_v3_qg_controlled); includes `best.pt`, `last.pt`, `results.csv`, `args.yaml`, plots and validation logs.
- Console log: [`seg_v3_qg_controlled_training.log`](../../work/audit/seg_v3_qg_controlled_training.log); GPU monitor: [`seg_v3_qg_controlled_gpu_monitor.csv`](../../work/audit/seg_v3_qg_controlled_gpu_monitor.csv).

**Final status:** one V3 training run completed successfully operationally. Optimizer and augmentation were not explicitly pinned in the quality-gate plan, so the run is documented with that limitation; no rerun or production promotion was performed. Test evaluation was not run.
