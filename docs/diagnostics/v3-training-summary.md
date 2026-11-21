# V3 controlled training summary

## Outcome

Training completed successfully: **40/40 epochs**, run `seg_v3_qg_controlled`, exit code 0. It fine-tuned the verified V2 checkpoint on the frozen quality-gate train list and used the 301-image validation split each epoch. The test split was not loaded. No backend changes or model promotion were performed.

- Best epoch: **38** (Ultralytics segmentation fitness; sum of box and mask mAP50–95, `1.37283`).
- Last epoch: **40**.
- Training time from `results.csv`: **4441.89 s** (74.0 min); host monitor wall interval **4450.51 s**.
- V3 best checkpoint SHA-256: `d6b598df678dd01df05198a9c46f1534e33903b0d4d699b790b052b3b48b7cdc`. Last checkpoint SHA-256: `46793f73a37a95dfcf08e1c58c1ea7e3390522121cfb588c879431d0154c02fb`.
- Initial V2 checkpoint SHA-256: `0366c022c85cf6a55c7cd01911444d60adec1d84fc3aa4820699952097783af6`.

## Frozen configuration

`imgsz=640`, `batch=1`, `device=0`, `workers=2`, `cache=False`, `amp=True`, `seed=42`, `epochs=40`, `conf=None` (Ultralytics validation default 0.001), `iou=0.70`, project `runs/seg`, name `seg_v3_qg_controlled`, `exist_ok=False`. Console logs confirm `CUDA:0 (NVIDIA GeForce RTX 3060 Laptop GPU)` and `Transferred 561/561 items from pretrained weights`.

- YAML SHA-256: `136642d3cef55160922665f7578051743255c5bd89fc8f9856ff43a75c5e72f9`; train-list SHA-256: `72c14ba131fca84730ac7a38e59f2553662cffe883095aa386ebe9aadf37dd96`.
- Training: 1,822 unique images; validation: 301 images; class order banana/apple/orange.

## V2 vs V3 on the same validation protocol

Both checkpoints were evaluated with the host RTX 3060 on the exact quality-gate `val` split (301 images, 362 instances), `imgsz=640`, `batch=1`, `conf=0.001`, `iou=0.70`, AMP enabled. This is validation-only; no test data was used.

| Metric | V2 | V3 best | V3 − V2 |
|---|---:|---:|---:|
| Box precision | 0.7730 | 0.7287 | -0.0443 |
| Box recall | 0.7928 | 0.7246 | -0.0682 |
| Box mAP50 | 0.8610 | 0.8234 | -0.0377 |
| Box mAP50–95 | 0.7537 | 0.7017 | -0.0520 |
| Mask precision | 0.8956 | 0.7373 | -0.1583 |
| Mask recall | 0.6932 | 0.6773 | -0.0159 |
| Mask mAP50 | 0.8332 | 0.7659 | -0.0673 |
| Mask mAP50–95 | 0.7464 | 0.6721 | -0.0743 |

V3 is lower than V2 on every listed aggregate metric. Banana remains the main weakness: box recall is **0.7102 → 0.5987**; mask recall is **0.5329 → 0.4934**; banana mask mAP50–95 is **0.5442 → 0.4142**. These validation results do not support promotion; no production model was changed.

## V3 per-class validation metrics

### Bounding boxes
| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| banana | 0.7583 | 0.5987 | 0.7258 | 0.4845 |
| apple | 0.8758 | 0.7222 | 0.8753 | 0.7822 |
| orange | 0.5518 | 0.8529 | 0.8690 | 0.8384 |

### Segmentation masks
| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| banana | 0.7516 | 0.4934 | 0.5940 | 0.4142 |
| apple | 0.8930 | 0.6954 | 0.8420 | 0.7793 |
| orange | 0.5674 | 0.8431 | 0.8616 | 0.8227 |

## Losses

| Loss | Train epoch 1 | Train epoch 40 | Validation at best epoch 38 | Validation epoch 40 |
|---|---:|---:|---:|---:|
| Box | 0.46906 | 0.29314 | 0.60740 | 0.61790 |
| Segmentation | 1.30089 | 0.79520 | 1.56359 | 1.71302 |
| Classification | 1.11873 | 0.51839 | 1.10801 | 1.08502 |
| DFL | 1.05657 | 0.84710 | 1.23416 | 1.26036 |

## GPU, errors and integrity

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB. Host samples: 446 at 10-second intervals; observed VRAM used peak **1167 MiB / 6,144 MiB** (system-wide, including the desktop), trainer-reported peak **0.775 GiB**; utilization max/mean **67% / 31.5%**; temperature max **72°C**.
- CUDA OOM/traceback markers: none. Ultralytics displayed an update notice; no package was updated.
- Ultralytics regenerated `train/labels.cache` and `val/labels.cache`. Their original bytes were restored from matching source-cache artifacts; all **5,070** dataset file hashes match the pre-training inventory again. Dataset tree SHA-256 before/after: `2efda8f9908eb1e0a249e73aadae9d6b4097ce4e678ec2097bc9dafa17563d1e`. YAML/train-list hashes also match.

## Artifacts

- Run directory: [`runs/seg/seg_v3_qg_controlled`](../../runs/seg/seg_v3_qg_controlled)
- V2/V3 validation records and per-class metrics: [`seg_v3_validation_comparison.json`](../../work/audit/seg_v3_validation_comparison.json)
- Pre-training input hash inventory: [`seg_v3_qg_controlled_pretraining_inventory.json`](../../work/audit/seg_v3_qg_controlled_pretraining_inventory.json)
- GPU samples: [`seg_v3_qg_controlled_gpu_monitor.csv`](../../work/audit/seg_v3_qg_controlled_gpu_monitor.csv)
- Console log: [`seg_v3_qg_controlled_training.log`](../../work/audit/seg_v3_qg_controlled_training.log)
- Machine-readable summary: [`seg_v3_training_summary.json`](../../work/audit/seg_v3_training_summary.json)

**Status:** training completed; candidate remains unpromoted. Test evaluation was not run.


## Execution configuration clarification

The saved `args.yaml` records `optimizer=auto`; the console log shows Ultralytics resolved it to `AdamW(lr=0.001429, momentum=0.9)`. The quality-gate recipe pinned epochs, resolution, batch, device, workers, cache, AMP and seed, but did not pin optimizer or training augmentations. The run used the saved Ultralytics augmentation defaults (mosaic 1.0, fliplr 0.5, scale 0.5, translate 0.1, and the other values in `args.yaml`). The run completed, but this is a configuration limitation; no second run was started.
