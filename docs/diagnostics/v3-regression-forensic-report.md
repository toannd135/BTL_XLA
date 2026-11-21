# V3 regression forensic analysis

**Scope:** Existing artifacts only. No training, inference, test evaluation, artifact modification (other than this report and its companion JSON), or promotion. Validation comparison is on 301 images / 362 instances.

## Executive summary

V3 is below V2 on all eight aggregate metrics in the same validation evaluation. The largest gap is mask precision (−0.1583 absolute; −17.7% relative); mask mAP50–95 is −0.0743 (−10.0%). Banana mask recall drops from 0.5329 to 0.4934, and banana mask mAP50–95 from 0.5442 to 0.4142. The artifacts support a real validation regression but do not identify one causal factor: V3 changed training data exposure and optimization settings together.

The run curves do **not** show a simple “training stopped improving” story: all four train losses fall over 40 epochs, V3 validation metrics improve from the first epoch through a best summed mAP50–95 at epoch 38, then soften at epoch 39–40. This is compatible with late overfit or distribution mismatch, but not enough to prove either. V3 remains **NO-GO for promotion**.

## 1. Artifact verification

| Artifact | SHA-256 | Verification |
|---|---|---|
| V2 best checkpoint | `0366c022c85cf6a55c7cd01911444d60adec1d84fc3aa4820699952097783af6` | matches expected |
| V3 best checkpoint | `d6b598df678dd01df05198a9c46f1534e33903b0d4d699b790b052b3b48b7cdc` | matches expected |
| V3 last checkpoint | `46793f73a37a95dfcf08e1c58c1ea7e3390522121cfb588c879431d0154c02fb` | matches expected |
| V2 args.yaml | `49a6d52e45aad5bfce3676c18e82e789552d3b064907ca33fe461e6ca41e1e52` | matches expected |
| V3 args.yaml | `a3d0964a7f90f677df5d770aaa2bae1bc2b811e482647e036188bf44f66d6e58` | matches expected |
| V2 results.csv | `717d9d89e322e7d8097a436b1d2ea67dd51f1f8967bab3b479ad22483eaf718e` | matches expected |
| V3 results.csv | `51d22ad08fa03d0e38e1d08f68a84c463a7f4a4d3a4405b166d13edcbe6491e9` | matches expected |
| Quality-gate YAML | `136642d3cef55160922665f7578051743255c5bd89fc8f9856ff43a75c5e72f9` | matches expected |
| Quality-gate train-list | `72c14ba131fca84730ac7a38e59f2553662cffe883095aa386ebe9aadf37dd96` | matches expected |

V3 has 40 `results.csv` rows; V2 has 50. A V2 console training log was not found in the V2 run directory, `work/audit`, or `runs`; therefore V2 optimizer and `lr0` are reported from `args.yaml` and its recorded LR curve, but its console-resolved optimizer state cannot be independently confirmed. The quality-gate YAML/list hashes match the previously frozen values. The recorded pre/post dataset tree hash is identical (`2efda8…63d1e`), and the training summary records all 5,070 dataset file hashes matching. V2 and V3 were compared on the same quality-gate validation split with `imgsz=640`, `conf=0.001`, `iou=0.70`, AMP enabled. Test was not loaded. See companion JSON for all source artifact paths and hashes.

## 2. V2 vs V3 aggregate metrics

| Metric | V2 | V3 best | Δ absolute | Δ relative |
|---|---:|---:|---:|---:|
| Box precision | 0.7730 | 0.7287 | -0.0443 | -5.7% |
| Box recall | 0.7928 | 0.7246 | -0.0682 | -8.6% |
| Box mAP50 | 0.8610 | 0.8234 | -0.0377 | -4.4% |
| Box mAP50–95 | 0.7537 | 0.7017 | -0.0520 | -6.9% |
| Mask precision | 0.8956 | 0.7373 | -0.1583 | -17.7% |
| Mask recall | 0.6932 | 0.6773 | -0.0159 | -2.3% |
| Mask mAP50 | 0.8332 | 0.7659 | -0.0673 | -8.1% |
| Mask mAP50–95 | 0.7464 | 0.6721 | -0.0743 | -10.0% |

Relative changes use `(V3−V2)/V2`. Class F1 below is the harmonic mean calculated from the saved per-class precision and recall; it is not an additional Ultralytics output metric.

### Per-class results

#### Boxes

| Class | P V2→V3 | R V2→V3 | F1 V2→V3 | mAP50 V2→V3 | mAP50–95 V2→V3 |
|---|---:|---:|---:|---:|---:|
| banana | 0.800→0.758 (-0.042) | 0.710→0.599 (-0.112) | 0.752→0.669 (-0.083) | 0.781→0.726 (-0.055) | 0.562→0.484 (-0.077) |
| apple | 0.784→0.876 (+0.092) | 0.806→0.722 (-0.083) | 0.794→0.792 (-0.003) | 0.891→0.875 (-0.015) | 0.812→0.782 (-0.030) |
| orange | 0.735→0.552 (-0.183) | 0.863→0.853 (-0.010) | 0.794→0.670 (-0.124) | 0.912→0.869 (-0.043) | 0.887→0.838 (-0.049) |
#### Masks

| Class | P V2→V3 | R V2→V3 | F1 V2→V3 | mAP50 V2→V3 | mAP50–95 V2→V3 |
|---|---:|---:|---:|---:|---:|
| banana | 0.861→0.752 (-0.109) | 0.533→0.493 (-0.039) | 0.658→0.596 (-0.062) | 0.743→0.594 (-0.149) | 0.544→0.414 (-0.130) |
| apple | 0.916→0.893 (-0.023) | 0.704→0.695 (-0.008) | 0.796→0.782 (-0.014) | 0.854→0.842 (-0.012) | 0.818→0.779 (-0.039) |
| orange | 0.910→0.567 (-0.343) | 0.843→0.843 (+0.000) | 0.875→0.678 (-0.197) | 0.903→0.862 (-0.041) | 0.877→0.823 (-0.054) |

Per-class F1 values are the harmonic means shown in the table. Banana F1 declines for both boxes and masks; apple/orange differences are class- and task-specific, as the table shows.

## 3. Forty-epoch training curves

The companion JSON stores every V2/V3 epoch row, including train/validation losses, eight validation metrics, and all three learning-rate parameter groups.

- V3 train loss, epoch 1→40: box 0.4691→0.2931; segmentation 1.3009→0.7952; classification 1.1187→0.5184; DFL 1.0566→0.8471.
- V3 validation first→last: box loss 0.7129→0.6179; seg loss 1.6872→1.7130; cls loss 1.1773→1.0850; DFL 1.4916→1.2604.
- V3 aggregate box+mask mAP50–95 peaks at epoch 38 (sum 1.37283). Epoch 38→40: box mAP50–95 0.7013→0.6897; mask mAP50–95 0.6716→0.6664; their sum 1.37283→1.35618.
- V3 validation loss epoch 38→40: box 0.6074→0.6179; segmentation 1.5636→1.7130; DFL 1.2342→1.2604. Classification loss changes 1.1080→1.0850.
- V3 LR pg0/pg1/pg2 at epoch 1: 0.0004761/0.0004761/0.0004761; epoch 40: 0.0000497/0.0000497/0.0000497. V2 has a distinct warmup trace (epoch-1 pg2 `0.0669038`) and finishes with pg0/pg1/pg2 `0.0000149/0.0000149/0.0000149.

**Interpretation:** The first recorded V3 validation row is below the saved V2 best checkpoint on all eight aggregate metrics. Within V3, the metrics subsequently rise unevenly and best fitness occurs at epoch 38. Therefore degradation is present by the first post-update epoch, not only late in training; the artifacts do not include a validation measurement of V3 weights before the first update, so they cannot show precisely when regression began. Epoch 38 is the best observed in-run validation checkpoint under the configured fitness, but the same validation split selected it, so this is not independent evidence of generalization.

**Overfitting:** possible, not proven. Decreasing train losses plus rising validation box/seg/DFL losses from epoch 38 to 40 and a small metric downturn are late-divergence signs. However, the validation metrics are noisy, box/mask losses do not trend monotonically, and most V3 improvements occur over its own run. A 40-epoch single-seed curve cannot distinguish overfit from data-distribution mismatch.

## 4. Banana regression

| Task metric | V2 | V3 | Δ |
|---|---:|---:|---:|
| Box precision | 0.7999 | 0.7583 | -0.0416 |
| Box recall | 0.7102 | 0.5987 | -0.1115 |
| Box map50 | 0.7809 | 0.7258 | -0.0551 |
| Box map50_95 | 0.5619 | 0.4845 | -0.0774 |
| Mask precision | 0.8605 | 0.7516 | -0.1089 |
| Mask recall | 0.5329 | 0.4934 | -0.0395 |
| Mask map50 | 0.7432 | 0.5940 | -0.1492 |
| Mask map50_95 | 0.5442 | 0.4142 | -0.1299 |

**Direct V3 evidence:** banana box recall fell by 0.1115, mask recall by 0.0395, mask mAP50 by 0.1492 and mask mAP50–95 by 0.1299. Banana mask precision fell 0.1089. This indicates weaker banana detection/segmentation on validation but cannot distinguish misses, merges, false positives, mask shape, or class confusion without V3 per-image predictions/matches.

**Related but not V3-specific evidence:** the previous frozen raw-detector benchmark report records 47 unmatched banana GT and 49 unmatched banana predictions among 156 GT / 158 predictions; it also labels several geometric cases as merge or poor-mask candidates, explicitly as unconfirmed triage. The P0 banana_10 review had 4 GT instances and one broad baseline prediction with IoUs 0.341 and 0.247 to two GT, and the user-authorized decision classified that prediction as `merge_error`. These are concrete examples from earlier model artifacts, not evidence that V3 has those same per-image errors. No test split or V3 image-level inference was run for this analysis.

## 5. Run configuration differences

| Field | V2 | V3 |
|---|---|---|
| `model` | `runs/seg/seg_boundary_candidate_gpu/weights/best.pt` | `/mnt/01DC7E1E309A75E0/runs/segment/runs/seg/seg_v2_finetune/weights/best.pt` |
| `data` | `work/obb/seg_data_v2/seg.yaml` | `/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/work/audit/seg_data_training_prep_v1_qg.yaml` |
| `epochs` | `50` | `40` |
| `patience` | `15` | `100` |
| `batch` | `4` | `1` |
| `imgsz` | `640` | `640` |
| `workers` | `4` | `2` |
| `project` | `runs/seg` | `/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/runs/seg` |
| `name` | `seg_v2_finetune` | `seg_v3_qg_controlled` |
| `save_dir` | `/mnt/01DC7E1E309A75E0/runs/segment/runs/seg/seg_v2_finetune` | `/mnt/01DC7E1E309A75E0/CODE/BTL_XLA/fruit/runs/seg/seg_v3_qg_controlled` |
| `optimizer` | `AdamW` | `auto` |
| `lr0` | `0.0005` | `0.01` |
| `lrf` | `0.01` | `0.01` |
| `momentum` | `0.937` | `0.937` |
| `weight_decay` | `0.0005` | `0.0005` |
| `warmup_epochs` | `3.0` | `3.0` |
| `seed` | `42` | `42` |
| `close_mosaic` | `10` | `10` |
| `mosaic` | `1.0` | `1.0` |
| `fliplr` | `0.5` | `0.5` |
| `scale` | `0.5` | `0.5` |
| `translate` | `0.1` | `0.1` |
| `auto_augment` | `randaugment` | `randaugment` |
| `erasing` | `0.4` | `0.4` |

These are all differing keys in the two saved `args.yaml` files. Other saved settings (imgsz, seed, AMP, cache, mask ratio, overlap mask, and listed augmentation fields) are the same. Augmentations are not the leading between-run main-effect hypothesis: the audited augmentation values match in both args files. The V2/V3 dataset paths differ, and the V3 quality gate excluded 62 train images linked to cross-split temporal groups. The quality-gate report records 1,822 remaining train images / 3,613 instances (banana 2,754; apple 448; orange 411), 250 true-crossing annotation rows in 231 files, 0 structurally unusable polygons, 301 validation images, and no negative validation images. The exact V2-vs-V3 image-level train-set difference was not reconstructed here, so its causal impact is unknown.

## 6. Root-cause hypotheses ranked by evidence

1. **Dataset exposure / split and annotation differences — strongest confound, not proven cause.** V2 trained using `work/obb/seg_data_v2/seg.yaml`; V3 used the quality-gate train list. The 62-image exclusion and changed dataset path coincide with the regression. The experiment changes multiple factors, so there is no causal attribution.
2. **Optimizer, learning rate and batch dynamics — settings demonstrably changed.** V2 args specify AdamW, lr0=0.0005, batch 4 (no V2 console log was found to verify any resolved override); V3 says `auto`, batch 1. The V3 log resolves auto to AdamW lr=0.001429, momentum 0.9 and explicitly says it ignored `lr0=0.01`. V3 LR traces differ from V2. This is a plausible mechanism, not proof.
3. **Late overfit or train/validation mismatch — moderate suggestive evidence.** Training losses fall; at epochs 38–40 validation box, segmentation and DFL losses rise and summed mAP50–95 declines. The best epoch is valid as the observed within-run selection, but not independent generalization evidence.
4. **Annotation quality / pseudo-label noise — material risk, unisolated.** Quality-gate found 250 genuine polygon crossing rows. They remain rasterizable, so this does not prove invalid training labels. Earlier banana P0 adjudication confirms one prior detector merge case, not a V3 cause.
5. **Augmentation defaults — weak as an isolated cause.** The audited V2/V3 augmentation settings match. An interaction with changed batch/data may exist but is untested.
6. **Initialization mismatch — evidence argues against it.** V3 references the exact V2 checkpoint hash; training log records 561/561 weights transferred.
7. **Simple banana minority imbalance — unsupported.** Banana is 2,754/3,613 train instances (about 76.3%), so it is not the minority by instance count. Image-level distribution, object scale and occlusion are unavailable; class count alone cannot explain behavior.

## 7. Recommended next experiment

Do not run it under this task. For a separately approved experiment, freeze one exact train manifest and the full resolved recipe. First reproduce V2 optimizer/LR/batch/schedule and augmentation settings on the quality-gate dataset, keeping initialization and validation protocol fixed. Then test the 62-image exclusion as a separate one-factor ablation. Record per-image/instance metrics stratified by class, size, occlusion and source group; retain test untouched. This separates data filtering from optimizer/batch effects better than changing them together.

## 8. Decision

**NO-GO: do not promote V3.** It is worse than V2 on all eight aggregate validation metrics, with pronounced mask precision and banana mask-quality regressions. This conclusion is validation-only; no test metrics were examined. No checkpoint, dataset, production code or configuration was modified.

## Sources

- [V2/V3 args, curves and checkpoints](../../work/audit/v3_regression_forensic_metrics.json) (source hash inventory included).
- [V3 validation and per-class comparison](../../work/audit/seg_v3_validation_comparison.json).
- [V3 training summary and curves](v3-training-summary.md).
- [Dataset quality gate](training-data-quality-gate.md).
- [Earlier detection error analysis](detection-error-analysis.md) and [P0 audit](p0-annotation-audit-report.md); banana findings are clearly qualified as pre-V3 evidence.
