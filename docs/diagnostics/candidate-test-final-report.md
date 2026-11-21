# Fresh CLIP swap-only test evaluation

**Status:** Complete, 345/345 images validated. Fresh experiment; historical partial evaluation was not reused.

Config SHA256: `65f552c3b25f545ab20643d986dd0777e8f7d30b482e889d4953ae223b0b53b4`  
Detector SHA256: `6da28310bb7627d00dfa574cc36147d215afb48e4d0c3d0705e62ecb05917982`  
CLIP weights SHA256: `5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b`  
Manifest: `94a849d857800128dbb9a8c8c1348d3b2f4f58134eed675ea042876f698d6698` (345 images)

## Detection and counting

| Metric | Baseline | Candidate |
|---|---:|---:|
| Detection precision | 0.7363 | 0.7363 |
| Detection recall | 0.7679 | 0.7679 |
| Detection F1 | 0.7518 | 0.7518 |
| False positives | 96.0000 | 96.0000 |
| False negatives | 81.0000 | 81.0000 |
| False positives / image | 0.2783 | 0.2783 |
| False negatives / image | 0.2348 | 0.2348 |

| Count metric | Baseline | Candidate |
|---|---:|---:|
| Total count MAE | 0.2348 | 0.2348 |
| Exact count match rate | 0.8290 | 0.8290 |
| Overcount images | 33.0000 | 33.0000 |
| Undercount images | 26.0000 | 26.0000 |

## Classification

| Class | Arm | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| banana | baseline | 0.6835 | 0.6923 | 0.6879 |
| apple | baseline | 0.7379 | 0.7755 | 0.7562 |
| orange | baseline | 0.7670 | 0.8316 | 0.7980 |
| banana | candidate | 0.6879 | 0.6923 | 0.6901 |
| apple | candidate | 0.7451 | 0.7755 | 0.7600 |
| orange | candidate | 0.7619 | 0.8421 | 0.8000 |

Matched-class accuracy: baseline 0.7536; candidate 0.7564.
Macro F1: baseline 0.7474; candidate 0.7500.
Swaps: 2. Correct-to-wrong: 0; wrong-to-correct: 1.

## Negative images

Negative images: 2 images with baseline FPs and 2 with candidate FPs among 71 negatives.

## Paired uncertainty

Image-level paired bootstrap, 2,000 replicates; intervals are 95% percentile CIs for candidate-minus-baseline differences.

- Matched-class accuracy difference CI: [0.0000, 0.0093]
- Macro F1 difference CI: [-0.0018, 0.0104]
- Exact count rate difference CI: [0.0000, 0.0000]
- Mean matched mask IoU difference CI: [0.0000, 0.0000]

## Completeness and recovery

Resume validation accepted 345/345 record triplets. Recovered incomplete attempts: 1.
The initial orchestration attempt for the recovered image is retained in the event/error records; its incomplete output was reprocessed and the final triplet passed validation.

## Artifacts

- Per-image paired metrics: `work/benchmarks/candidate_eval/full_runs/fresh_clip_swap_only_345/candidate-test-per-image.csv`
- Confusion matrix CSV: `work/benchmarks/candidate_eval/full_runs/fresh_clip_swap_only_345/candidate-test-confusion-matrix.csv`
- Negative-image CSV: `work/benchmarks/candidate_eval/full_runs/fresh_clip_swap_only_345/candidate-test-negative-images.csv`
- Machine-readable metrics: `work/benchmarks/candidate_eval/full_runs/fresh_clip_swap_only_345/report.json`
- Predictions, progress, execution summary, and logs are preserved in `work/benchmarks/candidate_eval/full_runs/fresh_clip_swap_only_345/`.

## Recommendation

**NO-GO** — Candidate did not meet the predeclared evidence gate for promotion consideration; no automatic deployment.

This report does not deploy or automatically promote the candidate.
