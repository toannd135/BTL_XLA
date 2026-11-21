# Fresh candidate evaluator implementation and validation audit

**Status:** implemented and validated; inference not run. This is a fresh experiment and makes no claim to recover historical predictions.

## Implemented files

- `work/benchmarks/candidate_eval/config.json` — frozen config with canonical config SHA256 `64d63ac777d656a03f8ea6a7990c4f8942916efacd3ac6ac818af59c0aa376a5`.
- `work/benchmarks/candidate_eval/evaluator.py` — standalone CLI, CLIP swap adapter, polygon rasterization, class-agnostic assignment, schema/invariant checks, atomic per-image persistence, resume gate, event log, and dry-run.
- `work/benchmarks/candidate_eval/test_evaluator.py` — synthetic unit tests; no dataset image is loaded.
- `work/benchmarks/candidate_eval/README.md` — exact behavior and operating instructions.
- `docs/diagnostics/candidate-evaluation-plan.md` — approved design, updated to implementation status.

No prediction JSONL, progress file, or metrics report exists because this validation did not run inference.

## CLIP source semantics pinned

Source `backend/clip_check.py` SHA256: `0adb44e382cb24bae93399b0c4d6f6891b3a1586ed973f69def9eb716fee941c`.

The source flattens prompts in this order:

1. Banana: `a banana`; `a bunch of bananas`; `a rotten banana`; `a green banana`.
2. Apple: `an apple`; `a red apple`; `a green apple`; `a yellow apple`; `a rotten apple`.
3. Orange: `an orange`; `a mandarin orange`; `a tangerine`; `a rotten orange`; `a sliced orange`.
4. Other, in order: `a strawberry`; `grapes`; `a mango`; `a pear`; `a plum`; `a lemon`; `a lime`; `a peach`; `a nectarine`; `an apricot`; `a kiwi fruit`; `a pomegranate`; `cherries`; `a tomato`; `a persimmon`; `a bell pepper`; `a papaya`; `a pineapple`; `a passion fruit`; `a dragon fruit`; `a lychee`; `a fig`; `berries`; `a watermelon`; `a melon`; `a coconut`; `an avocado`; `a potato`; `an onion`; `a vegetable`; `a ball`; `a cup`; `a bottle`; `a human face`; `a hand`; `a table`; `a leaf`; `an empty background`; `an object`; `a toy`; `food`.

Each prompt is formatted as `a photo of {prompt}.`. CLIP computes one softmax across all 55 prompt logits (`100 * normalized_image_features @ normalized_text_features.T`) and sums those probabilities by the four prompt groups using `index_add`. There is no second softmax over groups. Group order is banana=0, apple=1, orange=2, other=3. `argmax` ties select the lowest group ID. The canonical prompt payload SHA256 is `1504ef1f4b753b26c4aa77a5ad9f1f83f6726e1084a86e52934caeedd8900eb1`.

For each detector xyxy box, first-view expansion uses `px=(x2-x1)*pad`, `py=(y2-y1)*pad`; left/top are `int(max(0,min(size-1,edge-expansion)))`; right/bottom are `int(min(size,max(clamped_start+1,edge+expansion)))`. The configured first fraction is `0.10`. An inner view is requested exactly when expanded grouped `other >= 0.80`, with fraction `-0.10`. Its scores replace expanded scores exactly when `max(inner banana/apple/orange) >= 0.90`. Crops convert BGR to RGB PIL and use the validation transform returned by pinned OpenCLIP `create_model_and_transforms`. The fresh swap adapter applies `0.80` to the final grouped top-fruit probability; `other` never drops or relabels.

Malformed, non-normalized, NaN or infinite CLIP vectors raise an error and fail the image. No score is silently replaced or coerced.

## Geometry and matching pinned

- GT coordinates: normalized xy scaled by image width and height. Prediction coordinates: xy in pixel coordinates.
- Rasterization: installed OpenCV Python package `5.0.0.93`, runtime `cv2.__version__ == 5.0.0`; `numpy.rint` (ties-to-even), clamp vertices to `[0,width-1] × [0,height-1]`, `cv2.fillPoly(..., lineType=cv2.LINE_8)`, `uint8`, shape `(height,width)`.
- Fewer than three unique finite vertices, malformed input, and empty polygons rasterize to zero masks. Empty union has IoU 0.
- IoU eligible at `>=0.50`. Assignment uses SciPy `1.17.1` `linear_sum_assignment` to maximize total eligible IoU, with dummy columns for unmatched GT. No class data enters assignment. Among exact optimum ties, GT IDs are visited in order and the lowest available prediction ID that preserves the optimum is selected; unmatched is tried last.
- Prediction records preserve the polygon and also hash the rasterized mask. Candidate and baseline must have identical instance order/count, IDs, confidence, bbox, polygon, mask area and mask hash. Candidate decision metadata is replayed from stored grouped scores before success is permitted.

## Hashes and versions verified

| Artifact | SHA256 / version |
|---|---|
| Detector checkpoint `runs/seg/seg_boundary_candidate_gpu/weights/best.pt` | `6da28310bb7627d00dfa574cc36147d215afb48e4d0c3d0705e62ecb05917982` |
| CLIP weight `open_clip_pytorch_model.bin`, HF snapshot `1627032197142fbe2a7cfec626f4ced3ae60d07a` | `5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b`; 1,710,631,365 bytes |
| Test manifest file | `9b40bfb089e8697347fa1c34842b58c7046eda2f5c5757777cd275fc758fddc1` |
| Detector source `backend/detection.py` | `686a327f39973af224cd91dcca223c60cf78ca69bb5bf0f0be042eb8e9af7fd6` |
| Evaluator source | `40d55f0aacc7c6052730d3e4a916d2369eae3032f2383cb6970dd05aa01a9b86` |
| Test source | `67705fff930c9365a7f84fb3c6ba3d853efaa4045e5dde3f0a8863f232835e36` |
| Config canonical digest | `65f552c3b25f545ab20643d986dd0777e8f7d30b482e889d4953ae223b0b53b4` |
| Runtime | Python 3.11.8; OpenCLIP 2.30.0; torch 2.5.1+cu121; torchvision 0.20.1+cu121; Ultralytics 8.4.180; OpenCV 5.0.0.93/runtime 5.0.0; NumPy 2.4.6; Pillow 12.3.0; SciPy 1.17.1; huggingface_hub 2.4.0; jsonschema 4.26.0 |

The evaluator checks all these artifact hashes and versions before model construction. Config digest verification excludes only the `config_sha256` field itself. The implementation and test source hashes in the config are checked too.

## Validation performed

- Static compilation: `python -m py_compile work/benchmarks/candidate_eval/evaluator.py work/benchmarks/candidate_eval/test_evaluator.py` — passed.
- Synthetic tests: `python -m unittest discover -s work/benchmarks/candidate_eval -p 'test_*.py' -v` — **22 passed**. Coverage includes prompt aggregation/order hash, threshold/ties/NaN handling, crop clipping and second-view selection including float32 box arithmetic, class-agnostic matching, overlaps, equal-IoU ties, unmatched items, empty/invalid masks, class comparison after geometry match, schemas, swap invariants, config/artifact hashes, atomic write, and resume validation. No test images used.
- Dry-run: `python work/benchmarks/candidate_eval/evaluator.py --dry-run` — **ok**; 345 manifest entries, 345 unique images, 345 valid image/label pairs, 0 invalid entries. Config digest, checkpoint/CLIP/source/manifest hashes and all pinned package/runtime versions passed.
- Dry-run did not load either model, decode images, write output records, or modify dataset files.

## Scope and remaining gate

The evaluator only runs inference when called with `--run-inference`; the command used here was `--dry-run`. No inference, training, or metrics generation occurred. No dataset annotation, model checkpoint, or production backend source was changed by this implementation. The approved config is ready, but execution of the 345-image experiment remains behind the user's explicit approval.
