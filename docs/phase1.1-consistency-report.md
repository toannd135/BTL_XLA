# Phase 1.1 — Review package consistency check

## Scope

This check read the review manifest, dataset version manifest, review subset,
CVAT ZIP, contact sheets, and Phase 1 audit output. It did not change source
images/labels, checkpoints, or review decisions. Review CSV statuses remain
`pending`, with no reviewer notes entered.

Machine-readable results are in `work/audit/phase1.1_consistency.json`.

## Verified image counts

| Artifact | Unique images | Split counts | Result |
|---|---:|---|---|
| `phase1_review_manifest.csv` | 194 | train 143, val 27, test 24 | 664 instance/negative rows map to 194 unique paths and source paths |
| `seg_data_review_v1/review_subset` | 194 | train 143, val 27, test 24 | 194 distinct image paths and 194 distinct image SHA256 values |
| `cvat_import_review_v1.zip` | 194 | train 143, val 27, test 24 | 194 unique image members and 194 label members; ZIP integrity check passed |
| Contact sheets | 194 | 49 sheets | 48 sheets have 4 cards; the last sheet has 2 cards |

All 194 contact-sheet original-image panels were checked against the
corresponding source image thumbnail at the generator's position. All 194
matched within a mean absolute RGB difference of 15 (maximum observed: 5.70,
from JPEG compression). The CSV's 194 source paths, subset paths, archive image
members, and contact-sheet panels therefore represent the same image set.

## Why the package has 194 images while the old CVAT guide said 180

The selector currently ranks 180 images. The review-package builder also
preserves older pending candidates when its selection changes, rather than
discarding files already present in the review directory. Fourteen earlier
pending candidates were retained, giving the current package 180 ranked images
+ 14 retained images = **194**. The current dataset manifest records all 194;
all remain pending. `docs/cvat-workflow.md` now describes this actual scope and
the retention behavior.

## Split and hash consistency

- The review subset and ZIP both preserve the original train/val/test split:
  143/27/24.
- The version manifest contains hashes for all 2,530 source image/label pairs
  and separate records for all 194 review items.
- Every subset image and label matched its corresponding source SHA256 in the
  `dataset_manifest.json` review-item record; mismatch count: **0**.
- Every image and label member in the CVAT ZIP also matched the review-item
  SHA256; mismatch count: **0**. No source file or review label was rewritten.

## Separate validation results

The validator now has independent `--check structural` and `--check topology`
modes. Both ran on `work/obb/seg_data_review_v1/review_subset`:

| Check | Command result | Findings |
|---|---:|---:|
| Structural YOLO-seg validation | exit 0 | 0 problems; 194 images, 639 instances, 25 empty-label negatives |
| Polygon topology/self-intersection | exit 1 | 151 non-adjacent polygon edge crossing/touch candidates |
| Combined validation | exit 1 | 151 total, all from topology |

The earlier statement that validation found zero problems came from the
structural-only validator version, which did not check polygon topology. A
later Phase 1 report correctly added 151 topology candidates but retained one
stale sentence saying zero total problems. This consistency check separates the
two tests and updates both `docs/phase1-dataset-audit.md` and
`docs/cvat-workflow.md`. Structural acceptance does not clear topology
findings; the 151 polygons still need human inspection.

## Perceptual-hash leakage candidates

The 362 cross-split candidates have a reproducible pair list in
`work/audit/phase1_dataset_audit.json`, at
`leakage.near_duplicate_image_candidates_phash_le4`. Each row identifies both
split/path endpoints and the 64-bit DCT pHash Hamming distance. The canonical
JSON list SHA256 is
`d5e2e7043a9ea63c0db01312e034a036d81ffabea70799efe9b01e8bfb7191ba`.
The recipe is 32×32 grayscale DCT, 8×8 low-frequency hash using the median
excluding DC, cross-split pairs with Hamming distance <= 4.

There are **no dedicated side-by-side contact sheets** for those pairs. The
general 194-image review contact sheets contain both endpoints of zero
candidates; one candidate has a single endpoint represented. Thus the pair
list is reproducible, but manual pair-wise leakage review is not yet prepared.
The 362 pairs are screening candidates, not confirmed leakage.

To reproduce the candidate list without rewriting any review artifacts:

```bash
python - <<'PY'
from pathlib import Path
from tools.phase1_dataset_audit import audit_dataset

audit = audit_dataset(Path("work/obb/seg_data_candidate"))
for pair in audit["leakage"]["near_duplicate_image_candidates_phash_le4"]:
    print(pair)
PY
```

## Files updated

- `tools/validate_yolo_seg_export.py`: added separate `structural`, `topology`,
  and `all` modes so the two findings can be reproduced independently.
- `docs/phase1-dataset-audit.md`: removed the stale zero-total statement and
  documents both validator results and the 180+14 package scope.
- `docs/cvat-workflow.md`: corrected the current review count and documented
  the separate validation commands.
- `work/audit/phase1.1_consistency.json`: machine-readable verification,
  counts, hash results, validator outputs, and pHash-list digest.

No training was started. This phase stops here.
