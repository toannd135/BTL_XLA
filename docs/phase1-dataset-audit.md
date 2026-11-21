# Phase 1 — Dataset quality audit and review subset

## Scope and method

This is a read-only audit of `work/obb/seg_data_candidate/`. Source images and
labels were not repaired, moved, or overwritten. The existing masks are
SAM-derived pseudo-labels; structural validity does not establish visual
correctness. Polygon checks rasterize each normalized YOLO-seg instance at its
source image dimensions. Geometric overlap, mask area, perceptual hashes, and
model count disagreement are only review leads.

Reproduce the audit and review package with:

```bash
python tools/phase1_dataset_audit.py
python tools/validate_yolo_seg_export.py work/obb/seg_data_review_v1/review_subset --check structural
python tools/validate_yolo_seg_export.py work/obb/seg_data_review_v1/review_subset --check topology
```

The full per-image/per-instance metrics, file hashes, overlap pairs, issue
details, leakage candidates, and review-group aggregates are in
`work/audit/phase1_dataset_audit.json`.

## Integrity findings

All 2,530 image/label pairs were associated by matching filename stem, and all
source images were readable.

| Split | Images | Instances | Banana | Apple | Orange | Empty-label negatives | Missing label | Orphan label | Self-intersection candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,884 | 3,675 | 2,764 | 473 | 438 | 287 | 0 | 0 | 956 |
| Val | 301 | 362 | 152 | 108 | 102 | 0 | 0 | 0 | 41 |
| Test | 345 | 349 | 156 | 98 | 95 | 71 | 0 | 0 | 44 |
| **Total** | **2,530** | **4,386** | **3,072** | **679** | **635** | **358** | **0** | **0** | **1,041** |

No empty rasterized masks, exact duplicate polygon rows, or same-class
duplicate-mask candidates at IoU >= 0.98 were found. Every annotation row used a
known class and finite normalized coordinates within `[0,1]`. The additional
edge-crossing check found **1,041 polygon self-intersection candidates** (874
banana, 108 apple, 59 orange). A polygon with non-adjacent edges crossing or
touching is flagged as topologically suspicious; these are automated findings
and have not been individually adjudicated by a person. The older dataset
builder's `invalid_polygons: 0` count did not include this topology check.
Empty label files consistently mark 358 images as negative in the dataset
annotations; this is a label-pair consistency check, not a visual guarantee
that every image is free of a target fruit. Twenty-five negatives were included
in the pending manual review subset. There were no images without a label or
labels without an image.

No image contains more than 10 annotated instances (the audit's explicit
high-count review threshold). The dataset does contain **661 images with at
least 3 banana instances**: 631 train, 13 val, 17 test.

### Mask-size and overlap flags

- 425 instance masks cover at least 50% of their image: 274 train, 73 val, 78
  test. This deliberately broad threshold flags close-up crops and potentially
  merged regions for visual inspection; it does not mean these masks are wrong.
- No masks cover less than 0.01% of their image.
- 350 images contain at least one pair of polygons with intersecting raster
  masks, comprising 483 mask-overlap pairs. Split counts are 320/11/19 images
  and 433/22/28 pairs for train/val/test.
- Exact duplicate masks at IoU >= 0.98: 0 candidate pairs. Geometry cannot
  rule out a duplicate with a substantially different boundary.
- Structural validation reports **0 problems** in the prepared subset.
- Separate topology validation reports **151 self-intersection candidates**;
  it exits non-zero until these candidates are manually checked/corrected or
  explicitly adjudicated. See the Phase 1.1 consistency report for why an
  earlier statement of zero total validation problems was stale.

## Occlusion review groups

These groups are **non-exclusive**: an image with at least three bananas can
appear in both `B` (multiple annotations with no raster-mask intersection) and
`C` (dense banana cluster). `C` contains the 661 images with at least three
annotated bananas plus 11 additional banana mask-overlap candidates. `G` marks
an image when one mask intersection covers at least 50% of the smaller mask.
The other overlap groups require intersecting annotation masks. These rules
find candidates, not confirmed occlusion errors. Pair counts include all
intersecting mask pairs in each group image, so group pair counts can overlap.

| Group | Images | Instances | Class distribution | Mask-overlap pairs | Large-mask candidate images |
|---|---:|---:|---|---:|---:|
| A. Single visible fruit | 1,327 | 1,327 | apple 531, banana 390, orange 406 | 0 | 411 |
| B. Multiple, no polygon overlap | 495 | 1,691 | apple 88, banana 1,456, orange 147 | 0 | 10 |
| C. Banana clusters/overlap | 672 | 2,586 | banana 2,574, apple 6, orange 6 | 421 | 1 |
| D. Apple mask overlap | 13 | 51 | apple 51 | 21 | 1 |
| E. Orange mask overlap | 25 | 83 | banana 4, apple 3, orange 76 | 41 | 2 |
| F. Mixed-fruit mask overlap | 6 | 26 | banana 9, apple 6, orange 11 | 13 | 0 |
| G. Severe geometric overlap candidate | 25 | 94 | banana 94 | 64 | 1 |
| H. Negative images | 358 | 0 | none | 0 | 0 |

### Potential annotation risks

The automated signals are concentrated in banana clusters, mask-overlap
regions, and the 425 large-mask candidates. Model-versus-annotation count
differences from the frozen Phase 0.5 benchmark were also used to rank review
images. None of these signals proves that the pseudo-label is merged,
duplicated, or otherwise incorrect; a person must compare the source image to
each visible instance boundary. No human review results are claimed.

## Split leakage check

There were no exact image SHA256 duplicates across train/val/test. A 64-bit
perceptual-hash screen at Hamming distance <= 4 produced 362 cross-split
near-duplicate candidates. This permissive screen can match similar scenes,
backgrounds, or burst photography; manually inspect candidates listed in the
JSON before treating any as leakage. No split was changed.

## Manual review subset

- 194 unique images; 664 CSV rows (639 existing instances plus 25 negative
  image review rows).
- Split distribution: 143 train, 27 val, 24 test.
- All 25 selected negatives remain `pending`; the manifest includes every
  selected instance with class, instance ID, reason, status, and blank reviewer
  notes.
- 49 contact sheets are in `work/audit/phase1_contact_sheets/`. Each card shows
  the original beside its colored polygon overlay, instance IDs, class names,
  filename, split, and selection reason.
- The candidate subset and complete source-pair hash manifest are in
  `work/obb/seg_data_review_v1/`. It contains a review subset, not a copied
  full dataset. Its manifest tracks every source image and label hash, source
  split, source reference, annotation version, and pending/not-selected
  review state.
- The current package has 194 images: the selector ranked 180 and the builder
  retained 14 previously generated pending candidates during a rerun.
- CVAT import archive:
  `work/obb/seg_data_review_v1/cvat_import_review_v1.zip`.
- The structural validator returned **0 problems** on the prepared subset:
  194 images, 639 instances, 25 empty-label negatives. The separate topology
  validator found 151 polygon candidates for human review.

All review statuses are still pending. Geometry flags, benchmark differences,
and contact-sheet selection are not human annotation decisions. See
`docs/annotation-guidelines.md` and `docs/cvat-workflow.md` for the review
protocol. This phase stopped without training, changing source labels, or
starting Phase 2.
