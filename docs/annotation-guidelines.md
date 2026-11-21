# Visible-fruit instance annotation guidelines

These rules define the confirmed **visible instance** labels used by the fruit
segmentation dataset. The current source labels are SAM-derived pseudo-labels;
they are candidates for review, not ground truth.

## Instance rules

1. One visible fruit is one instance and receives one polygon mask.
2. A partially occluded fruit that can still be identified receives a mask of
   its visible pixels only.
3. Do not draw hidden boundaries as if they were observed. Trace the visible
   boundary where it meets an occluder.
4. Do not add a fully occluded fruit as a confirmed visible instance. If the
   reviewer has contextual evidence that a hidden fruit may exist, record that
   as a note for later review; do not add it to the segmentation labels.
5. Mark uncertain identity or boundary cases `needs_review` in the review
   manifest and explain the uncertainty. Do not guess a precise boundary.
6. Never combine two distinct visible fruits into one instance mask, even when
   they touch or share a color.
7. Do not create two masks for the same fruit. When two candidate masks appear
   to describe one fruit, compare the original image and resolve manually.
8. Use the fixed class mapping: `0 banana`, `1 apple`, `2 orange`.
9. A negative image has no visible banana, apple, or orange target instance;
   its label file remains present and empty.

Visible-instance masks must remain separate from any future inferred/amodal
labels. If inferred hidden-fruit information is ever collected, use a distinct
label field or dataset version; never mix it into the visible segmentation
polygons.

## Review protocol

- Inspect the original image alongside its overlay and instance IDs.
- Geometry flags, mask size, overlap, model count disagreements, and contact
  sheet selection are review cues only. They do not establish that an
  annotation is wrong.
- Record decisions in `review_status` and `reviewer_notes`. Allowed statuses
  are `pending`, `accepted`, `corrected`, and `needs_review`.
- Preserve source file hashes in the dataset manifest. Corrected exports form a
  new dataset version; do not edit the source candidate dataset in place.
- Have a second reviewer inspect severe occlusion or uncertain identity cases
  before treating them as adjudicated labels.
