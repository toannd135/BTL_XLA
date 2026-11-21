# Manual polygon review workflow with CVAT

CVAT Community is self-hosted and open source, so this workflow does not
require a paid annotation service. Use the [CVAT Community installation guide](https://docs.cvat.ai/docs/administration/community/basics/installation/)
for the current Docker Compose setup. This repository does not install or start
CVAT automatically.

## Prepared review package

The source dataset remains at `work/obb/seg_data_candidate/` and is unchanged.
The Phase 1 package contains 194 unique review images. The selector's current
deterministic quota is 180; 14 images from the earlier generated selection were
retained when the selector changed, so reruns would not silently discard
existing review files. All 194 remain pending. The package contains:

```text
work/obb/seg_data_review_v1/
├── dataset_manifest.json       # hashes/splits/version/status for all source pairs and subset
├── cvat_import_review_v1.zip   # subset images + starting polygon annotations + class YAML
└── review_subset/
    ├── data.yaml
    ├── seg.yaml
    ├── train/images, train/labels
    ├── val/images, val/labels
    └── test/images, test/labels
```

Class mapping is fixed and ordered: `0 banana`, `1 apple`, `2 orange`. The
starting polygons are the existing SAM pseudo-labels and are **not** marked as
human-reviewed. CVAT's Ultralytics YOLO format supports polygon/mask instance
segmentation and uses one normalized polygon row per instance. The archive
layout supports split folders containing `images/` and `labels/`.
[CVAT Ultralytics YOLO format reference](https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo-ultralytics/).

## Import and correct annotations

1. Start a local CVAT Community instance and open it in a supported browser.
2. Create a project named `fruit-visible-instances-review-v1`.
3. Add labels in exactly this order and spelling: `banana`, `apple`, `orange`.
4. Import `work/obb/seg_data_review_v1/cvat_import_review_v1.zip` as an
   **Ultralytics YOLO Segmentation 1.0** dataset. Confirm each image is paired
   with the same-stem text annotation. If your CVAT build asks to create a task
   from the data archive, create one task for this review project.
5. Review the original image and initial polygon overlay. Keep one mask per
   visible fruit, correct polygon vertices along visible boundaries, and never
   add hidden/amodal fruit as a confirmed instance. Follow
   [annotation-guidelines.md](annotation-guidelines.md).
6. For uncertain boundary or identity, leave the case unresolved and record
   `needs_review` plus a short note in
   `work/audit/phase1_review_manifest.csv`. Do not report a reviewer decision
   until a human has made it.
7. Export the corrected task/dataset as **Ultralytics YOLO Segmentation 1.0**.
   Save the export into a new immutable version directory, for example
   `work/obb/seg_data_review_v1_cvat_export/`; do not export over the source
   candidate or the prepared review subset.
8. Extract the export and run the validator before accepting it:

```bash
python tools/validate_yolo_seg_export.py work/obb/seg_data_review_v1_cvat_export
```

Run the checks separately so structural validity is not confused with polygon
topology:

```bash
python tools/validate_yolo_seg_export.py work/obb/seg_data_review_v1_cvat_export --check structural
python tools/validate_yolo_seg_export.py work/obb/seg_data_review_v1_cvat_export --check topology
```

The structural pass checks class order, image/label pairing, polygon row shape,
finite normalized coordinates, non-empty geometry, and duplicate masks. The
topology pass reports non-adjacent polygon edges that cross or touch. A topology
finding is a review candidate, not a decision about whether the polygon follows
the visible fruit boundary. The Phase 1 prepared subset currently has zero
structural problems and 151 topology findings; review them in CVAT before
accepting the export.

## Dataset versioning after review

Keep the CVAT export unchanged as the review source. Build any future training
candidate into another directory, preserving its source hash manifest and
mapping each corrected file to the original split. Record each image and label
as `accepted`, `corrected`, or `needs_review` in a new manifest. Only reviewed
and adjudicated visible-instance annotations should be eligible for later
training. Do not treat contact-sheet selection or geometric flags as review
outcomes.

The Ultralytics YOLO segmentation format's documented polygon row structure
and split layout are described in the [CVAT format reference](https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo-ultralytics/).
