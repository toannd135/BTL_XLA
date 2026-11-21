# Phase 1.4 — Annotation Review Progress

## Finding

**No human-reviewed CVAT annotation export is present in the repository.** The expected export location described by the workflow, `work/obb/seg_data_review_v1_cvat_export/`, is absent. The only CVAT package found is `work/obb/seg_data_review_v1/cvat_import_review_v1.zip` and its extracted import tree. It is the prepared import package, not a reviewed export.

The repository also contains `out/instance_review_*` diagnostic artifacts. Their manifests describe labels as initially unreviewed and do not provide a CVAT polygon export or explicit human decisions; they were not treated as reviewed annotations.

## Review progress from the original Phase 1 manifest

Counts below use only explicit `review_status` values in `work/audit/phase1_review_manifest.csv`. Unchanged labels do not imply acceptance.

| Split | Accepted | Corrected | Needs review | Pending |
|---|---:|---:|---:|---:|
| Train | 0 | 0 | 0 | 553 |
| Val | 0 | 0 | 0 | 51 |
| Test | 0 | 0 | 0 | 60 |
| **Total** | **0** | **0** | **0** | **664** |

The 664 rows cover 194 images and 639 fruit-instance annotations. The 25 remaining rows are negative-image review entries without instance IDs.

### Class distribution

| Class | Instances in review manifest | Reviewed | Pending |
|---|---:|---:|---:|
| Banana | 492 | 0 | 492 |
| Apple | 72 | 0 | 72 |
| Orange | 75 | 0 | 75 |

### Image-level progress

- Images in review subset: **194**.
- Images with any explicit reviewed decision: **0**.
- Images with recorded corrections: **0**.
- Images with an explicit `needs_review` instance: **0**.
- Images with pending/unresolved review work: **194**.
- Negative images in subset: **25**; negative images reviewed: **0**.

“Unresolved” here includes pending work. There are no explicit `needs_review` decisions in the manifest; all rows remain pending.

## Matching against source and available CVAT import

All 194 image paths in the Phase 1 manifest map to their original source images. The manifest's 639 instance IDs map by source label row order and class. All 194 image/label pairs are present in the available import package, with zero missing or unmatched pairs and no extra images or labels. The available import labels are byte-identical to the source labels for all 194 images; there are zero changed label files.

This equality documents that the available package still contains the original starting annotations. It does **not** establish that a reviewer accepted them. Since no reviewed export exists, **194 images, 639 instance rows, and 25 negative-image review entries have no reviewed-export counterpart**. No CVAT export decisions or corrections can be matched to source paths or instance IDs in this repository state.

## Polygon geometry in the available import package

The structural YOLO segmentation validator passed with zero structural problems: all 639 polygon rows are parseable and structurally valid, with zero empty or malformed masks. The separate topology validator found **151 self-intersection candidates**. The remaining **488** polygons pass that automated topology screen, but this is not human confirmation.

| Geometry measure | Count |
|---|---:|
| Structurally valid polygon rows | 639 |
| Pass automated topology screen | 488 |
| Self-intersection candidates remaining | 151 |
| Empty or malformed masks | 0 |
| Human-confirmed valid polygons | 0 |

The topology findings are candidates requiring review, not automatically assigned annotation errors or corrections.

## Test split protection

The review subset contains 143 train, 27 val, and 24 test images. All 24 test images are in the test package, none is present in the train package, and there are zero split mismatches. No split assignments were changed.

## Stop condition

This report records the available state only. It does not mark any annotation accepted, does not infer a human decision from unchanged labels, and does not start training. The CVAT-reviewed export and explicit reviewer decisions remain outstanding.

Machine-readable results: `work/audit/phase1.4_annotation_review_progress.json`.
