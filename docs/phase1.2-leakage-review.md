# Phase 1.2 — Cross-split leakage visual review package

## Source verification

The candidate pairs came from
`work/audit/phase1_dataset_audit.json` at
`leakage.near_duplicate_image_candidates_phash_le4`. The list contains 362
pairs. Its canonical JSON SHA256 was recomputed as
`d5e2e7043a9ea63c0db01312e034a036d81ffabea70799efe9b01e8bfb7191ba`, matching
the supplied Phase 1.1 value. The generator verified both source-image
SHA256 values for every pair against the original audit manifest before
writing the review files.

The pairs refer to 115 unique source images. Pair split combinations are 332
train–val and 30 train–test; there are no val–test pairs. Distances are 99
pairs at 0, 170 at 2, and 93 at 4. These remain pHash candidates; no decision
was inferred from distance or appearance.

## Generated files and verified counts

- `work/audit/phase1.2_leakage_review_manifest.csv`: 362 rows, IDs
  `leakage_pair_0001` through `leakage_pair_0362` in canonical source-list
  order.
- `work/audit/phase1.2_leakage_contact_sheets/`: 91 JPEG sheets. The first 90
  show four pairs each; the last shows two. Each pair shows both original
  images side by side, each relative path and filename, split, pHash Hamming
  distance, and original image dimensions. Images in a pair use the same
  pixel scale.
- `tools/phase1_2_leakage_review.py`: deterministic generator. It verifies the
  canonical pair-list checksum and source image hashes, and refuses to
  overwrite existing CSV or sheet outputs.

The CSV has all requested columns plus `image_a_sha256`, `image_b_sha256`, and
`split_recommendation`. All 362 `review_status` values are `pending`;
`reviewer_decision` and `reviewer_notes` are blank. Validation confirmed
pair IDs, source order, original distances, and both SHA256 values for every
row. Hash mismatch count: **0**.

Allowed review statuses are `pending`, `same_capture`, `near_duplicate`,
`similar_scene`, `unrelated`, and `needs_review`. Repeated images appear in
multiple candidate pairs; human reviewers should consider connected pair
groups together before recommending any future split change.

## Conditional split recommendations

Each CSV row has a recommendation based on the current split pair, but no
change was applied:

- If later confirmed to come from the same capture/session, keep the images in
  one split in a future dataset version. For train–val pairs, choose one common
  non-test split.
- For pairs involving test, keep both in test in a future version by moving
  only the non-test counterpart. This keeps that capture out of training and
  preserves test independence. Never move a test image into training.
- If the pair is only visually similar or unrelated, retain the current split
  assignments.

## Reproduction

From the repository root:

```bash
python tools/phase1_2_leakage_review.py
```

The command writes the CSV and contact sheets at the paths above. It stops if
those outputs already exist; use new `--csv` and `--sheets` destinations to
make another copy without overwriting review work.

The source dataset, train/val/test membership, existing 194-image annotation
review package, and review decisions remain unchanged. No model training was
started. This phase stops after preparing the pair-review package.
