# Phase 1.3 — Human-review workflow and decision validation

This workflow covers the existing 194-image annotation review set and all 362
cross-split pHash candidate pairs. It does not regenerate either package,
change source labels or split assignments, or train a model. Keep the original
CSV values `pending` until a reviewer has inspected the evidence.

## Annotation review

Use `work/audit/phase1_review_manifest.csv` with the contact sheets in
`work/audit/phase1_contact_sheets/` and the matching original images. Review
each image and visible instance against the source image:

- One visible fruit is one instance.
- For a partially occluded but identifiable fruit, annotate visible pixels
  only. Do not draw hidden boundaries as if they were visible.
- Never infer a fully hidden fruit as a confirmed visible instance.
- Never merge separate visible fruits into one mask or duplicate one fruit's
  mask.
- Mark uncertain identity or boundaries `needs_review` and explain why in
  `reviewer_notes`.
- Carefully inspect banana clusters and the polygon topology candidates
  reported by the structural/topology validator. These are review priorities,
  not evidence that a mask is wrong.
- Keep the fixed class mapping: banana, apple, orange. For negative-image rows,
  review the image itself; the blank class/instance fields represent the
  image-level negative check.

The annotation manifest is instance-oriented: an image with several instances
has several rows. Set `review_status` for each reviewed instance. Its allowed
values are `pending`, `accepted`, `corrected`, and `needs_review`. The current
CSV has no separate `reviewer_decision` column; `review_status` is the recorded
annotation outcome. If a decision column is added to a working copy, use the
same final value as the status. `needs_review` requires a non-empty note. Do
not write inferred/amodal instances into these visible-instance annotations.

## Leakage pair review

Use `work/audit/phase1.2_leakage_review_manifest.csv` and the side-by-side
pages in `work/audit/phase1.2_leakage_contact_sheets/`.

1. Build connected groups from pair endpoints: the same image can occur in
   several pairs. Review a connected group together before proposing any
   future split correction. A connected group does not itself prove that all
   images share a capture session.
2. Compare content and any available source evidence to distinguish the same
   capture/session from similar scenes or unrelated images.
3. Do not decide from pHash distance alone. A low distance is a candidate
   signal, not proof of identity or leakage.
4. Set both `review_status` and `reviewer_decision` to the same final category:
   `same_capture`, `near_duplicate`, `similar_scene`, `unrelated`, or
   `needs_review`. Leave the decision blank while status is `pending`.
5. Record the evidence and reasoning in `reviewer_notes` for every final pair
   decision. `needs_review` requires notes that state what is uncertain.

Allowed leakage statuses are `pending`, `same_capture`, `near_duplicate`,
`similar_scene`, `unrelated`, and `needs_review`. The validator checks that
non-pending status and decision agree and that final decisions have evidence
notes.

## Split protection

The manifests describe the current split and are not a place to apply changes.
Do not edit `split`, `split_a`, `split_b`, paths, or source assignments during
this phase. The validator verifies them against source paths and the canonical
Phase 1 audit.

- A confirmed same-capture train–test pair is flagged for a future correction:
  keep the capture in test and, if a future version is approved, move only the
  non-test counterpart to test. Never move a test image into training.
- A confirmed same-capture train–val pair is flagged for a future common
  non-test split.
- `similar_scene` and `unrelated` decisions retain the current assignments.
- The validator reports future-action flags but never performs the action.
- Any future split correction must be a separate, versioned dataset change
  with review/approval; it is outside Phase 1.3.

## Run the read-only validator

From the repository root:

```bash
python tools/validate_phase1_3_reviews.py
```

It reads the two review CSVs and the canonical Phase 1 audit, checks statuses,
decisions, notes, pair IDs, paths, split consistency, and test-image protection,
then writes only `work/audit/phase1.3_review_validation.json`. It does not edit
either manifest or any image, label, or split.

Exit code 0 means the current manifests are structurally consistent. Exit code
1 means validation issues were found; the JSON still records the issues and
pending counts. Future split correction flags are reported separately from
validation errors and always have `action_applied: false`.

## Current state and remaining human work

The initial validation reports 664 pending annotation rows covering 194 unique
images, and 362 pending leakage pairs. No human decisions or reviewer notes
exist yet. The first run is structurally valid, with no missing/duplicate pair
IDs, missing image paths, invalid split values, or test-to-train assignments.
No same-capture split flags are emitted until a reviewer records a
`same_capture` decision. All annotation, capture/session, and leakage decisions
remain manual.
