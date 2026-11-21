# Local fruit annotation review UI

## Start

From the `fruit/` project directory:

```bash
python tools/fruit_type_cls_experiment/review_app/app.py --host 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765/> in a browser. The server binds to loopback only and uses Python's standard library. Stop it with Ctrl-C; every saved decision is already on disk.

## Review flow

1. Enter a reviewer ID. It is remembered in this browser and recorded with each decision.
2. Filter by fruit, frozen split, candidate review cue, and status. The default view shows pending samples, with foreign-class intersections, multi-fruit crops, possible missing annotations, and duplicate candidates first.
3. Inspect the full source image with all original boxes and labels, then the single-object crop where one exists. Box coordinates use the original source-image pixel coordinates.
4. Choose a suggested class only when appropriate, add reviewer notes, then choose an explicit action: verify, incorrect label, missing fruit, incorrect box, ambiguous, or record another review cue.
5. Download `review_export.csv` for reviewed decisions. The CSV is refreshed after every saved decision.

The interface never edits `review_candidates.csv`, source images, YOLO annotation files, split manifests, or model checkpoints. For missing fruit, it records the finding and proposed class in the review ledger; it does not create an annotation or fabricate a box. The review status is limited to `pending`, `verified`, `incorrect`, or `ambiguous`.

## Persistence and exports

- `work/fruit_type_cls_experiment/review_progress.csv` stores the latest saved decision per review item; it is atomically replaced after each decision.
- `review_events.jsonl` appends a timestamped decision event for audit history.
- `review_export.csv` contains reviewed items only and retains the original candidate cue separately from the reviewer decision category.
- `review_progress_report.md` and `.json` show pending/reviewed totals and status/category/reviewer counts. The browser also shows progress.

On restart, saved decisions are reloaded and shown under their chosen status filter. The untouched queue remains the source of truth for original labels, boxes, hashes, and split membership.
