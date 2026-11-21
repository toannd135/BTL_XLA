#!/usr/bin/env python3
"""Local, append-only human review app for the fruit dataset v2 candidate."""
from __future__ import annotations

import csv
import contextlib
import datetime as dt
import json
import mimetypes
import os
import pathlib
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

csv.field_size_limit(sys.maxsize)

APP_DIR = pathlib.Path(__file__).resolve().parent
PHASE5_DIR = APP_DIR.parent
EXPERIMENT_DIR = PHASE5_DIR.parent
FRUIT_DIR = EXPERIMENT_DIR.parents[1]
CANDIDATE_DIR = EXPERIMENT_DIR / "dataset_v2_candidate"
QUEUE_PATH = PHASE5_DIR / "review_queue.csv"
PAIR_QUEUE_PATH = PHASE5_DIR / "near_duplicate_pair_queue.csv"
FAMILY_MANIFEST_PATH = CANDIDATE_DIR / "source_family_split_manifest.csv"
DATA_DIR = pathlib.Path(os.environ.get("FRUIT_REVIEW_DATA_DIR", PHASE5_DIR)).resolve()
DB_PATH = DATA_DIR / "review_state.sqlite3"
CSV_PATH = DATA_DIR / "review_decisions.csv"
PROGRESS_PATH = DATA_DIR / "review_progress_report.md"
HOST = os.environ.get("FRUIT_REVIEW_HOST", "127.0.0.1")
PORT = int(os.environ.get("FRUIT_REVIEW_PORT", "8766"))

CROP_DECISIONS = {
    "VERIFIED_CORRECT", "INCORRECT_LABEL", "AMBIGUOUS", "INVALID_CROP",
    "NEEDS_FURTHER_REVIEW",
}
PAIR_DECISIONS = {
    "PAIR_DISTINCT", "POSSIBLE_DUPLICATE", "NEEDS_PROVENANCE", "STILL_UNCLEAR",
}
REASON_REQUIRED = {"INCORRECT_LABEL", "AMBIGUOUS", "INVALID_CROP"}
LOCK = threading.RLock()


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_items() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    crops = read_csv(QUEUE_PATH)
    pairs = read_csv(PAIR_QUEUE_PATH) if PAIR_QUEUE_PATH.exists() else []
    return crops, pairs


CROP_ITEMS, PAIR_ITEMS = load_items()
CROP_BY_ID = {r["review_id"]: r for r in CROP_ITEMS}
PAIR_BY_ID = {r["pair_review_id"]: r for r in PAIR_ITEMS}
FAMILY_BY_ID = {r["source_family_id"]: r for r in read_csv(FAMILY_MANIFEST_PATH)}


def db_connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


@contextlib.contextmanager
def db_session():
    con = db_connect()
    try:
        with con:
            yield con
    finally:
        con.close()


def initialize_db() -> None:
    with db_session() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS decision_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            queue_id TEXT NOT NULL,
            item_type TEXT NOT NULL CHECK(item_type IN ('crop','source_pair')),
            decision TEXT NOT NULL,
            reviewer_id TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            timestamp_utc TEXT NOT NULL,
            previous_event_id INTEGER
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_events_queue ON decision_events(queue_id,event_id)")
        con.execute("""CREATE TRIGGER IF NOT EXISTS immutable_events_update
            BEFORE UPDATE ON decision_events BEGIN
                SELECT RAISE(ABORT, 'decision audit events are immutable');
            END""")
        con.execute("""CREATE TRIGGER IF NOT EXISTS immutable_events_delete
            BEFORE DELETE ON decision_events BEGIN
                SELECT RAISE(ABORT, 'decision audit events are immutable');
            END""")


def latest_events() -> dict[str, dict]:
    with db_session() as con:
        rows = con.execute("""SELECT e.* FROM decision_events e
            JOIN (SELECT queue_id,MAX(event_id) AS max_id FROM decision_events GROUP BY queue_id) x
            ON e.event_id=x.max_id""").fetchall()
    return {r["queue_id"]: dict(r) for r in rows}


def export_csv_and_progress() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with db_session() as con:
        events = [dict(r) for r in con.execute("SELECT * FROM decision_events ORDER BY event_id")]
    fields = ["event_id", "queue_id", "item_type", "decision", "reviewer_id", "reason", "notes", "timestamp_utc", "previous_event_id"]
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(events)
    tmp.replace(CSV_PATH)

    # Events are ordered oldest-to-newest; later entries overwrite earlier ones.
    latest = {r["queue_id"]: r for r in events}
    crop_latest = {k: v for k, v in latest.items() if v["item_type"] == "crop"}
    pair_latest = {k: v for k, v in latest.items() if v["item_type"] == "source_pair"}
    by_decision: dict[str, int] = {}
    by_class_split: dict[tuple[str, str], dict[str, int]] = {}
    for item in CROP_ITEMS:
        current = crop_latest.get(item["review_id"])
        if current:
            d = current["decision"]
            by_decision[d] = by_decision.get(d, 0) + 1
            key = (item["fruit_class"], item["split_v2"])
            by_class_split.setdefault(key, {"total": 0, "reviewed": 0})
            by_class_split[key]["reviewed"] += 1
        key = (item["fruit_class"], item["split_v2"])
        by_class_split.setdefault(key, {"total": 0, "reviewed": 0})
        by_class_split[key]["total"] += 1
    high = [r for r in CROP_ITEMS if r["priority"] == "HIGH"]
    high_reviewed = sum(r["review_id"] in crop_latest for r in high)
    lines = [
        "# Phase 5 review progress", "",
        f"Updated: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}", "",
        f"- Candidate crop queue: **{len(CROP_ITEMS)}**", 
        f"- Crop decisions saved: **{len(crop_latest)} / {len(CROP_ITEMS)}**",
        f"- Pending crops: **{len(CROP_ITEMS)-len(crop_latest)}**",
        f"- Prioritized flagged crops reviewed: **{high_reviewed} / {len(high)}**",
        f"- Near-duplicate pair tasks saved: **{len(pair_latest)} / {len(PAIR_ITEMS)}**",
        f"- Immutable audit events: **{len(events)}**", "",
        "## Current crop decisions", "",
        "| Decision | Items |", "|---|---:|",
    ]
    for decision in sorted(CROP_DECISIONS):
        lines.append(f"| {decision} | {by_decision.get(decision, 0)} |")
    lines += ["", "## Progress by class and split", "", "| Class | Split | Reviewed | Total |", "|---|---|---:|---:|"]
    for (cls, split), counts in sorted(by_class_split.items()):
        lines.append(f"| {cls} | {split} | {counts['reviewed']} | {counts['total']} |")
    lines += ["", "Decisions are append-only in `review_state.sqlite3`; a later review creates a new event and does not erase earlier decisions. `review_decisions.csv` is the full event export.", ""]
    PROGRESS_PATH.write_text("\n".join(lines), encoding="utf-8")


def record_decision(item_id: str, item_type: str, decision: str, reviewer: str,
                    reason: str = "", notes: str = "") -> dict:
    """Append one validated decision event; never updates an existing event."""
    reviewer, reason, notes = reviewer.strip(), reason.strip(), notes.strip()
    if not reviewer:
        raise ValueError("Reviewer ID is required")
    if item_type == "crop":
        if item_id not in CROP_BY_ID or decision not in CROP_DECISIONS:
            raise ValueError("Invalid crop item or decision")
        if decision in REASON_REQUIRED and not reason:
            raise ValueError("A reason is required for this decision")
    elif item_type == "source_pair":
        if item_id not in PAIR_BY_ID or decision not in PAIR_DECISIONS:
            raise ValueError("Invalid source-pair item or decision")
        if not reason:
            raise ValueError("Add a short rationale for a source-pair decision")
    else:
        raise ValueError("Invalid item type")
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")
    with LOCK:
        with db_session() as con:
            previous = con.execute("SELECT event_id FROM decision_events WHERE queue_id=? ORDER BY event_id DESC LIMIT 1", (item_id,)).fetchone()
            cur = con.execute("""INSERT INTO decision_events
                (queue_id,item_type,decision,reviewer_id,reason,notes,timestamp_utc,previous_event_id)
                VALUES (?,?,?,?,?,?,?,?)""", (item_id,item_type,decision,reviewer,reason,notes,stamp,previous["event_id"] if previous else None))
            event_id = cur.lastrowid
        export_csv_and_progress()
    return {"event_id": event_id, "timestamp_utc": stamp}


def safe_path(path: pathlib.Path, allowed_root: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve(strict=True)
    root = allowed_root.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise FileNotFoundError("asset not in the read-only allowlist")
    return resolved


def asset_path(item_id: str, which: str) -> pathlib.Path:
    if item_id in CROP_BY_ID:
        item = CROP_BY_ID[item_id]
        if which == "crop":
            return safe_path(CANDIDATE_DIR / item["candidate_crop_path"], CANDIDATE_DIR)
        if which == "source":
            return safe_path(FRUIT_DIR / item["source_image_path"], FRUIT_DIR)
    if item_id in PAIR_BY_ID and which in {"a", "b"}:
        item = PAIR_BY_ID[item_id]
        return safe_path(pathlib.Path(item[f"source_path_{which}"]), FRUIT_DIR)
    raise FileNotFoundError("unknown review item or asset")


def json_safe_item(item: dict[str, str], current: dict | None) -> dict:
    out = dict(item)
    out["current_review"] = current or None
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "FruitReview/1.0"

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def send_json(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            raw = (APP_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if parsed.path == "/api/queue":
            cls = query.get("class", [""])[0]
            split = query.get("split", [""])[0]
            status = query.get("status", ["ALL"])[0]
            search = query.get("q", [""])[0].casefold()
            try:
                offset = max(0, int(query.get("offset", ["0"])[0]))
                limit = min(200, max(1, int(query.get("limit", ["60"])[0])))
            except ValueError:
                self.send_json({"error": "offset/limit must be integers"}, 400); return
            current = latest_events()
            chosen = []
            for item in CROP_ITEMS:
                event = current.get(item["review_id"])
                decision = event["decision"] if event else "UNREVIEWED"
                if cls and item["fruit_class"] != cls: continue
                if split and item["split_v2"] != split: continue
                if status == "UNREVIEWED" and event: continue
                if status not in {"ALL", "UNREVIEWED"} and decision != status: continue
                if search and search not in (item["image"]+" "+item["source_family_id"]+" "+item["priority_reason"]+" "+item["audit_warning_flags"]).casefold(): continue
                chosen.append(json_safe_item(item, event))
            page = chosen[offset:offset+limit]
            self.send_json({"total": len(chosen), "offset": offset, "limit": limit, "items": page})
            return
        if parsed.path == "/api/item":
            item_id = query.get("id", [""])[0]
            item = CROP_BY_ID.get(item_id)
            if not item: self.send_json({"error": "not found"}, 404); return
            self.send_json(json_safe_item(item, latest_events().get(item_id))); return
        if parsed.path == "/api/family":
            family_id = query.get("id", [""])[0]
            family = FAMILY_BY_ID.get(family_id)
            if not family: self.send_json({"error": "family not found"}, 404); return
            self.send_json(family); return
        if parsed.path == "/api/pairs":
            latest = latest_events()
            self.send_json({"items": [json_safe_item(r, latest.get(r["pair_review_id"])) for r in PAIR_ITEMS]}); return
        if parsed.path == "/api/progress":
            latest = latest_events()
            crop_events = {k:v for k,v in latest.items() if v["item_type"] == "crop"}
            pair_events = {k:v for k,v in latest.items() if v["item_type"] == "source_pair"}
            high = sum(r["priority"] == "HIGH" for r in CROP_ITEMS)
            high_done = sum(r["priority"] == "HIGH" and r["review_id"] in crop_events for r in CROP_ITEMS)
            self.send_json({"total":len(CROP_ITEMS),"reviewed":len(crop_events),"pending":len(CROP_ITEMS)-len(crop_events),"priority_total":high,"priority_reviewed":high_done,"pair_total":len(PAIR_ITEMS),"pair_reviewed":len(pair_events),"events":sum(1 for _ in all_events())})
            return
        if parsed.path == "/asset":
            item_id = query.get("id", [""])[0]
            which = query.get("which", [""])[0]
            try:
                path = asset_path(item_id, which)
                raw = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "public, max-age=300")
                self.end_headers(); self.wfile.write(raw)
            except (FileNotFoundError, OSError, ValueError):
                self.send_error(404, "asset unavailable")
            return
        self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/decision":
            self.send_error(404); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64_000: raise ValueError("request too large")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            item_id = str(data.get("queue_id", ""))
            item_type = str(data.get("item_type", "crop"))
            decision = str(data.get("decision", ""))
            reviewer = str(data.get("reviewer_id", "")).strip()
            reason = str(data.get("reason", "")).strip()
            notes = str(data.get("notes", "")).strip()
            saved = record_decision(item_id,item_type,decision,reviewer,reason,notes)
            self.send_json({"ok":True,**saved,"message":"Decision saved as a new immutable audit event."})
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
            self.send_json({"error":str(e)},400)
        except Exception as e:
            self.send_json({"error":"Could not save decision: "+str(e)},500)


def all_events():
    with db_session() as con:
        return [dict(r) for r in con.execute("SELECT event_id FROM decision_events")]


def main():
    initialize_db()
    export_csv_and_progress()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Fruit dataset review: http://{HOST}:{PORT}/")
    print(f"Queue: {QUEUE_PATH} ({len(CROP_ITEMS)} crops; {sum(r['priority']=='HIGH' for r in CROP_ITEMS)} prioritized)")
    print(f"Persistence: {DB_PATH}")
    print("Ctrl-C stops the server; decisions are already persisted.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping review app.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
