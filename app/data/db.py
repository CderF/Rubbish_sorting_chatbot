"""SQLite schema + full-refresh writes for `waste_items` / `collection_schedule`
(documents/data_schema.md). STAGE02/03 operates in "is_current=1, single
snapshot" mode: `rebuild_db()` deletes all rows and re-inserts the latest crawl
in one transaction — no incremental diffing (that's STAGE04 scope; see
documents/data_schema.md 4章 "STAGE04担当者向けメモ" for how `is_current` /
`effective_from` / `superseded_by` are meant to be used once that lands).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from app.data.schema import CollectionSchedule, WasteItem

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "var" / "kitakyushu_waste.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS waste_items (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name          TEXT NOT NULL,
    item_name_raw      TEXT NOT NULL,
    disposal_method    TEXT NOT NULL,
    notes              TEXT NOT NULL DEFAULT '',
    source_url         TEXT NOT NULL,
    source_page_title  TEXT,
    crawled_at         TEXT NOT NULL,
    is_current         INTEGER NOT NULL DEFAULT 1,
    effective_from     TEXT,
    superseded_by      INTEGER REFERENCES waste_items(id)
);

CREATE INDEX IF NOT EXISTS idx_waste_items_item_name
    ON waste_items(item_name);
CREATE INDEX IF NOT EXISTS idx_waste_items_item_name_current
    ON waste_items(item_name, is_current);

CREATE TABLE IF NOT EXISTS collection_schedule (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ward           TEXT NOT NULL,
    town           TEXT NOT NULL,
    waste_type     TEXT NOT NULL,
    weekday        TEXT NOT NULL,
    source_url     TEXT NOT NULL,
    crawled_at     TEXT NOT NULL,
    is_current     INTEGER NOT NULL DEFAULT 1,
    effective_from TEXT,
    superseded_by  INTEGER REFERENCES collection_schedule(id)
);

CREATE INDEX IF NOT EXISTS idx_collection_schedule_lookup
    ON collection_schedule(ward, town, waste_type);
"""


def db_path() -> Path:
    """DB_PATH env var overrides the default, matching the MODEL_PATH pattern
    in app/server.py (per-machine override without touching code)."""
    return Path(os.environ.get("DB_PATH", DEFAULT_DB_PATH))


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    resolved = Path(path) if path else db_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def _waste_item_row(item: WasteItem) -> tuple:
    return (
        item.item_name,
        item.item_name_raw,
        item.disposal_method,
        item.notes,
        item.source_url,
        item.source_page_title,
        item.crawled_at.isoformat(),
        int(item.is_current),
        item.effective_from.isoformat() if item.effective_from else None,
        item.superseded_by,
    )


def _schedule_row(sched: CollectionSchedule) -> tuple:
    return (
        sched.ward,
        sched.town,
        sched.waste_type,
        sched.weekday,
        sched.source_url,
        sched.crawled_at.isoformat(),
        int(sched.is_current),
        sched.effective_from.isoformat() if sched.effective_from else None,
        sched.superseded_by,
    )


def rebuild_db(
    conn: sqlite3.Connection,
    waste_items: list[WasteItem],
    schedules: list[CollectionSchedule],
) -> None:
    """Delete all existing rows and insert the given crawl result, in one
    transaction (either the whole refresh lands or none of it does)."""
    init_db(conn)
    with conn:
        conn.execute("DELETE FROM waste_items")
        conn.execute("DELETE FROM collection_schedule")
        conn.executemany(
            """INSERT INTO waste_items
               (item_name, item_name_raw, disposal_method, notes, source_url,
                source_page_title, crawled_at, is_current, effective_from, superseded_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [_waste_item_row(i) for i in waste_items],
        )
        conn.executemany(
            """INSERT INTO collection_schedule
               (ward, town, waste_type, weekday, source_url, crawled_at,
                is_current, effective_from, superseded_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [_schedule_row(s) for s in schedules],
        )


def list_current_waste_items(conn: sqlite3.Connection) -> list[WasteItem]:
    """All current waste_items rows. The narrow lookup_* functions in
    app/data/lookup.py deliberately don't support "fetch everything" — this
    exists for the vector RAG path's index build (issue #7), which needs the
    full item set to embed, not a single lookup."""
    rows = conn.execute("SELECT * FROM waste_items WHERE is_current = 1").fetchall()
    return [
        WasteItem(
            id=r["id"],
            item_name=r["item_name"],
            item_name_raw=r["item_name_raw"],
            disposal_method=r["disposal_method"],
            notes=r["notes"],
            source_url=r["source_url"],
            source_page_title=r["source_page_title"],
            crawled_at=r["crawled_at"],
            is_current=bool(r["is_current"]),
            effective_from=r["effective_from"],
            superseded_by=r["superseded_by"],
        )
        for r in rows
    ]


def list_sources(conn: sqlite3.Connection) -> list[dict]:
    """Distinct crawled source pages, most recent first — feeds the WebUI's
    ingested-file listing (STAGE02 2-3)."""
    rows = conn.execute(
        """
        SELECT source_url, source_page_title AS title, MAX(crawled_at) AS crawled_at
        FROM waste_items WHERE is_current = 1
        GROUP BY source_url, source_page_title
        UNION
        SELECT source_url, ward AS title, MAX(crawled_at) AS crawled_at
        FROM collection_schedule WHERE is_current = 1
        GROUP BY source_url, ward
        ORDER BY crawled_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    from app.data.crawler import crawl_all

    waste_items, schedules = crawl_all()
    conn = connect()
    rebuild_db(conn, waste_items, schedules)
    print(f"{db_path()}: waste_items={len(waste_items)} collection_schedule={len(schedules)}")
    print(f"sources: {len(list_sources(conn))} distinct pages")
