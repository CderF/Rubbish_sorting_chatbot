"""SQLite access for STAGE 04 (dynamic knowledge expansion).

The `waste_items` / `collection_schedule` DDL here is a verbatim transcription of
`documents/data_schema.md` and is created with CREATE TABLE IF NOT EXISTS, so it
converges with `app/data/db.py` (Teamb2, issue #4) once that lands — whichever
runs first creates the tables, the other is a no-op.

`knowledge_sources` is owned by STAGE 04. It carries the fields the STAGE 04
charter asked for that the two shared tables do not have (`content_hash`,
`source_type`, the ingest timestamp), so ingesting distributed knowledge needs no
change to Teamb2's schema.

`valid_until` is likewise not a column: the end of a superseded rule is derived as
`effective_from(successor) - 1 day`, see `version_window`.

Target runtime is the B1 server's Python 3.10, so no 3.11+ syntax here.
"""

from __future__ import annotations

import sqlite3
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.data.schema import WasteItem

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS waste_items (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name         TEXT    NOT NULL,
    item_name_raw     TEXT    NOT NULL,
    disposal_method   TEXT    NOT NULL,
    notes             TEXT    NOT NULL DEFAULT '',
    source_url        TEXT    NOT NULL,
    source_page_title TEXT,
    crawled_at        TEXT    NOT NULL,
    is_current        INTEGER NOT NULL DEFAULT 1,
    effective_from    TEXT,
    superseded_by     INTEGER REFERENCES waste_items(id)
);
CREATE INDEX IF NOT EXISTS idx_waste_items_name         ON waste_items(item_name);
CREATE INDEX IF NOT EXISTS idx_waste_items_name_current ON waste_items(item_name, is_current);

CREATE TABLE IF NOT EXISTS collection_schedule (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ward           TEXT    NOT NULL,
    town           TEXT    NOT NULL,
    waste_type     TEXT    NOT NULL,
    weekday        TEXT    NOT NULL,
    source_url     TEXT    NOT NULL,
    crawled_at     TEXT    NOT NULL,
    is_current     INTEGER NOT NULL DEFAULT 1,
    effective_from TEXT,
    superseded_by  INTEGER REFERENCES collection_schedule(id)
);
CREATE INDEX IF NOT EXISTS idx_schedule_ward_town_type
    ON collection_schedule(ward, town, waste_type);

-- STAGE 04 only: one row per ingested notice section.
CREATE TABLE IF NOT EXISTS knowledge_sources (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file  TEXT    NOT NULL,
    source_type  TEXT    NOT NULL,
    anchor       TEXT,
    title        TEXT,
    content_hash TEXT    NOT NULL UNIQUE,
    ingested_at  TEXT    NOT NULL,
    record_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_knowledge_sources_file ON knowledge_sources(source_file);
"""


def normalize_item_name(name: str) -> str:
    """Same normalization the WasteItem validator applies, for query-side use.

    `documents/data_schema.md` §3 requires NFKC on both write and query.
    """
    return unicodedata.normalize("NFKC", name).strip()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_dt(value: datetime) -> str:
    """ISO8601 with a Z suffix, matching data_schema.md's sample rows."""
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc)
    return value.replace(tzinfo=None, microsecond=0).isoformat() + "Z"


def _parse_dt(value: str) -> datetime:
    # Python 3.10's fromisoformat does not accept the Z suffix.
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def connect(db_path: "str | Path") -> sqlite3.Connection:
    path = Path(db_path)
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


# --------------------------------------------------------------------------
# waste_items
# --------------------------------------------------------------------------
def _row_to_item(row: sqlite3.Row) -> WasteItem:
    return WasteItem(
        id=row["id"],
        item_name=row["item_name"],
        item_name_raw=row["item_name_raw"],
        disposal_method=row["disposal_method"],
        notes=row["notes"] or "",
        source_url=row["source_url"],
        source_page_title=row["source_page_title"],
        crawled_at=_parse_dt(row["crawled_at"]),
        is_current=bool(row["is_current"]),
        effective_from=(
            date.fromisoformat(row["effective_from"]) if row["effective_from"] else None
        ),
        superseded_by=row["superseded_by"],
    )


def insert_waste_item(conn: sqlite3.Connection, item: WasteItem) -> int:
    cur = conn.execute(
        """
        INSERT INTO waste_items (
            item_name, item_name_raw, disposal_method, notes, source_url,
            source_page_title, crawled_at, is_current, effective_from, superseded_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.item_name,
            item.item_name_raw,
            item.disposal_method,
            item.notes,
            item.source_url,
            item.source_page_title,
            _fmt_dt(item.crawled_at),
            1 if item.is_current else 0,
            item.effective_from.isoformat() if item.effective_from else None,
            item.superseded_by,
        ),
    )
    return int(cur.lastrowid)


def find_current_item(conn: sqlite3.Connection, item_name: str) -> "WasteItem | None":
    """The row a new notice would supersede: same name, still flagged current."""
    row = conn.execute(
        "SELECT * FROM waste_items WHERE item_name = ? AND is_current = 1 "
        "ORDER BY id DESC LIMIT 1",
        (normalize_item_name(item_name),),
    ).fetchone()
    return _row_to_item(row) if row else None


def mark_superseded(conn: sqlite3.Connection, old_id: int, new_id: int) -> None:
    """Retire a row in favour of its replacement (documents/data_schema.md §4).

    `effective_from` is deliberately NOT written here: it means "the date this
    record took effect", so the new rule's start date belongs on the new row.
    """
    conn.execute(
        "UPDATE waste_items SET is_current = 0, superseded_by = ? WHERE id = ?",
        (new_id, old_id),
    )


def lookup_item_versions(conn: sqlite3.Connection, item_name: str) -> "list[WasteItem]":
    """Every known version of one item, oldest first — the 4-2 read path.

    Deliberately has no `WHERE is_current = 1`: `app/data/lookup.py`'s
    `lookup_item_exact` pins that flag and so can only ever see the newest rule,
    which is exactly what requirement 4-2 must not do. Callers get the whole
    chain and decide what applies on a given date; see `version_window` for the
    validity span of each entry.

    Rows whose `effective_from` is NULL (the 早見表 crawl baseline) sort first,
    then by date ascending.
    """
    rows = conn.execute(
        """
        SELECT * FROM waste_items
        WHERE item_name = ?
        ORDER BY (effective_from IS NULL) DESC, effective_from ASC, id ASC
        """,
        (normalize_item_name(item_name),),
    ).fetchall()
    return [_row_to_item(r) for r in rows]


def version_window(
    versions: "list[WasteItem]", index: int
) -> "tuple[date | None, date | None]":
    """(valid_from, valid_until) for `versions[index]`, as produced by
    `lookup_item_versions`.

    `valid_until` is derived rather than stored: a rule runs until the day before
    its successor starts. The last version has no end date.
    """
    current = versions[index]
    valid_until = None
    for later in versions[index + 1 :]:
        if later.effective_from is not None:
            valid_until = later.effective_from - timedelta(days=1)
            break
    return current.effective_from, valid_until


# --------------------------------------------------------------------------
# knowledge_sources
# --------------------------------------------------------------------------
def is_ingested(conn: sqlite3.Connection, content_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM knowledge_sources WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return row is not None


def record_knowledge_source(
    conn: sqlite3.Connection,
    *,
    source_file: str,
    source_type: str,
    anchor: "str | None",
    title: "str | None",
    content_hash: str,
    ingested_at: datetime,
    record_count: int,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO knowledge_sources (
            source_file, source_type, anchor, title,
            content_hash, ingested_at, record_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_file,
            source_type,
            anchor,
            title,
            content_hash,
            _fmt_dt(ingested_at),
            record_count,
        ),
    )
    return int(cur.lastrowid)


def list_knowledge_sources(conn: sqlite3.Connection) -> "list[dict[str, Any]]":
    """Ingested knowledge sources, for the WebUI's file list (STAGE02 2-3)."""
    rows = conn.execute(
        "SELECT * FROM knowledge_sources ORDER BY source_file, anchor"
    ).fetchall()
    return [dict(r) for r in rows]
