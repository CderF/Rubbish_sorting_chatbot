"""SQLite access for STAGE 04 (dynamic knowledge expansion).

Built on top of `app/data/db.py` (Teamb2): the `waste_items` /
`collection_schedule` tables and the DB path come from there, so ingest writes
into the same DB the crawler builds. STAGE 04 only adds:

* `knowledge_sources` — one row per ingested お知らせ section, carrying the fields
  the shared tables don't have (`content_hash` for idempotency, `source_type`,
  which `waste_items` row it produced). This means no change to Teamb2's schema.
* `lookup_item_versions` / `version_window` — the read path requirement 4-2 needs.
  `app/data/lookup.py` pins `WHERE is_current = 1` by agreement, so it can only
  ever see the newest rule; these return the whole old→new chain instead.

`valid_until` is not a column either: a rule runs until the day before its
successor starts, derived in `version_window`.

Target runtime is the B1 server's Python 3.10 — no 3.11+ syntax.
"""

from __future__ import annotations

import sqlite3
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.data import db
from app.data.schema import WasteItem

# STAGE 04 only. `produced_item_id` / `is_change_notice` let `relink()` find the
# change-notice rows that still need a predecessor once the crawler has run.
KNOWLEDGE_SOURCES_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_sources (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file      TEXT    NOT NULL,
    source_type      TEXT    NOT NULL,
    anchor           TEXT,
    title            TEXT,
    content_hash     TEXT    NOT NULL UNIQUE,
    ingested_at      TEXT    NOT NULL,
    produced_item_id     INTEGER REFERENCES waste_items(id),
    is_change_notice     INTEGER NOT NULL DEFAULT 0,
    -- the notice's own account of the pre-change 早見表 treatment; kept so
    -- the answer layer has the prior state even when the crawl is missing
    -- the row or classifies it differently.
    prior_disposal_method TEXT
);
CREATE INDEX IF NOT EXISTS idx_knowledge_sources_file ON knowledge_sources(source_file);
"""

KNOWLEDGE_SCHEME = "knowledge://"


def normalize_item_name(name: str) -> str:
    """NFKC + strip, matching `WasteItem` and `app/data/lookup.py` on the query
    side (documents/data_schema.md §3 requires it on both write and query)."""
    return unicodedata.normalize("NFKC", name).strip()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_date(value: "str | date | None") -> "date | None":
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)


def connect(db_path: "str | Path | None" = None) -> sqlite3.Connection:
    """Same DB as the crawler (`app/data/db.py`); `DB_PATH` env var overrides."""
    return db.connect(db_path)


def init_schema(conn: sqlite3.Connection) -> None:
    db.init_db(conn)  # waste_items / collection_schedule
    conn.executescript(KNOWLEDGE_SOURCES_SQL)
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
        crawled_at=row["crawled_at"],
        is_current=bool(row["is_current"]),
        effective_from=_as_date(row["effective_from"]),
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
            item.crawled_at.isoformat(),
            1 if item.is_current else 0,
            item.effective_from.isoformat() if item.effective_from else None,
            item.superseded_by,
        ),
    )
    return int(cur.lastrowid)


def find_current_predecessor(
    conn: sqlite3.Connection, item_name: str, exclude_id: "int | None" = None
) -> "WasteItem | None":
    """The 早見表 row a change notice supersedes: same name, still current, and
    NOT itself a distributed-file row (so notices don't supersede each other)."""
    rows = conn.execute(
        """
        SELECT * FROM waste_items
        WHERE item_name = ? AND is_current = 1 AND source_url NOT LIKE ?
        ORDER BY id DESC
        """,
        (normalize_item_name(item_name), KNOWLEDGE_SCHEME + "%"),
    ).fetchall()
    for row in rows:
        if exclude_id is None or row["id"] != exclude_id:
            return _row_to_item(row)
    return None


def mark_superseded(conn: sqlite3.Connection, old_id: int, new_id: int) -> None:
    """Retire a row in favour of its replacement (documents/data_schema.md §4).

    `effective_from` is deliberately NOT touched here: it means "the date this
    record took effect", so the new rule's start date belongs on the new row.
    """
    conn.execute(
        "UPDATE waste_items SET is_current = 0, superseded_by = ? WHERE id = ?",
        (new_id, old_id),
    )


def has_predecessor(conn: sqlite3.Connection, new_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM waste_items WHERE superseded_by = ? LIMIT 1", (new_id,)
    ).fetchone()
    return row is not None


def lookup_item_versions(conn: sqlite3.Connection, item_name: str) -> "list[WasteItem]":
    """Every known version of one item, oldest first — the requirement 4-2 read
    path. No `WHERE is_current = 1`, unlike `app/data/lookup.py`: the caller gets
    the whole chain and decides what applies on a given date (see
    `version_window`). Rows with no `effective_from` (the 早見表 crawl baseline)
    sort first, then by date ascending.
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
    """(valid_from, valid_until) for `versions[index]` as returned by
    `lookup_item_versions`. `valid_until` is derived — a rule runs until the day
    before its successor starts; the last version has no end date.
    """
    valid_until = None
    for later in versions[index + 1 :]:
        if later.effective_from is not None:
            valid_until = later.effective_from - timedelta(days=1)
            break
    return versions[index].effective_from, valid_until


# --------------------------------------------------------------------------
# knowledge_sources
# --------------------------------------------------------------------------
def is_ingested(conn: sqlite3.Connection, content_hash: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM knowledge_sources WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        is not None
    )


def record_knowledge_source(
    conn: sqlite3.Connection,
    *,
    source_file: str,
    source_type: str,
    anchor: "str | None",
    title: "str | None",
    content_hash: str,
    ingested_at: datetime,
    produced_item_id: int,
    is_change_notice: bool,
    prior_disposal_method: "str | None" = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO knowledge_sources (
            source_file, source_type, anchor, title, content_hash,
            ingested_at, produced_item_id, is_change_notice, prior_disposal_method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_file,
            source_type,
            anchor,
            title,
            content_hash,
            ingested_at.isoformat(),
            produced_item_id,
            1 if is_change_notice else 0,
            prior_disposal_method,
        ),
    )
    return int(cur.lastrowid)


def unlinked_change_sources(conn: sqlite3.Connection) -> "list[sqlite3.Row]":
    """Change-notice rows whose produced item still has no predecessor — the
    work `relink()` does once the crawler has populated the 早見表."""
    return conn.execute(
        """
        SELECT ks.*, wi.item_name AS item_name
        FROM knowledge_sources ks
        JOIN waste_items wi ON wi.id = ks.produced_item_id
        WHERE ks.is_change_notice = 1
          AND NOT EXISTS (
              SELECT 1 FROM waste_items p WHERE p.superseded_by = ks.produced_item_id
          )
        ORDER BY ks.id
        """
    ).fetchall()


def list_knowledge_sources(conn: sqlite3.Connection) -> "list[dict[str, Any]]":
    """Ingested knowledge sources, for the WebUI file list (STAGE02 2-3).
    Pairs with `app/data/db.list_sources()` (the crawled pages)."""
    rows = conn.execute(
        "SELECT * FROM knowledge_sources ORDER BY source_file, anchor"
    ).fetchall()
    return [dict(r) for r in rows]
