"""Read-only queries against the structured DB (documents/data_schema.md).

Interface contract (documents/stage02&03_dev_plan.md 3章, agreed with the
application-layer owner — do not change these signatures without a heads-up):
  - "not found" returns None / an empty list; nothing here ever raises for
    a missing item, missing town, etc.
  - every returned record carries source_url for citation.
  - callers pass raw, un-normalized strings; normalization happens in here.

All queries filter on is_current = 1 (documents/data_schema.md 3章). Connection
management is internal: callers just import and call these functions, they
don't hold a sqlite3.Connection themselves.
"""

from __future__ import annotations

import sqlite3
import unicodedata

from app.data import db
from app.data.schema import CollectionSchedule, WasteItem

_conn: sqlite3.Connection | None = None


def _get_connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = db.connect()
        db.init_db(_conn)
        _conn.execute("PRAGMA query_only = ON")
    return _conn


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _like_pattern(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _row_to_waste_item(row: sqlite3.Row) -> WasteItem:
    return WasteItem(
        id=row["id"],
        item_name=row["item_name"],
        item_name_raw=row["item_name_raw"],
        disposal_method=row["disposal_method"],
        notes=row["notes"],
        source_url=row["source_url"],
        source_page_title=row["source_page_title"],
        crawled_at=row["crawled_at"],
        is_current=bool(row["is_current"]),
        effective_from=row["effective_from"],
        superseded_by=row["superseded_by"],
    )


def _row_to_schedule(row: sqlite3.Row) -> CollectionSchedule:
    return CollectionSchedule(
        id=row["id"],
        ward=row["ward"],
        town=row["town"],
        waste_type=row["waste_type"],
        weekday=row["weekday"],
        source_url=row["source_url"],
        crawled_at=row["crawled_at"],
        is_current=bool(row["is_current"]),
        effective_from=row["effective_from"],
        superseded_by=row["superseded_by"],
    )


def lookup_item_exact(item_name: str) -> WasteItem | None:
    normalized = _normalize(item_name)
    if not normalized:
        return None
    row = _get_connection().execute(
        "SELECT * FROM waste_items WHERE item_name = ? AND is_current = 1 LIMIT 1",
        (normalized,),
    ).fetchone()
    return _row_to_waste_item(row) if row else None


def lookup_item_partial(item_name: str, limit: int = 10) -> list[WasteItem]:
    normalized = _normalize(item_name)
    if not normalized:
        return []
    rows = _get_connection().execute(
        "SELECT * FROM waste_items "
        "WHERE item_name LIKE ? ESCAPE '\\' AND is_current = 1 LIMIT ?",
        (_like_pattern(normalized), limit),
    ).fetchall()
    return [_row_to_waste_item(r) for r in rows]


def lookup_collection_schedule(
    ward: str, town: str | None, waste_type: str | None = None
) -> list[CollectionSchedule] | None:
    if not town:
        return None
    ward, town = _normalize(ward), _normalize(town)
    query = "SELECT * FROM collection_schedule WHERE ward = ? AND town = ? AND is_current = 1"
    params: list[str] = [ward, town]
    if waste_type:
        query += " AND waste_type = ?"
        params.append(_normalize(waste_type))
    rows = _get_connection().execute(query, params).fetchall()
    return [_row_to_schedule(r) for r in rows] if rows else None
