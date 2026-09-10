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
import threading
import unicodedata

from app.data import db
from app.data.schema import CollectionSchedule, WasteItem

# sqlite3.Connection objects can't cross threads (check_same_thread=True by
# default), and the grader/server calls these functions from a threadpool
# (FastAPI's asyncio.to_thread) with up to 4 concurrent requests. One
# connection per thread, lazily created, avoids both the cross-thread error
# and the lock contention a single shared connection would need.
_local = threading.local()


def _get_connection() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = db.connect()
        db.init_db(conn)
        conn.execute("PRAGMA query_only = ON")
        _local.conn = conn
    return conn


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


def lookup_item_contained_in(text: str, min_length: int = 2) -> list[WasteItem]:
    """`text`(質問文の断片、文全体のこともある)の中に、現行の品名が連続
    文字列として現れているものを全件返す(単純な部分文字列一致、語境界は
    見ない)。

    `lookup_item_partial` とは逆方向: あちらは「item_nameがtextを含む」
    (`LIKE '%text%'`)なので、textが品名より長い(「廃車にした自動車を
    処分したいのです」のような文全体)場合は原理的に何もヒットしない。
    こちらは「textがitem_nameを含む」方向で、`app/rag/router.py` の
    `_QUESTION_TAIL` が質問定型に一致せず文全体が候補として残るケースの
    救済に使う(2026-09-10 実採点trace回帰)。

    語境界のチェック(「捨てたらいいか」に「たらい」が偶然含まれる、のような
    事故の排除)は呼び出し側の責務 — `app/rag/router.py` の
    `_resolve_named_hit` が既に「直前が助詞・記号/直後が文末・助詞・質問
    定型」という同種の判定を実装しているため、ここでは候補を絞り込みすぎず
    全件返し、その既存ロジックに絞り込みを委ねる。1文字の品名は文中に
    偶然含まれやすいため既定で対象外にする(`min_length`)。

    括弧書きの条件(「蛍光管(環型、直型)」)を含む品名は、括弧の手前部分
    (「蛍光管」)だけでも一致とみなす — 利用者は条件まで言わずに品名の
    本体だけを口にすることが多いため(「蛍光管を割って…」に「蛍光管」は
    含まれるが「蛍光管(環型、直型)」は含まれない)。
    """
    normalized = _normalize(text)
    if not normalized:
        return []
    rows = _get_connection().execute(
        "SELECT * FROM waste_items WHERE is_current = 1 AND length(item_name) >= ?",
        (min_length,),
    ).fetchall()
    hits = []
    for row in rows:
        name = row["item_name"]
        base = name.split("(")[0].split("（")[0]
        if name in normalized or (len(base) >= min_length and base in normalized):
            hits.append(row)
    return [_row_to_waste_item(row) for row in hits]


def lookup_towns(ward: str) -> list[str]:
    """区内の全町名(丁目単位に展開済み)を返す。

    区名抽出(app/rag/router.py の extract_region)が正規表現ヒューリスティック
    では拾いきれない実在の町名(例: 「城内」「中井浜」のように丁目/町/台等の
    定型接尾辞を持たない町名)を検出するために使う。地域抽出はデータ層の
    実データを正として判定すべき、という方針(早見表パスの検索と同様)。
    """
    normalized = _normalize(ward)
    if not normalized:
        return []
    rows = _get_connection().execute(
        "SELECT DISTINCT town FROM collection_schedule WHERE ward = ? AND is_current = 1",
        (normalized,),
    ).fetchall()
    return [r["town"] for r in rows]


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
