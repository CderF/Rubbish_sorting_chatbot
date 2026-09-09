"""SQLite access for STAGE 04 (dynamic knowledge expansion).

Built on top of `app/data/db.py` (Teamb2): the `waste_items` /
`collection_schedule` tables and the DB path come from there, so ingest writes
into the same DB the crawler builds. STAGE 04 only adds:

* `knowledge_sources` — one row per ingested お知らせ section, carrying the fields
  the shared tables don't have (`content_hash` for idempotency, `source_type`,
  which `waste_items` row it produced). This means no change to Teamb2's schema.
* `lookup_item_history` / `version_window` — the read path requirement 4-2 needs.
  `app/data/lookup.py`'s `lookup_item_exact` pins `WHERE is_current = 1` by
  agreement (issue #4), so it only ever sees the newest rule. `lookup_item_history`
  is the separate function for the whole old→new chain (FANG's naming, PR #3).

`valid_until` is not a column either: a rule runs until the day before its
successor starts, derived in `version_window`.

Target runtime is the B1 server's Python 3.10 — no 3.11+ syntax.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
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


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# お知らせ側だけに付く、品物の状態・用途を表す先頭修飾語。早見表は基礎名や
# 括弧つきの形で持つことがあるため、元の名前による exact/LIKE が空だった時に
# 限り、この修飾語を外してもう一度照合する。名称そのものを書き換えるためでは
# なく、旧行と新行を結ぶための限定的な照合用。
_PREDECESSOR_PREFIX_RE = re.compile(
    r"^(?:(?:使い捨て)|(?:使用済み(?:の)?)|(?:中古(?:の)?)|古い|(?:不要(?:な)?))+"
)


def _strip_predecessor_prefix(name: str) -> str:
    stripped = _PREDECESSOR_PREFIX_RE.sub("", name).strip()
    return stripped or name


@dataclass(frozen=True)
class PredecessorMatch:
    """`find_current_predecessor` の結果。

    * `item` あり + `exact` — 完全一致
    * `item` あり + not `exact` — 部分一致(LIKE)が1件だけ。早見表が複合名で持つ
      ケース(お知らせ「乾電池」→ 早見表「乾電池(充電式でないもの)」)。回答側は
      `TimelineSegment.matched_item_name` で限定を付けられる
    * `item` なし + `candidates` — 部分一致が複数。自動リンクせず人手判断
    * すべて空 — 前身なし(語順違い等。`relink(use_vector=True)` でベクトル候補を出せる)
    """

    item: "WasteItem | None" = None
    exact: bool = False
    candidates: "tuple[str, ...]" = ()


def find_current_predecessor(
    conn: sqlite3.Connection, item_name: str, exclude_id: "int | None" = None
) -> PredecessorMatch:
    """The 早見表 row a change notice supersedes: still current, and NOT itself a
    distributed-file row (so notices don't supersede each other).

    完全一致 → 部分一致(NFKC 正規化 + LIKE、`app/data/lookup.py` の
    `lookup_item_partial` と同じ方式。呼び出し側の接続で実行するだけ)。
    部分一致が複数なら曖昧としてリンクしない(お知らせ「乾電池」が早見表の
    充電式/非充電式のどちらを指すか通知文だけでは決まらない、等)。
    語順が違うケース(使い捨てライター→ライター)は LIKE では拾えない ―― これは
    ベクトル検索の担当(`relink(use_vector=True)`)。
    """
    norm = normalize_item_name(item_name)

    exact = conn.execute(
        "SELECT * FROM waste_items WHERE item_name = ? AND is_current = 1 "
        "AND source_url NOT LIKE ? AND id IS NOT ?",
        (norm, KNOWLEDGE_SCHEME + "%", exclude_id),
    ).fetchone()
    if exact:
        return PredecessorMatch(item=_row_to_item(exact), exact=True)

    def partial_rows(name: str):
        return conn.execute(
            "SELECT * FROM waste_items "
            "WHERE item_name LIKE ? ESCAPE '\\' AND is_current = 1 "
            "AND source_url NOT LIKE ? AND id IS NOT ? ORDER BY id",
            (f"%{_like_escape(name)}%", KNOWLEDGE_SCHEME + "%", exclude_id),
        ).fetchall()

    partial = partial_rows(norm)
    if len(partial) == 1:
        return PredecessorMatch(item=_row_to_item(partial[0]), exact=False)
    if len(partial) > 1:
        return PredecessorMatch(candidates=tuple(r["item_name"] for r in partial))

    # 早見表「花火」←お知らせ「使用済みの花火」、早見表「ライター」←
    # お知らせ「使い捨てライター」、早見表「カイロ(使い捨て)」←お知らせ
    # 「使い捨てカイロ」。元名で何も見つからなかった時だけ、状態修飾語を
    # 外して exact → LIKE を再試行する。LIKE は従来どおり一意な時だけ採用。
    base = _strip_predecessor_prefix(norm)
    if base != norm:
        stripped_exact = conn.execute(
            "SELECT * FROM waste_items WHERE item_name = ? AND is_current = 1 "
            "AND source_url NOT LIKE ? AND id IS NOT ?",
            (base, KNOWLEDGE_SCHEME + "%", exclude_id),
        ).fetchone()
        if stripped_exact:
            # 元の通知名との完全一致ではないため partial link として可視化する。
            return PredecessorMatch(item=_row_to_item(stripped_exact), exact=False)

        stripped_partial = partial_rows(base)
        if len(stripped_partial) == 1:
            return PredecessorMatch(item=_row_to_item(stripped_partial[0]), exact=False)
        if len(stripped_partial) > 1:
            return PredecessorMatch(
                candidates=tuple(r["item_name"] for r in stripped_partial)
            )
    return PredecessorMatch()


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


def _history_sort_key(row: "sqlite3.Row | dict") -> tuple:
    # NULL effective_from (the 早見表 baseline) first, then by date, then id.
    ef = row["effective_from"]
    return (ef is not None, ef or "", row["id"])


def lookup_item_history(conn: sqlite3.Connection, item_name: str) -> "list[WasteItem]":
    """Every known version of one item, oldest first — the requirement 4-2 read
    path. No `WHERE is_current = 1`, unlike `app/data/lookup.py`: the caller gets
    the whole chain and decides what applies on a given date (see
    `version_window`).

    The chain is followed through `superseded_by`, not just `item_name`: a
    change notice may be linked to a 早見表 row stored under a more specific name
    (お知らせ「乾電池」→ 早見表「乾電池(充電式でないもの)」), so name-only matching
    would miss the predecessor. Starting from every row whose name matches, the
    connected component in the supersede graph is collected.
    """
    seed = conn.execute(
        "SELECT * FROM waste_items WHERE item_name = ?",
        (normalize_item_name(item_name),),
    ).fetchall()
    if not seed:
        return []

    by_id: "dict[int, dict]" = {}
    frontier = [dict(r) for r in seed]
    while frontier:
        row = frontier.pop()
        if row["id"] in by_id:
            continue
        by_id[row["id"]] = row
        # newer: the row this one was superseded by
        nxt_id = row["superseded_by"]
        if nxt_id and nxt_id not in by_id:
            nxt = conn.execute(
                "SELECT * FROM waste_items WHERE id = ?", (nxt_id,)
            ).fetchone()
            if nxt:
                frontier.append(dict(nxt))
        # older: rows that were superseded by this one
        for prev in conn.execute(
            "SELECT * FROM waste_items WHERE superseded_by = ?", (row["id"],)
        ).fetchall():
            if prev["id"] not in by_id:
                frontier.append(dict(prev))

    return [_row_to_item(r) for r in sorted(by_id.values(), key=_history_sort_key)]


def version_window(
    versions: "list[WasteItem]", index: int
) -> "tuple[date | None, date | None]":
    """(valid_from, valid_until) for `versions[index]` as returned by
    `lookup_item_history`. `valid_until` is derived — a rule runs until the day
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
    """Return True only when the notice metadata and its produced row exist.

    ``app.data.db.rebuild_db()`` performs a full replacement of
    ``waste_items``.  ``knowledge_sources`` intentionally lives outside that
    shared schema, so a crawl can leave its idempotency record pointing at a
    deleted row.  Treating that orphan as ingested would make the following
    ingest skip the notice and silently disable STAGE04 4-1/4-2.
    """
    return (
        conn.execute(
            """
            SELECT 1
            FROM knowledge_sources ks
            JOIN waste_items wi ON wi.id = ks.produced_item_id
            WHERE ks.content_hash = ?
              AND wi.source_url LIKE ?
            """,
            (content_hash, f"{KNOWLEDGE_SCHEME}%"),
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
    values = (
        source_file,
        source_type,
        anchor,
        title,
        content_hash,
        ingested_at.isoformat(),
        produced_item_id,
        1 if is_change_notice else 0,
        prior_disposal_method,
    )

    # A full crawl replaces ``waste_items`` but leaves this STAGE04 manifest
    # intact.  Reuse the manifest row when its produced item disappeared;
    # inserting another row would violate the UNIQUE content_hash constraint.
    existing = conn.execute(
        "SELECT id FROM knowledge_sources WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    if existing is not None:
        conn.execute(
            """
            UPDATE knowledge_sources
            SET source_file = ?, source_type = ?, anchor = ?, title = ?,
                content_hash = ?, ingested_at = ?, produced_item_id = ?,
                is_change_notice = ?, prior_disposal_method = ?
            WHERE id = ?
            """,
            (*values, existing["id"]),
        )
        return int(existing["id"])

    cur = conn.execute(
        """
        INSERT INTO knowledge_sources (
            source_file, source_type, anchor, title, content_hash,
            ingested_at, produced_item_id, is_change_notice, prior_disposal_method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
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


# --------------------------------------------------------------------------
# item_timeline — a flat old→new view for the answer layer (requirement 4-2)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TimelineSegment:
    """One rule and the span it applies to.

    `valid_from = None` means "from the baseline" (no known start); `valid_until
    = None` means "still in force". `synthesized` marks a segment that has no row
    of its own — it was reconstructed from a notice's account of the prior 早見表
    treatment, used when the crawler never had (or has since lost) that row.
    """

    disposal_method: str
    valid_from: "date | None"
    valid_until: "date | None"
    source_url: str
    from_notice: bool
    synthesized: bool = False
    # set when the 早見表 row and a notice's stated 現行 disagree: `disposal_method`
    # holds the notice's account (authoritative "before"), `hayamihyo_disposal_method`
    # the crawled row's differing value.
    disputed: bool = False
    hayamihyo_disposal_method: "str | None" = None
    # set when this segment's 早見表 row is stored under a more specific name
    # than the query (e.g. queried 乾電池 → row 乾電池(充電式でないもの)).
    matched_item_name: "str | None" = None


@dataclass(frozen=True)
class ItemTimeline:
    item_name: str
    found: bool
    segments: "tuple[TimelineSegment, ...]"

    @property
    def current_segment(self) -> "TimelineSegment | None":
        """The rule with no end date (the latest one)."""
        for seg in reversed(self.segments):
            if seg.valid_until is None:
                return seg
        return self.segments[-1] if self.segments else None

    @property
    def change_dates(self) -> "tuple[date, ...]":
        """Every switch-over date (the `valid_from` of each non-first segment)."""
        return tuple(
            seg.valid_from for seg in self.segments[1:] if seg.valid_from is not None
        )

    def segment_on(self, when: date) -> "TimelineSegment | None":
        """Which rule applies on `when`."""
        for seg in self.segments:
            if (seg.valid_from is None or seg.valid_from <= when) and (
                seg.valid_until is None or when <= seg.valid_until
            ):
                return seg
        return None

    def changes_after(self, when: date) -> "tuple[date, ...]":
        """Switch-over dates strictly after `when` — non-empty ⇒ 4-2 applies."""
        return tuple(d for d in self.change_dates if d > when)


def knowledge_source_for_item(
    conn: sqlite3.Connection, produced_item_id: int
) -> "sqlite3.Row | None":
    return conn.execute(
        "SELECT * FROM knowledge_sources WHERE produced_item_id = ? LIMIT 1",
        (produced_item_id,),
    ).fetchone()


def item_timeline(conn: sqlite3.Connection, item_name: str) -> ItemTimeline:
    """`lookup_item_history` + `version_window`, flattened into an ordered list
    of (rule, span) so the answer layer never has to parse `notes` or reason
    about `is_current` / `superseded_by`.

    * 4-1 addition ....... 1 open-ended segment
    * 4-2, predecessor ... 2+ segments, real rows on both sides
    * 4-2, no predecessor  2 segments — the older one `synthesized=True`,
                           its rule taken from `knowledge_sources.prior_disposal_method`
    * unknown item ....... found=False, no segments (caller falls back to 4-3)
    """
    versions = lookup_item_history(conn, item_name)
    if not versions:
        return ItemTimeline(normalize_item_name(item_name), found=False, segments=())

    segments: "list[TimelineSegment]" = []

    first = versions[0]
    first_from_notice = first.source_url.startswith(KNOWLEDGE_SCHEME)
    if len(versions) == 1 and first_from_notice and first.effective_from is not None:
        # A change notice landed with no 早見表 row to supersede. Rebuild the
        # prior segment from the notice's own account of it.
        ks = knowledge_source_for_item(conn, first.id) if first.id else None
        prior = ks["prior_disposal_method"] if ks else None
        if prior:
            segments.append(
                TimelineSegment(
                    disposal_method=prior,
                    valid_from=None,
                    valid_until=first.effective_from - timedelta(days=1),
                    source_url=first.source_url,
                    from_notice=True,
                    synthesized=True,
                )
            )

    queried = normalize_item_name(item_name)
    for i, v in enumerate(versions):
        vf, vu = version_window(versions, i)
        from_notice = v.source_url.startswith(KNOWLEDGE_SCHEME)
        method, disputed, hayamihyo_method = v.disposal_method, False, None
        matched_name = (
            v.item_name if not from_notice and v.item_name != queried else None
        )
        successor = versions[i + 1] if i + 1 < len(versions) else None
        if not from_notice and successor is not None:
            ks = (
                knowledge_source_for_item(conn, successor.id)
                if successor.id
                and successor.source_url.startswith(KNOWLEDGE_SCHEME)
                else None
            )
            notice_prior = ks["prior_disposal_method"] if ks else None
            if notice_prior and notice_prior not in v.disposal_method:
                # the お知らせ is a targeted statement about this item's prior
                # rule; prefer it over a possibly stale / mis-parsed 早見表 row.
                method, disputed, hayamihyo_method = notice_prior, True, v.disposal_method
        segments.append(
            TimelineSegment(
                disposal_method=method,
                valid_from=vf,
                valid_until=vu,
                source_url=v.source_url,
                from_notice=from_notice,
                disputed=disputed,
                hayamihyo_disposal_method=hayamihyo_method,
                matched_item_name=matched_name,
            )
        )

    return ItemTimeline(
        normalize_item_name(item_name), found=True, segments=tuple(segments)
    )
