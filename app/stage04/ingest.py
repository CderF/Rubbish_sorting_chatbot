"""Ingest distributed お知らせ knowledge files into the structured DB (STAGE 04).

Covers two assignment requirements:

* **4-1** — a notice about a品目 that is not in the 早見表 is inserted standalone,
  so the next question about it is answered from the distributed file.
* **4-2** — a notice that changes an existing rule is inserted *alongside* the row
  it replaces: the new row carries `effective_from`, the old row is flagged
  `is_current = 0` with `superseded_by` pointing at the new one. Both stay
  readable through `store.lookup_item_versions`, which is what lets the answer
  layer say what applies now, from when it changes, and to what — instead of
  asserting either rule on its own.

This follows the operating rule in `documents/data_schema.md` §4. That section
says to set "`effective_from` に新情報の適用開始日" without naming the row; it is
written on the **new** row here, because the column means "the date this record
took effect" and putting the successor's start date on the retired row would
contradict its own definition.

Ordering with the crawler: if the 早見表 crawl (`app/data/crawler.py`) has not
run yet, a change notice has nothing to link to. It is still inserted, listed in
`IngestResult.unlinked`, and picked up later by `relink()` — or by re-running
ingest, which is a no-op for the row itself (idempotent on the notice's
SHA-256) but retries the link.

Usage::

    python -m app.stage04.ingest knowledge/incoming/お知らせ-4-1.md
    python -m app.stage04.ingest --relink        # after the crawler has run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.data.schema import WasteItem
from app.stage04 import store
from app.stage04.notice_parser import Notice, parse_notice_file

SOURCE_TYPE = "notice"


@dataclass
class IngestResult:
    """What one ingest run did, in terms a human can check against the file."""

    source_file: str
    inserted: "list[str]" = field(default_factory=list)
    linked: "list[tuple[str, int, int]]" = field(default_factory=list)
    unlinked: "list[str]" = field(default_factory=list)
    skipped: "list[str]" = field(default_factory=list)
    disputed: "list[tuple[str, str, str]]" = field(default_factory=list)
    warnings: "list[str]" = field(default_factory=list)

    @property
    def record_count(self) -> int:
        return len(self.inserted)

    def summary(self) -> str:
        lines = [
            f"{self.source_file}: {self.record_count}件を取り込みました "
            f"(新旧リンク {len(self.linked)}件 / 前身なし {len(self.unlinked)}件 / "
            f"取込済みスキップ {len(self.skipped)}件)"
        ]
        if self.unlinked:
            lines.append(
                "  前身の早見表レコードが見つかりませんでした"
                "（クローラー実行後に `--relink` でリンクされます）: "
                + "、".join(self.unlinked)
            )
        for name, notice_says, crawl_says in self.disputed:
            lines.append(
                f"  不一致: {name} — お知らせは現行「{notice_says}」だが"
                f"早見表は「{crawl_says}」。両方を残しリンクは張った"
            )
        for w in self.warnings:
            lines.append(f"  警告: {w}")
        return "\n".join(lines)


@dataclass
class RelinkResult:
    linked: "list[tuple[str, int, int]]" = field(default_factory=list)
    still_unlinked: "list[str]" = field(default_factory=list)
    # linked, but the crawled 早見表 row's disposal_method does not corroborate
    # the notice's stated 現行 treatment — surfaced, not blocked.
    disputed: "list[tuple[str, str, str]]" = field(default_factory=list)

    def summary(self) -> str:
        line = (
            f"relink: {len(self.linked)}件をリンクしました "
            f"(未リンクのまま {len(self.still_unlinked)}件 / 現行表記の不一致 {len(self.disputed)}件)"
        )
        if self.still_unlinked:
            line += "\n  早見表に該当なし: " + "、".join(self.still_unlinked)
        for name, notice_says, crawl_says in self.disputed:
            line += (
                f"\n  不一致: {name} — お知らせは現行「{notice_says}」だが"
                f"早見表は「{crawl_says}」。両方を残しリンクは張った"
            )
        return line


def build_source_url(notice: Notice) -> str:
    """Citation target for a distributed file.

    These notices are handed to us as files, not web pages, so there is no http
    URL to cite. `knowledge://<file>#<anchor>` keeps `source_url` populated (the
    column is NOT NULL and requirement 2-2 needs a citation) while making it
    obvious the source is a distributed file rather than the city's site.
    """
    return f"{store.KNOWLEDGE_SCHEME}{notice.source_file}#{notice.anchor}"


def build_notes(notice: Notice) -> str:
    parts = [notice.title]
    if notice.is_change_notice:
        until = (
            notice.prior_valid_until.isoformat() if notice.prior_valid_until else "不明"
        )
        parts.append(
            f"現行（早見表）の「{notice.prior_disposal_method}」としての扱いは{until}まで有効。"
        )
    if notice.issuer:
        parts.append(notice.issuer)
    return " / ".join(p for p in parts if p)


def notice_to_waste_item(notice: Notice, ingested_at: datetime) -> WasteItem:
    return WasteItem(
        item_name=notice.item_name,  # NFKC-normalized by the model validator
        item_name_raw=notice.heading,
        disposal_method=notice.body,
        notes=build_notes(notice),
        source_url=build_source_url(notice),
        source_page_title=notice.title or None,
        crawled_at=ingested_at,
        is_current=True,
        effective_from=notice.effective_from,
        superseded_by=None,
    )


def ingest_notices(
    notices: "list[Notice]",
    conn: sqlite3.Connection,
    *,
    source_file: str,
    now: "datetime | None" = None,
) -> IngestResult:
    ingested_at = now or store.utc_now()
    result = IngestResult(source_file=source_file)

    for notice in notices:
        result.warnings.extend(f"{notice.anchor}: {w}" for w in notice.warnings)

        if store.is_ingested(conn, notice.content_hash):
            result.skipped.append(notice.item_name)
            continue

        # Find the predecessor *before* inserting, else the new row matches its
        # own is_current = 1 lookup.
        predecessor = (
            store.find_current_predecessor(conn, notice.item_name)
            if notice.is_change_notice
            else None
        )

        new_id = store.insert_waste_item(conn, notice_to_waste_item(notice, ingested_at))
        result.inserted.append(notice.item_name)

        if notice.is_change_notice:
            if predecessor is not None and predecessor.id is not None:
                store.mark_superseded(conn, predecessor.id, new_id)
                result.linked.append((notice.item_name, predecessor.id, new_id))
                if notice.prior_disposal_method and (
                    notice.prior_disposal_method not in (predecessor.disposal_method or "")
                ):
                    result.disputed.append(
                        (
                            notice.item_name,
                            notice.prior_disposal_method,
                            predecessor.disposal_method,
                        )
                    )
            else:
                result.unlinked.append(notice.item_name)

        store.record_knowledge_source(
            conn,
            source_file=notice.source_file,
            source_type=SOURCE_TYPE,
            anchor=notice.anchor,
            title=notice.title,
            content_hash=notice.content_hash,
            ingested_at=ingested_at,
            produced_item_id=new_id,
            is_change_notice=notice.is_change_notice,
            prior_disposal_method=notice.prior_disposal_method,
        )

    conn.commit()
    return result


def ingest_notice_file(
    path: "str | Path",
    conn: sqlite3.Connection,
    *,
    now: "datetime | None" = None,
) -> IngestResult:
    path = Path(path)
    return ingest_notices(
        parse_notice_file(path), conn, source_file=path.name, now=now
    )


def relink(conn: sqlite3.Connection) -> RelinkResult:
    """Link change-notices that had no predecessor at ingest time to the 早見表
    row that now exists (e.g. after `app/data/crawler.py` populated it).

    Safe to run repeatedly: rows that are already linked are skipped by
    `store.unlinked_change_sources`.
    """
    result = RelinkResult()
    for row in store.unlinked_change_sources(conn):
        new_id = row["produced_item_id"]
        predecessor = store.find_current_predecessor(
            conn, row["item_name"], exclude_id=new_id
        )
        if predecessor is None or predecessor.id is None:
            result.still_unlinked.append(row["item_name"])
            continue
        store.mark_superseded(conn, predecessor.id, new_id)
        result.linked.append((row["item_name"], predecessor.id, new_id))

        notice_says = row["prior_disposal_method"]
        if notice_says and notice_says not in (predecessor.disposal_method or ""):
            result.disputed.append(
                (row["item_name"], notice_says, predecessor.disposal_method)
            )
    conn.commit()
    return result


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="配布されたお知らせファイルを構造化DBに取り込む（STAGE 04）"
    )
    parser.add_argument("files", nargs="*", type=Path, help="取り込むお知らせ .md")
    parser.add_argument(
        "--relink",
        action="store_true",
        help="取り込み済みの変更通知を、いま存在する早見表レコードにリンクし直す",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite DB (既定: app/data/db.py の DB_PATH / var/kitakyushu_waste.db)",
    )
    args = parser.parse_args(argv)

    if not args.files and not args.relink:
        parser.error("取り込むファイルか --relink のいずれかを指定してください")

    conn = store.connect(args.db)
    store.init_schema(conn)
    try:
        for path in args.files:
            if not path.exists():
                print(f"ファイルが見つかりません: {path}", file=sys.stderr)
                return 1
            print(ingest_notice_file(path, conn).summary())
        if args.relink:
            print(relink(conn).summary())
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
