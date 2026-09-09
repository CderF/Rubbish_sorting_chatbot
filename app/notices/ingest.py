"""Ingest distributed お知らせ knowledge files into the structured DB (STAGE 04).

Covers two assignment requirements:

* **4-1** — a notice about a品目 that is not in the 早見表 is inserted standalone,
  so the next question about it is answered from the distributed file.
* **4-2** — a notice that changes an existing rule is inserted *alongside* the row
  it replaces: the new row carries `effective_from`, the old row is flagged
  `is_current = 0` with `superseded_by` pointing at the new one. Both stay
  readable through `store.lookup_item_history`, which is what lets the answer
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

    python -m app.notices.ingest knowledge/incoming/お知らせ-4-1.md
    python -m app.notices.ingest --relink        # after the crawler has run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.data.schema import WasteItem
from app.notices import store
from app.rag import guard
from app.notices.notice_parser import Notice, parse_notice_file

# 語順違い(使い捨てライター→ライター)は LIKE では拾えないので、任意でベクトル
# 検索(issue #7)を使う。未インストール/インデックス未構築でも動くようフォール
# バック。vector_search_items は現状スコアを返さないため、ヒットしても自動リンク
# せず人手レビュー行き(RelinkResult.vector_candidates)にとどめる。
try:
    from app.rag.vector_store import vector_search_items  # type: ignore[import-not-found]
except ImportError:
    def vector_search_items(query: str, top_k: int = 3):  # noqa: ARG001
        return []

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
    # linked via a base-name match, not an identical name (お知らせ「乾電池」→
    # 早見表「乾電池(充電式でないもの)」). (notice_name, hayamihyo_name).
    partial_links: "list[tuple[str, str]]" = field(default_factory=list)
    # base-name matched several 早見表 rows — NOT linked, needs a human.
    ambiguous: "list[tuple[str, tuple[str, ...]]]" = field(default_factory=list)
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
        for notice_name, hayamihyo_name in self.partial_links:
            lines.append(
                f"  部分一致リンク: お知らせ「{notice_name}」→ 早見表「{hayamihyo_name}」"
            )
        for name, cands in self.ambiguous:
            lines.append(
                f"  曖昧: お知らせ「{name}」が早見表の複数行に該当（{'、'.join(cands)}）。"
                "自動リンクせず。人手で確認"
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
    partial_links: "list[tuple[str, str]]" = field(default_factory=list)
    ambiguous: "list[tuple[str, tuple[str, ...]]]" = field(default_factory=list)
    # vector search found candidates for a name LIKE can't reach; NOT linked
    # (no similarity score yet) — for human review. (notice_name, candidate names).
    vector_candidates: "list[tuple[str, tuple[str, ...]]]" = field(default_factory=list)
    # linked, but the crawled 早見表 row's disposal_method does not corroborate
    # the notice's stated 現行 treatment — surfaced, not blocked.
    disputed: "list[tuple[str, str, str]]" = field(default_factory=list)

    def summary(self) -> str:
        line = (
            f"relink: {len(self.linked)}件をリンクしました "
            f"(未リンクのまま {len(self.still_unlinked)}件 / 部分一致 {len(self.partial_links)}件 / "
            f"曖昧 {len(self.ambiguous)}件 / 現行表記の不一致 {len(self.disputed)}件)"
        )
        if self.still_unlinked:
            line += "\n  早見表に該当なし: " + "、".join(self.still_unlinked)
        for notice_name, hayamihyo_name in self.partial_links:
            line += f"\n  部分一致: お知らせ「{notice_name}」→ 早見表「{hayamihyo_name}」"
        for name, cands in self.ambiguous:
            line += (
                f"\n  曖昧: お知らせ「{name}」→ 複数該当（{'、'.join(cands)}）。人手で確認"
            )
        for name, cands in self.vector_candidates:
            line += (
                f"\n  ベクトル候補: お知らせ「{name}」→ {'、'.join(cands)}"
                "（スコア未対応のため自動リンクせず。人手で確認）"
            )
        for name, notice_says, crawl_says in self.disputed:
            line += (
                f"\n  不一致: {name} — お知らせは現行「{notice_says}」だが"
                f"早見表は「{crawl_says}」。両方を残しリンクは張った"
            )
        return line


def _url_part(text: str) -> str:
    """Sanitize a filename / anchor before it goes into `source_url`.

    `source_url` is appended to the answer body verbatim by `format_citations`,
    so anything here reaches the graded text. Spaces are dropped as well —
    `sanitize_text` folds newlines to spaces, which a URL must not carry.
    """
    return guard.sanitize_text(text, max_chars=120).replace(" ", "")


def build_source_url(notice: Notice) -> str:
    """Citation target for a distributed file.

    These notices are handed to us as files, not web pages, so there is no http
    URL to cite. `knowledge://<file>#<anchor>` keeps `source_url` populated (the
    column is NOT NULL and requirement 2-2 needs a citation) while making it
    obvious the source is a distributed file rather than the city's site.
    """
    return (
        f"{store.KNOWLEDGE_SCHEME}{_url_part(notice.source_file)}"
        f"#{_url_part(notice.anchor)}"
    )


def build_notes(notice: Notice) -> str:
    parts = [guard.sanitize_text(notice.title, max_chars=guard.MAX_TITLE_CHARS)]
    if notice.is_change_notice:
        until = (
            notice.prior_valid_until.isoformat() if notice.prior_valid_until else "不明"
        )
        prior = guard.sanitize_text(
            notice.prior_disposal_method or "", max_chars=guard.MAX_TITLE_CHARS
        )
        parts.append(f"現行（早見表）の「{prior}」としての扱いは{until}まで有効。")
    if notice.issuer:
        parts.append(guard.sanitize_text(notice.issuer, max_chars=guard.MAX_TITLE_CHARS))
    return " / ".join(p for p in parts if p)


def notice_to_waste_item(notice: Notice, ingested_at: datetime) -> WasteItem:
    """Convert one parsed notice into a `waste_items` row.

    Every field taken from the distributed file is sanitized here (PIJ
    「間接注入」対策). These files arrive during grading and their text ends up
    inside the *system* prompt via `build_system_prompt` — a stronger position
    than user input — so the cleaning happens once, at the only write path into
    the DB, rather than on every read.
    """
    title = guard.sanitize_text(notice.title, max_chars=guard.MAX_TITLE_CHARS)
    return WasteItem(
        item_name=guard.sanitize_item_name(notice.item_name),  # validator も NFKC する
        item_name_raw=guard.sanitize_item_name(notice.heading),
        disposal_method=guard.sanitize_text(notice.body),
        notes=build_notes(notice),
        source_url=build_source_url(notice),
        source_page_title=title or None,
        crawled_at=ingested_at,
        is_current=True,
        effective_from=notice.effective_from,
        superseded_by=None,
    )


def _record_link(
    result: "IngestResult | RelinkResult",
    notice_name: str,
    prior_disposal_method: "str | None",
    match: "store.PredecessorMatch",
    new_id: int,
    conn: sqlite3.Connection,
) -> None:
    """Apply one predecessor match: link it, or record why it wasn't linked."""
    if match.item is not None and match.item.id is not None:
        store.mark_superseded(conn, match.item.id, new_id)
        result.linked.append((notice_name, match.item.id, new_id))
        if not match.exact:
            result.partial_links.append((notice_name, match.item.item_name))
        if prior_disposal_method and (
            prior_disposal_method not in (match.item.disposal_method or "")
        ):
            result.disputed.append(
                (notice_name, prior_disposal_method, match.item.disposal_method)
            )
    elif match.candidates:
        result.ambiguous.append((notice_name, match.candidates))
    elif isinstance(result, IngestResult):
        result.unlinked.append(notice_name)
    else:
        result.still_unlinked.append(notice_name)


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

        # 取り込みは止めない(正当なお知らせを落とす方が損害が大きい)。
        # 本文は sanitize 済みでシステムプロンプトの構造は壊せないが、
        # 配布ファイルに指示文が混ざっていた事実自体は運用者に見せる。
        attack = guard.detect_attack(notice.body)
        if attack:
            result.warnings.append(
                f"{notice.anchor}: 本文に指示文らしき記述({attack})。"
                "取り込みは行うが内容を確認すること"
            )

        if store.is_ingested(conn, notice.content_hash):
            result.skipped.append(notice.item_name)
            continue

        # Find the predecessor *before* inserting, else the new row matches its
        # own is_current = 1 lookup.
        match = (
            store.find_current_predecessor(conn, notice.item_name)
            if notice.is_change_notice
            else store.PredecessorMatch()
        )

        new_id = store.insert_waste_item(conn, notice_to_waste_item(notice, ingested_at))
        result.inserted.append(notice.item_name)

        if notice.is_change_notice:
            _record_link(
                result, notice.item_name, notice.prior_disposal_method, match, new_id, conn
            )

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


def relink(conn: sqlite3.Connection, *, use_vector: bool = False) -> RelinkResult:
    """Link change-notices that had no predecessor at ingest time to the 早見表
    row that now exists (e.g. after `app/data/crawler.py` populated it).

    Safe to run repeatedly: rows that are already linked are skipped by
    `store.unlinked_change_sources`. With `use_vector=True`, names that exact +
    LIKE can't reach (語順違い: 使い捨てライター→ライター) are looked up with
    `vector_search_items`; the hits are reported in `vector_candidates` for a
    human — never auto-linked, since there is no similarity score to threshold on.
    """
    result = RelinkResult()
    for row in store.unlinked_change_sources(conn):
        name, new_id = row["item_name"], row["produced_item_id"]
        match = store.find_current_predecessor(conn, name, exclude_id=new_id)

        if match.item is None and not match.candidates and use_vector:
            hits = [
                h.item_name
                for h in vector_search_items(name, top_k=3)
                if not h.source_url.startswith(store.KNOWLEDGE_SCHEME)
            ]
            if hits:
                result.vector_candidates.append((name, tuple(hits)))
                result.still_unlinked.append(name)
                continue

        _record_link(
            result, name, row["prior_disposal_method"], match, new_id, conn
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
        "--vector",
        action="store_true",
        help="--relink 時、LIKE で拾えない品名をベクトル検索で候補出しする"
        "（自動リンクはしない）",
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
            print(relink(conn, use_vector=args.vector).summary())
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
