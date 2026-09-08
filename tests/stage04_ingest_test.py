"""Tests for the STAGE 04 ingest path (requirements 4-1 and 4-2).

Everything here runs offline against an in-memory SQLite DB — no GPU, no server,
no network. The 早見表 side is stubbed from `tests/fixtures/hayamihyo_seed.json`
because the real crawler (issue #4, Teamb2) does not exist yet; see that file's
_README for why those values are safe to assert against but not to ship.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from app.data.schema import WasteItem
from app.stage04 import store
from app.stage04.ingest import ingest_notice_file, relink
from app.stage04.notice_parser import parse_notice_file

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE = REPO_ROOT / "knowledge" / "incoming"
ADDITIONS = KNOWLEDGE / "お知らせ-4-1.md"
CHANGES = KNOWLEDGE / "お知らせ-4-2.md"
SEED = Path(__file__).resolve().parent / "fixtures" / "hayamihyo_seed.json"

CHANGE_START = date(2026, 10, 1)
PRIOR_END = date(2026, 9, 30)
INGEST_AT = datetime(2026, 9, 8, 3, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    c = store.connect(":memory:")
    store.init_schema(c)
    yield c
    c.close()


def seed_rows() -> "list[dict]":
    return json.loads(SEED.read_text(encoding="utf-8"))["rows"]


@pytest.fixture
def seeded_conn(conn):
    """A DB pre-loaded with the 15 pre-change 早見表 entries."""
    for row in seed_rows():
        store.insert_waste_item(
            conn,
            WasteItem(
                item_name=row["item_name"],
                item_name_raw=row["item_name"],
                disposal_method=row["disposal_method"],
                notes="",
                source_url="fixture://test-only",
                source_page_title=row["source_page_title"],
                crawled_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                is_current=True,
            ),
        )
    conn.commit()
    return conn


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def test_each_file_yields_fifteen_notices_and_skips_the_toc():
    assert len(parse_notice_file(ADDITIONS)) == 15
    assert len(parse_notice_file(CHANGES)) == 15
    # 目次 is a `##` section too, so it would show up if it were not filtered.
    assert all(n.item_name != "目次" for n in parse_notice_file(ADDITIONS))


def test_addition_file_strips_the_kaishu_suffix_from_headings():
    notices = {n.number: n for n in parse_notice_file(ADDITIONS)}
    assert notices[1].heading == "ホバーボード回収"
    assert notices[1].item_name == "ホバーボード"
    assert notices[14].item_name == "USB卓上扇風機"
    assert all(not n.item_name.endswith("回収") for n in notices.values())


def test_change_file_headings_are_bare_item_names():
    notices = {n.number: n for n in parse_notice_file(CHANGES)}
    assert notices[1].item_name == "スプレー缶" == notices[1].heading
    assert notices[15].item_name == "乾電池"


def test_title_body_and_issuer_are_separated():
    notice = parse_notice_file(CHANGES)[0]
    assert notice.title == "【市からのお知らせ】スプレー缶の出し方が変わります"
    assert notice.issuer == "北九州市 環境局 循環社会推進部業務課"
    assert notice.body.startswith("2026年10月から、スプレー缶の出し方を変更します。")
    # the 現行 clause is context, not an instruction, so it is lifted out of body
    assert "現行" not in notice.body
    assert notice.issuer not in notice.body


def test_notices_are_classified_by_content_not_filename():
    assert all(not n.is_change_notice for n in parse_notice_file(ADDITIONS))
    assert all(n.is_change_notice for n in parse_notice_file(CHANGES))


def test_change_notices_all_start_on_the_same_date_with_no_gap():
    for notice in parse_notice_file(CHANGES):
        assert notice.effective_from == CHANGE_START
        assert notice.prior_valid_until == PRIOR_END
        assert not notice.warnings  # continuity check passed


def test_addition_notices_have_no_effective_date():
    for notice in parse_notice_file(ADDITIONS):
        assert notice.effective_from is None
        assert notice.prior_disposal_method is None


def test_prior_treatment_is_extracted_for_all_three_phrasings():
    """The 現行 clause is written four different ways across the 15 notices."""
    parsed = {n.item_name: n.prior_disposal_method for n in parse_notice_file(CHANGES)}
    expected = {r["item_name"]: r["disposal_method"] for r in seed_rows()}
    assert parsed == expected
    # spot-check the awkward shapes explicitly
    assert parsed["スプレー缶"] == "燃えないごみ"                      # 「…」としての
    assert parsed["蛍光管"] == "家庭ごみ"                            # bare + としての
    assert parsed["使用済みの花火"] == "そのまま家庭ごみに出してよい"      # 「…」扱い
    assert parsed["乾電池"].startswith("電池類。")                    # 「…」という扱い


def test_content_hash_is_unique_per_notice():
    hashes = [n.content_hash for n in parse_notice_file(CHANGES)]
    assert len(set(hashes)) == 15


# --------------------------------------------------------------------------
# 4-1 — additions
# --------------------------------------------------------------------------
def test_addition_notices_insert_standalone_rows(conn):
    result = ingest_notice_file(ADDITIONS, conn, now=INGEST_AT)

    assert result.record_count == 15
    assert result.linked == []
    assert result.unlinked == []

    rows = conn.execute("SELECT * FROM waste_items").fetchall()
    assert len(rows) == 15
    assert all(r["is_current"] == 1 for r in rows)
    assert all(r["effective_from"] is None for r in rows)
    assert all(r["superseded_by"] is None for r in rows)


def test_addition_row_carries_a_citable_source(conn):
    ingest_notice_file(ADDITIONS, conn, now=INGEST_AT)
    versions = store.lookup_item_history(conn, "ドローン")

    assert len(versions) == 1
    item = versions[0]
    assert item.source_url == "knowledge://お知らせ-4-1.md#お知らせ-05-ドローン回収"
    assert item.source_page_title.startswith("【市からのお知らせ】")
    assert "無料で回収" in item.disposal_method
    assert item.item_name_raw == "ドローン回収"  # page wording preserved


# --------------------------------------------------------------------------
# 4-2 — changes forming a version chain
# --------------------------------------------------------------------------
def test_change_notices_supersede_the_existing_rows(seeded_conn):
    result = ingest_notice_file(CHANGES, seeded_conn, now=INGEST_AT)

    assert result.record_count == 15
    assert len(result.linked) == 15
    assert result.unlinked == []

    old = seeded_conn.execute("SELECT * FROM waste_items WHERE is_current = 0").fetchall()
    new = seeded_conn.execute("SELECT * FROM waste_items WHERE is_current = 1").fetchall()
    assert len(old) == 15
    assert len(new) == 15
    assert all(r["superseded_by"] is not None for r in old)
    assert all(r["effective_from"] == "2026-10-01" for r in new)
    # the retired row keeps its own effective_from (see data_schema.md §4 note)
    assert all(r["effective_from"] is None for r in old)


def test_lookup_item_history_returns_both_rules_oldest_first(seeded_conn):
    ingest_notice_file(CHANGES, seeded_conn, now=INGEST_AT)
    versions = store.lookup_item_history(seeded_conn, "スプレー缶")

    assert len(versions) == 2
    old, new = versions
    assert old.disposal_method == "燃えないごみ"
    assert old.is_current is False
    assert old.superseded_by == new.id
    assert old.effective_from is None

    assert new.is_current is True
    assert new.effective_from == CHANGE_START
    assert "かん・びん" in new.disposal_method
    assert "燃えないごみ" in new.notes  # what it used to be, kept for the answer


def test_lookup_item_history_ignores_the_is_current_flag(seeded_conn):
    """The whole point of this function: `lookup_item_exact` pins is_current = 1
    and so can only see the future rule, which is what 4-2 must not do."""
    ingest_notice_file(CHANGES, seeded_conn, now=INGEST_AT)
    for item_name in (r["item_name"] for r in seed_rows()):
        assert len(store.lookup_item_history(seeded_conn, item_name)) == 2


def test_version_window_derives_the_end_of_the_superseded_rule(seeded_conn):
    ingest_notice_file(CHANGES, seeded_conn, now=INGEST_AT)
    versions = store.lookup_item_history(seeded_conn, "乾電池")

    old_from, old_until = store.version_window(versions, 0)
    new_from, new_until = store.version_window(versions, 1)

    assert old_from is None
    assert old_until == PRIOR_END          # 2026-10-01 minus one day
    assert new_from == CHANGE_START
    assert new_until is None               # the latest rule has no end


def test_query_is_nfkc_normalized(seeded_conn):
    ingest_notice_file(CHANGES, seeded_conn, now=INGEST_AT)
    # full-width alphanumerics / half-width kana must resolve to the same key
    assert store.lookup_item_history(seeded_conn, "  スプレー缶 ") != []


# --------------------------------------------------------------------------
# missing predecessor, idempotency, manifest
# --------------------------------------------------------------------------
def test_change_notices_still_ingest_when_the_crawler_has_not_run(conn):
    """Nothing to link to yet — insert anyway and report it, so re-running after
    the 早見表 crawler lands can complete the chain."""
    result = ingest_notice_file(CHANGES, conn, now=INGEST_AT)

    assert result.record_count == 15
    assert result.linked == []
    assert len(result.unlinked) == 15
    assert "スプレー缶" in result.unlinked
    assert all(r["is_current"] == 1 for r in conn.execute("SELECT * FROM waste_items"))


def test_reingesting_the_same_file_changes_nothing(seeded_conn):
    first = ingest_notice_file(CHANGES, seeded_conn, now=INGEST_AT)
    before = seeded_conn.execute("SELECT COUNT(*) FROM waste_items").fetchone()[0]

    second = ingest_notice_file(CHANGES, seeded_conn, now=INGEST_AT)
    after = seeded_conn.execute("SELECT COUNT(*) FROM waste_items").fetchone()[0]

    assert first.record_count == 15
    assert second.record_count == 0
    assert len(second.skipped) == 15
    assert before == after == 30


def test_both_files_ingest_together(seeded_conn):
    ingest_notice_file(ADDITIONS, seeded_conn, now=INGEST_AT)
    ingest_notice_file(CHANGES, seeded_conn, now=INGEST_AT)

    # 15 seeded + 15 additions + 15 changes
    assert seeded_conn.execute("SELECT COUNT(*) FROM waste_items").fetchone()[0] == 45
    assert len(store.lookup_item_history(seeded_conn, "ホバーボード")) == 1
    assert len(store.lookup_item_history(seeded_conn, "スプレー缶")) == 2


def test_knowledge_sources_lists_every_ingested_notice(conn):
    ingest_notice_file(ADDITIONS, conn, now=INGEST_AT)
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)

    sources = store.list_knowledge_sources(conn)
    assert len(sources) == 30
    assert {s["source_file"] for s in sources} == {
        "お知らせ-4-1.md",
        "お知らせ-4-2.md",
    }
    assert all(s["source_type"] == "notice" for s in sources)
    assert all(s["content_hash"] for s in sources)


# --------------------------------------------------------------------------
# relink — connecting change-notices to the 早見表 once the crawler has run
# --------------------------------------------------------------------------
def _add_hayamihyo_row(conn, item_name: str, disposal_method: str) -> int:
    """Simulate a row the real crawler (app/data/crawler.py) would produce."""
    return store.insert_waste_item(
        conn,
        WasteItem(
            item_name=item_name,
            item_name_raw=item_name,
            disposal_method=disposal_method,
            notes="",
            source_url="https://www.city.kitakyushu.lg.jp/contents/924_10041.html",
            source_page_title="ごみの分別早見表（さ行）",
            crawled_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
            is_current=True,
        ),
    )


def test_relink_connects_change_notices_after_the_crawler_runs(conn):
    # notices ingested first, 早見表 still empty -> all unlinked
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    assert all(r["is_current"] == 1 for r in conn.execute("SELECT * FROM waste_items"))

    # crawler runs later
    for row in seed_rows():
        _add_hayamihyo_row(conn, row["item_name"], row["disposal_method"])
    conn.commit()

    result = relink(conn)

    assert len(result.linked) == 15
    assert result.still_unlinked == []
    versions = store.lookup_item_history(conn, "スプレー缶")
    assert len(versions) == 2
    assert versions[0].disposal_method == "燃えないごみ"
    assert versions[0].is_current is False
    assert versions[0].superseded_by == versions[1].id
    assert versions[1].effective_from == CHANGE_START


def test_relink_is_a_noop_when_everything_is_already_linked(seeded_conn):
    ingest_notice_file(CHANGES, seeded_conn, now=INGEST_AT)  # links during ingest
    result = relink(seeded_conn)
    assert result.linked == []
    assert result.still_unlinked == []


def test_relink_reports_items_with_no_hayamihyo_entry(conn):
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    _add_hayamihyo_row(conn, "スプレー缶", "燃えないごみ")  # only one of the 15
    conn.commit()

    result = relink(conn)

    assert [name for name, _, _ in result.linked] == ["スプレー缶"]
    assert len(result.still_unlinked) == 14
    assert "乾電池" in result.still_unlinked


def test_relink_never_makes_notices_supersede_each_other(conn):
    """Only real 早見表 rows (non-knowledge:// source_url) are eligible
    predecessors, so ingesting the same change twice from different files does
    not chain notice-onto-notice."""
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    result = relink(conn)  # nothing crawled yet
    assert result.linked == []
    # the 15 change rows are all still current, none superseded by another notice
    rows = conn.execute("SELECT * FROM waste_items").fetchall()
    assert all(r["is_current"] == 1 and r["superseded_by"] is None for r in rows)


def test_relink_via_cli(tmp_path, monkeypatch):
    from app.stage04 import ingest as ingest_mod

    db_file = tmp_path / "kb.db"
    conn = store.connect(db_file)
    store.init_schema(conn)
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    for row in seed_rows():
        _add_hayamihyo_row(conn, row["item_name"], row["disposal_method"])
    conn.commit()
    conn.close()

    rc = ingest_mod.main(["--relink", "--db", str(db_file)])
    assert rc == 0

    conn = store.connect(db_file)
    assert len(store.lookup_item_history(conn, "乾電池")) == 2
    conn.close()


def test_reingest_then_relink_completes_the_chain(conn):
    """The documented recovery path: ingest before the crawler, crawl, then
    re-run — the re-run skips the row but relink finishes the link."""
    first = ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    assert len(first.unlinked) == 15

    for row in seed_rows():
        _add_hayamihyo_row(conn, row["item_name"], row["disposal_method"])
    conn.commit()

    second = ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    assert second.record_count == 0 and len(second.skipped) == 15  # idempotent

    result = relink(conn)
    assert len(result.linked) == 15


def test_relink_flags_but_does_not_block_a_disposal_method_mismatch(conn):
    """The distributed file's stated 現行 treatment may disagree with the crawl
    (or the crawl may be incomplete). relink still links on item identity and
    reports the discrepancy rather than dropping the link."""
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    # crawl says スプレー缶 is 家庭ごみ; the notice says its 現行 was 燃えないごみ
    _add_hayamihyo_row(conn, "スプレー缶", "家庭ごみ")
    conn.commit()

    result = relink(conn)

    assert [n for n, _, _ in result.linked] == ["スプレー缶"]
    assert result.disputed == [("スプレー缶", "燃えないごみ", "家庭ごみ")]
    versions = store.lookup_item_history(conn, "スプレー缶")
    assert len(versions) == 2  # link was still made
    # the notice's own account of the prior state is preserved on the new row
    assert "燃えないごみ" in versions[1].notes


def test_prior_disposal_method_is_persisted(conn):
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    rows = {
        r["title"].split("】")[1].split("の")[0] if "】" in (r["title"] or "") else r["anchor"]: r
        for r in store.list_knowledge_sources(conn)
    }
    by_anchor = {r["anchor"]: r for r in store.list_knowledge_sources(conn)}
    assert by_anchor["お知らせ-01-スプレー缶"]["prior_disposal_method"] == "燃えないごみ"
    assert by_anchor["お知らせ-15-乾電池"]["prior_disposal_method"].startswith("電池類。")


# --------------------------------------------------------------------------
# item_timeline — the flat old→new view for the answer layer
# --------------------------------------------------------------------------
def test_item_timeline_linked_pair(seeded_conn):
    ingest_notice_file(CHANGES, seeded_conn, now=INGEST_AT)
    tl = store.item_timeline(seeded_conn, "スプレー缶")

    assert tl.found is True
    assert len(tl.segments) == 2
    old, new = tl.segments
    assert old.synthesized is False
    assert old.disposal_method == "燃えないごみ"
    assert (old.valid_from, old.valid_until) == (None, PRIOR_END)
    assert new.from_notice is True
    assert (new.valid_from, new.valid_until) == (CHANGE_START, None)
    assert tl.change_dates == (CHANGE_START,)
    assert tl.segment_on(date(2026, 9, 8)).disposal_method == "燃えないごみ"
    assert "かん・びん" in tl.segment_on(date(2026, 10, 2)).disposal_method
    assert tl.changes_after(date(2026, 9, 8)) == (CHANGE_START,)
    assert tl.changes_after(date(2026, 11, 1)) == ()


def test_item_timeline_synthesizes_prior_when_crawler_lacks_the_row(conn):
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)  # empty DB, nothing to link
    tl = store.item_timeline(conn, "乾電池")

    assert len(tl.segments) == 2
    old, new = tl.segments
    assert old.synthesized is True
    assert old.disposal_method.startswith("電池類。")
    assert (old.valid_from, old.valid_until) == (None, PRIOR_END)
    assert new.synthesized is False and new.valid_from == CHANGE_START
    assert tl.segment_on(date(2026, 9, 8)).synthesized is True
    assert tl.changes_after(date(2026, 9, 8)) == (CHANGE_START,)


def test_item_timeline_addition_is_one_open_segment(conn):
    ingest_notice_file(ADDITIONS, conn, now=INGEST_AT)
    tl = store.item_timeline(conn, "ホバーボード")

    assert tl.found is True
    assert len(tl.segments) == 1
    seg = tl.segments[0]
    assert (seg.valid_from, seg.valid_until) == (None, None)
    assert seg.from_notice is True and seg.synthesized is False
    assert tl.change_dates == ()
    assert tl.changes_after(date(2026, 9, 8)) == ()


def test_item_timeline_unknown_item(conn):
    tl = store.item_timeline(conn, "宇宙服")
    assert tl.found is False
    assert tl.segments == ()
    assert tl.current_segment is None
    assert tl.segment_on(date(2026, 9, 8)) is None
