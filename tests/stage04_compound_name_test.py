"""早見表が品目を複合名で持つケース(FANG の PR #6 検証結果)。

お知らせ「乾電池」/「使い捨てライター」/「テレビ」に対し、早見表は
`乾電池(充電式でないもの)` / `ライター` / `テレビ(ブラウン管式・液晶式・プラズマ式)`。

前身の突き合わせ:
  完全一致 → 部分一致(NFKC + LIKE、app/data/lookup.lookup_item_partial と同方式)
  → 語順違い(使い捨てライター→ライター)はベクトル検索(任意、候補出しのみ)
チェーンは `item_name` ではなく `superseded_by` リンクで辿る。
"""

from __future__ import annotations

import datetime as dt
from datetime import date, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.data.schema import WasteItem
from app.stage04 import ingest as ingest_mod
from app.stage04 import store
from app.stage04.ingest import ingest_notice_file, relink

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGES = REPO_ROOT / "knowledge" / "incoming" / "お知らせ-4-2.md"
INGEST_AT = dt.datetime(2026, 9, 8, 3, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    c = store.connect(":memory:")
    store.init_schema(c)
    yield c
    c.close()


def _hayamihyo(conn, item_name: str, method: str = "家庭ごみ") -> int:
    return store.insert_waste_item(
        conn,
        WasteItem(
            item_name=item_name,
            item_name_raw=item_name,
            disposal_method=method,
            notes="",
            source_url="https://www.city.kitakyushu.lg.jp/contents/924_10041.html",
            source_page_title="ごみの分別早見表",
            crawled_at=dt.datetime(2026, 9, 1, tzinfo=timezone.utc),
        ),
    )


# --- exact + LIKE predecessor matching --------------------------------
def test_like_match_links_a_notice_to_its_compound_name_predecessor(conn):
    _hayamihyo(conn, "乾電池(充電式でないもの)", "電池類")
    _hayamihyo(conn, "スプレー缶", "燃えないごみ")  # exact-match control

    result = ingest_notice_file(CHANGES, conn, now=INGEST_AT)

    assert ("乾電池", "乾電池(充電式でないもの)") in result.partial_links
    assert ("スプレー缶", "乾電池(充電式でないもの)") not in result.partial_links  # exact, no flag
    assert any(n == "スプレー缶" for n, _, _ in result.linked)
    assert "乾電池" not in result.unlinked


def test_history_follows_the_supersede_link_across_a_name_change(conn):
    _hayamihyo(conn, "乾電池(充電式でないもの)", "電池類")
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)

    tl = store.item_timeline(conn, "乾電池")  # queried by the short name
    assert tl.found and len(tl.segments) == 2
    old, new = tl.segments
    assert old.synthesized is False
    assert old.matched_item_name == "乾電池(充電式でないもの)"
    assert old.valid_until == date(2026, 9, 30)
    assert new.from_notice is True and new.valid_from == date(2026, 10, 1)
    assert tl.changes_after(date(2026, 9, 8)) == (date(2026, 10, 1),)


def test_multiple_like_matches_are_ambiguous_not_linked(conn):
    _hayamihyo(conn, "乾電池(充電式でないもの)")
    _hayamihyo(conn, "乾電池(ボタン電池を除く)", "電池類")

    result = ingest_notice_file(CHANGES, conn, now=INGEST_AT)

    amb = dict(result.ambiguous)
    assert set(amb["乾電池"]) == {"乾電池(充電式でないもの)", "乾電池(ボタン電池を除く)"}
    assert not any(n == "乾電池" for n, _, _ in result.linked)
    # notice self-describes its prior state, so the timeline still has 2 segments
    tl = store.item_timeline(conn, "乾電池")
    assert len(tl.segments) == 2 and tl.segments[0].synthesized is True


def test_genuinely_different_names_stay_unlinked(conn):
    # LIKE '%蛍光管%' does not match '蛍光灯'
    _hayamihyo(conn, "蛍光灯")
    result = ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    assert "蛍光管" in result.unlinked
    assert not any(n == "蛍光管" for n, _, _ in result.partial_links)


def test_relink_also_does_like_matching(conn):
    r1 = ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    assert "乾電池" in r1.unlinked

    _hayamihyo(conn, "乾電池(充電式でないもの)", "電池類")
    conn.commit()

    r2 = relink(conn)
    assert ("乾電池", "乾電池(充電式でないもの)") in r2.partial_links
    assert "乾電池" not in r2.still_unlinked
    assert len(store.item_timeline(conn, "乾電池").segments) == 2


# --- word-order differences: vector candidates (opt-in, never auto-linked) ---
def test_reverse_order_names_are_unlinked_without_vector(conn):
    # お知らせ「使い捨てライター」, 早見表「ライター」: LIKE '%使い捨てライター%' misses
    _hayamihyo(conn, "ライター", "資源化物")
    result = ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    assert "使い捨てライター" in result.unlinked
    assert not any(n == "使い捨てライター" for n, _, _ in result.partial_links)


def test_relink_vector_reports_candidates_but_does_not_link(conn, monkeypatch):
    _hayamihyo(conn, "ライター", "資源化物")
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)  # 使い捨てライター unlinked

    def fake_vector(query, top_k=3):
        if "ライター" in query:
            return [SimpleNamespace(item_name="ライター", source_url="https://x/y.html")]
        return []

    monkeypatch.setattr(ingest_mod, "vector_search_items", fake_vector)

    result = relink(conn, use_vector=True)

    assert ("使い捨てライター", ("ライター",)) in result.vector_candidates
    assert "使い捨てライター" in result.still_unlinked          # NOT auto-linked
    assert not any(n == "使い捨てライター" for n, _, _ in result.linked)
    assert len(store.item_timeline(conn, "使い捨てライター").segments) == 2  # notice-only, synthesized


def test_vector_is_off_by_default(conn, monkeypatch):
    _hayamihyo(conn, "ライター", "資源化物")
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    called = []
    monkeypatch.setattr(ingest_mod, "vector_search_items", lambda *a, **k: called.append(1) or [])
    relink(conn)  # no use_vector
    assert called == []
