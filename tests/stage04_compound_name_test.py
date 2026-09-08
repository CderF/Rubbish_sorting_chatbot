"""早見表が品目を複合名で持つケース(FANG の PR #6 検証結果)。

例: お知らせ「乾電池」/「使い捨てライター」/「テレビ」に対し、早見表は
`乾電池(充電式でないもの)` / `ライター` / `テレビ(ブラウン管式・液晶式・プラズマ式)`。
完全一致だけだと前身を取りこぼすので、基底名で突き合わせて部分一致リンクする。
チェーンは `item_name` ではなく `superseded_by` で辿る。
"""

from __future__ import annotations

import datetime as dt
from datetime import date, timezone
from pathlib import Path

import pytest

from app.data.schema import WasteItem
from app.stage04 import store
from app.stage04.ingest import ingest_notice_file, relink
from app.stage04.store import _base_name, _name_relates

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGES = REPO_ROOT / "knowledge" / "incoming" / "お知らせ-4-2.md"
INGEST_AT = dt.datetime(2026, 9, 8, 3, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    c = store.connect(":memory:")
    store.init_schema(c)
    yield c
    c.close()


def _hayamihyo(conn, item_name: str, method: str) -> int:
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


# --- name helpers -------------------------------------------------------
@pytest.mark.parametrize(
    "raw,base",
    [
        ("乾電池(充電式でないもの)", "乾電池"),
        ("テレビ（ブラウン管式・液晶式・プラズマ式）", "テレビ"),
        ("ライター", "ライター"),
        ("使い捨てカイロ等", "使い捨てカイロ"),
        ("充電式電池（ニカド・ニッケル水素・リチウムイオン）類", "充電式電池"),
    ],
)
def test_base_name_strips_qualifiers(raw, base):
    assert _base_name(raw) == base


def test_name_relates_matches_compound_and_prefixed_names():
    assert _name_relates("乾電池", "乾電池(充電式でないもの)")
    assert _name_relates("テレビ", "テレビ(ブラウン管式・液晶式・プラズマ式)")
    assert _name_relates("使い捨てライター", "ライター")
    # genuinely different items must NOT auto-relate
    assert not _name_relates("蛍光管", "蛍光灯")
    assert not _name_relates("傘", "日傘")  # len(base) guard on the 1-char side


# --- ingest / relink linking -----------------------------------------
def test_ingest_links_a_notice_to_its_compound_name_predecessor(conn):
    _hayamihyo(conn, "乾電池(充電式でないもの)", "家庭ごみ")
    _hayamihyo(conn, "ライター", "資源化物")

    result = ingest_notice_file(CHANGES, conn, now=INGEST_AT)

    assert ("乾電池", "乾電池(充電式でないもの)") in result.partial_links
    assert ("使い捨てライター", "ライター") in result.partial_links
    assert "乾電池" not in result.unlinked
    assert "使い捨てライター" not in result.unlinked
    # exact-match items are still linked without the partial flag
    _hayamihyo(conn, "スプレー缶", "家庭ごみ")  # (no-op: already ingested run)


def test_history_follows_the_supersede_link_across_a_name_change(conn):
    _hayamihyo(conn, "乾電池(充電式でないもの)", "電池類")
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)

    tl = store.item_timeline(conn, "乾電池")  # queried by the short name
    assert tl.found and len(tl.segments) == 2

    old, new = tl.segments
    assert old.synthesized is False                      # a real row, not reconstructed
    assert old.matched_item_name == "乾電池(充電式でないもの)"
    assert old.valid_until == date(2026, 9, 30)
    assert new.from_notice is True and new.valid_from == date(2026, 10, 1)
    assert tl.changes_after(date(2026, 9, 8)) == (date(2026, 10, 1),)


def test_ambiguous_base_name_is_reported_not_linked(conn):
    # two 早見表 rows both reduce to base "乾電池"
    _hayamihyo(conn, "乾電池(充電式でないもの)", "家庭ごみ")
    _hayamihyo(conn, "乾電池(ボタン電池を除く)", "電池類")

    result = ingest_notice_file(CHANGES, conn, now=INGEST_AT)

    amb = dict(result.ambiguous)
    assert "乾電池" in amb
    assert set(amb["乾電池"]) == {"乾電池(充電式でないもの)", "乾電池(ボタン電池を除く)"}
    assert not any(n == "乾電池" for n, _, _ in result.linked)
    # not linked to a specific row; `relink` retries it later (a more precise
    # crawl, or a human, decides which variant applies). item_timeline still
    # reconstructs the prior rule from the notice itself (synthesized).
    tl = store.item_timeline(conn, "乾電池")
    assert len(tl.segments) == 2 and tl.segments[0].synthesized is True
    assert tl.segments[0].matched_item_name is None


def test_relink_also_does_compound_name_matching(conn):
    # notices ingested before the crawler -> all unlinked
    r1 = ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    assert "乾電池" in r1.unlinked

    # crawler lands, with the compound name
    _hayamihyo(conn, "乾電池(充電式でないもの)", "電池類")
    conn.commit()

    r2 = relink(conn)
    assert ("乾電池", "乾電池(充電式でないもの)") in r2.partial_links
    assert "乾電池" not in r2.still_unlinked
    assert len(store.item_timeline(conn, "乾電池").segments) == 2
