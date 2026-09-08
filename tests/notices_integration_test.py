"""STAGE 04 integration with the merged application layer.

Covers the two seams STAGE 04 plugs into after issues #6-#10 landed:

* `app/notices/lookup.py` — the connection-free, is_current-agnostic read API the
  router calls (mirrors `app/data/lookup.py`'s contract).
* `app/rag/knowledge.py` — the WebUI's source list (STAGE02 2-3): ingested お知らせ
  must appear as one entry per file, and must not leak into the crawled-page
  grouping as per-anchor junk.

All offline; a fresh SQLite file per test via DB_PATH / RAG_DB_PATH.
"""

from __future__ import annotations

import datetime as dt
import importlib
from datetime import date, timezone
from pathlib import Path

import pytest

from app.data.schema import WasteItem
from app.notices import store
from app.notices.ingest import ingest_notice_file

REPO_ROOT = Path(__file__).resolve().parent.parent
ADDITIONS = REPO_ROOT / "knowledge" / "incoming" / "お知らせ-4-1.md"
CHANGES = REPO_ROOT / "knowledge" / "incoming" / "お知らせ-4-2.md"
INGEST_AT = dt.datetime(2026, 9, 8, 3, 0, 0, tzinfo=timezone.utc)
TODAY = date(2026, 9, 8)


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "kb.db"
    monkeypatch.setenv("DB_PATH", str(path))
    monkeypatch.setenv("RAG_DB_PATH", str(path))
    conn = store.connect()
    store.init_schema(conn)
    # a real crawled row + one that a change notice will supersede
    for name, method in [("テレビ", "粗大ごみ"), ("スプレー缶", "家庭ごみ")]:
        store.insert_waste_item(
            conn,
            WasteItem(
                item_name=name,
                item_name_raw=name,
                disposal_method=method,
                notes="",
                source_url="https://www.city.kitakyushu.lg.jp/contents/924_10041.html",
                source_page_title="ごみの分別早見表（た行）",
                crawled_at=dt.datetime(2026, 9, 1, tzinfo=timezone.utc),
            ),
        )
    conn.commit()
    ingest_notice_file(ADDITIONS, conn, now=INGEST_AT)
    ingest_notice_file(CHANGES, conn, now=INGEST_AT)
    conn.close()
    return path


# --- app/notices/lookup.py -------------------------------------------------
def test_conn_free_lookup_returns_the_full_timeline(db):
    from app.notices import lookup

    importlib.reload(lookup)  # drop any thread-local conn from a prior test
    tl = lookup.item_timeline("スプレー缶")
    assert tl.found and len(tl.segments) == 2
    assert tl.segment_on(TODAY).disposal_method == "燃えないごみ"  # notice's account


def test_pending_change_gates_on_an_upcoming_switch(db):
    from app.notices import lookup

    importlib.reload(lookup)
    assert lookup.pending_change("スプレー缶", TODAY) is not None      # 4-2 applies
    assert lookup.pending_change("ホバーボード", TODAY) is None        # addition, no switch
    assert lookup.pending_change("宇宙服", TODAY) is None              # unknown item
    # after the switch-over there is no longer a *pending* change
    assert lookup.pending_change("スプレー缶", date(2026, 11, 1)) is None


def test_conn_free_lookup_is_read_only(db):
    from app.notices import lookup

    importlib.reload(lookup)
    lookup.item_timeline("テレビ")
    with pytest.raises(Exception):
        lookup._conn().execute("INSERT INTO waste_items (item_name) VALUES ('x')")


# --- app/rag/knowledge.py -----------------------------------------------
def test_ingested_notices_listed_as_one_entry_per_file(db):
    from app.rag import knowledge

    result = knowledge.list_sources()
    assert result["available"] is True
    notices = [s for s in result["sources"] if s["kind"] == "knowledge_notice"]
    assert {s["title"] for s in notices} == {"お知らせ-4-1.md", "お知らせ-4-2.md"}
    assert all(s["record_count"] == 15 for s in notices)


def test_notice_rows_do_not_leak_into_the_crawled_page_grouping(db):
    from app.rag import knowledge

    result = knowledge.list_sources()
    for s in result["sources"]:
        if s["kind"] != "knowledge_notice":
            assert not s["url"].startswith("knowledge://")
    # テレビ still shows; スプレー缶's crawled row was superseded by the notice
    hayamihyo = [s for s in result["sources"] if s["kind"] == "waste_items"]
    assert len(hayamihyo) == 1 and hayamihyo[0]["record_count"] == 1


def test_notice_source_detail_shows_each_notice(db):
    from app.rag import knowledge

    result = knowledge.list_sources()
    src_id = next(
        s["id"] for s in result["sources"]
        if s.get("title") == "お知らせ-4-2.md"
    )
    detail = knowledge.get_source(src_id)
    assert detail is not None
    assert detail["headers"] == ["品名", "お知らせ", "適用開始", "反映後の内容"]
    assert len(detail["rows"]) == 15
    names = {row[0] for row in detail["rows"]}
    assert "スプレー缶" in names and "乾電池" in names
    spray = next(row for row in detail["rows"] if row[0] == "スプレー缶")
    assert spray[2] == "2026-10-01"  # 適用開始


# --- app/rag/router.py : 4-2 が /chat に反映される ------------------------
def test_router_surfaces_both_rules_for_a_pending_change(db):
    from app.rag import router
    from app.rag.router import _blocking_route, build_system_prompt

    importlib.reload(router)
    from app.rag.router import _blocking_route, build_system_prompt

    ans = _blocking_route("スプレー缶はどう捨てますか")

    assert ans.timeline_change is True
    assert "現在(〜2026-09-30): 燃えないごみ" in ans.context     # 旧ルール(お知らせの記述)
    assert "2026-10-01 から:" in ans.context                    # 切替日 + 新ルール
    assert "かん・びん" in ans.context                           # 新ルール本文
    assert "家庭ごみ" in ans.context                             # disputed: 早見表側も注記
    assert "knowledge://お知らせ-4-2.md#お知らせ-01-スプレー缶" in ans.sources
    assert "どちらか一方だけを答えて断定しない" in build_system_prompt(ans)


def test_router_leaves_unchanged_items_alone(db):
    from app.rag import router

    importlib.reload(router)
    from app.rag.router import _blocking_route

    for prompt in ("ホバーボードはどう捨てますか", "テレビの捨て方"):
        ans = _blocking_route(prompt)
        assert ans.timeline_change is False
        assert "出し方が変わります" not in ans.context


def test_router_imports_without_notices_lookup(monkeypatch):
    """PR #5 未マージ環境でも router.py が壊れないこと(guarded import)。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "app.notices.lookup":
            raise ImportError("simulated: PR #5 not merged")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import importlib

    import app.rag.router as router

    importlib.reload(router)
    assert router.pending_change("スプレー缶", date(2026, 9, 8)) is None
    importlib.reload(router)  # restore the real import for other tests
