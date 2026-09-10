"""STAGE07 7-1: `app/mcp_server.py` が公開する各ツールが、正しく
`app/rag/tools.execute_tool` / `app/rag/router._blocking_route` に
委譲していることを検証する。

MCP の配線(stdio・スキーマ生成)自体はここでは検証しない(`FastMCP` の責務であり、
外部クライアント(Claude Desktop 等)との実接続でしか確認できない、
`app/server.py` のルーティングと同じ理由)。ここで見るのは「ツール関数を直接
呼んだときに正しい結果が返るか」で、`tests/agentic_tools_test.py` と同じ
孤立SQLiteフィクスチャ + reload の手法を使う。
"""

from __future__ import annotations

import datetime as dt
import importlib
from datetime import timezone
from pathlib import Path

import pytest

from app.data.schema import WasteItem

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGES = REPO_ROOT / "tests" / "fixtures" / "notices" / "お知らせ-4-2.md"
INGEST_AT = dt.datetime(2026, 9, 8, 3, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mcp_server(tmp_path, monkeypatch):
    """孤立したSQLiteに既知のフィクスチャを入れ、`app.mcp_server` がそれを見る
    よう依存モジュールを reload してから返す(`tests/agentic_tools_test.py` と同じ)。
    """
    from app.notices import store

    path = tmp_path / "kb.db"
    monkeypatch.setenv("DB_PATH", str(path))
    monkeypatch.setenv("RAG_DB_PATH", str(path))

    conn = store.connect()
    store.init_schema(conn)

    store.insert_waste_item(
        conn,
        WasteItem(
            item_name="テレビ",
            item_name_raw="テレビ(ブラウン管式を除く)",
            disposal_method="粗大ごみ",
            notes="事前申込制。粗大ごみ受付センターへ",
            source_url="https://www.city.kitakyushu.lg.jp/contents/924_10041.html",
            source_page_title="ごみの分別早見表(た行)",
            crawled_at=dt.datetime(2026, 9, 1, tzinfo=timezone.utc),
        ),
    )
    store.insert_waste_item(
        conn,
        WasteItem(
            item_name="スプレー缶",
            item_name_raw="スプレー缶",
            # お知らせ-4-2.md のスプレー缶notice_priorと一致させる(不一致だと
            # item_timeline が notice 側の申告を優先し disputed 扱いになるため)。
            disposal_method="燃えないごみ",
            notes="",
            source_url="https://www.city.kitakyushu.lg.jp/contents/924_10041.html",
            source_page_title="ごみの分別早見表(さ行)",
            crawled_at=dt.datetime(2026, 9, 1, tzinfo=timezone.utc),
        ),
    )
    now_iso = dt.datetime(2026, 9, 1, tzinfo=timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO collection_schedule "
        "(ward, town, waste_type, weekday, source_url, crawled_at, is_current) "
        "VALUES (?, ?, ?, ?, ?, ?, 1)",
        (
            "小倉北区",
            "浅野1丁目",
            "粗大ごみ",
            "第2水曜日",
            "https://www.city.kitakyushu.lg.jp/contents/924_10102.html",
            now_iso,
        ),
    )
    conn.commit()

    from app.notices.ingest import ingest_notice_file

    ingest_notice_file(CHANGES, conn, now=INGEST_AT)  # スプレー缶の変更予定
    conn.close()

    import app.data.lookup as data_lookup
    import app.notices.lookup as notices_lookup
    import app.rag.router as router_module
    import app.rag.tools as tools_module
    import app.mcp_server as mcp_server_module

    importlib.reload(data_lookup)
    importlib.reload(notices_lookup)
    importlib.reload(router_module)
    importlib.reload(tools_module)
    importlib.reload(mcp_server_module)
    monkeypatch.setattr(router_module, "vector_search_items", lambda *a, **k: [])
    return mcp_server_module


# --------------------------------------------------------------------------
# 原子ツール — tools.execute_tool への委譲を確認
# --------------------------------------------------------------------------
def test_search_waste_item_delegates_to_tools_module(mcp_server):
    result = mcp_server.search_waste_item("テレビ")
    assert result["match_stage"] == "exact"
    assert result["items"][0]["disposal_method"] == "粗大ごみ"


def test_search_waste_item_empty_query_is_a_tool_error(mcp_server):
    assert "error" in mcp_server.search_waste_item("")


def test_lookup_collection_schedule_delegates_to_tools_module(mcp_server):
    result = mcp_server.lookup_collection_schedule("小倉北区", "浅野1丁目", "粗大ごみ")
    assert result["schedules"] == [
        {
            "ward": "小倉北区",
            "town": "浅野1丁目",
            "waste_type": "粗大ごみ",
            "weekday": "第2水曜日",
            "source_url": "https://www.city.kitakyushu.lg.jp/contents/924_10102.html",
        }
    ]


def test_lookup_collection_schedule_missing_town_is_a_tool_error(mcp_server):
    assert "error" in mcp_server.lookup_collection_schedule("小倉北区", "")


def test_lookup_item_change_notice_reports_pending_change(mcp_server):
    result = mcp_server.lookup_item_change_notice("スプレー缶")
    assert result["has_pending_change"] is True
    assert result["current"]["disposal_method"] == "燃えないごみ"
    assert result["future"][0]["valid_from"] == "2026-10-01"


def test_lookup_item_change_notice_none_for_unaffected_item(mcp_server):
    assert mcp_server.lookup_item_change_notice("テレビ") == {"has_pending_change": False}


def test_search_general_info_missing_query_is_a_tool_error(mcp_server):
    assert "error" in mcp_server.search_general_info("")


# --------------------------------------------------------------------------
# 統合ツール — router._blocking_route への委譲を確認
# --------------------------------------------------------------------------
def test_get_disposal_guidance_returns_context_and_sources(mcp_server):
    result = mcp_server.get_disposal_guidance("テレビはどう捨てますか")

    assert result["found"] is True
    assert result["match_stage"] == "exact"
    assert "粗大ごみ" in result["context"]
    assert result["sources"] == [
        "https://www.city.kitakyushu.lg.jp/contents/924_10041.html"
    ]


def test_get_disposal_guidance_never_calls_the_llm(mcp_server):
    """このサーバに vLLM は無い — 返り値は決定論ルータの材料だけであること。"""
    result = mcp_server.get_disposal_guidance("テレビはどう捨てますか")
    assert set(result) == {
        "context", "sources", "detected", "match_stage",
        "item_match", "found", "missing", "markdown",
    }


def test_get_disposal_guidance_includes_a_markdown_table(mcp_server):
    """STAGE07 7-3: テキスト以外の出力として表(GFM markdown)を返す。"""
    result = mcp_server.get_disposal_guidance("テレビはどう捨てますか")
    assert "| 品名 | 出し方 | 備考 |" in result["markdown"]
    assert "テレビ" in result["markdown"]
    assert "粗大ごみ" in result["markdown"]


def test_get_disposal_guidance_reports_missing_town_for_schedule_only_ward(mcp_server):
    result = mcp_server.get_disposal_guidance("小倉北区の収集日はいつですか")
    assert "町名" in result["missing"]


def test_get_disposal_guidance_total_miss_has_no_markdown_table(mcp_server):
    result = mcp_server.get_disposal_guidance("これはごみ分別と無関係な架空の相談です")
    assert result["found"] is False
    assert result["markdown"] == ""
