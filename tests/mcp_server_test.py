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
    # 7-4 の記憶ストアも隔離する。既定(var/user_memory.db)のままだとテストが
    # 本物の記憶ファイルを読み書きしてしまう。
    monkeypatch.setenv("USER_MEMORY_DB_PATH", str(tmp_path / "mem.db"))

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
        "context", "sources", "detected", "match_stage", "item_match",
        "found", "missing", "canned_response", "markdown", "used_memory",
        "remembered",
    }
    assert result["used_memory"] is False
    assert result["remembered"] is None  # user_id を渡していない


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


def test_get_disposal_guidance_forwards_the_routers_canned_response(mcp_server, monkeypatch):
    """レビュー指摘: 確度ゲート(4-3)・類別質問・安全拒否でルータが
    `canned_response` を確定させても返り値に載せていなかったため、MCP
    クライアントには `found=False` / `context=""` としか見えず、本当に見せる
    べき固定文が消えていた。ルータが立てた文をそのまま透過する。

    (`category_procedure_intent` の固定文自体は STAGE05 ブランチの
    `fix/stage05-multiturn-state` にあり、この STAGE07 ブランチには未マージ
    なので、ここではルータ出力を差し替えて「透過するか」だけを見る。)
    """
    from app.rag.types import RoutedAnswer

    fixed = "早見表に個別の記載がなく、分別を断定できません。窓口でご確認ください。"
    monkeypatch.setattr(
        mcp_server.router,
        "_blocking_route",
        lambda *a, **k: RoutedAnswer(found=False, canned_response=fixed),
    )

    result = mcp_server.get_disposal_guidance("早見表にも類似品にも根拠がない品目")

    assert result["canned_response"] == fixed
    assert result["found"] is False
    assert result["markdown"] == ""


# --------------------------------------------------------------------------
# 7-4 パーソナライズ — remember/recall/forget ツール
# --------------------------------------------------------------------------
def test_remember_and_recall_facts_round_trip(mcp_server):
    assert mcp_server.remember_fact("mentor-1", "ward", "小倉北区") == {"ok": True}
    assert mcp_server.recall_facts("mentor-1") == {"facts": {"ward": "小倉北区"}}


def test_recall_facts_for_unknown_user_is_empty(mcp_server):
    assert mcp_server.recall_facts("ghost") == {"facts": {}}


def test_forget_facts_one_key(mcp_server):
    mcp_server.remember_fact("mentor-1", "ward", "小倉北区")
    mcp_server.remember_fact("mentor-1", "town", "浅野1丁目")

    assert mcp_server.forget_facts("mentor-1", "ward") == {"removed": 1}
    assert mcp_server.recall_facts("mentor-1") == {"facts": {"town": "浅野1丁目"}}


def test_forget_facts_all(mcp_server):
    mcp_server.remember_fact("mentor-1", "ward", "小倉北区")
    mcp_server.remember_fact("mentor-1", "town", "浅野1丁目")

    assert mcp_server.forget_facts("mentor-1") == {"removed": 2}
    assert mcp_server.recall_facts("mentor-1") == {"facts": {}}


def test_forget_facts_rejects_empty_key_without_deleting_all(mcp_server):
    mcp_server.remember_fact("mentor-1", "ward", "小倉北区")
    mcp_server.remember_fact("mentor-1", "town", "浅野1丁目")

    assert mcp_server.forget_facts("mentor-1", "") == {
        "error": "key は空にできません"
    }
    assert mcp_server.recall_facts("mentor-1") == {
        "facts": {"town": "浅野1丁目", "ward": "小倉北区"}
    }


def test_remember_fact_rejects_empty_key(mcp_server):
    assert "error" in mcp_server.remember_fact("mentor-1", "", "x")


# --------------------------------------------------------------------------
# 7-4 パーソナライズ — get_disposal_guidance への配線
# --------------------------------------------------------------------------
def test_get_disposal_guidance_saves_the_region_for_next_time(mcp_server):
    result = mcp_server.get_disposal_guidance(
        "小倉北区浅野1丁目の粗大ごみの収集日は?", user_id="mentor-1"
    )
    assert result["used_memory"] is False  # 今回の発話だけで足りている
    assert mcp_server.recall_facts("mentor-1") == {
        "facts": {"ward": "小倉北区", "town": "浅野1丁目"}
    }


def test_get_disposal_guidance_fills_in_the_remembered_region(mcp_server):
    mcp_server.remember_fact("mentor-1", "ward", "小倉北区")
    mcp_server.remember_fact("mentor-1", "town", "浅野1丁目")

    result = mcp_server.get_disposal_guidance("テレビの収集日は?", user_id="mentor-1")

    assert result["used_memory"] is True
    assert result["found"] is True
    assert "町名" not in result["missing"]


def test_get_disposal_guidance_without_user_id_does_not_touch_memory(mcp_server):
    mcp_server.get_disposal_guidance("小倉北区浅野1丁目の粗大ごみの収集日は?")
    assert mcp_server.recall_facts("mentor-1") == {"facts": {}}


def test_get_disposal_guidance_without_a_saved_region_leaves_missing_alone(mcp_server):
    result = mcp_server.get_disposal_guidance("収集日はいつですか", user_id="mentor-new")
    assert result["used_memory"] is False
    assert "区名と町名" in result["missing"] or "町名" in result["missing"]


def test_memory_does_not_override_a_ward_written_in_the_question(mcp_server):
    """レビュー #47-1: 記憶に別の区・町名があっても、質問文に区名が書かれて
    いるターンでは記憶を注入しない。注入すると `extract_region` が WARDS の
    並び順で1つ選ぶため記憶側が勝ち、ユーザーの区が黙って上書きされる。"""
    mcp_server.remember_fact("mentor-1", "ward", "小倉北区")
    mcp_server.remember_fact("mentor-1", "town", "浅野1丁目")

    result = mcp_server.get_disposal_guidance(
        "八幡西区の粗大ごみの収集日は?", user_id="mentor-1"
    )

    assert result["used_memory"] is False
    # フィクスチャの収集日は小倉北区浅野1丁目のものだけ。記憶側の区に化けて
    # その収集日(第2水曜日)を返していない。
    assert "第2水曜日" not in result["markdown"]
    assert "町名" in result["missing"]


def test_memory_augmented_retry_that_still_finds_nothing_is_not_adopted(mcp_server):
    """レビュー #47-2: 記憶で補って引き直しても空振りなら、その再検索結果は
    採用しない。採用すると、ユーザーが一度も言っていない住所について
    「データがありません」と答えてしまう。元の「町名を教えて」を保つ。"""
    mcp_server.remember_fact("m2", "ward", "門司区")
    mcp_server.remember_fact("m2", "town", "老松町")  # フィクスチャに収集日なし

    result = mcp_server.get_disposal_guidance("粗大ごみの収集日は?", user_id="m2")

    assert result["used_memory"] is False
    assert result["found"] is False
    assert any(("町名" in m) for m in result["missing"])


def test_memory_is_not_corrupted_on_a_ward_only_turn(mcp_server):
    """レビュー #47-3: 区だけ言った(町名なし)ターンで記憶の区を上書きすると、
    記憶が「同時に存在したことのない区/町名の組」に壊れる。完全な住所を
    言ったターンだけ書き戻す。"""
    mcp_server.remember_fact("m3", "ward", "小倉北区")
    mcp_server.remember_fact("m3", "town", "浅野1丁目")

    mcp_server.get_disposal_guidance("八幡西区の粗大ごみの収集日は?", user_id="m3")

    assert mcp_server.recall_facts("m3") == {
        "facts": {"ward": "小倉北区", "town": "浅野1丁目"}
    }


# --------------------------------------------------------------------------
# 7-1 リソース — ナレッジ一覧
# --------------------------------------------------------------------------
def test_knowledge_sources_resource_lists_ingested_data(mcp_server):
    result = mcp_server.knowledge_sources()
    assert result["available"] is True
    kinds = {s["kind"] for s in result["sources"]}
    assert "waste_items" in kinds


def test_knowledge_source_resource_returns_the_detail(mcp_server):
    listing = mcp_server.knowledge_sources()
    source_id = listing["sources"][0]["id"]

    detail = mcp_server.knowledge_source(source_id)

    assert detail["id"] == source_id
    assert "headers" in detail and "rows" in detail


def test_knowledge_source_resource_unknown_id_is_an_error(mcp_server):
    assert "error" in mcp_server.knowledge_source("does-not-exist")


# --------------------------------------------------------------------------
# 7-3 画像出力 — 収集日カレンダーSVG
# --------------------------------------------------------------------------
def test_marked_weekdays_extracts_and_orders_kanji_from_weekday_strings(mcp_server):
    schedules = [{"weekday": "木曜日・月曜日"}, {"weekday": "第2水曜日"}]
    assert mcp_server._marked_weekdays(schedules) == ["月", "水", "木"]


def test_marked_weekdays_empty_when_no_schedules(mcp_server):
    assert mcp_server._marked_weekdays([]) == []


def test_render_week_svg_is_well_formed_and_highlights_marked_days(mcp_server):
    svg = mcp_server._render_week_svg(["月", "木"], "小倉北区浅野1丁目の収集日")

    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert svg.count("<rect") == 7  # 月〜日の7マス
    assert "小倉北区浅野1丁目の収集日" in svg
    assert "#1a56b8" in svg  # 強調色が使われている(該当曜日あり)


def test_render_week_svg_escapes_title_xml(mcp_server):
    svg = mcp_server._render_week_svg([], "A < B & C")

    assert "A &lt; B &amp; C" in svg
    assert "A < B & C" not in svg


def test_markdown_cell_escapes_table_delimiters(mcp_server):
    assert mcp_server._markdown_cell("a|b\nc") == r"a\|b<br>c"


def test_schedule_calendar_svg_resource_reflects_the_real_schedule(mcp_server):
    svg = mcp_server.schedule_calendar_svg("小倉北区", "浅野1丁目")
    assert svg.startswith("<svg")
    assert "浅野1丁目" in svg


def test_schedule_calendar_svg_resource_degrades_gracefully_when_no_hits(mcp_server):
    svg = mcp_server.schedule_calendar_svg("門司区", "存在しない町")
    assert svg.startswith("<svg")
    assert "#1a56b8" not in svg  # 強調無しの週表


def test_schedule_calendar_svg_resource_shows_lookup_errors_distinctly(mcp_server):
    """レビュー #47-5: lookup がエラー(空 town・DB 破損など)を返したときは、
    「この地区に収集日がゼロ」の素の週表と混同しないよう、エラー文をタイトルに
    出す。"""
    svg = mcp_server.schedule_calendar_svg("小倉北区", "")

    assert svg.startswith("<svg")
    assert "#1a56b8" not in svg
    assert "ward and town are both required" in svg  # エラーが見える形で出ている
    assert "小倉北区 の収集日" not in svg  # 通常タイトルではない
