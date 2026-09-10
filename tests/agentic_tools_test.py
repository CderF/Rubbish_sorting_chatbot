"""app/rag/tools.py の各ツールが既存のデータ層・ベクトルRAGパスの関数に
正しくルーティングし、JSON化可能な形で結果を返すことを検証する。

ツール自体は検索ロジックを再実装しない(単なるラッパー)ので、ここでの
関心は「ディスパッチが正しいか」「戻り値の形がFunction Calling結果として
使える形か」「LLM由来の不正な引数で例外を投げないか」。検索ロジック自体の
正しさ(完全一致→部分一致→ベクトル、収集日パスの絞り込み等)は
`tests/notices_integration_test.py` / `tests/collection_schedule_town_test.py`
/ `tests/vector_store_test.py` 側で別途担保されている。

早見表パス・収集日パス・お知らせパスは孤立したSQLiteフィクスチャで検証する
(`tests/notices_integration_test.py` と同じ DB_PATH 差し替え + reload の
手法)。search_general_info(ベクトルRAGパス)と search_waste_item のベクトル
フォールバック段は、このリポジトリに構築済みのローカル var/chroma インデッ
クスに対して検証する(GPU不要・ネットワーク不要、ただし事前に
`python -m app.rag.vector_store` でビルド済みであることが前提)。
"""

from __future__ import annotations

import datetime as dt
import importlib
from datetime import timezone
from pathlib import Path

import pytest

from app.data.schema import WasteItem

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGES = REPO_ROOT / "knowledge" / "incoming" / "お知らせ-4-2.md"
INGEST_AT = dt.datetime(2026, 9, 8, 3, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def tools(tmp_path, monkeypatch):
    """孤立したSQLiteに既知のフィクスチャを入れ、app.rag.tools がそれを見る
    よう依存モジュールを reload してから返す。

    `DB_PATH` を差し替えても、既にimport済みのモジュールが束縛済みの関数
    参照までは更新されないため、依存元(`app.rag.tools`)まで含めて reload
    する(`tests/notices_integration_test.py` と同じ手法)。
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
            item_name_raw="テレビ（ブラウン管式を除く）",
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
            disposal_method="家庭ごみ",
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
            "家庭ごみ",
            "火曜日・金曜日",
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
    import app.rag.tools as tools_module

    importlib.reload(data_lookup)
    importlib.reload(notices_lookup)
    importlib.reload(tools_module)
    return tools_module


# --- search_waste_item -----------------------------------------------------
def test_search_waste_item_exact_match(tools):
    result = tools.execute_tool("search_waste_item", {"query": "テレビ"})
    assert result["match_stage"] == "exact"
    assert result["items"][0]["item_name"] == "テレビ"
    assert result["items"][0]["disposal_method"] == "粗大ごみ"
    assert "事前申込制" in result["items"][0]["notes"]


def test_search_waste_item_no_hit_falls_back_without_raising(tools):
    # 早見表に無い品目でも例外を投げず、vector段の結果(0件含む)を返す
    result = tools.execute_tool("search_waste_item", {"query": "宇宙服"})
    assert result["match_stage"] in ("vector", "none")
    assert isinstance(result["items"], list)


def test_search_waste_item_missing_query_is_a_tool_error(tools):
    result = tools.execute_tool("search_waste_item", {"query": ""})
    assert "error" in result


# --- lookup_collection_schedule --------------------------------------------
def test_lookup_collection_schedule_returns_matching_rows(tools):
    result = tools.execute_tool(
        "lookup_collection_schedule",
        {"ward": "小倉北区", "town": "浅野1丁目", "waste_type": "家庭ごみ"},
    )
    assert result["schedules"] == [
        {
            "ward": "小倉北区",
            "town": "浅野1丁目",
            "waste_type": "家庭ごみ",
            "weekday": "火曜日・金曜日",
            "source_url": "https://www.city.kitakyushu.lg.jp/contents/924_10102.html",
        }
    ]


def test_lookup_collection_schedule_missing_town_is_a_tool_error(tools):
    result = tools.execute_tool("lookup_collection_schedule", {"ward": "小倉北区", "town": ""})
    assert "error" in result


def test_lookup_collection_schedule_unknown_town_returns_empty_not_error(tools):
    result = tools.execute_tool(
        "lookup_collection_schedule", {"ward": "小倉北区", "town": "存在しない町"}
    )
    assert result["schedules"] == []


# --- lookup_item_change_notice ----------------------------------------------
def test_lookup_item_change_notice_reports_pending_change(tools):
    result = tools.execute_tool("lookup_item_change_notice", {"item_name": "スプレー缶"})
    assert result["has_pending_change"] is True
    assert result["current"]["disposal_method"]
    assert result["future"]


def test_lookup_item_change_notice_none_for_unaffected_item(tools):
    result = tools.execute_tool("lookup_item_change_notice", {"item_name": "テレビ"})
    assert result == {"has_pending_change": False}


# --- 未知のツール名/例外の握り潰し -----------------------------------------
def test_execute_tool_unknown_name(tools):
    result = tools.execute_tool("not_a_real_tool", {})
    assert "error" in result


def test_execute_tool_bad_argument_type_does_not_raise(tools):
    # LLMがJSONを生成する以上、想定外の形は必ず来る。例外にしない。
    result = tools.execute_tool("lookup_collection_schedule", {"ward": None, "town": None})
    assert "error" in result


# --- search_general_info(ローカルの実 var/chroma インデックスを使用) -------
def test_search_general_info_returns_shaped_results():
    import app.rag.tools as tools_module

    result = tools_module.execute_tool("search_general_info", {"query": "ごみステーション"})
    assert isinstance(result["results"], list)
    for r in result["results"]:
        assert set(r) == {"text", "source_url", "source_page_title", "score"}


def test_search_general_info_missing_query_is_a_tool_error():
    import app.rag.tools as tools_module

    result = tools_module.execute_tool("search_general_info", {"query": ""})
    assert "error" in result
