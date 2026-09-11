"""`app/mcp_bridge.py`(STAGE07 7-2: WebUI から MCP サーバへの実プロトコル接続)。

WebUI が Claude Desktop 等と同じ経路(公式 `mcp` クライアントSDK、stdio)で
`app/mcp_server.py` を子プロセスとして実際に起動し、ツール呼び出し・リソース
読み取りができることを検証する。ここでは `app.mcp_server` の実装(検索ロジック
そのもの)は再検証しない(`tests/mcp_server_test.py` の役割) — 見るのは
「ブリッジがホワイトリスト外の呼び出しを拒否するか」「実プロトコル経由で
往復できるか」だけ。

このリポジトリの既存の非同期テストと同じ `asyncio.run()` 方式(テストごとに
イベントループが変わる)を踏襲するため、`MCPBridge()` はテストごとに新しく
作り、必ず `close()` する。子プロセス起動(chroma/sentence-transformers の
ロードを含む)を挟むため実行に数秒かかる — `tests/spec_test.py` と同じ、
手元で確認する部類の遅いテストなので、1本の bridge に複数の確認をまとめて
プロセス起動回数を抑えている。
"""

from __future__ import annotations

import asyncio
import datetime as dt
from datetime import timezone

import pytest

from app.mcp_bridge import ALLOWED_RESOURCE_SCHEMES, ALLOWED_TOOLS, BridgeError, MCPBridge


def _run(coro):
    return asyncio.run(coro)


def test_bridge_round_trips_tools_and_resources_over_the_real_stdio_protocol():
    async def body():
        bridge = MCPBridge()
        try:
            tools = await bridge.list_tools()
            names = {t["name"] for t in tools}
            assert names == set(ALLOWED_TOOLS)
            for t in tools:
                assert t["inputSchema"]  # WebUI がフォームを組み立てるのに使う

            listing = await bridge.list_resources()
            assert {r["uri"] for r in listing["resources"]} == {"knowledge://sources"}
            templates = {t["uriTemplate"] for t in listing["resource_templates"]}
            assert templates == {
                "knowledge://{source_id}",
                "schedule-calendar://{ward}/{town}",
            }
            for t in listing["resource_templates"]:
                assert t["uriTemplate"].startswith(ALLOWED_RESOURCE_SCHEMES)

            result = await bridge.call_tool("search_waste_item", {"query": "テレビ"})
            assert "match_stage" in result
            assert "items" in result

            # 空queryはツール自体が {"error": ...} を返す仕様(isError は立たない)。
            # ブリッジはそのまま透過する。
            errored = await bridge.call_tool("search_waste_item", {"query": ""})
            assert "error" in errored

            svg = await bridge.read_resource("schedule-calendar://小倉北区/浅野1丁目")
            assert svg["mimeType"] == "image/svg+xml"
            assert svg["text"].startswith("<svg")
        finally:
            await bridge.close()

    _run(body())


def test_bridge_passes_the_parent_processes_db_path_through_to_the_subprocess(
    tmp_path, monkeypatch
):
    """`mcp.client.stdio.stdio_client` は既定で `get_default_environment()`
    (PATH/HOME 等の最小限)しか子プロセスへ渡さない。`_launch_params()` が
    `env=` を指定し忘れると、WebUIサーバ側で `DB_PATH`/`RAG_DB_PATH` を
    カスタムに設定していても、ブリッジの子プロセスは既定のDBを見てしまう
    (実測で判明、レビュー対応)。隔離DBを実際に作って、子プロセスがそこを
    見ていることを確認する。"""
    from app.data.schema import WasteItem
    from app.notices import store

    db_path = tmp_path / "isolated.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("RAG_DB_PATH", str(db_path))

    conn = store.connect(db_path)
    store.init_schema(conn)
    store.insert_waste_item(
        conn,
        WasteItem(
            item_name="ブリッジ疎通確認用の架空品目",
            item_name_raw="ブリッジ疎通確認用の架空品目",
            disposal_method="燃えないごみ",
            notes="",
            source_url="https://example.invalid/",
            source_page_title="test",
            crawled_at=dt.datetime(2026, 9, 1, tzinfo=timezone.utc),
        ),
    )
    conn.commit()
    conn.close()

    async def body():
        bridge = MCPBridge()
        try:
            return await bridge.call_tool(
                "search_waste_item", {"query": "ブリッジ疎通確認用の架空品目"}
            )
        finally:
            await bridge.close()

    result = _run(body())
    assert result["match_stage"] == "exact"
    assert result["items"][0]["item_name"] == "ブリッジ疎通確認用の架空品目"


def test_bridge_rejects_calls_and_resources_outside_the_allowlist():
    async def body():
        bridge = MCPBridge()
        try:
            with pytest.raises(BridgeError):
                await bridge.call_tool("not_a_real_tool", {})
            with pytest.raises(BridgeError):
                await bridge.read_resource("file:///etc/passwd")
            # `startswith` を細工(`knowledge-evil://`)で抜けないことも確認する。
            with pytest.raises(BridgeError):
                await bridge.read_resource("knowledge-evil://sources")
        finally:
            await bridge.close()

    _run(body())
