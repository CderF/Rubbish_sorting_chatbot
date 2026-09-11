"""STAGE07 7-2/7-3/7-4: WebUI から STAGE07 MCP サーバを呼ぶための薄いブリッジ。

課題PDF 7-2「外部クライアントから接続して呼び出せる状態にする」を、Claude
Desktop のような外部AIクライアントだけでなく **自分たちの chatbot WebUI**
からも証明するために追加した(2026-09-11、メンター指摘対応)。

`app/server.py` の `require_login` 済みルート(`/stage7/mcp/*`)だけがこの
モジュールを呼ぶ。設計上の制約:

  - 起動するコマンドは ``sys.executable -m app.mcp_server`` に固定し、引数・
    環境変数のどちらにもリクエスト由来の値を一切混ぜない(任意プロセス実行の
    経路にしない)。
  - 呼べるツール名は `ALLOWED_TOOLS`、読めるリソースの scheme は
    `ALLOWED_RESOURCE_SCHEMES` でホワイトリストする。ここに無い名前・scheme
    は `BridgeError` で拒否する(`app/server.py` 側で 400 に変換)。
  - 子プロセスは遅延起動・使い回し(1リクエストごとに起動すると chroma /
    sentence-transformers のロードで数秒かかる)。`asyncio.Lock` で直列化し、
    stdio 1本を複数リクエストが同時に叩かない単純な設計にしている。

STAGE07 の実装(`app/mcp_server.py`)そのものは変更しない — ここは公式
`mcp` クライアントSDK(`mcp.client.stdio`)で接続する「もう1つの外部
クライアント」を WebUI の裏に置くだけ。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parent.parent

# app/mcp_server.py が @mcp.tool() で公開している名前と完全一致させる。
ALLOWED_TOOLS = frozenset(
    {
        "search_waste_item",
        "lookup_collection_schedule",
        "lookup_item_change_notice",
        "search_general_info",
        "get_disposal_guidance",
        "remember_fact",
        "recall_facts",
        "forget_facts",
    }
)

# 7-4: この4つは呼び出し元が渡した user_id をそのまま信用せず、認証済み
# ブラウザの stage7_uid Cookie で必ず上書きする(app/server.py 側の責務)。
MEMORY_SCOPED_TOOLS = frozenset(
    {"get_disposal_guidance", "remember_fact", "recall_facts", "forget_facts"}
)

ALLOWED_RESOURCE_SCHEMES = ("knowledge://", "schedule-calendar://")
STATIC_RESOURCE_URIS = frozenset({"knowledge://sources"})


class BridgeError(Exception):
    """許可されていない呼び出し、またはMCPサーバ側のエラー応答。"""


def _child_environment() -> dict[str, str]:
    """Build the MCP child environment without losing deployment-specific paths.

    The chatbot venv on B1 contains MCP 2.x for the WebUI client, while the
    existing FastMCP server is intentionally kept on MCP 1.x in the isolated
    ``stage07-mcp/.deps`` directory.  Prefer that directory when it exists;
    local development without it keeps the normal parent environment.
    """
    env = dict(os.environ)
    candidates = []
    configured = env.get("STAGE7_MCP_DEPS")
    if configured:
        candidates.append(Path(configured))
    candidates.append(REPO_ROOT.parent / "stage07-mcp" / ".deps")
    candidates.append(Path("/home/teamb2/stage07-mcp/.deps"))
    deps = next((p for p in candidates if p.is_dir()), None)
    if deps is not None:
        old = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(deps)] + ([old] if old else [])
        )
    return env


def _launch_params() -> StdioServerParameters:
    # 公式SDKの既定(`get_default_environment()`)は PATH/HOME 等の最小限しか
    # 子プロセスへ渡さない — 任意の(信頼できない)MCPサーバに接続する場合の
    # 安全側の既定だが、ここで起動するのは自分たちの `app.mcp_server`(コマンド
    # 自体がリクエスト由来の値を含まない固定文字列)なので、その安全策は不要。
    # むしろ親プロセス(WebUIサーバ)に設定された `DB_PATH` / `RAG_DB_PATH` /
    # `CHROMA_PATH` / `USER_MEMORY_DB_PATH`(さらにB1の隔離デプロイで使う
    # `PYTHONPATH` 差し替え)がそのまま子プロセスにも効いてほしいので、
    # 親の環境をそのまま渡す。
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=str(REPO_ROOT),
        env=_child_environment(),
    )


class MCPBridge:
    """1つの stdio セッションを保持し、直列に呼び出す薄いクライアント。

    `app/server.py` から使う本番用インスタンスはモジュール直下の `_bridge`
    (プロセス寿命で1本、uvicorn の単一イベントループ上で使い回す)。
    テストは(このリポジトリの既存の非同期テストと同じ `asyncio.run()` 方式
    ではイベントループが呼び出しごとに変わるため)自分で `MCPBridge()` を
    作り、`close()` してから次のループへ進む。
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._stack: "AsyncExitStack | None" = None
        self._session: "ClientSession | None" = None

    async def _ensure_session(self) -> ClientSession:
        if self._session is not None:
            return self._session
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(_launch_params()))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        return session

    async def close(self) -> None:
        async with self._lock:
            stack, self._stack = self._stack, None
            self._session = None
        if stack is not None:
            await stack.aclose()

    async def list_tools(self) -> "list[dict[str, Any]]":
        async with self._lock:
            session = await self._ensure_session()
            result = await session.list_tools()
        return [
            {
                "name": t.name,
                "description": (t.description or "").strip(),
                "inputSchema": t.inputSchema,
            }
            for t in result.tools
            if t.name in ALLOWED_TOOLS
        ]

    async def list_resources(self) -> "dict[str, Any]":
        async with self._lock:
            session = await self._ensure_session()
            statics = await session.list_resources()
            templates = await session.list_resource_templates()
        return {
            "resources": [
                {
                    "uri": str(r.uri),
                    "name": r.name,
                    "description": (r.description or "").strip(),
                }
                for r in statics.resources
                if str(r.uri) in STATIC_RESOURCE_URIS
            ],
            "resource_templates": [
                {
                    "uriTemplate": t.uriTemplate,
                    "name": t.name,
                    "description": (t.description or "").strip(),
                    "mimeType": t.mimeType,
                }
                for t in templates.resourceTemplates
                if t.uriTemplate.startswith(ALLOWED_RESOURCE_SCHEMES)
            ],
        }

    async def call_tool(self, name: str, arguments: "dict[str, Any]") -> "dict[str, Any]":
        if name not in ALLOWED_TOOLS:
            raise BridgeError(f"許可されていないツールです: {name}")
        async with self._lock:
            session = await self._ensure_session()
            result = await session.call_tool(name, arguments)

        if result.structuredContent is not None:
            payload: Any = result.structuredContent
        else:
            text = result.content[0].text if result.content else ""
            try:
                payload = json.loads(text) if text else {}
            except json.JSONDecodeError:
                payload = {"text": text}

        if result.isError:
            message = payload.get("error") if isinstance(payload, dict) else None
            raise BridgeError(message or "MCPツールがエラーを返しました")
        return payload if isinstance(payload, dict) else {"result": payload}

    async def read_resource(self, uri: str) -> "dict[str, Any]":
        if not uri.startswith(ALLOWED_RESOURCE_SCHEMES):
            raise BridgeError(f"許可されていないリソースです: {uri}")
        async with self._lock:
            session = await self._ensure_session()
            try:
                result = await session.read_resource(uri)
            except Exception as exc:  # noqa: BLE001 — 不正なパスセグメント等
                raise BridgeError(f"リソースの読み取りに失敗しました: {exc}") from exc

        if not result.contents:
            raise BridgeError("空のリソース応答です")
        content = result.contents[0]
        mime = content.mimeType or "application/octet-stream"
        text = getattr(content, "text", None)
        if text is not None:
            return {"uri": uri, "mimeType": mime, "text": text}
        blob = getattr(content, "blob", None)
        return {"uri": uri, "mimeType": mime, "blob": blob}


# --------------------------------------------------------------------------
# 本番用シングルトン(app/server.py が使う)
# --------------------------------------------------------------------------
_bridge = MCPBridge()


async def list_tools() -> "list[dict[str, Any]]":
    return await _bridge.list_tools()


async def list_resources() -> "dict[str, Any]":
    return await _bridge.list_resources()


async def call_tool(name: str, arguments: "dict[str, Any]") -> "dict[str, Any]":
    return await _bridge.call_tool(name, arguments)


async def read_resource(uri: str) -> "dict[str, Any]":
    return await _bridge.read_resource(uri)


async def shutdown() -> None:
    """FastAPI の shutdown イベントから呼ぶ想定(子プロセスを残さない)。"""
    await _bridge.close()
