"""`app/server.py` の STAGE07 7-2/7-3/7-4 ブリッジAPI(`/stage7/mcp/*`)。

`app/server.py` は `pynvml` / `vllm` をモジュールレベルで import する
(GPU実機専用の STAGE1 チャットサーバのため)。このリポジトリのどのテストも
これまで `app.server` を import したことが無い(オフライン開発環境には
そもそも `fastapi` すら入っていなかった — このテストのために追加インストール
した。`requirements.txt` に既に `fastapi==0.136.3` として宣言済みのものを
オフライン開発用 venv にも入れただけで、`requirements.txt` 自体は変えて
いない)。

ここでは `pynvml`/`vllm` を「import が通るだけ」のダミーモジュールに差し替え、
FastAPI の `TestClient` で **STAGE07 のルーティングだけ** を検証する。
`/chat`・`/ui/chat`・`/vram` 等、実エンジンが要る経路はこのテストの対象外
(このリポジトリでは検証不能 — GPU実機でしか確認できない、既存の
`tests/spec_test.py` と同じ「手元で確認する」領域)。

`app.mcp_bridge` は実子プロセスを起動すると遅い(`tests/mcp_bridge_test.py`
の役割)ので、ここではフェイクに差し替え、配線(認証・Cookie・allowlist越しの
受け渡し・user_id の強制上書き)だけを見る。
"""

from __future__ import annotations

import sys
import types

import pytest

TEST_PASSWORD = "test-password-2026"


def _install_fake_gpu_stack(monkeypatch):
    """`pynvml` / `vllm` を、`app.server` の import を通すだけのダミーに差し替える。"""
    fake_pynvml = types.ModuleType("pynvml")
    fake_pynvml.nvmlInit = lambda: None
    fake_pynvml.nvmlDeviceGetHandleByIndex = lambda i: object()
    fake_pynvml.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=0, total=1)
    fake_pynvml.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=0)
    fake_pynvml.nvmlDeviceGetName = lambda h: "fake-gpu"
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.AsyncEngineArgs = object
    fake_vllm.AsyncLLMEngine = object
    fake_vllm.SamplingParams = object
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)


class _FakeBridge:
    """`app.mcp_bridge` の代役。実子プロセスは起動せず、呼ばれた引数を記録する。"""

    def __init__(self):
        self.calls: "list[tuple[str, dict]]" = []
        self.memory: "dict[str, dict]" = {}

    async def list_tools(self):
        return [{"name": "search_waste_item", "description": "", "inputSchema": {}}]

    async def list_resources(self):
        return {
            "resources": [{"uri": "knowledge://sources", "name": "x", "description": ""}],
            "resource_templates": [
                {"uriTemplate": "schedule-calendar://{ward}/{town}",
                 "name": "x", "description": "", "mimeType": "image/svg+xml"}
            ],
        }

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        uid = arguments.get("user_id")
        if name == "remember_fact":
            self.memory.setdefault(uid, {})[arguments["key"]] = arguments["value"]
            return {"ok": True}
        if name == "recall_facts":
            return {"facts": dict(self.memory.get(uid, {}))}
        if name == "get_disposal_guidance":
            remembered = self.memory.get(uid, {})
            used_memory = bool(uid) and "ward" in remembered and "町名" not in arguments["prompt"]
            return {
                "found": True,
                "context": "",
                "markdown": "| 品名 | 出し方 |\n|---|---|\n| ソファー | 粗大ごみ |",
                "canned_response": None,
                "used_memory": used_memory,
                "remembered": remembered,
                "missing": [],
            }
        return {"result": "ok"}

    async def read_resource(self, uri):
        return {"uri": uri, "mimeType": "image/svg+xml", "text": "<svg>fake</svg>"}


@pytest.fixture
def app_client(monkeypatch):
    _install_fake_gpu_stack(monkeypatch)
    monkeypatch.setenv("WEBUI_PASSWORD", TEST_PASSWORD)

    import importlib

    import app.server as server_module

    importlib.reload(server_module)  # 環境変数を反映させ、前のテストの状態も捨てる

    fake_bridge = _FakeBridge()
    monkeypatch.setattr(server_module.mcp_bridge, "list_tools", fake_bridge.list_tools)
    monkeypatch.setattr(server_module.mcp_bridge, "list_resources", fake_bridge.list_resources)
    monkeypatch.setattr(server_module.mcp_bridge, "call_tool", fake_bridge.call_tool)
    monkeypatch.setattr(server_module.mcp_bridge, "read_resource", fake_bridge.read_resource)

    from fastapi.testclient import TestClient

    # `with` を使わない(= lifespan/startup を発火させない): 本物のエンジン
    # 起動(`load_model`)を素通りさせるため。STAGE07 ルートは engine/processor
    # に依存しないので、このテストの範囲では問題ない。
    client = TestClient(server_module.app)
    return client, server_module, fake_bridge


def _login(client) -> None:
    res = client.post("/login", data={"password": TEST_PASSWORD}, follow_redirects=False)
    assert res.status_code == 303
    assert "rag_session" in res.cookies


def test_stage7_endpoints_require_login(app_client):
    client, _server, _bridge = app_client
    assert client.get("/stage7/mcp/tools").status_code == 401
    assert client.get("/stage7/mcp/resources").status_code == 401
    assert client.post("/stage7/mcp/call", json={"tool": "search_waste_item"}).status_code == 401
    assert client.get("/stage7/mcp/resource", params={"uri": "knowledge://sources"}).status_code == 401


def test_login_issues_a_stable_stage7_uid_cookie(app_client):
    client, _server, _bridge = app_client
    _login(client)
    assert "stage7_uid" in client.cookies
    uid1 = client.cookies["stage7_uid"]

    # 別リクエスト(WebUI の「新しい会話」相当 — フロント側は session_id しか
    # 変えない)でも同じCookieのまま。サーバ側は毎回それを読むだけで上書きしない。
    res = client.get("/stage7/mcp/tools")
    assert res.status_code == 200
    assert client.cookies["stage7_uid"] == uid1
    # パスワードや完全なCookie値は返り値に出さない。
    assert res.json()["stage7_uid_short"] != uid1
    assert TEST_PASSWORD not in res.text


def test_tools_and_resources_pass_through_the_allowlisted_bridge(app_client):
    client, _server, _bridge = app_client
    _login(client)
    tools = client.get("/stage7/mcp/tools").json()["tools"]
    assert any(t["name"] == "search_waste_item" for t in tools)

    resources = client.get("/stage7/mcp/resources").json()
    assert resources["resources"][0]["uri"] == "knowledge://sources"
    assert resources["resource_templates"][0]["uriTemplate"] == "schedule-calendar://{ward}/{town}"


def test_call_rejects_a_missing_tool_name(app_client):
    client, _server, _bridge = app_client
    _login(client)
    res = client.post("/stage7/mcp/call", json={"arguments": {}})
    assert res.status_code == 400


def test_call_forwards_a_normal_tool_call(app_client):
    client, _server, _bridge = app_client
    _login(client)
    res = client.post(
        "/stage7/mcp/call", json={"tool": "search_waste_item", "arguments": {"query": "テレビ"}}
    )
    assert res.status_code == 200
    assert res.json()["result"]["result"] == "ok"
    assert _bridge.calls[-1] == ("search_waste_item", {"query": "テレビ"})


def test_call_overrides_a_client_supplied_user_id_for_memory_scoped_tools(app_client):
    """7-4: クライアントが勝手な `user_id` を送っても、Cookie の値で必ず
    上書きする(他ブラウザの記憶を読める穴を作らない)。"""
    client, _server, _bridge = app_client
    _login(client)
    my_uid = client.cookies["stage7_uid"]

    client.post(
        "/stage7/mcp/call",
        json={"tool": "recall_facts", "arguments": {"user_id": "someone-elses-id"}},
    )
    assert _bridge.calls[-1] == ("recall_facts", {"user_id": my_uid})


def test_resource_proxies_the_svg_calendar(app_client):
    client, _server, _bridge = app_client
    _login(client)
    res = client.get(
        "/stage7/mcp/resource", params={"uri": "schedule-calendar://小倉北区/浅野1丁目"}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["mimeType"] == "image/svg+xml"
    assert body["text"].startswith("<svg")


def test_resource_rejects_a_bridge_error_as_400(app_client, monkeypatch):
    client, server_module, _bridge = app_client
    _login(client)

    async def _boom(uri):
        raise server_module.mcp_bridge.BridgeError("許可されていないリソースです")

    monkeypatch.setattr(server_module.mcp_bridge, "read_resource", _boom)
    res = client.get("/stage7/mcp/resource", params={"uri": "file:///etc/passwd"})
    assert res.status_code == 400


def test_memory_persists_across_a_simulated_new_conversation(app_client):
    """7-4 のデモ本体: ①住所付きで訊く → ②「新しい会話」(=session_idの
    リセットだけ、stage7_uid Cookie はそのまま) → ③住所なしで訊く →
    記憶から補われ used_memory=true になる。"""
    client, _server, _bridge = app_client
    _login(client)

    r1 = client.post(
        "/stage7/mcp/call",
        json={
            "tool": "get_disposal_guidance",
            "arguments": {"prompt": "小倉北区浅野1丁目のソファの収集日は?"},
        },
    )
    assert r1.status_code == 200

    client.post(
        "/stage7/mcp/call",
        json={"tool": "remember_fact", "arguments": {"key": "ward", "value": "小倉北区"}},
    )

    # 「新しい会話」ボタンはフロント側の session_id だけを変える操作なので、
    # ここでは何もしない(stage7_uid Cookie に触れる処理が無いことの確認は
    # test_login_issues_a_stable_stage7_uid_cookie が担う)。同じ client
    # (= 同じブラウザ Cookie)のまま次のターンを送る。
    r2 = client.post(
        "/stage7/mcp/call",
        json={"tool": "get_disposal_guidance", "arguments": {"prompt": "ソファの収集日は?"}},
    )
    assert r2.status_code == 200
    assert r2.json()["result"]["used_memory"] is True
