"""app/rag/agent.py の制御フロー(ツール呼び出しループ・反復上限・フォール
バック条件)の単体テスト。

実際のQwen3-VL-8B-Instructがどんなtool_callを出すかはこの開発マシンでは
検証できない(GPU無し)ため、スクリプト化した偽の `engine`(1回呼ばれる
ごとに用意したテキストを順に返す)を注入してループ制御だけを検証する。
`app.rag.tools.execute_tool` もモンキーパッチし、ツール実行そのものの
正しさ(`tests/agentic_tools_test.py` で別途検証済み)とは切り離す。

pytest-asyncio はこのプロジェクトの依存(requirements.txt)に無いので、
各テストは `asyncio.run()` で素の async 関数を実行する薄い同期テストにする
(既存の他テストファイルと同じくpytestの追加プラグイン無しで動く)。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.rag import agent


class _FakeOutput:
    def __init__(self, text: str):
        self.outputs = [SimpleNamespace(text=text)]


class _ScriptedEngine:
    """`generate` が呼ばれるたびに、あらかじめ渡したテキストを順番に返す
    偽エンジン。台本を使い切ったら空文字(ツール呼び出し無し = 終了)を返す。"""

    def __init__(self, script: list[str]):
        self._script = list(script)
        self.call_count = 0

    async def generate(self, prompt_input, sampling_params, request_id):
        self.call_count += 1
        text = self._script.pop(0) if self._script else ""
        yield _FakeOutput(text)


def _build_prompt(messages: list, tools: list | None = None) -> dict:
    return {"messages": messages, "tools": tools}


_TOOL_CALL_TV = (
    "<tool_call>\n"
    '{"name": "search_waste_item", "arguments": {"query": "テレビ"}}\n'
    "</tool_call>"
)
_PLAIN_TEXT = "十分な情報が集まりました。"


def test_no_tool_call_returns_none_so_caller_falls_back():
    async def body():
        engine = _ScriptedEngine([_PLAIN_TEXT])
        result = await agent.route_agentically(
            "テレビはどう捨てますか", engine=engine, build_prompt=_build_prompt,
            sampling_params=object(),
        )
        assert result is None
        assert engine.call_count == 1

    asyncio.run(body())


def test_one_tool_call_then_stop_builds_a_routed_answer(monkeypatch):
    def fake_execute_tool(name, arguments):
        assert name == "search_waste_item"
        assert arguments == {"query": "テレビ"}
        return {
            "match_stage": "exact",
            "items": [
                {
                    "item_name": "テレビ",
                    "disposal_method": "粗大ごみ",
                    "notes": "事前申込制",
                    "source_url": "https://example.test/tv",
                }
            ],
        }

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    async def body():
        engine = _ScriptedEngine([_TOOL_CALL_TV, _PLAIN_TEXT])
        result = await agent.route_agentically(
            "テレビはどう捨てますか", engine=engine, build_prompt=_build_prompt,
            sampling_params=object(),
        )

        assert result is not None
        assert result.found is True
        assert result.item_match == "exact"
        assert result.item_hits[0].item_name == "テレビ"
        assert result.item_hits[0].disposal_method == "粗大ごみ"
        assert "https://example.test/tv" in result.sources
        assert engine.call_count == 2  # ツール呼び出し1回 + 終了確認の1回

    asyncio.run(body())


def test_unknown_tool_error_result_does_not_crash_the_loop(monkeypatch):
    def fake_execute_tool(name, arguments):
        return {"error": f"unknown tool: {name}"}

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    async def body():
        bogus_call = '<tool_call>\n{"name": "not_a_tool", "arguments": {}}\n</tool_call>'
        engine = _ScriptedEngine([bogus_call, _PLAIN_TEXT])
        result = await agent.route_agentically(
            "テレビはどう捨てますか", engine=engine, build_prompt=_build_prompt,
            sampling_params=object(),
        )
        # エラー結果しか無いので何も収集できず、呼び出し側にフォールバックさせる
        assert result is None

    asyncio.run(body())


def test_hits_max_iterations_still_uses_whatever_was_collected(monkeypatch):
    def fake_execute_tool(name, arguments):
        return {
            "match_stage": "exact",
            "items": [
                {
                    "item_name": "テレビ",
                    "disposal_method": "粗大ごみ",
                    "notes": "",
                    "source_url": "https://example.test/tv",
                }
            ],
        }

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    async def body():
        # 毎回ツール呼び出しを返し続け、決して自分から終了しないモデルを模す。
        engine = _ScriptedEngine([_TOOL_CALL_TV] * 10)
        result = await agent.route_agentically(
            "テレビはどう捨てますか", engine=engine, build_prompt=_build_prompt,
            sampling_params=object(), max_iterations=3,
        )
        assert result is not None
        assert engine.call_count == 3  # 上限で打ち切られる
        assert len(result.item_hits) == 3  # 3回分の結果を集めている

    asyncio.run(body())


def test_tool_execution_exception_is_caught_and_returns_none(monkeypatch):
    def raising_execute_tool(name, arguments):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent, "execute_tool", raising_execute_tool)

    async def body():
        engine = _ScriptedEngine([_TOOL_CALL_TV])
        result = await agent.route_agentically(
            "テレビはどう捨てますか", engine=engine, build_prompt=_build_prompt,
            sampling_params=object(),
        )
        assert result is None

    asyncio.run(body())


def test_engine_generate_exception_is_caught_and_returns_none():
    class _ExplodingEngine:
        async def generate(self, prompt_input, sampling_params, request_id):
            raise RuntimeError("engine down")
            yield  # pragma: no cover - never reached, keeps this an async generator

    async def body():
        result = await agent.route_agentically(
            "テレビはどう捨てますか", engine=_ExplodingEngine(), build_prompt=_build_prompt,
            sampling_params=object(),
        )
        assert result is None

    asyncio.run(body())


def test_image_item_name_is_passed_through_as_context():
    captured_messages = []

    def capturing_build_prompt(messages, tools=None):
        captured_messages.append(messages)
        return {"messages": messages, "tools": tools}

    async def body():
        engine = _ScriptedEngine([_PLAIN_TEXT])
        await agent.route_agentically(
            "これは何ですか", engine=engine, build_prompt=capturing_build_prompt,
            sampling_params=object(), image_item_name="テレビ",
        )
        user_msg = captured_messages[0][1]["content"]
        assert "テレビ" in user_msg

    asyncio.run(body())
