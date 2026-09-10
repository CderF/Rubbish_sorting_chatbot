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
from pathlib import Path
from types import SimpleNamespace

from app.rag import agent

_REAL_DB_PATH = Path(__file__).resolve().parent.parent / "var" / "kitakyushu_waste.db"


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
        # 3回とも同じ品目の exact ヒットなので、素朴に集約した重複3件ではなく
        # 絞り込み後の1件になる(`_narrow_item_hits` 参照)。
        assert len(result.item_hits) == 1
        assert result.item_hits[0].item_name == "テレビ"

    asyncio.run(body())


def test_unrelated_extra_search_call_does_not_pollute_context(monkeypatch):
    """モデルが本命の品目を見つけた後、探索的に別の(無関係な)品目を検索
    しても、その備考が最終的な item_hits に混ざらないことを確認する。

    2026-09-10、実機で「たんす」の質問に無関係な品目(かご等)の「50センチ
    未満なら…」という材質・サイズ条件が context に紛れ込み、たんす自体には
    無いはずの条件をLLMが答えに書いてしまう不具合が見つかった。原因は
    `search_waste_item` の呼び出しごとの結果を単純に全部 collected_items へ
    足し込んでいたこと。"""

    responses = iter(
        [
            {
                "match_stage": "exact",
                "items": [
                    {
                        "item_name": "たんす",
                        "disposal_method": "粗大ごみ",
                        "notes": "事前申込制",
                        "source_url": "https://example.test/tansu",
                    }
                ],
            },
            {
                "match_stage": "vector",
                "items": [
                    {
                        "item_name": "かご",
                        "disposal_method": "家庭ごみ",
                        "notes": "プラスチックだけでできており、1辺の長さが50センチ未満のもので指定袋に入る大きさのものは、プラスチック",
                        "source_url": "https://example.test/kago",
                    }
                ],
            },
        ]
    )

    def fake_execute_tool(name, arguments):
        return next(responses)

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    _TOOL_CALL_TANSU = (
        "<tool_call>\n"
        '{"name": "search_waste_item", "arguments": {"query": "たんす"}}\n'
        "</tool_call>"
    )
    _TOOL_CALL_KAGO = (
        "<tool_call>\n"
        '{"name": "search_waste_item", "arguments": {"query": "かご"}}\n'
        "</tool_call>"
    )

    async def body():
        engine = _ScriptedEngine([_TOOL_CALL_TANSU, _TOOL_CALL_KAGO, _PLAIN_TEXT])
        result = await agent.route_agentically(
            "たんすを捨てたいです", engine=engine, build_prompt=_build_prompt,
            sampling_params=object(),
        )

        assert result is not None
        # exact(たんす)の方が vector(かご)より match_stage の順位が高いので、
        # かご の50センチ条件は item_hits にも context にも残らない。
        assert len(result.item_hits) == 1
        assert result.item_hits[0].item_name == "たんす"
        assert "センチ" not in result.context
        assert "https://example.test/kago" not in result.sources

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


def test_conversation_history_is_passed_to_function_calling_prompt():
    captured_messages = []

    def capturing_build_prompt(messages, tools=None):
        captured_messages.append(messages)
        return {"messages": messages, "tools": tools}

    history = [
        {"role": "user", "content": "ペットボトルの分別を教えて"},
        {"role": "assistant", "content": "ペットボトルは資源ごみです。"},
    ]

    async def body():
        await agent.route_agentically(
            "小倉北区浅野1丁目です。収集日も教えてください。",
            engine=_ScriptedEngine([_PLAIN_TEXT]),
            build_prompt=capturing_build_prompt,
            sampling_params=object(),
            history=history,
        )

    asyncio.run(body())
    messages = captured_messages[0]
    assert messages[1:3] == history
    assert messages[3]["content"].startswith("小倉北区浅野1丁目")


def test_district_fallback_note_reaches_context_and_states_the_weekday(monkeypatch):
    """丁目を省略した地区名(「黒崎」等)で lookup_collection_schedule を呼んだ
    場合、`tools._tool_lookup_collection_schedule` が付ける `district_note` が
    最終的な context に載り、収集日そのものは省略されないことを確認する。

    2026-09-10、実機で「八幡西区黒崎でソファを処分する方法を教えてください」
    に対し、town="黒崎"(丁目無し)で lookup_collection_schedule を呼んでも
    直接一致する行が無いため何も返らず、収集日が一切案内されない不具合が
    見つかった(router.py の決定論的経路には既にあった `_match_district`/
    `_uniform_district_schedule` による丁目省略地区名の解決が、エージェント
    経路の `tools.py` には無かったのが原因)。"""

    def fake_execute_tool(name, arguments):
        if name == "search_waste_item":
            return {
                "match_stage": "analogy",
                "items": [
                    {
                        "item_name": "ソファーベッド",
                        "disposal_method": "粗大ごみ",
                        "notes": "事前申込制",
                        "source_url": "https://example.test/sofa",
                    }
                ],
            }
        if name == "lookup_collection_schedule":
            assert arguments == {"ward": "八幡西区", "town": "黒崎"}
            return {
                "schedules": [
                    {
                        "ward": "八幡西区",
                        "town": "黒崎1丁目",
                        "waste_type": "粗大ごみ",
                        "weekday": "第2水曜日",
                        "source_url": "https://example.test/schedule",
                    }
                ],
                "district_note": (
                    "「黒崎」は丁目の指定がなかったが、地区内の全ての町"
                    "(黒崎1丁目・黒崎2丁目)で収集日が一致することを確認済み。"
                ),
            }
        raise AssertionError(f"unexpected tool call: {name}")

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    _TOOL_CALL_SOFA = (
        "<tool_call>\n"
        '{"name": "search_waste_item", "arguments": {"query": "ソファ"}}\n'
        "</tool_call>"
    )
    _TOOL_CALL_SCHEDULE = (
        "<tool_call>\n"
        '{"name": "lookup_collection_schedule", '
        '"arguments": {"ward": "八幡西区", "town": "黒崎"}}\n'
        "</tool_call>"
    )

    async def body():
        engine = _ScriptedEngine([_TOOL_CALL_SOFA, _TOOL_CALL_SCHEDULE, _PLAIN_TEXT])
        result = await agent.route_agentically(
            "八幡西区黒崎でソファを処分する方法を教えてください",
            engine=engine, build_prompt=_build_prompt, sampling_params=object(),
        )

        assert result is not None
        assert len(result.schedule_hits) == 1
        assert result.schedule_hits[0].weekday == "第2水曜日"
        assert "第2水曜日" in result.context
        assert "地区内の全ての町" in result.context

    asyncio.run(body())


def test_schedule_is_filled_in_even_when_the_model_never_calls_the_tool(monkeypatch):
    """実機で実際に起きたケース: モデルが search_waste_item で品目(粗大ごみ)
    を見つけたあと、lookup_collection_schedule を一度も呼ばずに終了しても、
    収集曜日が省略されないことを確認する。

    2026-09-10、B1実機で「八幡西区黒崎でソファを処分する方法を教えてくだ
    さい」に対し、`tools.py` 側の地区名解決(直前のコミット)を入れたあとも
    モデルが lookup_collection_schedule を一度も呼ばず、事前申込制の案内
    だけで収集曜日が出ない状態が再現した。指示(システムプロンプト)だけに
    頼らず、品目が見つかっていて収集日がまだ無ければコード側で
    lookup_collection_schedule を直接実行する安全策(`_fill_missing_schedule`)
    を追加した。

    `_fill_missing_schedule` は(執行順が違うと壊れるモック経由ではなく)
    `extract_region`/`_match_district` を通じて実際の本番DBを読む。それらは
    `functools.lru_cache` されているので、`pytest tests/` 全体を実行した
    ときに限り、無関係な `collection_schedule_town_test.py`(pytestのテスト
    ではなく単体で実行するスクリプトだが `_test.py` という名前のため収集は
    される)がトップレベルで `os.environ["DB_PATH"]` を空の一時DBへ恒久的に
    書き換えてしまい、このテストが最初に `_town_candidates`/`_district_groups`
    を呼ぶ側になった場合にその汚染を拾って落ちることがある
    (2026-09-10 に発覚。テストスイート側の孤立性の問題で本コミットの範囲
    外のため、ここでは実DBのパスを明示することで自衛するに留める)。"""
    monkeypatch.setenv("DB_PATH", str(_REAL_DB_PATH))

    def fake_execute_tool(name, arguments):
        if name == "search_waste_item":
            assert arguments == {"query": "ソファ"}
            return {
                "match_stage": "analogy",
                "items": [
                    {
                        "item_name": "ソファーベッド",
                        "disposal_method": "粗大ごみ",
                        "notes": "事前申込制",
                        "source_url": "https://example.test/sofa",
                    }
                ],
            }
        if name == "lookup_collection_schedule":
            # モデルではなく _fill_missing_schedule がコード側から直接呼ぶ。
            # 区名・町名は extract_region/_match_district がプロンプトの原文
            # から実データを使って解決した結果("黒崎"は丁目省略の地区名)。
            assert arguments == {
                "ward": "八幡西区",
                "town": "黒崎",
                "waste_type": "粗大ごみ",
            }
            return {
                "schedules": [
                    {
                        "ward": "八幡西区",
                        "town": "黒崎1丁目",
                        "waste_type": "粗大ごみ",
                        "weekday": "第1月曜日",
                        "source_url": "https://example.test/schedule",
                    }
                ],
                "district_note": "「黒崎」は丁目の指定がなかったが…確認済み。",
            }
        raise AssertionError(f"unexpected tool call: {name}")

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    _TOOL_CALL_SOFA = (
        "<tool_call>\n"
        '{"name": "search_waste_item", "arguments": {"query": "ソファ"}}\n'
        "</tool_call>"
    )

    async def body():
        # モデルは search_waste_item だけ呼んで終了する(lookup_collection_schedule
        # を一度も呼ばない、実機で観測された挙動そのまま)。
        engine = _ScriptedEngine([_TOOL_CALL_SOFA, _PLAIN_TEXT])
        result = await agent.route_agentically(
            "八幡西区黒崎でソファを処分する方法を教えてください",
            engine=engine, build_prompt=_build_prompt, sampling_params=object(),
        )

        assert result is not None
        assert len(result.schedule_hits) == 1
        assert result.schedule_hits[0].weekday == "第1月曜日"
        assert "第1月曜日" in result.context

    asyncio.run(body())
