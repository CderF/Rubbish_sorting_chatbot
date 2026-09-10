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


def test_item_search_with_a_query_absent_from_the_prompt_is_not_trusted(monkeypatch):
    """5-1 の回帰テスト: 品目名を一切含まない発言(地域名だけ)に対して
    モデルが search_waste_item を存在しない品目名で呼んでも、それを根拠に
    しない。

    2026-09-10、B1実機の複数ターン会話で「小倉北区浅野1丁目です」
    (品目名を一切含まない発言)に対し、モデルが `車いす(電動式を除く)` を
    検索して見つけたことにし、それを根拠に無関係な回答を生成する事例が
    見つかった(要件5-1「推測せず確認する」への違反)。他に何も拾えなければ
    `route_agentically` は None を返し、呼び出し側が決定論的経路
    (「該当する情報が見つかりませんでした」)にフォールバックすること。"""

    def fake_execute_tool(name, arguments):
        assert name == "search_waste_item"
        assert arguments == {"query": "車いす"}
        return {
            "match_stage": "vector",
            "items": [
                {
                    "item_name": "車いす(電動式を除く)",
                    "disposal_method": "粗大ごみ",
                    "notes": "",
                    "source_url": "https://example.test/wheelchair",
                }
            ],
        }

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    _TOOL_CALL_MADE_UP = (
        "<tool_call>\n"
        '{"name": "search_waste_item", "arguments": {"query": "車いす"}}\n'
        "</tool_call>"
    )

    async def body():
        engine = _ScriptedEngine([_TOOL_CALL_MADE_UP, _PLAIN_TEXT])
        result = await agent.route_agentically(
            "小倉北区浅野1丁目です", engine=engine, build_prompt=_build_prompt,
            sampling_params=object(),
        )
        # 品目・収集日・一般説明のいずれも実質的に拾えていないので、呼び出し
        # 側で決定論的経路にフォールバックさせる None を返す。
        assert result is None

    asyncio.run(body())


def test_item_search_with_a_query_present_in_the_prompt_is_still_trusted(monkeypatch):
    """上のガードが、本文に実在する品目名まで誤って弾かないことを確認する
    (既存の正常系の維持)。"""

    def fake_execute_tool(name, arguments):
        if name == "search_waste_item":
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
        if name == "lookup_collection_schedule":
            # _fill_missing_schedule がコード側から直接呼ぶ(品目は見つかった
            # が、モデルがこのツールを自発的に呼ばなかった場合の安全策)。
            return {"schedules": []}
        raise AssertionError(f"unexpected tool call: {name}")

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    async def body():
        engine = _ScriptedEngine([_TOOL_CALL_TV, _PLAIN_TEXT])
        result = await agent.route_agentically(
            "小倉北区城内でテレビはどう捨てますか", engine=engine,
            build_prompt=_build_prompt, sampling_params=object(),
        )
        assert result is not None
        assert result.item_hits[0].item_name == "テレビ"

    asyncio.run(body())


def test_single_character_item_name_is_not_falsely_blocked_by_the_grounding_gate(
    monkeypatch,
):
    """2026-09-10 レビュー指摘の回帰: `_query_grounded_in_text` が「2文字以上
    の共通する連続部分列」だけを見ていると、「傘」のような1文字の品目名は
    それ自体の長さが1しかないため常に弾かれてしまっていた。本文にそのまま
    含まれる(部分文字列として一致する)場合は文字数に関わらず信頼する。"""

    def fake_execute_tool(name, arguments):
        if name == "search_waste_item":
            assert arguments == {"query": "傘"}
            return {
                "match_stage": "exact",
                "items": [
                    {
                        "item_name": "傘",
                        "disposal_method": "燃えないごみ",
                        "notes": "",
                        "source_url": "https://example.test/kasa",
                    }
                ],
            }
        if name == "lookup_collection_schedule":
            return {"schedules": []}
        raise AssertionError(f"unexpected tool call: {name}")

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    _TOOL_CALL_KASA = (
        "<tool_call>\n"
        '{"name": "search_waste_item", "arguments": {"query": "傘"}}\n'
        "</tool_call>"
    )

    async def body():
        engine = _ScriptedEngine([_TOOL_CALL_KASA, _PLAIN_TEXT])
        result = await agent.route_agentically(
            "傘はどう捨てますか", engine=engine, build_prompt=_build_prompt,
            sampling_params=object(),
        )
        assert result is not None
        assert result.item_hits[0].item_name == "傘"

    asyncio.run(body())


def test_single_hiragana_character_query_is_not_trusted_even_if_substring_matches(
    monkeypatch,
):
    """2026-09-10 レビュー指摘の回帰: 「傘」のような1文字品目名を通すために
    部分文字列一致を先に見るようにした結果、query が「す」のようなひらがな
    1文字だと「です」「ます」等の語尾にほぼ必ず含まれてしまい、部分文字列
    一致だけでは弾けなくなっていた。「小倉北区浅野1丁目です」
    (品目名を一切含まない発言)に対しモデルが query="す" で
    search_waste_item を呼んでも、それを根拠にしない。"""

    def fake_execute_tool(name, arguments):
        assert name == "search_waste_item"
        assert arguments == {"query": "す"}
        return {
            "match_stage": "vector",
            "items": [
                {
                    "item_name": "車いす(電動式を除く)",
                    "disposal_method": "粗大ごみ",
                    "notes": "",
                    "source_url": "https://example.test/wheelchair",
                }
            ],
        }

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    _TOOL_CALL_SU = (
        "<tool_call>\n"
        '{"name": "search_waste_item", "arguments": {"query": "す"}}\n'
        "</tool_call>"
    )

    async def body():
        engine = _ScriptedEngine([_TOOL_CALL_SU, _PLAIN_TEXT])
        result = await agent.route_agentically(
            "小倉北区浅野1丁目です", engine=engine, build_prompt=_build_prompt,
            sampling_params=object(),
        )
        assert result is None

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


# --------------------------------------------------------------------------
# STAGE06: 品目検索が失敗したのに search_general_info の弱い結果で found=True
# になってしまう回帰(コードレビュー PR #45 で発見)
# --------------------------------------------------------------------------
_TOOL_CALL_UFO = (
    "<tool_call>\n"
    '{"name": "search_waste_item", "arguments": {"query": "謎の物体"}}\n'
    "</tool_call>"
)
_TOOL_CALL_GENERAL_INFO = (
    "<tool_call>\n"
    '{"name": "search_general_info", "arguments": {"query": "謎の物体"}}\n'
    "</tool_call>"
)


def test_failed_item_search_plus_weak_explanation_falls_back_to_none(monkeypatch):
    """品目検索(search_waste_item)が空振りしたあと、search_general_info が
    (弱いかもしれない)説明チャンクを拾っただけでは found=True にしない。

    修正前は `collected_explanations` が非空というだけで found=True が確定し、
    STAGE06 6-3(完全空振りの固定拒絶)が `_blocking_route` 側にしか配線されて
    いないため、この経路だけ6-3を素通りしていた。
    """

    def fake_execute_tool(name, arguments):
        if name == "search_waste_item":
            return {"match_stage": "none", "items": []}
        if name == "search_general_info":
            return {
                "results": [
                    {
                        "text": "ごみステーションの利用ルールについて…",
                        "source_url": "https://example.test/station",
                        "source_page_title": "ごみステーションの散乱対策について",
                    }
                ]
            }
        raise AssertionError(f"unexpected tool call: {name}")

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    async def body():
        engine = _ScriptedEngine([_TOOL_CALL_UFO, _TOOL_CALL_GENERAL_INFO, _PLAIN_TEXT])
        result = await agent.route_agentically(
            "謎の物体はどう捨てればいいですか", engine=engine,
            build_prompt=_build_prompt, sampling_params=object(),
        )
        # 呼び出し側(_route_safely)を決定論的経路にフォールバックさせ、
        # そちら側で6-3の固定拒絶が正しく発火する。
        assert result is None

    asyncio.run(body())


def test_pure_general_info_question_without_item_search_still_succeeds(monkeypatch):
    """search_waste_item を一度も呼ばず(＝最初から一般説明質問として扱われ)
    search_general_info だけで答えたケースは、修正後も found=True のまま
    (ベクトルRAGパスの正規の使い方を壊さないことの確認)。
    """

    def fake_execute_tool(name, arguments):
        assert name == "search_general_info"
        return {
            "results": [
                {
                    "text": "ごみステーションの防鳥ネットは補助金の対象です。",
                    "source_url": "https://example.test/station",
                    "source_page_title": "ごみステーションの散乱対策について",
                }
            ]
        }

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    async def body():
        engine = _ScriptedEngine([_TOOL_CALL_GENERAL_INFO, _PLAIN_TEXT])
        result = await agent.route_agentically(
            "ごみステーションの防鳥ネットは借りられますか", engine=engine,
            build_prompt=_build_prompt, sampling_params=object(),
        )
        assert result is not None
        assert result.found is True
        assert "防鳥ネット" in result.context

    asyncio.run(body())
