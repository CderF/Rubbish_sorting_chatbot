"""app/rag/tool_call_parser.py の単体テスト。

Qwen3-VL-8B-Instruct 本体はこの開発マシンでは動かせない(推論はB1/B2サーバー
のGPUでのみ可能)ため、実際のモデル出力ではなく手書きの模擬テキストで
パーサーのロジックだけを検証する。実機のtool_call実フォーマットが想定と
異なる場合は `app/rag/tool_call_parser.py` の `_CANDIDATE_PATTERNS` を追う
形で対応する(このファイルのテストは形式追加時の回帰検知に使う)。
"""

from __future__ import annotations

from app.rag.tool_call_parser import ToolCall, parse_tool_calls, strip_tool_call_blocks


def test_parses_a_single_hermes_style_tool_call():
    text = (
        "<tool_call>\n"
        '{"name": "search_waste_item", "arguments": {"query": "テレビ"}}\n'
        "</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert calls == [ToolCall(name="search_waste_item", arguments={"query": "テレビ"})]


def test_parses_multiple_tool_calls_in_one_turn():
    text = (
        "<tool_call>\n"
        '{"name": "search_waste_item", "arguments": {"query": "テレビ"}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "lookup_item_change_notice", "arguments": {"item_name": "テレビ"}}\n'
        "</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["search_waste_item", "lookup_item_change_notice"]


def test_accepts_arguments_as_a_json_encoded_string_openai_style():
    text = (
        "<tool_call>\n"
        '{"name": "search_waste_item", "arguments": "{\\"query\\": \\"テレビ\\"}"}\n'
        "</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert calls == [ToolCall(name="search_waste_item", arguments={"query": "テレビ"})]


def test_falls_back_to_fenced_json_block_when_no_hermes_tags():
    text = '```json\n{"name": "search_general_info", "arguments": {"query": "指定袋"}}\n```'
    calls = parse_tool_calls(text)
    assert calls == [ToolCall(name="search_general_info", arguments={"query": "指定袋"})]


def test_plain_answer_text_yields_no_tool_calls():
    text = "テレビは粗大ごみとして、事前申込のうえ収集日に出してください。"
    assert parse_tool_calls(text) == []


def test_malformed_json_inside_tool_call_tags_is_ignored_not_raised():
    text = "<tool_call>\n{not valid json at all\n</tool_call>"
    assert parse_tool_calls(text) == []


def test_tool_call_missing_name_key_is_ignored():
    text = '<tool_call>\n{"arguments": {"query": "テレビ"}}\n</tool_call>'
    assert parse_tool_calls(text) == []


def test_empty_and_none_text_yield_no_tool_calls():
    assert parse_tool_calls("") == []
    assert parse_tool_calls(None) == []


def test_strip_tool_call_blocks_removes_hermes_and_fenced_forms():
    text = (
        "前置き\n"
        "<tool_call>\n"
        '{"name": "search_waste_item", "arguments": {"query": "テレビ"}}\n'
        "</tool_call>\n"
        "後書き"
    )
    cleaned = strip_tool_call_blocks(text)
    assert "tool_call" not in cleaned
    assert "前置き" in cleaned and "後書き" in cleaned
