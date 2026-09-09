"""モデルの生テキスト出力から Function Calling の tool_call を抽出するパーサー。

`app/rag/agent.py` の `route_agentically` から、非ストリームの短い generate()
結果(生テキスト)を渡して使う。

Hermes方式(`<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`)
を第一候補として実装している。これは推測ではなく、Hugging Face Hub上の
`Qwen/Qwen3-VL-8B-Instruct` の `chat_template.json` を直接取得して確認した
実際のテンプレートの記述と一致する(2026-09-09、このマシンではVLM本体の
重みは無くGPU推論はできないが、テンプレート自体は数KBのテキストファイルな
ので取得できた)。テンプレートは `tools` が渡されるとツール一覧を
`<tools>...</tools>` XMLタグ内にJSONで列挙し、モデルには「関数呼び出しは
`<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n
</tool_call>` の形で返せ」と明示的に指示する — つまり `arguments` は
(文字列化したJSONではなく)ネストしたJSONオブジェクトとして出力される想定。
このパーサーはその形式を主に扱うが、単一の固定フォーマットに決め打ちせず
「候補パターンを順に試す」設計は維持してある(実際の生成結果でずれが見つ
かった場合に `_CANDIDATE_PATTERNS` へ追加するだけで対応できるように)。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


# Qwen系(Hermes方式)の標準フォーマット。第一候補。
_HERMES_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# 一部のモデル/テンプレートは ```json ... ``` のフェンス付きコードブロックで
# 関数呼び出しを書くことがある。Hermes形式が見つからないときの第二候補。
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\"name\"\s*:.*?\})\s*```", re.DOTALL)

_CANDIDATE_PATTERNS = (_HERMES_TOOL_CALL_RE, _FENCED_JSON_RE)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """モデルの生テキスト出力から tool_call を抽出する。

    見つからなければ空リストを返す(=通常の回答文として扱う、というのが
    呼び出し側 `app/rag/agent.py` の解釈)。
    """
    if not text:
        return []

    for pattern in _CANDIDATE_PATTERNS:
        matches = pattern.findall(text)
        if not matches:
            continue
        calls = [c for c in (_parse_one(raw) for raw in matches) if c is not None]
        if calls:
            return calls
    return []


def _parse_one(raw_json: str) -> ToolCall | None:
    try:
        obj = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None

    arguments = obj.get("arguments", {})
    # OpenAI形式の慣習では arguments はJSON文字列のことがある。Hermes形式は
    # ネストしたオブジェクトのことが多い。実機のテンプレートがどちらで来るか
    # 未確認なので両方受け付ける。
    if isinstance(arguments, str):
        if not arguments.strip():
            arguments = {}
        else:
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
    if not isinstance(arguments, dict):
        return None

    return ToolCall(name=name, arguments=arguments)


def strip_tool_call_blocks(text: str) -> str:
    """tool_call ブロックを取り除いた残りのテキストを返す(ログ・デバッグ用)。"""
    cleaned = _HERMES_TOOL_CALL_RE.sub("", text or "")
    cleaned = _FENCED_JSON_RE.sub("", cleaned)
    return cleaned.strip()
