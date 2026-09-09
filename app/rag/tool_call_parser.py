"""モデルの生テキスト出力から Function Calling の tool_call を抽出するパーサー。

`app/rag/agent.py` の `route_agentically` から、非ストリームの短い generate()
結果(生テキスト)を渡して使う。

Qwen3-VL-8B-Instruct の実際のチャットテンプレートは、このリポジトリの開発
マシンにVLM本体の重みが無く(推論はB1/B2サーバーのGPUでのみ可能)実地検証
できていない。Qwenの2.5/3系はHermes方式(`<tool_call>{...}</tool_call>`)を
標準とするため、それを第一候補として実装するが、単一の固定フォーマットに
決め打ちせず「候補パターンを順に試す」設計にしてある — 実機接続後に異なる
形式が確認された場合は `_CANDIDATE_PATTERNS` に追加するだけで対応できる。
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
