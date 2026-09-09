"""Agentic RAG のオーケストレーションループ(Function Calling)。

`app/rag/router.py` の正規表現ヒューリスティックによる決定論的ルーティング
とは別に、LLM自身に `app/rag/tools.py` のツールをどう呼ぶか決めさせる経路。
指導員から課された「Function CallingによるRAG呼び出し」の要件対応。

設計方針(`documents` 内の計画メモ参照):
  - `route_agentically` は成功すれば `RoutedAnswer` を返し、失敗(パース不能・
    ツール0件ヒット・想定外の例外)なら None を返す。呼び出し側
    (`app/server.py`)は None のとき現行の決定論的 `router.handle_query` に
    フォールバックする — `_route_safely` と同じ安全側の方針。
  - `vllm` への直接依存を持たない: `engine`/`sampling_params`/`build_prompt`
    はすべて呼び出し側から注入する。これによりこのモジュール自体はGPU無しの
    ローカル環境でもimport・テストできる(このセッションはB1/B2サーバーに
    接続できないため、この制約は意図的)。
  - `build_prompt` は `app/server.py._build_prompt` 互換のシグネチャ
    `(messages: list, tools: list | None = None) -> dict` を想定する。

Hugging Face Hub上の `Qwen/Qwen3-VL-8B-Instruct` の `chat_template.json` を
直接取得して確認済み(2026-09-09、テンプレート自体は数KBのテキストファイル
なのでGPU無しのこのマシンでも取得できた。VLM本体の重みでの実推論はまだ
未検証):
  - `tools=` 引数は実際にレンダリングされる(渡すと `<|im_start|>system` に
    続けて `# Tools` セクションが挿入され、各ツールを `<tools>` タグ内に
    JSONで列挙し、モデルへ `<tool_call>{"name":...,"arguments":{...}}
    </tool_call>` の形で応答するよう明示的に指示する)。
    `app/rag/tool_call_parser.py` の想定形式と一致する。
  - `{"role": "tool", "content": ...}` はこのテンプレートで正しく扱われる。
    ただし個別の `<|im_start|>tool` ターンにはならず、連続する tool ロール
    のメッセージは1つの `<|im_start|>user ... <tool_response>...
    </tool_response> ... <|im_end|>` ブロックにまとめられる(テンプレートの
    仕様であり、こちらの実装は `content` を渡すだけでよく変更不要)。

未検証(実機のGPUでの実推論が必要):
  - モデルが実際にこの指示通りの tool_call を安定して生成するか(フォー
    マットの「あるべき姿」は確認できたが、8Bモデルの実際の遵守率は未検証)。
  - レイテンシ(ツール呼び出し往復で generate() が最低2回になる)。
"""

from __future__ import annotations

import json
import unicodedata
import uuid
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from app.rag.tool_call_parser import ToolCall, parse_tool_calls
from app.rag.tools import TOOL_SCHEMAS, execute_tool
from app.rag.types import RoutedAnswer

_SYSTEM_PROMPT = (
    "あなたは北九州市のごみ分別・収集日を案内するアシスタントの検索エージェントです。\n"
    "ユーザーの質問に直接答えず、まずは提供されたツールを使って必要な情報を"
    "集めることに専念してください。\n"
    "- 品目の分別方法を調べるには search_waste_item を使う。\n"
    "- 収集曜日を調べるには lookup_collection_schedule を使う"
    "(区名・町名の両方が分かっているときだけ呼ぶこと。町名が不明なら呼ばない)。\n"
    "- 品目や収集日に当てはまらない一般的な質問には search_general_info を使う。\n"
    "- search_waste_item で品目が見つかったら、その品目名で"
    "lookup_item_change_notice も必ず呼び、出し方が変わる予定が無いか確認する。\n"
    "十分な情報が集まったら、それ以上ツールを呼ばずに終了してください"
    "(このやり取り自体はユーザーには見えません。最終的な回答は別途生成されます)。\n"
)

_MAX_ITERATIONS_DEFAULT = 3
_TOOL_CALL_MAX_TOKENS = 300


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()


async def route_agentically(
    prompt: str,
    engine: Any,
    build_prompt: Callable[..., dict],
    sampling_params: Any,
    image_item_name: str | None = None,
    max_iterations: int = _MAX_ITERATIONS_DEFAULT,
) -> RoutedAnswer | None:
    """Function Calling でツールを呼びながら検索する。

    成功すれば `RoutedAnswer` を、失敗すれば None を返す(呼び出し側は None の
    とき決定論的経路にフォールバックすること)。この関数自体は例外を投げない
    — 想定外の失敗もすべて捕まえて None を返す。
    """
    try:
        return await _run_loop(
            prompt, engine, build_prompt, sampling_params, image_item_name, max_iterations
        )
    except Exception as exc:  # noqa: BLE001 — 呼び出し側のフォールバックに委ねる
        print(f"[agent] route_agentically failed: {exc!r}", flush=True)
        return None


async def _run_loop(
    prompt: str,
    engine: Any,
    build_prompt: Callable[..., dict],
    sampling_params: Any,
    image_item_name: str | None,
    max_iterations: int,
) -> RoutedAnswer | None:
    user_content = prompt
    if image_item_name:
        user_content = f"(画像から推定した品目名: {image_item_name})\n\n{prompt}"

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    collected_items: list[SimpleNamespace] = []
    collected_schedules: list[SimpleNamespace] = []
    collected_explanations: list[dict] = []
    collected_notices: list[dict] = []
    sources: list[str] = []
    match_stage: str | None = None
    detected_item: str | None = image_item_name
    detected_ward: str | None = None
    detected_town: str | None = None

    for _ in range(max_iterations):
        prompt_input = build_prompt(messages, tools=TOOL_SCHEMAS)
        text = await _generate_once(engine, prompt_input, sampling_params)

        calls = parse_tool_calls(text)
        if not calls:
            break

        messages.append({"role": "assistant", "content": text})
        for call in calls:
            result = execute_tool(call.name, call.arguments)
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

            if call.name == "search_waste_item":
                if "match_stage" in result:
                    match_stage = result["match_stage"]
                if not detected_item:
                    detected_item = call.arguments.get("query")
                for item in result.get("items", []):
                    collected_items.append(SimpleNamespace(**item))
                    if item.get("source_url"):
                        sources.append(item["source_url"])

            elif call.name == "lookup_collection_schedule":
                detected_ward = detected_ward or call.arguments.get("ward")
                detected_town = detected_town or call.arguments.get("town")
                for sched in result.get("schedules", []):
                    collected_schedules.append(SimpleNamespace(**sched))
                    if sched.get("source_url"):
                        sources.append(sched["source_url"])

            elif call.name == "search_general_info":
                for chunk in result.get("results", []):
                    collected_explanations.append(chunk)
                    if chunk.get("source_url"):
                        sources.append(chunk["source_url"])

            elif call.name == "lookup_item_change_notice":
                if result.get("has_pending_change"):
                    collected_notices.append(result)
                    for seg in [result.get("current")] + result.get("future", []):
                        if seg and seg.get("source_url"):
                            sources.append(seg["source_url"])

    if not (collected_items or collected_schedules or collected_explanations):
        # 何も拾えなければ、呼び出し側で決定論的経路にフォールバックさせる。
        return None

    return _build_routed_answer(
        query=detected_item or prompt,
        items=collected_items,
        schedules=collected_schedules,
        explanations=collected_explanations,
        notices=collected_notices,
        sources=sources,
        match_stage=match_stage,
        detected_item=detected_item,
        detected_ward=detected_ward,
        detected_town=detected_town,
    )


async def _generate_once(engine: Any, prompt_input: dict, sampling_params: Any) -> str:
    """1回分の(非ストリーム扱いの)生成。ユーザーには見せない中間ステップ
    なので、ストリーミングはせず最終テキストだけを取り出す。"""
    request_id = f"agent-{uuid.uuid4().hex[:8]}"
    text = ""
    async for request_output in engine.generate(prompt_input, sampling_params, request_id):
        text = request_output.outputs[0].text
    return text


def _dedupe(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _build_context_text(
    items: list[SimpleNamespace],
    schedules: list[SimpleNamespace],
    explanations: list[dict],
    notices: list[dict],
) -> str:
    blocks: list[str] = []

    if items:
        lines = ["## 分別早見表の検索結果(エージェントが検索)"]
        for it in items:
            lines.append(f"- 品名: {it.item_name}")
            lines.append(f"  出し方: {it.disposal_method}")
            if it.notes:
                lines.append(f"  備考: {it.notes}")
        blocks.append("\n".join(lines))

    if schedules:
        lines = ["## 収集日の検索結果(エージェントが検索)"]
        for s in schedules:
            lines.append(f"- {s.ward}{s.town} / {s.waste_type}: {s.weekday}")
        blocks.append("\n".join(lines))

    if explanations:
        lines = ["## 一般説明の検索結果(エージェントが検索)"]
        for e in explanations:
            title = e.get("source_page_title") or "説明ページ"
            lines.append(f"- {title}: {e.get('text', '')}")
        blocks.append("\n".join(lines))

    if notices:
        lines = ["## この品目は出し方が変わります(旧・新の両方を必ず時系列で案内する)"]
        for n in notices:
            current = n.get("current")
            if current:
                until = current.get("valid_until") or "当面"
                lines.append(f"- 現在(〜{until}): {current.get('disposal_method')}")
            for seg in n.get("future", []):
                lines.append(f"- {seg.get('valid_from')} から: {seg.get('disposal_method')}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _build_routed_answer(
    *,
    query: str,
    items: list[SimpleNamespace],
    schedules: list[SimpleNamespace],
    explanations: list[dict],
    notices: list[dict],
    sources: list[str],
    match_stage: str | None,
    detected_item: str | None,
    detected_ward: str | None,
    detected_town: str | None,
) -> RoutedAnswer:
    q = _norm(query)
    item_match = "none"
    if items:
        item_match = "exact" if any(_norm(it.item_name) == q for it in items) else "analogy"

    ans = RoutedAnswer(
        found=True,
        item_match=item_match,
        item_hits=items,
        schedule_hits=schedules,
        item_match_stage=match_stage,
        detected_item=detected_item,
        detected_ward=detected_ward,
        detected_town=detected_town,
        timeline_change=bool(notices),
        sources=_dedupe(sources),
    )
    ans.context = _build_context_text(items, schedules, explanations, notices)
    return ans
