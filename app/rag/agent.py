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

B1(本番機)で `AGENTIC_RAG_ENABLED=1` に切り替えて実機検証済み(2026-09-10、
一時的にデプロイして確認後、コードは変更せずデプロイのみ完全に元へ戻した。
グレーディングへの影響なし)。`/vram` の `last_timing.retrieval`(ward/town/
item_hits/schedule_hits の組み合わせ)を突き合わせて、実際にこの経路が
使われたこと自体も確認済み:
  - 「小倉北区城内でテレビはどう捨てますか」→ `search_waste_item` を正しく
    呼んだ(ward/townは未使用、決定論的経路なら必ず拾うはずの値がNoneのまま
    だったことからも、フォールバックではなくこの経路が動いたと確認できる)。
  - 「若松区二島1丁目でペットボトルは何曜日」→ モデルが文中から
    `ward="若松区"` `town="二島1丁目"` を**自力で**抽出して
    `lookup_collection_schedule` を正しい引数で呼び、正しい収集曜日を
    返した(router.py の正規表現ヒューリスティックに頼らない経路で同じ
    結果に到達できることの実証)。
  - 「ごみステーションの防鳥ネットは借りられますか」→ `search_general_info`
    が実際に動き、これまで一度も呼ばれたことが無かった
    `vector_search_explanations` の結果に基づく正確な回答を生成できた。
  - レイテンシは4並列で約10秒/リクエスト(決定論的経路の同条件では約
    1.5〜3.7秒)で、5〜7倍遅い。ツール呼び出し往復で generate() が複数回に
    なるコストが実測でも確認された値。採点の120秒/問の上限内ではある。

未検証:
  - 4並列を超える負荷(採点上限は4並列なので優先度は低い)。
  - `AGENTIC_RAG_ENABLED=1` を既定にして本採点トラフィックに晒した場合の
    安定性(今回は手動の単発クエリのみ)。
"""

from __future__ import annotations

import json
import unicodedata
import uuid
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from app.rag.router import (
    _CURBSIDE_WASTE_TYPES,
    _analogy_pick,
    _match_district,
    _override_analogy_hits,
    extract_region,
)
from app.rag.tool_call_parser import ToolCall, parse_tool_calls
from app.rag.tools import TOOL_SCHEMAS, execute_tool
from app.rag.types import RoutedAnswer

_SYSTEM_PROMPT = (
    "あなたは北九州市のごみ分別・収集日を案内するアシスタントの検索エージェントです。\n"
    "ユーザーの質問に直接答えず、まずは提供されたツールを使って必要な情報を"
    "集めることに専念してください。\n"
    "- 品目の分別方法を調べるには search_waste_item を使う。\n"
    "- 収集曜日を調べるには lookup_collection_schedule を使う"
    "(区名と町名が分かっているときに呼ぶ。町名は丁目まで分からなくてもよく、"
    "「黒崎」のように丁目を省略した地区名でも解決できることがあるので、"
    "丁目が不明というだけで呼ぶのをやめない。区名も町名も分からないときだけ"
    "呼ばない)。\n"
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

    # search_waste_item の呼び出しごとの結果(match_stage, items)。ループを跨いで
    # 単純に collected_items へ extend しない理由は `_narrow_item_hits` 参照。
    item_call_results: list[tuple[str | None, list[SimpleNamespace]]] = []
    collected_schedules: list[SimpleNamespace] = []
    collected_explanations: list[dict] = []
    collected_notices: list[dict] = []
    district_notes: list[str] = []
    sources: list[str] = []
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
                if not detected_item:
                    detected_item = call.arguments.get("query")
                items = [SimpleNamespace(**item) for item in result.get("items", [])]
                item_call_results.append((result.get("match_stage"), items))

            elif call.name == "lookup_collection_schedule":
                detected_ward = detected_ward or call.arguments.get("ward")
                detected_town = detected_town or call.arguments.get("town")
                for sched in result.get("schedules", []):
                    collected_schedules.append(SimpleNamespace(**sched))
                    if sched.get("source_url"):
                        sources.append(sched["source_url"])
                if result.get("district_note"):
                    district_notes.append(result["district_note"])

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

    collected_items, match_stage = _narrow_item_hits(item_call_results, detected_item, user_content)
    for item in collected_items:
        if item.source_url:
            sources.append(item.source_url)

    if collected_items and not collected_schedules:
        # モデルが lookup_collection_schedule を自発的に呼ばなかった場合の
        # 安全策。「事前予約が必要な品目でも収集曜日そのものは必ず案内する」
        # という要件(STAGE03採点フィードバック由来、router.py側は既に対応
        # 済み)を、モデルの自発的なツール呼び出しに頼らずコード側で保証する
        # (4-3確度ゲートと同じ「小型モデルへの指示だけでは守れない」という
        # 設計思想)。2026-09-10、実機で「八幡西区黒崎でソファを処分する
        # 方法」に対し、tools.py 側の地区名解決(`_narrow_item_hits` 直後の
        # 修正)を入れてもなお、モデルが lookup_collection_schedule を
        # 一度も呼ばずに終わる(＝ツール自体が実行されない)ケースを確認した。
        filled = _fill_missing_schedule(
            prompt, collected_items[0], detected_ward, detected_town
        )
        if filled is not None:
            schedules, note, ward, town = filled
            for sched in schedules:
                collected_schedules.append(SimpleNamespace(**sched))
                if sched.get("source_url"):
                    sources.append(sched["source_url"])
            detected_ward = detected_ward or ward
            detected_town = detected_town or town
            if note:
                district_notes.append(note)

    if not (collected_items or collected_schedules or collected_explanations):
        # 何も拾えなければ、呼び出し側で決定論的経路にフォールバックさせる。
        return None

    return _build_routed_answer(
        query=detected_item or prompt,
        items=collected_items,
        schedules=collected_schedules,
        explanations=collected_explanations,
        notices=collected_notices,
        district_notes=district_notes,
        sources=sources,
        match_stage=match_stage,
        detected_item=detected_item,
        detected_ward=detected_ward,
        detected_town=detected_town,
    )


def _fill_missing_schedule(
    prompt: str,
    item: SimpleNamespace,
    existing_ward: str | None,
    existing_town: str | None,
) -> tuple[list[dict], str | None, str | None, str | None] | None:
    """品目は見つかったが収集日が一件も無いとき、区名・町名をプロンプトから
    直接抜き出してコード側で収集日を引く(router.py の `extract_region`/
    `_match_district` を再利用。「黒崎」のような丁目省略の地区名も解決できる)。

    出し方が収集曜日を持たない区分(回収ボックス等、`_CURBSIDE_WASTE_TYPES`
    以外)なら曜日という概念自体が無いので何もしない。
    """
    disposal_method = getattr(item, "disposal_method", None)
    if disposal_method not in _CURBSIDE_WASTE_TYPES:
        return None

    ward = existing_ward
    town = existing_town
    if not ward or not town:
        r_ward, r_town = extract_region(prompt)
        ward = ward or r_ward
        town = town or r_town
        if ward and not town:
            district = _match_district(ward, prompt)
            if district:
                town = district[0]
    if not ward or not town:
        return None

    result = execute_tool(
        "lookup_collection_schedule",
        {"ward": ward, "town": town, "waste_type": disposal_method},
    )
    schedules = result.get("schedules") or []
    if not schedules:
        return None
    return schedules, result.get("district_note"), ward, town


_MATCH_STAGE_RANK = {"exact": 3, "partial": 2, "vector": 1}


def _narrow_item_hits(
    item_call_results: list[tuple[str | None, list[SimpleNamespace]]],
    detected_item: str | None,
    prompt_text: str,
) -> tuple[list[SimpleNamespace], str | None]:
    """search_waste_item の全呼び出し結果から、最終回答の context に載せてよい
    品目を絞り込む。

    `router.py`(決定論的経路)は `_search_item` が1回しか呼ばれず、しかも
    `item_match == "analogy"` のときは `_override_analogy_hits`/`_analogy_pick`
    で候補を1件に絞ってから context・`_has_conditional_notes` の判定に渡す
    (line 996〜1018, 1226〜1238)。Agentic 経路はモデル自身がツールを何度でも
    呼べるため、素朴に全呼び出しの `items` を集約すると、質問された品目とは
    無関係な候補(モデルが確認のために追加で呼んだ検索の結果)の備考
    ―特に材質・サイズによる分別区分の条件文― まで context に混入し、
    `_has_conditional_notes` が誤って発火して「この品目にはサイズ条件がある」
    という事実に基づかない回答を生む(2026-09-10、実機で「たんす」の質問に
    「かご」等の50センチ条件が紛れ込む形で再現・確認済み)。

    決定論的経路と同じ絞り込みをここでも適用する:
    1. match_stage が最も良い(exact > partial > vector)呼び出しの結果だけを
       採用し、それより弱い呼び出しの結果は捨てる(モデルが探索的に呼んだ
       追加の検索が本命の結果を汚さないようにする)。
    2. その中に品目名が `detected_item` と正規化一致するものがあれば、それ
       1件だけに絞る(完全一致)。
    3. 複数件のままなら `_override_analogy_hits`/`_analogy_pick` で1件に
       絞る(router.py と同じ規則)。
    """
    if not item_call_results:
        return [], None

    best_stage: str | None = None
    best_items: list[SimpleNamespace] = []
    best_rank = -1
    for stage, items in item_call_results:
        if not items:
            continue
        rank = _MATCH_STAGE_RANK.get(stage, 0)
        if rank > best_rank:
            best_rank = rank
            best_stage = stage
            best_items = items

    if not best_items:
        return [], None
    if len(best_items) == 1:
        return best_items, best_stage

    if detected_item:
        q = _norm(detected_item)
        named = [it for it in best_items if _norm(getattr(it, "item_name", "")) == q]
        if named:
            return named[:1], best_stage

    narrowed = _override_analogy_hits(detected_item or prompt_text, best_items)
    if len(narrowed) == 1:
        return narrowed, best_stage
    pick = _analogy_pick(narrowed, detected_item)
    return ([pick] if pick is not None else narrowed[:1]), best_stage


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
    district_notes: list[str] | None = None,
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
        if any(s.waste_type == "粗大ごみ" for s in schedules):
            # router.py の同じ注記と揃える(2026-09-10 追加)。粗大ごみは
            # 事前申込制だが、申込の説明だけで済ませて収集曜日そのものを
            # 省略されると「具体的な収集曜日を漏らす」という別の問題になる。
            # 曜日の明記を先に、申込の要否は追加条件として明示する順で書く
            # (どちらも必須、二択にしない)。
            lines.append(
                "  ※ 事前申込制。上記の収集曜日(申込後にこの曜日に収集される)を"
                "必ず回答に明記すること。申込の説明だけで済ませて曜日を省略しない。"
                "その上で、申込なしにはこの曜日へ出せないこと、"
                "事前に粗大ごみ受付センター(電話: 093-513-3005)へ電話、"
                "またはインターネットで申込む必要があることも併せて伝える。"
            )
        for note in district_notes or ():
            lines.append(f"  ※ {note}")
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
    district_notes: list[str] | None = None,
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
    ans.context = _build_context_text(items, schedules, explanations, notices, district_notes)
    return ans
