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
    _AFTER_NAME_OK,
    _CURBSIDE_WASTE_TYPES,
    _KANA_ONLY_RE,
    _analogy_pick,
    _base_name,
    _listed_item_extras,
    _match_district,
    _override_analogy_hits,
    _resolve_alias_hit,
    _shared_run_len,
    category_procedure_intent,
    extract_region,
)
from app.rag.tool_call_parser import ToolCall, parse_tool_calls
from app.rag.tools import TOOL_SCHEMAS, _serialize_item, execute_tool
from app.rag.types import RoutedAnswer

_SYSTEM_PROMPT = (
    "あなたは北九州市のごみ分別・収集日を案内するアシスタントの検索エージェントです。\n"
    "ユーザーの質問に直接答えず、まずは提供されたツールを使って必要な情報を"
    "集めることに専念してください。\n"
    "この会話では、現在の発話だけでなく、それまでの会話履歴も前提にして"
    "検索条件を判断してください。過去の品目・地区が現在の発話で省略されて"
    "いる場合は引き継ぎ、「追加」「も」「一緒に」は既存の条件を保持して追加、"
    "「変更」「訂正」「やっぱり」「ではなく」は該当する過去の条件を今回の内容"
    "に置き換えてください。品目と地区が会話全体で揃ったら、両方を使って"
    "分別・出し方・収集日を検索してください。\n"
    "- 品目の分別方法を調べるには search_waste_item を使う。"
    "現在の発話にもそれまでの利用者の発言にも品目名として解釈できる語が"
    "一切含まれていない場合は、品目名を作り出して呼んではならない。"
    "その場合は品目名が分からないものとして扱う。"
    "「粗大ごみってどうやって出すんですか」「粗大ごみの申込み方法は」"
    "「古紙を出したいです」のように、特定の品目ではなくごみ種別そのものの"
    "出し方・手続きを尋ねる質問でも呼んではならない"
    "(「ごみ箱」等、無関係な品目名を検索キーにしないこと)。\n"
    "- 一度に複数の品目を尋ねられた場合(「たんすと布団と、あと段ボール」等)は、"
    "列挙された品目を一つも省かず、それぞれについて search_waste_item を呼ぶ。\n"
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


def _message_text(message: dict) -> str:
    """会話履歴のメッセージから本文テキストだけを取り出す(画像付きのターンは
    `content` が `[{"type": "image", ...}, {"type": "text", "text": ...}]` の
    リスト形式になる)。"""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def build_history_query(history: list[dict], prompt: str) -> str | None:
    """会話履歴の利用者発言を、古い順にこのターンの発話へ前置した文字列。

    決定論的経路(`router._blocking_route`)は会話履歴を受け取らないため、
    「え、知らなかったです。じゃあどうすればいいんですか？」のように品名を
    省いた追問では品目を特定できず6-3に落ちる。過去の利用者発言を一緒に
    渡して品名の省略を補う(`app/server.py._route_safely` から呼ばれる。
    2026-09-10 第6回採点回帰)。

    直前の3発言までに限るのは、話題が変わった古い発言まで混ぜると、無関係な
    品名を拾って別の回答になるため。利用者発言が1つも無ければ None を返し、
    呼び出し側は引き直しを行わない。
    """
    past = [
        text
        for text in (_message_text(m) for m in history if m.get("role") == "user")
        if text.strip()
    ]
    if not past:
        return None
    return "\n".join([*past[-3:], prompt])


def _query_grounded_in_text(query: str, text: str) -> bool:
    """モデルが呼んだ search_waste_item の `query` が、実際にこのターンの
    発言に基づくものかを検査する。

    2026-09-10、B1実機の複数ターン会話で「小倉北区浅野1丁目です」
    (品目名を一切含まない、地域名だけの発言)に対し、モデルが
    search_waste_item を実在しない品目名(会話に一度も出ていない語)で
    呼び、それを根拠に無関係な回答を生成する事例を確認した
    (要件5-1「推測せず確認する」への違反)。プロンプトへの指示だけでは
    小型モデルが必ず守るとは限らない(4-3確度ゲート等、既存の設計判断と
    同じ理由)ため、コード側でも検証する。

    発言中に `query` がそのまま(正規化後)部分文字列として現れるか、2文字
    以上の共通する連続部分列があれば「本文から拾った」とみなす。前者を先に
    見るのは、「傘」のような1文字の品目名だと連続部分列の長さが1にしか
    ならず、後者の条件だけでは常に弾かれてしまうため(2026-09-10 レビュー
    指摘で発覚)。

    ただし、ひらがなだけの1文字(「す」等)は、部分文字列として容易に
    見つかってしまう(「です」「ます」等の語尾・助詞にほぼ必ず含まれる)ため
    部分文字列一致だけでは弾けない。2026-09-10、「小倉北区浅野1丁目です」
    (品目名を一切含まない発言)から、モデルが query="す" で
    search_waste_item を呼び、「す」が発言に含まれることを理由に
    採用されてしまう再発を確認した(レビュー指摘)。ひらがな1文字はこの
    判定より先に弾く(「傘」のような漢字1文字の品目名はこの対象外)。
    どちらの条件も満たさなければ、本文には無い語をモデルが作り出したとみなし、
    結果を採用しない。
    """
    q, t = _norm(query), _norm(text)
    if not q:
        return False
    if _KANA_ONLY_RE.match(q) and len(q) < 2:
        return False
    if q in t:
        return True
    return _shared_run_len(q, t) >= 2


async def route_agentically(
    prompt: str,
    engine: Any,
    build_prompt: Callable[..., dict],
    sampling_params: Any,
    image_item_name: str | None = None,
    max_iterations: int = _MAX_ITERATIONS_DEFAULT,
    history: list[dict] | None = None,
) -> RoutedAnswer | None:
    """Function Calling でツールを呼びながら検索する。

    `history` は同じ `session_id` の過去の user/assistant ターン。検索条件の
    省略・追加・変更・訂正をFunction Calling側でも解釈できるよう、現在の
    発話と一緒に検索エージェントへ渡す。

    成功すれば `RoutedAnswer` を、失敗すれば None を返す(呼び出し側は None の
    とき決定論的経路にフォールバックすること)。この関数自体は例外を投げない
    — 想定外の失敗もすべて捕まえて None を返す。
    """
    try:
        return await _run_loop(
            prompt, engine, build_prompt, sampling_params, image_item_name,
            history, max_iterations
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
    history: list[dict] | None,
    max_iterations: int,
) -> RoutedAnswer | None:
    user_content = prompt
    if image_item_name:
        user_content = f"(画像から推定した品目名: {image_item_name})\n\n{prompt}"

    # 特定品目ではなく、ごみ種別そのものの手続き・収集日を尋ねる質問
    # (要件5-1、`router.category_procedure_intent` 参照)。この経路でも
    # 品目検索の結果を一切信用しない(下の search_waste_item 分岐で判定)。
    _pre_ward, _pre_town = extract_region(prompt)
    category_intent = (
        None if image_item_name
        else category_procedure_intent(prompt, _pre_ward, _pre_town)
    )

    # search_waste_item の query が「利用者が実際に言った語」かを検査する対象
    # (5-1 ガード、`_query_grounded_in_text`)。STAGE05 では過去ターンの品目を
    # 会話履歴から引き継いで検索させる(「ソファを捨てたい」→「え、じゃあ
    # どうすれば」)ため、今回の発話だけを見ると引き継いだ品目名が必ず弾かれて
    # しまう。このセッションで利用者が話した全ターンを対象にする。
    # assistant の発言は含めない — 案内文中の例示(6-3 の定型文にある
    # 「電子レンジ」「ソファ」等)を根拠に、利用者が一度も言っていない品目を
    # 「本文から拾った」と誤認しないため。
    grounding_text = "\n".join(
        [_message_text(m) for m in (history or []) if m.get("role") == "user"]
        + [user_content]
    )

    # `server.sessions` stores only user/assistant turns. Copy the list so the
    # tool-call loop can append its own assistant/tool messages without
    # modifying the session history that is concurrently used by `/chat`.
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *(list(history) if history else []),
        {"role": "user", "content": user_content},
    ]

    # search_waste_item の呼び出しごとの結果(match_stage, items)。ループを跨いで
    # 単純に collected_items へ extend しない理由は `_narrow_item_hits` 参照。
    item_call_results: list[tuple[str | None, list[SimpleNamespace]]] = []
    # 要件6-4: モデルが呼んだ search_waste_item ごとの検索過程(却下された
    # 呼び出しも含む)。`/trace` でエージェント経路の判断を事後に追えるよう
    # `RoutedAnswer.search_trail` に載せる。
    search_trails: list[dict] = []
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
                query = call.arguments.get("query") or ""
                trail_entry = {"query": query, "attempts": result.get("trail", [])}
                if category_intent is not None or not _query_grounded_in_text(query, grounding_text):
                    # このターンが特定品目ではなくごみ種別そのものの質問
                    # (category_intent)か、利用者がこのセッションで一度も
                    # 言っていない品目名をモデルが作り出した(5-1 ガード)。
                    # ツール応答はループの文脈には残すが、検索結果としては
                    # 採用しない。理由も記録して `/trace` で「なぜ却下したか」
                    # まで追えるようにする。
                    trail_entry["rejected_reason"] = (
                        "ごみ種別そのものの質問(category_intent)"
                        if category_intent is not None
                        else "会話に基づかないクエリ(5-1 ガード)"
                    )
                    search_trails.append(trail_entry)
                    continue
                search_trails.append(trail_entry)
                if not detected_item:
                    detected_item = query
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
                    if category_intent is not None:
                        # 種別そのものの質問では、埋め込み類似度だけで
                        # 無関係なページ(「ふれあい収集」等)が高スコアで
                        # 混入しうる(router.py.general_info_for_category と
                        # 同じ理由)。本文に種別名が実際に含まれているものだけ
                        # 信頼する。
                        text = chunk.get("text") or ""
                        if category_intent[0] not in text:
                            continue
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

    # 質問文に複数の品目が列挙されていても、モデルが全品目を検索したとは限らない
    # (小型モデルは列挙の一部しか呼ばないことがある)。コード側で列挙を拾い直し、
    # 回答から品目が抜け落ちないようにする(router._blocking_route と同じ規則。
    # 2026-09-10 第6回採点回帰)。
    if collected_items and match_stage == "exact":
        exclude = {_norm(getattr(it, "item_name", "")) for it in collected_items}
        if detected_item:
            exclude.add(_norm(detected_item))
        extras = _listed_item_extras(user_content, exclude)
        if extras:
            collected_items = collected_items + [
                SimpleNamespace(**_serialize_item(r)) for r in extras
            ]
        if len(collected_items) > 1:
            # `_build_routed_answer` は `query`(=detected_item)と品目名を突き合わせて
            # exact/analogy を決めるため、detected_item が収集済みの品目に含まれて
            # いないと analogy に落ち、context が `_analogy_pick` で1件に絞られて
            # 他の品目が回答から消える。複数品目の質問では先頭の品目を代表にする。
            detected_item = (
                getattr(collected_items[0], "item_name", None) or detected_item
            )
        else:
            # 言い換え表(`_CANONICAL_ALIASES`)の解決で名前が変わった場合も同じ問題が
            # 起きる。モデルが query="硫酸" で検索し、ツールが包括行
            # 「薬品(家庭医薬品でないもの)」を返すと、名前が一致しないため
            # analogy 扱いになり、「類似する『薬品…』に準じると」という
            # ためらいのある言い回しと、類推用のプロンプト規則になってしまう
            # (B1実機のスモークテストで確認。決定論的経路には同じ解決の
            # 引き上げ `_resolve_alias_hit` が入っている)。
            # 完全一致の段で確定しているときだけ、解決先の正式名に合わせる。
            alias_hit = _resolve_alias_hit(
                user_content,
                [detected_item or ""],
                list(collected_items),
            )
            if alias_hit is not None:
                detected_item = (
                    getattr(alias_hit, "item_name", None) or detected_item
                )

    if collected_items:
        # 品目が確定している質問では、無関係なページのチャンクを根拠から外す。
        collected_explanations = _filter_explanations_to_items(
            collected_explanations, collected_items
        )

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

    if category_intent is not None and not collected_schedules:
        # 種別そのものの収集日質問(「家庭ごみの日を教えてください」)。
        # search_waste_item は上のガードで一切採用しないので、品目を介さず
        # 種別名を直接 waste_type として収集日を引く。
        category_waste_type, category_schedule_like = category_intent
        if category_schedule_like:
            filled = _fill_schedule_for_waste_type(
                prompt, category_waste_type, detected_ward, detected_town
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

    # コードレビュー(PR #45)で発見: 品目・収集日がどちらもヒットせず、
    # search_general_info だけが何か(弱関連かもしれない説明チャンク)を拾った
    # 場合、_build_routed_answer は found=True を確定させてしまい、STAGE06
    # 6-3(完全空振りの固定拒絶)が `_blocking_route` 側にしか無いため素通り
    # していた(4-3 確度ゲートが防ぐはずの「モデルが弱い根拠から自由に
    # 生成する」失敗モードが、Agentic 経路からだけ再現する)。
    #
    # 「search_general_info だけを使う一般説明質問」(ごみステーションの
    # ルール等、ベクトルRAGパスが正規に担当する範囲)は壊さないよう、
    # search_waste_item が一度も呼ばれていない(＝最初から一般質問として
    # 扱われた)場合はそのまま found=True にする。search_waste_item が
    # 呼ばれたのに品目が確定しなかった場合(＝品目質問のつもりが失敗した)
    # だけ、explanations があっても解決済み扱いにせず決定論的経路へ回す
    # (そちらの `_blocking_route` 末尾フックが6-3を正しく発火させる)。
    item_search_attempted_and_failed = bool(item_call_results) and not collected_items
    if not (collected_items or collected_schedules) and (
        not collected_explanations or item_search_attempted_and_failed
    ):
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
        search_trail=search_trails,
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
    return _fill_schedule_for_waste_type(prompt, disposal_method, existing_ward, existing_town)


def _fill_schedule_for_waste_type(
    prompt: str,
    disposal_method: str | None,
    existing_ward: str | None,
    existing_town: str | None,
) -> tuple[list[dict], str | None, str | None, str | None] | None:
    """`_fill_missing_schedule` の中身(品目に依存しない部分)。ごみ種別
    そのものの質問(`category_procedure_intent`)にも同じ解決ロジックを
    再利用するため、品目から切り離してある。"""
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


def _items_mentioned_in_text(
    text: str, items: list[SimpleNamespace]
) -> list[SimpleNamespace]:
    """`text` の中に品名(または括弧を外した主名)が主辞として現れている候補を
    全部返す。

    多品目の質問(「たんすと古い布団と、あと段ボールがたくさんあります」)では、
    モデルが品目ごとに `search_waste_item` を呼んでいても、`_narrow_item_hits`
    が1件に絞ると残りの品目が回答から抜け落ちる。質問文に実際に現れている候補は
    根拠があるので全部残す。

    境界検査は `router._pick_embedded_item` と同じ規則(`_AFTER_NAME_OK`)を使う。
    これが無いと「テレビ台」の中の「テレビ」を拾ってしまう。
    """
    norm_text = _norm(text)
    found: list[SimpleNamespace] = []
    for it in items:
        full = _norm(getattr(it, "item_name", ""))
        if not full:
            continue
        for name in (full, _base_name(full)):
            if len(name) < 2:
                continue
            idx = norm_text.find(name)
            if idx == -1:
                continue
            if _AFTER_NAME_OK.match(norm_text[idx + len(name):]):
                found.append(it)
                break
    return found


def _filter_explanations_to_items(
    explanations: list[dict], items: list[SimpleNamespace]
) -> list[dict]:
    """一般説明チャンクのうち、見つかった品目に関係するものだけを残す。

    モデルは `search_general_info` を探索的に呼ぶため、無関係なページ
    (「ふれあい収集」「ごみステーションの散乱対策」等)のチャンクが context に
    混ざり、小型モデルがそれを回答に混ぜてしまう。品目が確定している質問では、
    品目名(括弧を外した主名)か、その品目の分別区分に触れているチャンクだけを
    根拠として残す。

    品目名だけを条件にすると「粗大ごみの申込み方法」のような、品目名を含まない
    が的確なページまで落ちてしまうため、分別区分も条件に含める。品目が
    見つかっていない一般質問は対象外(絞り込む根拠が無い)。
    """
    if not items:
        return explanations
    names: list[str] = []
    waste_types: list[str] = []
    for it in items:
        full = _norm(getattr(it, "item_name", ""))
        for name in (full, _base_name(full)):
            if len(name) >= 2 and name not in names:
                names.append(name)
        method = getattr(it, "disposal_method", "") or ""
        if method and method not in waste_types:
            waste_types.append(method)

    kept: list[dict] = []
    for entry in explanations:
        text = _norm(entry.get("text") or "")
        if any(name in text for name in names) or any(
            wt in text for wt in waste_types
        ):
            kept.append(entry)
    return kept


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

    if best_stage == "exact":
        # 質問文に複数の品名がそのまま現れているなら、それは品目の列挙
        # (「たんすと古い布団と、あと段ボール」)。完全一致の段でだけ、現れて
        # いる候補を全部残す(部分一致・ベクトルの段では質問の品目が一意でなく、
        # 複数残すとモデルが各候補の出し方を混ぜて答えてしまう)。
        mentioned = _items_mentioned_in_text(prompt_text, best_items)
        if len(mentioned) > 1:
            return mentioned, best_stage

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


def _filter_schedules_to_known_items(
    items: list[SimpleNamespace], schedules: list[SimpleNamespace]
) -> list[SimpleNamespace]:
    """`schedules` を、実際に見つかっている品目の `disposal_method` に
    対応するものだけへ絞り込み、かつ重複を除く。

    2026-09-14 レビュー指摘: モデルが `lookup_collection_schedule` を
    見つかった品目とは無関係な `waste_type` で(自発的に、または混乱して)
    呼んだ場合、そのまま context に混ざると「市が収集しないもの」の品目
    (洗濯機・冷蔵庫等)に無関係な粗大ごみ収集日が付いてしまう。`items` が
    空(品目とは無関係の収集日質問)のときはこの絞り込みをしない。
    """
    deduped: list[SimpleNamespace] = []
    seen: set[tuple] = set()
    known_types = {it.disposal_method for it in items} if items else None
    for s in schedules:
        if known_types is not None and s.waste_type not in known_types:
            continue
        key = (s.ward, s.town, s.waste_type, s.weekday)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped


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

    schedules = _filter_schedules_to_known_items(items, schedules)
    if schedules:
        lines = ["## 収集日の検索結果(エージェントが検索)"]
        for s in schedules:
            lines.append(f"- {s.ward}{s.town} / {s.waste_type}: {s.weekday}")
            if s.waste_type == "粗大ごみ":
                # 注記はこの行(粗大ごみ)にだけ付ける — 他のごみ種別
                # (家庭ごみ等)の収集日に誤って適用されないようにするため
                # (2026-09-14 レビュー指摘: 毛布(家庭ごみ)が布団(粗大ごみ)
                # の事前申込条件を引き継いでしまう不具合の再発防止。以前は
                # ブロック全体に1回だけ付けていた)。粗大ごみは事前申込制だが、
                # 申込の説明だけで済ませて収集曜日そのものを省略されると
                # 「具体的な収集曜日を漏らす」という別の問題になる。曜日の
                # 明記を先に、申込の要否は追加条件として明示する順で書く
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
    search_trail: list[dict] | None = None,
) -> RoutedAnswer:
    q = _norm(query)
    item_match = "none"
    if items:
        item_match = "exact" if any(_norm(it.item_name) == q for it in items) else "analogy"

    ans = RoutedAnswer(
        found=True,
        search_trail=list(search_trail or []),
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
