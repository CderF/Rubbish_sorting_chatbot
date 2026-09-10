"""Agentic RAG のツール定義とディスパッチャ。

LLM(Function Calling)が呼び出す「ツール」は、既存のデータ層・ベクトル
RAGパスの関数をそのままラップしたものだけで、検索ロジック自体はここでは
再実装しない(`app/data/lookup.py` / `app/rag/vector_store.py` /
`app/notices/lookup.py` が唯一の実装)。

`documents/CONTEXT.md` の語彙にそのまま対応させている:
  - search_waste_item          -> 早見表パス
  - lookup_collection_schedule -> 収集日パス
  - search_general_info        -> ベクトルRAGパス(vector_search_explanations、
    これまで router.py/server.py のどこからも呼ばれていなかった未接続の
    機能。Function Calling で初めて実戦投入する)
  - lookup_item_change_notice  -> お知らせパス(STAGE04 4-2)

画像分類パスはここではツール化しない。画像が来たら常に品目分類が必要で
選択の余地が無いため、呼び出し側(`app/rag/agent.py`)が事前に
`classify_image_to_item_name` を実行し、その結果を品名として他のツールに
渡す(`router.handle_query` の既存の `image_item_name` 優先ロジックと同じ
考え方)。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

from app.data.lookup import lookup_collection_schedule, lookup_item_exact, lookup_item_partial
from app.notices.lookup import pending_change
from app.rag.router import _district_groups, _uniform_district_schedule
from app.rag.vector_store import vector_search_explanations, vector_search_items

# --- OpenAI-style function-calling スキーマ ---------------------------------
# `processor.apply_chat_template(messages, tools=TOOL_SCHEMAS, ...)` にそのまま
# 渡す想定。Hugging Face Hub上の `Qwen/Qwen3-VL-8B-Instruct` の
# `chat_template.json` を取得して確認済み(2026-09-09): `tools` はリストの
# 各要素をそのまま `<tools>` タグ内にJSONで列挙するだけなので、このリストの
# 各要素がOpenAI形式(`{"type": "function", "function": {...}}`)かどうかは
# テンプレート機構上は問わない。Qwen公式ドキュメント・vLLMのHermes tool
# parserとの整合を優先し、この形式で統一している。テンプレートの実地確認は
# 済んでいるが、モデルが実際にこの形式のfunction定義を安定して解釈するかは
# 実機のGPUでの推論確認が必要(このマシンではVLM本体の重みでの推論は未検証)。

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_waste_item",
            "description": (
                "ごみの品目名から、分別区分(出し方)と備考(持ち込み先・"
                "事前申込の要否・サイズ条件等)を検索する(早見表パス)。"
                "利用者が特定の品物の捨て方を聞いているときに使う。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索したい品目名(例: 'テレビ'、'乾電池')",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_collection_schedule",
            "description": (
                "区名・町名・ごみ種別から、収集曜日を検索する(収集日パス)。"
                "ごみ種別が分かっている場合はそれで絞り込むと精度が上がる。"
                "町名は丁目まで分からなくてもよい(「黒崎」のように丁目を"
                "省略した地区名でも、地区内の全丁目で収集日が一致する場合は"
                "解決できる)。区名も町名も一切分からない場合だけこのツールを"
                "呼ばず、利用者に尋ねること。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ward": {"type": "string", "description": "区名(例: '小倉北区')"},
                    "town": {
                        "type": "string",
                        "description": (
                            "町名・丁目、または丁目を省略した地区名"
                            "(例: '浅野1丁目'、または単に '黒崎')。省略不可。"
                        ),
                    },
                    "waste_type": {
                        "type": "string",
                        "description": (
                            "ごみ種別(例: '家庭ごみ'、'プラスチック'、'ペットボトル'、"
                            "'かん・びん'、'粗大ごみ')。分かっていれば指定する。"
                            "不明なら省略してよい(その区・町の全種別が返る)。"
                        ),
                    },
                },
                "required": ["ward", "town"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_general_info",
            "description": (
                "特定の品目や収集日ではなく、ごみステーション・指定ごみ袋・"
                "ごみ出し支援制度など一般的な説明を検索する(ベクトルRAGパス)。"
                "search_waste_item / lookup_collection_schedule のどちらにも"
                "当てはまらない一般的な質問のときに使う。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索したい内容(自然文でよい)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_item_change_notice",
            "description": (
                "指定した品目に、今後(または最近)出し方が変わる予定がないかを"
                "確認する(お知らせパス)。search_waste_item で見つかった品目に"
                "ついて、変更予定の有無を必ず確認すること。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string", "description": "確認したい品目名"},
                },
                "required": ["item_name"],
            },
        },
    },
]


# --- 実行 --------------------------------------------------------------


def _serialize_item(item: Any) -> dict:
    return {
        "item_name": getattr(item, "item_name", None),
        "disposal_method": getattr(item, "disposal_method", None),
        "notes": getattr(item, "notes", None) or "",
        "source_url": getattr(item, "source_url", None),
    }


def _tool_search_waste_item(query: str) -> dict:
    query = (query or "").strip()
    if not query:
        return {"error": "query is required"}

    exact = lookup_item_exact(query)
    if exact:
        return {"match_stage": "exact", "items": [_serialize_item(exact)]}

    partial = lookup_item_partial(query)[:5]
    if partial:
        return {"match_stage": "partial", "items": [_serialize_item(i) for i in partial]}

    vector = vector_search_items(query, top_k=3)
    return {
        "match_stage": "vector" if vector else "none",
        "items": [_serialize_item(i) for i in vector],
    }


def _tool_lookup_collection_schedule(
    ward: str, town: str, waste_type: str | None = None
) -> dict:
    ward = (ward or "").strip()
    town = (town or "").strip()
    if not ward or not town:
        return {"error": "ward and town are both required"}

    hits = lookup_collection_schedule(ward, town, waste_type or None)
    district_note: str | None = None
    if not hits:
        # 丁目を省略した地区名(「黒崎」等)の可能性を試す。地区内の全ての町
        # (丁目)で収集日が完全に一致する場合に限り、それをそのまま使う
        # (`router.py` の `_match_district`/`_uniform_district_schedule` と
        # 同じ規則。エージェント経路にはこの絞り込みが元々無く、モデルが
        # `town="黒崎"` のように丁目無しで呼んだ場合に空振りしたまま収集日を
        # 一切案内しない不具合があった。2026-09-10、実機で
        # 「八幡西区黒崎でソファを処分する方法」に再現・確認済み)。
        towns = _district_groups(ward).get(town)
        if towns:
            uniform = _uniform_district_schedule(ward, towns, waste_type or None)
            if uniform:
                hits = uniform
                district_note = (
                    f"「{town}」は丁目の指定がなかったが、地区内の全ての町"
                    f"({'・'.join(towns)})で収集日が一致することを確認済み。"
                    "上記の収集日をそのまま案内してよい"
                    "(特定の1丁目だけを見て答えたわけではない)。"
                )

    if not hits:
        return {"schedules": []}
    result: dict = {
        "schedules": [
            {
                "ward": h.ward,
                "town": h.town,
                "waste_type": h.waste_type,
                "weekday": h.weekday,
                "source_url": h.source_url,
            }
            for h in hits
        ]
    }
    if district_note:
        result["district_note"] = district_note
    return result


def _tool_search_general_info(query: str) -> dict:
    query = (query or "").strip()
    if not query:
        return {"error": "query is required"}

    chunks = vector_search_explanations(query, top_k=3)
    return {
        "results": [
            {
                "text": c["text"],
                "source_url": c["source_url"],
                "source_page_title": c["source_page_title"],
                "score": c["score"],
            }
            for c in chunks
        ]
    }


def _tool_lookup_item_change_notice(item_name: str) -> dict:
    item_name = (item_name or "").strip()
    if not item_name:
        return {"error": "item_name is required"}

    tl = pending_change(item_name, date.today())
    if tl is None:
        return {"has_pending_change": False}

    ref = date.today()
    now_seg = tl.segment_on(ref)
    future = [s for s in tl.segments if s.valid_from and s.valid_from > ref]
    return {
        "has_pending_change": True,
        "current": None
        if now_seg is None
        else {
            "disposal_method": now_seg.disposal_method,
            "valid_until": now_seg.valid_until.isoformat() if now_seg.valid_until else None,
            "source_url": now_seg.source_url,
        },
        "future": [
            {
                "disposal_method": s.disposal_method,
                "valid_from": s.valid_from.isoformat() if s.valid_from else None,
                "source_url": s.source_url,
            }
            for s in future
        ],
    }


_TOOL_HANDLERS: dict[str, Callable[..., dict]] = {
    "search_waste_item": _tool_search_waste_item,
    "lookup_collection_schedule": _tool_lookup_collection_schedule,
    "search_general_info": _tool_search_general_info,
    "lookup_item_change_notice": _tool_lookup_item_change_notice,
}


def execute_tool(name: str, arguments: dict | None) -> dict:
    """ツール名と引数(LLMが生成したJSON由来)を受け取り、対応する検索関数を
    呼んで結果をdictで返す。

    LLM由来の引数は必須キーの欠落・型違いなど何でも起こりうるので、ここで
    例外を握り潰して `{"error": ...}` を返す(`app/server.py` の
    `_route_safely` と同じ、エージェントループを止めない安全側の方針)。
    """
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return handler(**(arguments or {}))
    except TypeError as exc:
        return {"error": f"invalid arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 — ツール実行の失敗でループを止めない
        return {"error": f"{name} failed: {exc}"}
