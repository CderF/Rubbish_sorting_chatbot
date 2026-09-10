"""STAGE06 6-4: `var/stage4_trace.jsonl` を WebUI 用に読み取る(閲覧専用)。

`app/server.py` の `/trace` エンドポイントから呼ぶ。ファイルの読み取りと整形
だけを行い、DB・LLM・ネットワークには触れないので、`app/rag/session_state.py`
と同じくオフラインで全分岐を単体テストできる。

1行 = 1リクエストで、`router.build_trace_record()` の出力に `server._write_trace`
が `session_id` と `final_response` を足したもの(JSON)。
"""

from __future__ import annotations

import json
import os

# router.MATCH_STAGE_LABELS と揃える(重い依存を持ち込まないためここに複製)。
_MATCH_STAGE_LABELS = {
    "exact": "完全一致",
    "partial": "部分一致",
    "vector": "類似検索",
    "analogy": "類推",
}

# safety_gate 値のラベル。
_SAFETY_GATE_LABELS = {
    "6-1a": "6-1 担当範囲外",
    "6-2": "6-2 危険処分の誘導",
    "6-3": "6-3 完全空振り",
}


def read_trace_tail(path: str, limit: int = 100) -> "list[dict]":
    """`path` の末尾から最大 `limit` 行を、新しい順(末尾が先頭)で返す。

    ファイルが無ければ空リスト。壊れた行(JSON でない・オブジェクトでない)は
    黙って読み飛ばす — トレースはベストエフォートの追記ログなので、途中に
    不完全な行が混ざり得る。
    """
    if limit <= 0 or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return []

    rows: "list[dict]" = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        if len(rows) >= limit:
            break
    return rows


def _truncate(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + "…（省略）"


def summarize_trace_rows(rows: "list[dict]") -> "list[dict]":
    """トレース行を `/trace` の表に出す形へ整える。

    生の prompt / 回答文もそのまま含める(閲覧は `require_login` 必須。
    STAGE06 §2.4 の注意点)。欠けているキーは古い形式の行なので既定値で埋める。
    """
    out: "list[dict]" = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        gate = r.get("safety_gate")
        stage = r.get("item_match_stage")
        out.append(
            {
                "session_id": r.get("session_id"),
                "prompt": _truncate(str(r.get("prompt") or "")),
                "ward": r.get("detected_ward"),
                "town": r.get("detected_town"),
                "item": r.get("detected_item"),
                "item_candidates": list(r.get("item_candidates") or []),
                "match_stage": stage,
                "match_stage_label": _MATCH_STAGE_LABELS.get(stage or "", "該当なし"),
                "item_hit_names": list(r.get("item_hit_names") or []),
                "schedule_hits": r.get("schedule_hits") or 0,
                "missing": list(r.get("missing") or []),
                "timeline_change": bool(r.get("timeline_change")),
                "timeline_item_name": r.get("timeline_item_name"),
                "analogy": r.get("item_match") == "analogy",
                "analogy_unreliable": bool(r.get("analogy_unreliable")),
                "safety_gate": gate,
                "safety_gate_label": (
                    _SAFETY_GATE_LABELS.get(gate, gate)
                    if gate
                    else ("固定回答" if r.get("canned_response") else None)
                ),
                "canned_response": bool(r.get("canned_response")),
                "sources": list(r.get("sources") or []),
                "final_response": _truncate(str(r.get("final_response") or "")),
            }
        )
    return out
