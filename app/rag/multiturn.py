"""STAGE05(5-1/5-2)の複数ターン一貫性を保証するセッション横断のマージ層。

背景(2026-09-10、B1実機の生会話6本で検証・再現。詳細は
`documents/stage05_verification.md`):

  `app/server.py` は生の会話履歴をそのままモデルに渡すだけで、検索結果
  (`RoutedAnswer.context`)は「このターンで見つかった分」しか積まない。

  - 複数品目を跨ぐ追加(例:「ソファを捨てたい」→(地区)→「冷蔵庫も
    捨てたいです」)では、今回見つかった品目(冷蔵庫)の検索結果だけが
    システムプロンプトに載り、前ターンで案内済みの品目(ソファ)は検索結果に
    無いため、モデルは『検索結果にない内容を書き足さない』という既存ルール
    (`router._RULES_WITH_CONTEXT`)に従って前の品目を回答から省いてしまう
    (要件5-2「追加なら前の案内を保ったまま足す」への違反)。
  - 地域だけを訂正するターン(例:「やっぱり八幡西区黒崎でした」)では今回の
    発言に品目名が一切出てこないため検索そのものが走らず、モデルは会話履歴
    からの記憶だけで品目名を書き直す→類推の根拠(類似品目名)がターンごとに
    揺れる(実機で「ソファーベッド」→「つっぱり棒」)。

このモジュールは、そのどちらのケースでも「前ターンまでに確立した品目を、
今回のツール群で毎回コード側から再検索し直して合算する」ことで、モデルの
記憶だけに頼らない再現性のある検索結果を作る。`app/rag/agent.py` が既に
持つ複数品目対応の context 組み立て(`_build_context_text`/
`_build_routed_answer`)をそのまま再利用する — この2関数は元々agentic経路が
1ターンで複数回ツールを呼んだ結果を束ねるために作られており、複数ターンを
跨いで束ねる今回の用途にもそのまま使える。

vllm/GPU に依存しない(`app/rag/agent.py` と同じ制約: このセッションは
GPU無し環境でしか開発できないため、テストできる設計を優先する)。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from types import SimpleNamespace

from app.rag.agent import _build_context_text, _build_routed_answer
from app.rag.router import _CURBSIDE_WASTE_TYPES
from app.rag.tools import execute_tool
from app.rag.types import RoutedAnswer

# 「前の品目を置き換える」ことを明示する表層パターン。該当しなければ
# 既定は「保ったまま足す」(要件5-2の既定側)。
#
# 「間違い」「間違え」は末尾に「なく」「ありません」が続くと「間違いなく
# 粗大ごみですか」のように「確実に」という意味の慣用句になり、訂正ではない
# (2026-09-10 レビュー指摘で発覚)。その形だけ除外する。
_CORRECTION_MARKERS = re.compile(
    r"(じゃなく|ではなく|じゃなかった|ではなかった|訂正|やっぱ|"
    r"間違[えい](?!なく|ありません)|"
    r"やめて|取り消|キャンセル)"
)


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()


@dataclass
class SessionState:
    """セッション1件分の「これまでに確立した」状態。

    推測は含まない — `RoutedAnswer` が実際に検索で確定した値
    (`item_hits`/`detected_ward`/`detected_town`)だけを積む。
    """

    items: list[str] = field(default_factory=list)
    """確立した品目名(早見表の正式名。確立順、重複無し)。"""

    ward: str | None = None
    town: str | None = None


def classify_turn(prompt: str) -> str:
    """'correction'(前の品目を置き換える) か 'keep'(追加/維持) かを、
    日本語の表層パターンで判定する。曖昧なときは 'keep'
    (要件5-2の既定は「追加なら保ったまま足す」なので、安全側は保持)。
    """
    return "correction" if _CORRECTION_MARKERS.search(prompt or "") else "keep"


def update_state(state: SessionState, prompt: str, routed: RoutedAnswer) -> None:
    """今回のターンの確定結果(`routed`)でセッション状態を更新する。

    品目名は `routed.detected_item`(生成前のクエリ文字列)ではなく
    `routed.item_hits[*].item_name`(早見表側で確定した正式名)を使う —
    再検索を常に完全一致で行えるようにするため(類推・ベクトル検索の
    揺れを再発させないのが目的)。`item_hits` は現行の設計上ほぼ常に0〜1件
    (agent.py の `_narrow_item_hits`/router.py の類推絞り込みが、1ターンに
    複数ヒットしても最終的に1件へ絞る)だが、`item_hits[0]` だけを見ると
    将来その絞り込みが緩んだ場合に静かに品目を取りこぼすため、全件を見る
    (2026-09-10 レビュー指摘)。

    確度が低い類推(`analogy_unreliable`、4-3確度ゲート対象)は書き込まない
    ― `item_hits` はWebUI表示のために残っているだけで、根拠として信頼できる
    候補ではないため。これをそのまま書き込むと、次のターン以降でも
    信頼できない品目が「確立した品目」として再検索され続けてしまう
    (2026-09-14 レビュー指摘)。
    """
    if routed.detected_ward:
        state.ward = routed.detected_ward
    if routed.detected_town:
        state.town = routed.detected_town

    if routed.analogy_unreliable:
        return

    resolved_names = []
    for hit in routed.item_hits:
        name = _norm(getattr(hit, "item_name", "") or "")
        if name and name not in resolved_names:
            resolved_names.append(name)
    if not resolved_names:
        return

    if classify_turn(prompt) == "correction" or not state.items:
        state.items = resolved_names
        return
    for name in resolved_names:
        if name not in state.items:
            state.items.append(name)


def _refetch_item(name: str) -> dict | None:
    result = execute_tool("search_waste_item", {"query": name})
    hits = result.get("items") or []
    exact = next((h for h in hits if _norm(h.get("item_name") or "") == name), None)
    return exact or (hits[0] if hits else None)


def _refetch_notice(name: str) -> dict | None:
    result = execute_tool("lookup_item_change_notice", {"item_name": name})
    return result if result.get("has_pending_change") else None


def merge_with_state(
    prompt: str, state: SessionState, routed: RoutedAnswer
) -> RoutedAnswer:
    """`state.items` が複数、または今回品目が見つからず前ターンの品目が
    残っている場合に、それらを毎回コード側で再検索して束ねた
    `RoutedAnswer` を作る。

    1品目だけで今回もそれが見つかっている通常時(これまでずっと動いていた
    大半のケース)は `routed` をそのまま返す — 挙動を変えない。
    固定回答(`canned_response`)・安全ゲート(`safety_gate`)発火時も対象外
    (ただし6-3の「品名を特定できない」兜底は例外 — 前ターンの品目が残って
    いれば合流で解決できるため)。
    """
    # 合流しても意味が無い固定回答だけを対象外にする:
    # - 6-1a(担当範囲外)・6-2(危険処分)・ごみ出しルール確定回答
    #   (safety_gate が "6-1a"/"6-2")。
    # - 4-3 類推不可(_ANALOGY_UNRELIABLE_REPLY。canned が立つが safety_gate
    #   は None)。
    # 6-3(品名を特定できない兜底)は対象外にしない — 「え、じゃあどうすれば」
    # のような追問が「品名を教えてください」に落ちず、前の品目を引き継げる。
    if routed.safety_gate in ("6-1a", "6-2"):
        return routed
    if routed.canned_response is not None and routed.safety_gate != "6-3":
        return routed

    carried_names = list(state.items)
    this_turn_name = (
        _norm(getattr(routed.item_hits[0], "item_name", "") or "")
        if routed.item_hits
        else None
    )

    # 2026-09-10 レビュー指摘: 以前は地域(ward/town)か `routed.found` が
    # 立っているときだけ合流させていたが、これだと「品目だけ確立していて
    # 地域はまだ一度も出てきていないセッション」で品目に関する追問
    # (例:「ペットボトルの分別を教えて」→「それは何色の袋に入れますか」)
    # が毎回合流の対象から外れ、モデルの記憶頼みに戻ってしまっていた。今回
    # 新たに品目が見つからず、セッションに既に確立した品目があるだけで
    # 合流の対象にする。
    needs_merge = len(carried_names) > 1 or (
        this_turn_name is None and bool(carried_names)
    )
    if not needs_merge:
        return routed

    items: list[SimpleNamespace] = []
    sources = list(routed.sources)
    notices: list[dict] = []
    for name in carried_names:
        chosen = _refetch_item(name)
        if chosen is None:
            continue
        items.append(SimpleNamespace(**chosen))
        if chosen.get("source_url"):
            sources.append(chosen["source_url"])
        notice = _refetch_notice(name)
        if notice is not None:
            notices.append(notice)
            for seg in [notice.get("current")] + notice.get("future", []):
                if seg and seg.get("source_url"):
                    sources.append(seg["source_url"])

    if not items:
        # 再検索でも何も取れなければ元のまま返す(壊さない)。
        return routed

    schedules: list[SimpleNamespace] = []
    district_notes: list[str] = []
    ward, town = state.ward, state.town
    if ward and town:
        for it in items:
            if it.disposal_method not in _CURBSIDE_WASTE_TYPES:
                continue
            result = execute_tool(
                "lookup_collection_schedule",
                {"ward": ward, "town": town, "waste_type": it.disposal_method},
            )
            for sched in result.get("schedules", []):
                schedules.append(SimpleNamespace(**sched))
                if sched.get("source_url"):
                    sources.append(sched["source_url"])
            if result.get("district_note"):
                district_notes.append(result["district_note"])
    else:
        # 地域がまだ確定していなければ、今回のターンで既に見つかっている分
        # (あれば)をそのまま使う。
        schedules = list(routed.schedule_hits)

    merged = _build_routed_answer(
        query=carried_names[-1],
        items=items,
        schedules=schedules,
        explanations=[],
        notices=notices,
        district_notes=district_notes,
        sources=sources,
        match_stage="exact",
        detected_item=carried_names[-1],
        detected_ward=ward,
        detected_town=town,
    )
    merged.missing = [] if schedules else list(routed.missing)
    merged.found = merged.found or routed.found

    # 今回のターン自身が(品目とは無関係に)何かを見つけていた場合
    # (一般説明・収集日のみ等)、その内容を捨てずに前に残す。ただし今回の
    # ターンで新しい品目を見つけていた場合(`this_turn_name` あり)は、その
    # 品目の情報は既に `items`(carried_names に含まれる)の再検索結果として
    # 下でも組み立て直されるため、二重掲載を避けてここでは付け足さない。
    if this_turn_name is None and routed.context:
        merged.context = routed.context + "\n\n" + merged.context

    return merged
