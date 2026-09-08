"""要件4-3: 早見表に個別記載のない品目は、類似品目に準じた案内 + 断り書き。

PR #14 レビュー(藤井)を反映:
- 断り書きの要否は「どの検索段でヒットしたか」ではなく「返ってきた行の名前が
  質問の品名と実際に一致するか」で決める(`_classify_item_match`)。
- 断り書きは早見表セクションだけに寄せ、収集日など他の欄は巻き込まない。
- 無関係なベクトルヒットは距離しきい値で弾く(`_filter_by_distance`)。

LLM 出力そのものは対象外(決定的な部分のみ)。
"""

from __future__ import annotations

from types import SimpleNamespace

import app.rag.router as router
from app.rag.router import (
    _build_context,
    _classify_item_match,
    _search_item,
    build_system_prompt,
)
from app.rag.types import RoutedAnswer
from app.rag.vector_store import _ITEM_MATCH_MAX_DISTANCE, _filter_by_distance


def _rec(name: str):
    return SimpleNamespace(
        item_name=name,
        disposal_method="粗大ごみ",
        notes="",
        source_url="https://www.city.kitakyushu.lg.jp/contents/x.html",
        source_page_title="早見表",
        score=0.9,
    )


def _sched():
    return SimpleNamespace(
        ward="小倉北区",
        town="浅野1丁目",
        waste_type="家庭ごみ",
        weekday="火,金",
        source_url="https://www.city.kitakyushu.lg.jp/contents/z.html",
    )


# --- vector distance threshold (blocker 1) ----------------------------
def test_filter_by_distance_drops_far_hits():
    metas = [{"n": "a"}, {"n": "b"}, {"n": "c"}]
    dists = [0.20, 0.90, 0.50]
    assert _filter_by_distance(metas, dists, 0.55) == [{"n": "a"}, {"n": "c"}]


def test_filter_by_distance_passes_through_when_distances_missing():
    metas = [{"n": "a"}, {"n": "b"}]
    assert _filter_by_distance(metas, [], 0.55) == metas
    assert _filter_by_distance(metas, [0.1], 0.55) == metas  # length mismatch


def test_threshold_constant_is_a_sane_cosine_distance():
    assert 0.0 < _ITEM_MATCH_MAX_DISTANCE < 1.0


# --- name-based classification (blockers 2 & 3) -----------------------
def test_classify_item_match_is_name_based_not_tier_based(monkeypatch):
    assert _classify_item_match("宇宙服", []) == "none"
    # exact: a returned row's name equals the query
    assert _classify_item_match("テレビ", [_rec("テレビ")]) == "exact"
    # partial hits (LIKE '%テレビ%') return *different* items -> analogy, not exact
    assert _classify_item_match("テレビ", [_rec("テレビ台"), _rec("テレビアンテナ")]) == "analogy"
    # vector hits are different items -> analogy
    assert _classify_item_match("電動スケートボード", [_rec("電動キックボード")]) == "analogy"


def test_search_item_returns_hits_and_stage(monkeypatch):
    # the stage is kept for the WebUI 根拠 panel (#12); the answer logic uses
    # _classify_item_match instead.
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: _rec(n))
    assert _search_item("テレビ") == ([_rec("テレビ")], "exact")
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [])
    monkeypatch.setattr(router, "vector_search_items", lambda q, top_k=3: [])
    assert _search_item("宇宙服") == ([], None)


# --- the caveat is scoped to the item section (blocker 2) ------------
def test_analogy_relabels_only_the_item_section():
    ans = RoutedAnswer(
        found=True,
        item_match="analogy",
        detected_item="テレビ",
        item_hits=[_rec("テレビ台")],
        schedule_hits=[_sched()],
    )
    ctx = _build_context(ans)
    assert "早見表に「テレビ」の個別記載なし" in ctx
    assert "## 分別早見表の検索結果" not in ctx
    # the collection-day section is untouched
    assert "## 収集日の検索結果" in ctx
    assert "浅野1丁目" in ctx


def test_exact_match_keeps_the_normal_header():
    ans = RoutedAnswer(found=True, item_match="exact", detected_item="テレビ", item_hits=[_rec("テレビ")])
    assert "## 分別早見表の検索結果" in _build_context(ans)
    assert "個別記載なし" not in _build_context(ans)


# --- the prompt rule (blockers 2, 4, 5) -----------------------------
def test_analogy_rule_only_for_analogy_and_names_the_real_item():
    for m in ("exact", "none"):
        assert "類似する" not in build_system_prompt(RoutedAnswer(found=True, item_match=m))

    prompt = build_system_prompt(RoutedAnswer(found=True, item_match="analogy"))
    assert "類似する" in prompt
    assert "実際の品目名を挙げて" in prompt          # substitute, don't echo a placeholder
    assert "収集日など他の欄の情報はこの限りではない" in prompt  # don't blur the schedule
    assert "確認" in prompt


def test_blocking_route_sets_item_match_from_the_returned_name(monkeypatch):
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [_rec("テレビ台")])
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)

    ans = router._blocking_route("テレビはどう捨てますか")
    assert ans.item_match == "analogy"
    assert "類似する" in build_system_prompt(ans)
