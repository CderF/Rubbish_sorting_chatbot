"""要件4-3: 早見表に個別記載のない品目は、類似品目に準じた案内 + 断り書き。

PR #14 レビュー(藤井)を反映:
- 断り書きの要否は「どの検索段でヒットしたか」ではなく「返ってきた行の名前が
  質問の品名と実際に一致するか」で決める(`_classify_item_match`)。
- 断り書きは早見表セクションだけに寄せ、収集日など他の欄は巻き込まない。
- 無関係なベクトルヒットは距離しきい値で弾く(`_filter_by_distance`)。

LLM 出力そのものは対象外(決定的な部分のみ)。
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import app.rag.router as router
from app.rag.router import (
    _analogy_pick,
    _build_context,
    _classify_item_match,
    _item_name_candidates,
    _search_item,
    _timeline_method_text,
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


def _rec_src(name: str, source_url: str):
    r = _rec(name)
    r.source_url = source_url
    return r


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
    # 修飾語つき原文と清掃後候補のどちらで一致しても、別品目扱いにしない
    assert _classify_item_match(["壊れたテレビ", "テレビ"], [_rec("テレビ")]) == "exact"


def test_item_candidates_keep_raw_official_name_and_cleaned_fallback():
    candidates, vector_query = _item_name_candidates("使用済みの花火はどう捨てますか")
    assert candidates[0] == "使用済みの花火"
    assert "花火" in candidates
    assert vector_query == "使用済みの花火"

    candidates, _ = _item_name_candidates("壊れたテレビはどう捨てますか")
    assert "壊れたテレビ" in candidates
    assert "テレビ" in candidates

    candidates, vector_query = _item_name_candidates("カイロ(使い捨て)はどう捨てますか")
    assert candidates[0] == "カイロ(使い捨て)"
    assert "カイロ" in candidates and "使い捨て" in candidates
    assert vector_query == "カイロ(使い捨て)"


def test_timeline_method_uses_only_the_structured_exact_date():
    start = date(2026, 10, 1)
    assert _timeline_method_text(
        "2026年10月から、蛍光管は家庭ごみでは収集しなくなります。", start
    ) == "蛍光管は家庭ごみでは収集しなくなります。"
    assert _timeline_method_text(
        "2026年10月1日から、乾電池を家庭ごみへ出すことはできません。", start
    ) == "乾電池を家庭ごみへ出すことはできません。"
    assert _timeline_method_text(
        "2026年10月から、安全のため花火の出し方を変更します。", start
    ) == "安全のため花火の出し方を変更します。"


def test_search_item_returns_hits_and_stage(monkeypatch):
    # the stage is kept for the WebUI 根拠 panel (#12); the answer logic uses
    # _classify_item_match instead.
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: _rec(n))
    assert _search_item("テレビ") == ([_rec("テレビ")], "exact")
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [])
    monkeypatch.setattr(router, "vector_search_items", lambda q, top_k=3: [])
    assert _search_item("宇宙服") == ([], None)


# --- code-side single candidate for 4-3 (採点フィードバック) -----------
def test_analogy_pick_prefers_hayamihyo_row_over_4_1_event_notice():
    # VRヘッドセット の実例: ヘッドホン(早見表) と VRゴーグル(4-1告知)が
    # 候補に並ぶ。区分を持たないイベント告知ではなく早見表行を選ぶ。
    hits = [
        _rec_src("VRゴーグル", "knowledge://お知らせ-4-1.md#お知らせ-07-VRゴーグル回収"),
        _rec_src("ヘッドホン", "https://www.city.kitakyushu.lg.jp/contents/y.html"),
    ]
    assert _analogy_pick(hits).item_name == "ヘッドホン"


def test_analogy_pick_falls_back_to_top1_when_all_are_notices():
    hits = [
        _rec_src("VRゴーグル", "knowledge://お知らせ-4-1.md#お知らせ-07-VRゴーグル回収"),
        _rec_src("ドローン", "knowledge://お知らせ-4-1.md#お知らせ-05-ドローン回収"),
    ]
    assert _analogy_pick(hits).item_name == "VRゴーグル"


def test_analogy_context_contains_exactly_one_candidate():
    # 複数候補があっても LLM に渡す context には1件だけ入れる(混用防止)。
    ans = RoutedAnswer(
        found=True,
        item_match="analogy",
        detected_item="VRヘッドセット",
        item_hits=[
            _rec_src("VRゴーグル", "knowledge://お知らせ-4-1.md#お知らせ-07-VRゴーグル回収"),
            _rec_src("ヘッドホン", "https://www.city.kitakyushu.lg.jp/contents/y.html"),
        ],
    )
    ctx = _build_context(ans)
    assert "ヘッドホン" in ctx
    assert "VRゴーグル" not in ctx  # 2件目は context に出さない
    assert ctx.count("類似品目:") == 1


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
    assert "類似品目: テレビ台" in ctx
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
    # 候補は _analogy_pick がコード側で1件に絞る。プロンプトは「その1件だけを
    # 根拠に、勝手に足さない」ことを課す。
    assert "1件だけ" in prompt
    assert "新たに付け加えてはならない" in prompt
    assert "収集日など他の欄の情報はこの限りではない" in prompt  # don't blur the schedule
    assert "確認" in prompt


def test_blocking_route_sets_item_match_from_the_returned_name(monkeypatch):
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [_rec("テレビ台")])
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)

    ans = router._blocking_route("テレビはどう捨てますか")
    assert ans.item_match == "analogy"
    assert "類似する" in build_system_prompt(ans)
