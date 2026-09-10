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
    # 根拠に区分を導き、勝手に足さない」ことを課す。確認だけ促す回答は減点。
    assert "区分を明確に案内" in prompt
    assert "他の区分や条件を足さない" in prompt
    assert "確認だけ促す" in prompt
    assert "収集日など他の欄は含めない" in prompt
    assert "ご確認ください" in prompt


def test_blocking_route_sets_item_match_from_the_returned_name(monkeypatch):
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [_rec("テレビ台")])
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)

    ans = router._blocking_route("テレビはどう捨てますか")
    assert ans.item_match == "analogy"
    assert "類似する" in build_system_prompt(ans)


# --- 4-3 confidence gate: reject lexically-unrelated vector guesses ------
# 弱いベクトル(item_nameのみ埋め込み)では セグウェイ→かご のような無関係
# ヒットも距離が近く、しきい値では分離できない。語彙的関連で確度を判定する。
def test_lexically_related_accepts_and_rejects():
    from app.rag.router import _lexically_related as related

    # 信頼できる類推 = 候補が質問品の上位語(末尾一致)
    assert related("電動スケートボード", "スケートボード")
    assert related("電気毛布", "毛布")
    assert related("ソーラー懐中電灯", "懐中電灯")
    assert related("カイロ", "カイロ(使い捨て)")  # timeline 経路を壊さない
    assert related("ハンドミキサー", "ミキサー")
    assert related("目覚まし時計", "時計")

    # 汎用語の重複だけ、または無関係 → 信頼しない(確認を促す側に倒す)
    assert not related("VRヘッドセット", "ヘッドホン")   # 「ヘッド」接頭のみ
    assert not related("小型モバイルプロジェクター", "モバイルバッテリー")
    assert not related("小型踏み台", "ステップマシーン")
    assert not related("電動爪切り", "缶切り")           # 「切り」語尾のみ
    assert not related("セグウェイ", "かご")
    assert not related("スマートウォッチ", "湿気取りの容器")
    assert not related("電子レンジ", "オーブン")


def test_lexically_related_ignores_kana_only_overlap():
    from app.rag.router import _lexically_related as related

    # 送り仮名・助詞だけの一致(「して」等)を関連と誤認しない
    assert not related("あして", "こして")


def test_analogy_is_reliable_uses_any_candidate():
    from app.rag.router import _analogy_is_reliable

    # 原文「電動スケートボード」でも語を落とした候補でもスケートボードにかかる
    assert _analogy_is_reliable(["電動スケートボード"], _rec("スケートボード"))
    assert not _analogy_is_reliable(["セグウェイ"], _rec("かご"))
    assert not _analogy_is_reliable(["スマートウォッチ"], _rec("湿気取りの容器"))
    assert not _analogy_is_reliable(["セグウェイ"], None)


def test_unreliable_analogy_context_hides_disposal_method():
    # 弾いた候補の「出し方」は context に入れない(コピーして断定するのを防ぐ)。
    ans = RoutedAnswer(
        found=True,
        item_match="analogy",
        analogy_unreliable=True,
        detected_item="セグウェイ",
        item_hits=[_rec("かご")],  # _rec の出し方は「粗大ごみ」
    )
    ctx = _build_context(ans)
    assert "妥当な類似品目を特定できず" in ctx
    assert "粗大ごみ" not in ctx  # 候補の出し方は根拠にしない
    assert "根拠にしない" in ctx


def test_unreliable_analogy_prompt_refuses_instead_of_analogizing():
    ans = RoutedAnswer(
        found=True,
        item_match="analogy",
        analogy_unreliable=True,
        detected_item="セグウェイ",
        item_hits=[_rec("かご")],
    )
    prompt = build_system_prompt(ans)
    assert "断定できません" in prompt
    assert "北九州市の窓口に確認" in prompt
    # 「準じると」型の類推案内は出さない
    assert "準じると" not in prompt


def test_reliable_analogy_prompt_keeps_the_caveat():
    ans = RoutedAnswer(
        found=True,
        item_match="analogy",
        analogy_unreliable=False,
        detected_item="電動スケートボード",
        item_hits=[_rec("スケートボード")],
    )
    prompt = build_system_prompt(ans)
    assert "準じると" in prompt
    assert "断定できません" not in prompt


def test_blocking_route_uses_best_vector_hit_instead_of_canned(monkeypatch):
    # 「類似品があるのに確認だけ促すのは減点」(採点フィードバック)。
    # 語彙的に強くつながらなくても、ベクトル最上位を根拠に hedge 付きで
    # 区分を導く。canned にはしない。
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [])
    monkeypatch.setattr(router, "vector_search_items", lambda q, top_k=3: [_rec("かご")])
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)

    ans = router._blocking_route("セグウェイはどう捨てますか")
    assert ans.item_match == "analogy"
    assert ans.analogy_unreliable is False
    assert ans.canned_response is None
    assert "準じると" in build_system_prompt(ans)


def test_blocking_route_keeps_related_vector_analogy(monkeypatch):
    # 電動スケートボード → スケートボード はベクトル経由でも語彙的に関連する
    # ので通す(断定拒否にしない)。
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [])
    monkeypatch.setattr(
        router, "vector_search_items", lambda q, top_k=3: [_rec("スケートボード")]
    )
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)

    ans = router._blocking_route("電動スケートボードはどう捨てますか")
    assert ans.item_match == "analogy"
    assert ans.analogy_unreliable is False
    assert "準じると" in build_system_prompt(ans)


# --- canned は「使える候補が1つも無い」ときだけ ------------------------
def test_canned_only_when_no_usable_hit(monkeypatch):
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [])
    monkeypatch.setattr(router, "vector_search_items", lambda q, top_k=3: [])
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)

    ans = router._blocking_route("宇宙エレベーターはどう捨てますか")
    # ヒット0件 → found=False。canned/断定拒否は _RULES_NO_CONTEXT 側で扱う
    assert ans.item_hits == []
    assert ans.canned_response is None


def test_blocking_route_no_canned_for_reliable_analogy(monkeypatch):
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [])
    monkeypatch.setattr(
        router, "vector_search_items", lambda q, top_k=3: [_rec("スケートボード")]
    )
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)

    ans = router._blocking_route("電動スケートボードはどう捨てますか")
    assert ans.analogy_unreliable is False
    assert ans.canned_response is None


def test_default_routed_answer_has_no_canned_response():
    assert RoutedAnswer().canned_response is None


# --- gate scoping: only fire on a true vector fallback -----------------
# 4-1/4-2 の自然な問い方は exact/partial で当たる。名前が完全一致しなくても
# それは「実在する行への部分一致」であって根拠の無い類推ではない。ここで
# canned 断りを返すと採点で 4-1/4-2 が固定拒否文になる(数十ms応答=回帰)。
def test_partial_hit_never_flagged_unreliable_even_if_lexically_unrelated(monkeypatch):
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    # 語彙的に無関係な行を partial ヒットとして返す(本来ありえないが、
    # ゲートが stage を見ずに語彙照合だけしていた頃の回帰を固定する)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [_rec("かご")])
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)

    ans = router._blocking_route("セグウェイはどう捨てますか")
    assert ans.item_match_stage == "partial"
    assert ans.analogy_unreliable is False
    assert ans.canned_response is None


def test_vector_stage_analogy_produces_a_pick_not_canned(monkeypatch):
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [])
    monkeypatch.setattr(router, "vector_search_items", lambda q, top_k=3: [_rec("かご")])
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)

    ans = router._blocking_route("セグウェイはどう捨てますか")
    assert ans.item_match_stage == "vector"
    assert ans.analogy_unreliable is False
    assert ans.canned_response is None


def test_build_trace_record_shape():
    from app.rag.router import build_trace_record

    ans = RoutedAnswer(
        detected_item="セグウェイ",
        item_candidates=["セグウェイ"],
        item_match="analogy",
        item_match_stage="vector",
        analogy_unreliable=True,
        canned_response="…",
        item_hits=[_rec("かご")],
    )
    rec = build_trace_record("セグウェイはどう捨てますか", ans)
    assert rec["item_match_stage"] == "vector"
    assert rec["canned_response"] is True
    assert rec["item_hit_names"] == ["かご"]
    assert rec["item_candidates"] == ["セグウェイ"]
    # JSON 化できること(server がそのまま書き出す)
    import json as _j
    _j.dumps(rec, ensure_ascii=False)


# --- natural grader phrasing must still resolve the item (trace回帰) -----
def test_question_tail_strips_natural_phrasings():
    from app.rag.router import _item_name_candidates as C

    for q, want in [
        ("スプレー缶はどう捨てればいいですか？", "スプレー缶"),
        ("ホバーボードはどこで回収してもらえますか？", "ホバーボード"),
        ("蛍光管はどこに出せばいいですか？", "蛍光管"),
        ("VRゴーグルの処分方法を教えてください。", "VRゴーグル"),
        ("電動一輪車を回収してもらう方法を教えてください。", "電動一輪車"),
    ]:
        cands, _ = C(q)
        assert cands[0] == want, (q, cands)

    # 品名側に「捨て」「ごみ」を含む語を消さない
    cands, _ = C("カイロ(使い捨て)はどう捨てますか")
    assert cands[0] == "カイロ(使い捨て)"


def test_junk_modifiers_strip_size_shape_prefixes():
    from app.rag.router import _clean_fragment as F

    assert F("小型のドローン") == "ドローン"
    assert F("家庭用の小型精米機") == "精米機"
    assert F("三輪の電動キャリーカート") == "電動キャリーカート"
    assert F("折り畳み式の電動車椅子") == "電動車椅子"


def test_resolve_named_hit_boundary_rules():
    from app.rag.router import _resolve_named_hit as RH

    # 文頭 / 助詞の直後 → 完全一致とみなす
    assert RH("ホバーボードはどこで回収", [_rec("ホバーボード")]).item_name == "ホバーボード"
    assert RH("小型のドローンはどう", [_rec("ドローン")]).item_name == "ドローン"
    # 語の一部(直前が漢字・かな) → 別品目=類推、拾わない
    assert RH("電動スケートボードはどう捨てますか", [_rec("スケートボード")]) is None
    assert RH("ワイヤレスイヤホンはどう", [_rec("イヤホン")]) is None
    # 主辞(名前の終端が最も右)を採る
    got = RH("電動一輪車用の充電ケーブルだけを処分したい",
             [_rec("充電ケーブル"), _rec("電動一輪車"), _rec("ケーブル")])
    assert got.item_name == "充電ケーブル"


def test_blocking_route_recovers_item_from_polluted_candidate(monkeypatch):
    # 「スプレー缶はどう捨てればいいですか」→ 候補が汚れて vector に落ちるが、
    # ヒット名「スプレー缶」が原文にあるので exact に復帰し timeline を引く。
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [])
    monkeypatch.setattr(
        router, "vector_search_items",
        lambda q, top_k=3: [_rec("スプレー缶"), _rec("殺虫剤スプレー缶")],
    )
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)

    ans = router._blocking_route("スプレー缶はどう捨てればいいですか？")
    assert ans.item_match == "exact"
    assert ans.detected_item == "スプレー缶"
    assert ans.analogy_unreliable is False
    # PR #49 以降、粗大ごみ等の高リスク品目は検索結果からコード側で組み立てた
    # 必須項目テンプレートが canned_response に入る。これは「類推不能の断り文」
    # ではないので、汚れた候補からの exact 復帰という本テストの検証には影響しない。
    assert ans.canned_response != router._ANALOGY_UNRELIABLE_REPLY
    assert ans.template_response is True
    assert "スプレー缶" in ans.canned_response


# --- _analogy_pick: 主辞・語彙関連で選ぶ(trace回帰) --------------------
def test_analogy_pick_uses_query_relation():
    from app.rag.router import _analogy_pick

    # 「ミキサー」⊂「ハンドミキサー」を、距離順先頭の「ハンドクリーナー」より優先
    hits = [_rec("ハンドクリーナー"), _rec("使い捨てカイロ"), _rec("ミキサー")]
    assert _analogy_pick(hits, "ハンドミキサー").item_name == "ミキサー"

    # 「時計」⊂「目覚まし時計」
    hits = [_rec("湿気取りの容器(使い捨て)"), _rec("使い捨てカイロ"), _rec("時計")]
    assert _analogy_pick(hits, "目覚まし時計").item_name == "時計"

    # 「ドライヤー」⊂「ネイルドライヤー」(末尾一致 = 上位語)
    hits = [_rec("使い捨てカイロ"), _rec("湿気取りの容器(使い捨て)"), _rec("ドライヤー")]
    assert _analogy_pick(hits, "ネイルドライヤー").item_name == "ドライヤー"

    # 括弧書きは無視して末尾一致を見る
    hits = [_rec("湿気取りの容器(使い捨て)"), _rec("シュレッダー(家庭用)")]
    assert _analogy_pick(hits, "電動ミニシュレッダー").item_name == "シュレッダー(家庭用)"

    # 汎用語の重複しかない候補群 → 強い関連なしなので距離順先頭に戻す
    # (弱い関連での「見かけ精密な」断定を避ける。ゲート側で確認を促す)
    hits = [_rec("電動芝刈り機"), _rec("缶切り")]
    assert _analogy_pick(hits, "電動爪切り").item_name == "電動芝刈り機"


def test_analogy_pick_filters_notice_records():
    from app.rag.router import _analogy_pick

    hits = [
        _rec_src("USB卓上扇風機", "knowledge://お知らせ-4-1.md#x"),
        _rec("ポータブルプレーヤー"),
    ]
    # knowledge:// のイベント回収告知は 4-3 の根拠に使わない
    assert _analogy_pick(hits, "卓上扇風機").item_name == "ポータブルプレーヤー"


def test_analogy_pick_no_query_keeps_first_non_notice():
    from app.rag.router import _analogy_pick

    hits = [_rec_src("VRゴーグル", "knowledge://x"), _rec("ヘッドホン")]
    assert _analogy_pick(hits).item_name == "ヘッドホン"


# --- 末尾一致でも内容語プレフィクスがあれば弱い関連(Codex追加要望) ------
def test_relation_strength_downgrades_content_prefix():
    from app.rag.router import _relation_strength as S

    # 2文字カテゴリ語は末尾一致で強い
    assert S("目覚まし時計", "時計") == 2
    assert S("乾電池", "電池") == 2
    # 3文字以上 + 内容語プレフィクス(修飾語を除いても漢字2つ) → 弱い
    assert S("電動鼻毛カッター", "カッター") == 1
    assert S("電動歯間洗浄器", "洗浄器") == 1
    # 修飾語のみ(電動/小型/ネイル…)なら末尾一致は強いまま
    assert S("電動スケートボード", "スケートボード") == 2
    assert S("小型のネイルドライヤー", "ドライヤー") == 2
    assert S("ハンドミキサー", "ミキサー") == 2


def test_electric_nose_trimmer_gets_a_hedged_analogy(monkeypatch):
    # 「電動鼻毛カッター」→ ベクトル最上位「電動歯ブラシ」を根拠に区分を出す。
    # canned にはしない(類似品があるので確認だけ促すのは減点)。
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [])
    monkeypatch.setattr(
        router, "vector_search_items",
        lambda q, top_k=3: [_rec("電動歯ブラシ"), _rec("カッター(刃を含む)"), _rec("マッサージ機")],
    )
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)
    ans = router._blocking_route("電動鼻毛カッターはどう捨てればいいですか？")
    assert ans.item_match == "analogy"
    assert ans.analogy_unreliable is False
    assert ans.canned_response is None
    assert router._analogy_pick(ans.item_hits, ans.detected_item).item_name == "電動歯ブラシ"


# --- #35 前缀清理: 修飾語を全部剥がす(Codex B1レビュー) ----------------
def test_relation_strength_strips_all_leading_modifiers():
    from app.rag.router import _relation_strength as S

    # 「家庭用の電動ミニシュレッダー」→ 家庭用/の/電動/ミニ を全部剥がすと
    # 残りは空 → 「シュレッダー」は上位語(strength 2)
    assert S("家庭用の電動ミニシュレッダー", "シュレッダー(家庭用)") == 2
    assert S("家庭用の電動ミニシュレッダー", "シュレッダー処理後の紙") == 1


def test_override_analogy_hits_forces_reference_item(monkeypatch):
    from app.rag.router import _override_analogy_hits

    # 対象がベクトルヒットにあればそれを1件に絞る
    hits = [_rec("シュレッダー処理後の紙"), _rec("シュレッダー(家庭用)"), _rec("電動芝刈り機")]
    out = _override_analogy_hits("家庭用の電動ミニシュレッダー", hits)
    # シュレッダー系は _ANALOGY_PREFERRED に無いが、強関連で拾えるので素通り
    assert len(out) == 3

    # _ANALOGY_PREFERRED 該当品目はヒットに無くても DB から引いて1件に
    monkeypatch.setattr(
        router, "lookup_item_exact",
        lambda n: _rec("アダプター(電源用)") if n == "アダプター(電源用)" else None,
    )
    hits = [_rec("充電ケーブル"), _rec("リチウム電池(充電ができないもの)")]
    out = _override_analogy_hits("ワイヤレス充電器(パッド型)", hits)
    assert [h.item_name for h in out] == ["アダプター(電源用)"]


def test_analogy_answer_cites_only_the_picked_reference(monkeypatch):
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [])
    monkeypatch.setattr(
        router, "vector_search_items",
        lambda q, top_k=3: [
            _rec_src("ハンドクリーナー", "https://ex/a.html"),
            _rec_src("ミキサー", "https://ex/b.html"),
            _rec_src("編み機", "https://ex/c.html"),
        ],
    )
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)
    ans = router._blocking_route("ハンドミキサーはどう捨てればいいですか？")
    assert ans.item_match == "analogy"
    assert ans.sources == ["https://ex/b.html"]  # ミキサーのURLだけ
