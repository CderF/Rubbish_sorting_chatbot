"""第6回実採点(2026-09-10)の回帰テスト。

対象は3つ:
  - 多品目の質問(「たんすと古い布団と、あと段ボール」)で、主候補の検索だけでは
    1件しか拾えず、残りの品目が回答から抜け落ちていた(必須項目テンプレートが
    1品目しか表せないことも重なっていた)。
  - 「水と魚が入ったまま」のように状態・内容物を述べた質問に、必須項目
    テンプレート(項目表)が当たり、述べられた条件に一切触れない回答になっていた。
  - 品名を省いた追問(「え、知らなかったです。じゃあどうすればいいんですか？」)が、
    会話履歴を受け取らない決定論的経路に落ちて6-3(品名を教えてください)に
    なっていた。

実DB(`var/kitakyushu_waste.db`)を使う。`app/server.py` は vllm を import する
ためテストから読めないので、履歴の引き継ぎは「サーバーが行う手順」を
`agent.build_history_query` + `router.handle_query` で再現して検証する。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.rag import agent, router
from app.rag.agent import _filter_explanations_to_items, _narrow_item_hits
from app.rag.router import (
    _build_required_fields_response,
    _has_stated_condition,
    _is_waste_type_only,
    _listed_item_extras,
    _norm,
    apply_response_template,
    needs_history_retry,
)

ENUMERATED = "たんすと古い布団と、あと段ボールがたくさんあります"


def _route(prompt: str):
    return asyncio.run(router.handle_query(prompt))


# --- 品目の列挙 -------------------------------------------------------------


def test_enumerated_items_are_all_searched():
    names = [h.item_name for h in _route(ENUMERATED).item_hits]
    assert "たんす" in names
    assert "布団" in names
    assert "段ボール" in names


def test_enumerated_question_is_exact_and_cannot_use_the_item_table():
    """項目表は1品目しか表せないので、多品目では必ず外す。"""
    ans = _route(ENUMERATED)
    assert ans.item_match == "exact"
    assert _build_required_fields_response(ans) is None
    assert apply_response_template(ans, ENUMERATED).template_response is False


def test_two_item_correction_turn_keeps_both_items():
    names = [h.item_name for h in _route("えっと、たんすと洗濯機です").item_hits]
    assert "たんす" in names
    assert "洗濯機(乾燥機付きのものを含む)" in names


def test_synonym_of_the_first_item_is_also_collected():
    """「ソファ」は早見表の表記が「ソファー」。列挙の2品目目も落とさない。"""
    names = [h.item_name for h in _route("ソファと食器棚です。").item_hits]
    assert "ソファー" in names
    assert "食器棚" in names


def test_waste_type_words_are_not_treated_as_listed_items():
    """「かん・びんとペットボトル」は品目の列挙ではなく種別の質問。"""
    assert _is_waste_type_only("ペットボトル") is True
    assert _is_waste_type_only("かん・びん") is True
    assert _is_waste_type_only("たんす") is False
    assert _listed_item_extras("かん・びんとペットボトルは同じ袋にまとめて出していいですよね？", set()) == []


def test_listed_item_extras_skips_what_was_already_searched():
    extras = _listed_item_extras("たんすと本棚です。", {_norm("たんす")})
    assert [r.item_name for r in extras] == ["本棚"]


# --- 述べられた条件 ---------------------------------------------------------


def test_stated_condition_skips_the_item_table():
    prompt = "水と魚が入ったままの大きな水槽を処分したいのですが、何ごみですか？"
    ans = _route(prompt)
    assert _has_stated_condition(prompt) is True
    assert ans.item_hits and ans.item_hits[0].item_name == "水槽"
    assert _build_required_fields_response(ans) is None
    assert apply_response_template(ans, prompt).template_response is False


def test_stated_condition_does_not_fire_on_ordinary_questions():
    for prompt in (
        "スプレー缶はどう捨てればいいですか",
        "エアコンは燃えないごみの日に出せばいいんですよね？",
        "たんすと本棚です。",
        "使用済みの注射器の本体（針を除いた部分）は燃えるごみに混ぜていいですか？",
    ):
        assert _has_stated_condition(prompt) is False


# --- 会話履歴の引き継ぎ -----------------------------------------------------


def test_needs_history_retry_only_for_the_empty_fallback():
    assert needs_history_retry(None) is False
    empty = _route("え、知らなかったです。じゃあどうすればいいんですか？")
    assert empty.safety_gate == "6-3"
    assert needs_history_retry(empty) is True
    found = _route("エアコンはどう捨てればいいですか？")
    assert needs_history_retry(found) is False


def test_history_query_prepends_earlier_user_turns():
    history = [
        {"role": "user", "content": "エアコンは燃えないごみの日に出せばいいんですよね？"},
        {"role": "assistant", "content": "いいえ、燃えないごみの日には出せません。"},
    ]
    query = agent.build_history_query(history, "え、知らなかったです。じゃあどうすればいいんですか？")
    assert "エアコン" in query
    assert query.splitlines()[-1] == "え、知らなかったです。じゃあどうすればいいんですか？"


def test_history_query_is_none_without_user_turns():
    assert agent.build_history_query([], "どうすればいいですか？") is None
    assert (
        agent.build_history_query(
            [{"role": "assistant", "content": "いいえ、出せません。"}], "どうすれば？"
        )
        is None
    )


def test_history_query_keeps_only_the_last_three_user_turns():
    history = [{"role": "user", "content": f"発言{i}"} for i in range(5)]
    query = agent.build_history_query(history, "最後の質問")
    assert "発言1" not in query
    assert "発言2" in query
    assert "発言4" in query


def test_history_query_reads_text_out_of_image_turns():
    """画像付きのターンは content がリスト形式になる。"""
    history = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "..."},
                {"type": "text", "text": "エアコンです"},
            ],
        }
    ]
    assert "エアコン" in agent.build_history_query(history, "どうすればいいですか？")


def test_follow_up_without_item_recovers_the_item_from_history():
    """サーバー(`_route_safely`)が行う引き直しの手順そのもの。"""
    cases = [
        ("エアコンは燃えないごみの日に出せばいいんですよね？", "エアコン"),
        ("洗濯機は粗大ごみとして受付センターに申し込めばいいんですよね？", "洗濯機(乾燥機付きのものを含む)"),
        ("冷蔵庫は家庭ごみの収集日に出せばいいんですよね？", "冷蔵庫(冷凍冷蔵庫)"),
    ]
    for first, expected in cases:
        history = [
            {"role": "user", "content": first},
            {"role": "assistant", "content": "（回答）"},
        ]
        follow_up = "そうなんですね、じゃあどこに持って行けばいいですか？"
        direct = _route(follow_up)
        assert needs_history_retry(direct), first
        retried = _route(agent.build_history_query(history, follow_up))
        assert expected in [h.item_name for h in retried.item_hits], first


def test_needs_history_retry_does_not_fire_when_more_input_is_being_asked_for():
    """「町名が要る」等、あと一歩の情報を求めている回答は引き直さない。

    引き直すと、品名ではなく町名を聞いている状況で過去の発話を混ぜてしまい、
    `missing` による追加入力の案内(3-1)が消える。
    """
    ans = _route("小倉北区の粗大ごみの収集日を教えてください")
    assert "町名" in ans.missing
    assert needs_history_retry(ans) is False


def test_needs_history_retry_does_not_fire_for_the_other_fixed_answers():
    """6-1a(確定回答)・6-2(危険)は 6-3 ではないので引き直さない。"""
    from app.rag.types import RoutedAnswer

    for gate in ("6-1a", "6-2"):
        assert needs_history_retry(RoutedAnswer(safety_gate=gate, canned_response="x")) is False
    assert needs_history_retry(RoutedAnswer(canned_response="x")) is False


def test_single_item_question_still_gets_the_item_table():
    """テンプレートを外す条件(多品目・述べられた条件)に当たらない質問は従来どおり。"""
    prompt = "布団を捨てたいです。"
    ans = _route(prompt)
    assert [h.item_name for h in ans.item_hits] == ["布団"]
    assert apply_response_template(ans, prompt).template_response is True


# --- agentic 経路の絞り込み -------------------------------------------------


def _ns(name: str, method: str = "") -> SimpleNamespace:
    return SimpleNamespace(item_name=name, disposal_method=method, notes="", source_url="https://ex/")


def test_narrow_item_hits_keeps_every_enumerated_item():
    items = [_ns("たんす"), _ns("布団"), _ns("段ボール")]
    kept, stage = _narrow_item_hits([("exact", items)], "たんす", ENUMERATED)
    assert stage == "exact"
    assert [k.item_name for k in kept] == ["たんす", "布団", "段ボール"]


def test_narrow_item_hits_still_narrows_when_only_one_item_is_named():
    items = [_ns("たんす"), _ns("布団"), _ns("段ボール")]
    kept, _ = _narrow_item_hits([("exact", items)], "たんす", "たんすを捨てたいです")
    assert [k.item_name for k in kept] == ["たんす"]


def test_narrow_item_hits_does_not_keep_several_vector_candidates():
    """部分一致・ベクトルの段では質問の品目が一意でないので1件に絞る。"""
    items = [_ns("たんす"), _ns("布団")]
    kept, _ = _narrow_item_hits(
        [("vector", items)], "たんす", "たんすと布団を捨てたいです"
    )
    assert len(kept) == 1


def test_explanations_unrelated_to_the_item_are_dropped():
    items = [_ns("たんす", "粗大ごみ")]
    kept = _filter_explanations_to_items(
        [
            {"text": "たんすは事前申込制です"},
            {"text": "ふれあい収集のご案内"},
            {"text": "粗大ごみの申込み方法"},
        ],
        items,
    )
    assert [e["text"] for e in kept] == ["たんすは事前申込制です", "粗大ごみの申込み方法"]


def test_explanations_are_untouched_when_no_item_was_found():
    """品目が見つかっていない一般質問では、絞り込む根拠が無い。"""
    explanations = [{"text": "ふれあい収集のご案内"}]
    assert _filter_explanations_to_items(explanations, []) == explanations
