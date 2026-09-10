"""`app/rag/router.py` の `category_procedure_intent`(STAGE05 実採点の
失敗題レビュー、2026-09-14)の単体テスト。

B1実機の失敗トレース(ユーザー提供)で確認された、特定品目ではなく
ごみ種別そのものの手続き・収集日を尋ねる質問が品目検索に落ちる不具合:
  - 5-1 #03: 「粗大ごみの出し方」→「ごみ箱」に化ける
  - 5-1 #11: 「粗大ごみの申込み方法」→「ふれあい収集」(無関係な制度)に化ける
  - 5-1 #04: 「古紙を出したい」→「銀紙」に化ける

DB に依存しない部分(`category_procedure_intent` そのもの)と、実DBに依存する
`_blocking_route` 経由の確認を分けている。
"""

from __future__ import annotations

from app.rag.router import _blocking_route, category_procedure_intent, extract_region


def _intent(prompt: str):
    ward, town = extract_region(prompt)
    return category_procedure_intent(prompt, ward, town)


# --- category_procedure_intent (DB非依存) ----------------------------------
def test_detects_bulky_waste_procedure_question():
    assert _intent("粗大ごみってどうやって出すんですか") == ("粗大ごみ", False)


def test_detects_bulky_waste_application_question():
    assert _intent("粗大ごみってどうやって申し込むんですか") == ("粗大ごみ", False)


def test_detects_waste_paper_procedure_question():
    assert _intent("古紙を出したいです") == ("古紙", False)


def test_detects_household_waste_schedule_question():
    assert _intent("家庭ごみの日を教えてください") == ("家庭ごみ", True)


def test_does_not_swallow_a_real_item_next_to_a_category_word():
    """「粗大ごみの布団」のように具体的な品目が併記されているときは対象外
    (通常の品目検索に任せる)。"""
    assert _intent("粗大ごみの布団はどう捨てますか") is None
    assert _intent("布団を捨てたい") is None


def test_does_not_swallow_hiragana_item_names_with_a_category_prefix():
    assert _intent("プラスチックのおもちゃはどう捨てますか") is None
    assert _intent("古紙のかごはどう捨てますか") is None


def test_does_not_swallow_a_normal_item_schedule_question():
    assert _intent("小倉北区浅野1丁目でペットボトルは何曜日") is None


# --- _blocking_route 経由(実DB) ---------------------------------------------
def test_procedure_question_does_not_hit_item_search():
    """要件5-1: 品目検索(exact/partial/ベクトル)に一切渡らず、無関係な
    品目(「ごみ箱」等)を見つけない。"""
    for prompt in [
        "粗大ごみってどうやって出すんですか",
        "粗大ごみってどうやって申し込むんですか",
        "古紙を出したいです",
    ]:
        ans = _blocking_route(prompt, None)
        assert ans.item_hits == []
        assert ans.detected_item is None


def test_schedule_category_question_without_region_asks_for_region():
    ans = _blocking_route("家庭ごみの日を教えてください", None)
    assert ans.item_hits == []
    assert ans.missing == ["区名と町名"]


def test_schedule_category_question_with_region_resolves_directly():
    """品目を介さず、種別名を直接 waste_type として収集日を引く。"""
    ans = _blocking_route("小倉北区浅野1丁目で家庭ごみの日を教えてください", None)
    assert ans.item_hits == []
    assert ans.found is True
    assert [(s.waste_type, s.weekday) for s in ans.schedule_hits] == [
        ("家庭ごみ", "火曜日・金曜日")
    ]


def test_real_item_next_to_category_word_still_gets_a_normal_item_search():
    ans = _blocking_route("粗大ごみの布団はどう捨てますか", None)
    assert ans.detected_item is not None
    assert any(getattr(h, "item_name", "") == "布団" for h in ans.item_hits)


# --- 2026-09-14 レビュー指摘の回帰: procedure_like の「無結果」パス --------
def test_pure_procedure_questions_never_fall_through_to_free_generation():
    """要件5-1: 知識ベースに根拠が無い手続き質問は、found=False・context=""
    のまま関数を抜けて LLM の自由生成に委ねてはならない(小型モデルが
    「ごみ箱」「ふれあい収集」のような無関係な内容を作り出す)。
    固定の確認文(canned_response)を必ず立て、生成そのものをスキップする。
    """
    for prompt in [
        "粗大ごみってどうやって出すんですか",
        "粗大ごみってどうやって申し込むんですか",
        "古紙を出したいです",
    ]:
        ans = _blocking_route(prompt, None)
        assert ans.item_hits == []
        # found=False かつ context 空 かつ canned_response も無い、という
        # 「自由生成に丸投げ」の組み合わせは絶対に発生させない。
        assert not (not ans.found and not ans.context and ans.canned_response is None)
        assert ans.canned_response == "具体的な品目名と、お住まいの区・町名を教えてください。"


def test_procedure_question_canned_reply_does_not_mention_unrelated_systems():
    ans = _blocking_route("粗大ごみってどうやって申し込むんですか", None)
    assert "ふれあい収集" not in (ans.canned_response or "")
    assert "ごみ箱" not in (ans.canned_response or "")


def test_category_question_with_punctuation():
    """2026-09-14 レビュー指摘の回帰: 全角「？」「。」が付くと
    `_PUNCT_OR_SPACE_RE` が拾わず、除去後の残りに記号が残って「実質的な
    内容がある」と誤判定され、category_procedure_intent が None を返して
    しまっていた(=固定回答が発火せず自由生成に落ちる経路が残っていた)。"""
    for prompt in [
        "粗大ごみってどうやって出すんですか？",
        "粗大ごみってどうやって申し込むんですか？",
        "古紙を出したいです。",
    ]:
        ans = _blocking_route(prompt, None)
        assert ans.canned_response == (
            "具体的な品目名と、お住まいの区・町名を教えてください。"
        )
