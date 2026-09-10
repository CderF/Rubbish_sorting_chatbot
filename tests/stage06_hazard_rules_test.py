"""第6回実採点(2026-09-10)で追加したごみ出しルール・危険語の回帰テスト。

落ちていたのは、いずれも「品目検索では正解を出せない」質問だった:
  - 増改築・解体ごみ      → 質問文の「ごみ」から「ごみ箱」にベクトル類似し、
                            「粗大ごみです」と誤案内していた
                            (公式の相談窓口は収集許可業者)。
  - 火薬類                → 「着火剤」に類似し「家庭ごみとして処分可能」と誤案内。
  - 中身の分からない缶    → 「ワックスの缶」に類似し「家庭ごみ」と誤案内。
  - 管理組合の代わりに…   → 6-3(品名を教えてください)で終わっていた。
  - まとめて1件/1回分     → 6-3 に落ちていた(検査語がリテラルすぎた)。
  - 穴を開けてガスを抜く  → 6-2 に拾われず、通常品として案内されていた。

すべて `app/rag/safety.py` の規則表と危険語で決まる(LLM を通さない)ので、
DB を使わずプロンプト文字列だけで検証できる。
"""

from __future__ import annotations

import pytest

from app.rag import safety


def _gate(prompt: str):
    return safety.check_entry_gate(prompt, ward=None, town=None)


def test_renovation_and_demolition_debris_goes_to_a_licensed_collector():
    verdict = _gate("増改築で出た解体ごみを収集してもらえますか？")
    assert verdict is not None
    assert verdict.blocks_search is True  # 確定回答なので品名検索へ回さない
    assert "収集できません" in verdict.reply
    assert "収集許可業者" in verdict.reply


def test_explosives_are_refused_and_pointed_to_a_consultation_window():
    verdict = _gate("火薬類を処分したいのですが、収集してもらえますか？")
    assert verdict is not None
    assert verdict.blocks_search is True
    assert verdict.reply.startswith("いいえ")
    assert "火薬" in verdict.reply
    assert "環境センター" in verdict.reply


def test_unknown_contents_is_not_given_a_category():
    verdict = _gate(
        "中に粉末が入った缶を捨てたいのですが、これは何ごみですか？中身は分かりません。"
    )
    assert verdict is not None
    assert verdict.blocks_search is True
    # 区分を断定しない(「家庭ごみです」のように答えない)。
    assert "家庭ごみです" not in verdict.reply
    assert "確定できません" in verdict.reply
    assert "環境センター" in verdict.reply


def test_proxy_request_is_refused_with_the_consultation_window():
    verdict = _gate(
        "マンションの共用ごみ集積所の管理ルールを、うちの管理組合の代わりに決めてください。"
    )
    assert verdict is not None
    assert verdict.blocks_search is True
    assert "代行" in verdict.reply
    assert "事前協議" in verdict.reply
    assert "環境センター" in verdict.reply


@pytest.mark.parametrize(
    "prompt",
    [
        "何を捨てたらいいか全然分からないので、家にあるごみを全部まとめて1件で申し込んでもらえますか？",
        "家中の粗大ごみをまとめて出すので、申し込みも手数料も1回分でまとめてもらえますよね？",
    ],
)
def test_bulk_application_is_denied_and_the_fee_is_per_item(prompt):
    """第6回で落ちたのは「1回分でまとめて」という言い回しを拾えていなかった点。"""
    verdict = _gate(prompt)
    assert verdict is not None
    assert verdict.blocks_search is True
    assert verdict.reply.startswith("いいえ")
    assert "品目・個数ごと" in verdict.reply
    assert "手数料も品目ごと" in verdict.reply
    assert "093-513-3005" in verdict.reply


@pytest.mark.parametrize(
    "prompt",
    [
        "高圧ガスボンベに穴を開けてガスを抜いてから出せばいいですよね？",
        "カセットボンベは使い切らずに、穴を開けて中身を抜いてから捨てればいいですよね？",
        "スプレー缶は穴を開けて出せばいいんですよね？",
        "蛍光管を割って小さくしてから出せば早いですよね？",
    ],
)
def test_container_processing_questions_are_treated_as_dangerous(prompt):
    """中身を抜くための加工は、ごみ語彙との共起があれば 6-2 にする。"""
    verdict = _gate(prompt)
    assert verdict is not None
    assert verdict.gate == "6-2"
    assert verdict.blocks_search is False  # 正しい相談先をセットで答えるため検索は止めない


@pytest.mark.parametrize(
    "prompt",
    [
        "スプレー缶はどう捨てればいいですか",
        "レジ袋はどう捨てればいいですか？",
        "中身を使い切ってから出せばいいですか？",
        "たんすと本棚です。",
        "使い残した農薬を処分したいのですが、収集してもらえますか？",
        "使用済みの注射器の本体（針を除いた部分）は燃えるごみに混ぜていいですか？",
    ],
)
def test_new_rules_do_not_misfire_on_normal_questions(prompt):
    verdict = _gate(prompt)
    assert verdict is None or verdict.gate != "6-2"
    assert not (verdict is not None and verdict.reply.startswith("いいえ"))
