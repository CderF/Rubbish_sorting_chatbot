"""`app/rag/session_state.py` の畳み込み方針の単体テスト(STAGE05)。

DB・GPU・ネットワーク・サーバーのいずれも要らない。STAGE05 のバグはほぼ全て
「どのターンでスロットをどう更新するか」に出るため、そこだけを純関数として
切り出してここで押さえる(`tests/stage04_ingest_test.py` と同じ方針)。

    python3 -m pytest tests/session_state_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.session_state import (  # noqa: E402
    MAX_ITEMS,
    DialogState,
    TurnInput,
    is_demonstrative,
    is_plausible_item_name,
    merge,
)


def turn(text: str, **kw) -> TurnInput:
    return TurnInput(text=text, **kw)


# --- 品名候補のガード -------------------------------------------------------
def test_rejects_particle_and_polite_form_leftovers():
    # 「小倉北区浅野1丁目です」から実際に落ちてきていた候補。
    assert not is_plausible_item_name("す")
    assert not is_plausible_item_name("です")
    assert not is_plausible_item_name("")


def test_keeps_single_kanji_items():
    # 早見表には「傘」「本」のような1文字の品目が実在する。
    assert is_plausible_item_name("傘")
    assert is_plausible_item_name("ソファ")


def test_demonstratives_are_not_item_names():
    assert is_demonstrative("それ")
    assert is_demonstrative("さっきの")
    assert not is_plausible_item_name("それ")


# --- 5-1: 数ターンに分かれた相談を引き継ぐ ---------------------------------
def test_item_from_an_earlier_turn_survives_a_region_only_turn():
    s = merge(None, turn("ソファを捨てたい", item="ソファ"))
    s = merge(s, turn("小倉北区浅野1丁目です", ward="小倉北区", town="浅野1丁目"))

    assert s.items == ("ソファ",)
    assert s.ward == "小倉北区" and s.town == "浅野1丁目"
    assert s.has_region


def test_schedule_intent_is_sticky_across_turns():
    s = merge(None, turn("粗大ごみの収集日は?", asks_schedule=True))
    s = merge(s, turn("小倉北区浅野1丁目です", ward="小倉北区", town="浅野1丁目"))
    assert s.wants_schedule


# --- 5-2: 追加 ------------------------------------------------------------
def test_adding_an_item_keeps_the_previous_one():
    s = merge(None, turn("ソファを捨てたい", item="ソファ"))
    s = merge(s, turn("冷蔵庫も追加でお願いします", item="冷蔵庫"))

    assert s.items == ("ソファ", "冷蔵庫")
    assert s.last_item_update == "add"
    assert not s.corrected


def test_a_bare_new_item_is_treated_as_an_addition():
    """マーカーの無い言い換え(「冷蔵庫は?」)も追加として扱う。

    前ターンの案内を保つほうが 5-2 に対して安全側に倒れるため。
    """
    s = merge(None, turn("ソファを捨てたい", item="ソファ"))
    s = merge(s, turn("冷蔵庫は?", item="冷蔵庫"))
    assert s.items == ("ソファ", "冷蔵庫")


def test_item_list_is_capped_oldest_first():
    s = None
    for i in range(MAX_ITEMS + 2):
        s = merge(s, turn(f"品目{i}も", item=f"品目{i}"))
    assert len(s.items) == MAX_ITEMS
    assert s.items[0] == f"品目{2}"


# --- 5-2: 変更・訂正 --------------------------------------------------------
def test_correction_replaces_the_latest_item():
    s = merge(None, turn("ソファを捨てたい", item="ソファ"))
    s = merge(s, turn("ソファじゃなくて冷蔵庫でした", item="冷蔵庫"))

    assert s.items == ("冷蔵庫",)
    assert s.last_item_update == "replace"
    assert s.corrected


def test_correction_after_an_addition_only_drops_the_latest():
    s = merge(None, turn("ソファ", item="ソファ"))
    s = merge(s, turn("冷蔵庫も", item="冷蔵庫"))
    s = merge(s, turn("やっぱり洗濯機", item="洗濯機"))
    assert s.items == ("ソファ", "洗濯機")


def test_changing_the_ward_discards_the_old_town():
    """区が変われば町名は必ず捨てる。

    残すと `八幡西区 + 浅野1丁目`(実在しない組み合わせ)を引くことになる。
    """
    s = merge(None, turn("小倉北区浅野1丁目", ward="小倉北区", town="浅野1丁目"))
    s = merge(s, turn("やっぱり八幡西区です", ward="八幡西区"))

    assert s.ward == "八幡西区"
    assert s.town is None
    assert s.last_region_update == "replace"
    assert s.corrected


def test_changing_only_the_town_keeps_the_ward():
    s = merge(None, turn("小倉北区浅野1丁目", ward="小倉北区", town="浅野1丁目"))
    s = merge(s, turn("すみません、篠崎1丁目でした", town="篠崎1丁目"))

    assert s.ward == "小倉北区"
    assert s.town == "篠崎1丁目"


def test_repeating_a_known_item_refocuses_instead_of_duplicating():
    s = merge(None, turn("ソファ", item="ソファ"))
    s = merge(s, turn("冷蔵庫も", item="冷蔵庫"))
    s = merge(s, turn("ソファの収集日は?", item="ソファ", asks_schedule=True))

    assert s.items == ("冷蔵庫", "ソファ")
    assert s.primary_item == "ソファ"
    # docstring が公開している4値のうちの1つ。並べ替えただけなので訂正ではない
    # (PR #24 レビュー指摘4: 実装だけが返していて契約に書かれていなかった)。
    assert s.last_item_update == "refocus"
    assert not s.corrected


def test_refocusing_an_item_already_in_focus_is_a_no_op():
    s = merge(None, turn("ソファ", item="ソファ"))
    s = merge(s, turn("ソファの収集日は?", item="ソファ", asks_schedule=True))

    assert s.items == ("ソファ",)
    assert s.last_item_update == "none"


# --- 6-4 の布石: 判断の記録 -------------------------------------------------
def test_turn_log_records_what_was_extracted_and_decided():
    s = merge(None, turn("ソファを捨てたい", item="ソファ"))
    s = merge(s, turn("小倉北区浅野1丁目です", ward="小倉北区", town="浅野1丁目"))

    assert len(s.turn_log) == 2
    last = s.turn_log[-1]
    assert last["extracted"]["ward"] == "小倉北区"
    assert last["region_update"] == "set"
    assert last["items_after"] == ["ソファ"]


def test_state_updates_do_not_mutate_the_previous_state():
    first = merge(None, turn("ソファ", item="ソファ"))
    second = merge(first, turn("冷蔵庫も", item="冷蔵庫"))

    assert first.items == ("ソファ",)
    assert second.items == ("ソファ", "冷蔵庫")
    assert isinstance(first, DialogState)
