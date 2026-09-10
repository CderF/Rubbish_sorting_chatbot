"""`_describe_non_uniform_district`(STAGE05 5-2 #28、2026-09-14)の単体テスト。

B1実機の失敗トレース(ユーザー提供)で確認された不具合: 「企救丘」(小倉南区)
は丁目によって家庭ごみの収集日が異なる(1〜3丁目/5〜6丁目の2グループ、
加えて4丁目(一部)は同じごみ種別でも収集日が2通り存在する)のに、回答は
「町名が不明です」とだけ案内し、そもそも丁目によって収集日が違うこと自体が
伝わらなかった。丁目ごとの内訳を案内し、どちらのグループかを選べるように
する。

実DB(`var/kitakyushu_waste.db`)の実データに依存する ― 「企救丘」の構成
(1・2・3丁目/5・6丁目の2グループ + 4丁目(一部)の要確認)は crawl 結果
そのものなので、DB が入れ替わっても構成自体が変わることはない前提
(北九州市の行政区画)。
"""

from __future__ import annotations

from app.rag.router import _blocking_route


def test_kikuoka_schedule_question_explains_the_chome_split_instead_of_just_asking_for_town():
    ans = _blocking_route("小倉南区の企救丘で家庭ごみの日を教えてください", None)

    assert ans.schedule_hits == []  # 丁目を勝手に決めつけて答えない
    assert len(ans.missing) == 1
    message = ans.missing[0]
    assert "企救丘" in message
    assert "月曜日・木曜日" in message
    assert "火曜日・金曜日" in message
    # 実際のグループ分け(1・2・3丁目 / 5・6丁目)がそのまま出ている。
    assert "企救丘1丁目" in message
    assert "企救丘5丁目" in message


def test_kikuoka_chome4_partial_is_not_dropped_and_is_flagged_as_ambiguous():
    """2026-09-14 レビュー指摘の回帰: 「企救丘4丁目(一部)」は末尾が
    「丁目」ではなく「丁目(一部)」のため、以前は `_town_base` の正規表現に
    マッチせず地区のグループから漏れていた(=案内から丸ごと消えていた)。
    しかもこの丁目は同じごみ種別でも収集日が2通り存在する(住所の一部で
    分かれる)ため、他の丁目とまとめて1つのグループにもできない ―
    単独で「要確認」として出す。"""
    ans = _blocking_route("小倉南区の企救丘で家庭ごみの日を教えてください", None)
    message = ans.missing[0]

    assert "企救丘4丁目(一部)" in message
    assert "要確認" in message
    # 1〜3丁目・5〜6丁目のどちらのグループの曜日とも同じ扱いで結合されて
    # いない(「4丁目」がグループの町名一覧に紛れ込んでいない)。
    assert "企救丘4丁目(一部)は収集日が複数存在するため要確認" in message


def test_kikuoka_followup_can_then_resolve_with_a_chome(monkeypatch):
    """次のターンで丁目が分かれば普通に収集日を返す(追加の分岐は不要 ―
    既存の完全一致の町名検索がそのまま効く)。"""
    ans = _blocking_route("小倉南区企救丘1丁目の家庭ごみの日を教えてください", None)
    assert [(s.waste_type, s.weekday) for s in ans.schedule_hits] == [
        ("家庭ごみ", "月曜日・木曜日")
    ]
