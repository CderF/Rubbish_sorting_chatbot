"""`app/rag/router.py` の `_has_plausible_item_text`(STAGE05 5-1 の決定論的
経路向けガード)の単体テスト。DB に依存しない(`_match_district` を直接
モンキーパッチして、地区名グルーピングが引ける場合の挙動だけを検証する)。

2026-09-10 レビュー指摘: agentic 経路にだけ「本文に無い品目名を作り出さない」
ガード(`app/rag/agent.py` の `_query_grounded_in_text`)を入れても、
`AGENTIC_RAG_ENABLED=0` や agentic 経路が失敗したときにフォールバックする
決定論的経路(`router._blocking_route`)には同じ穴が残っていた。実際に本物の
DB(`var/kitakyushu_waste.db`)に対して `_blocking_route` を実行すると、
地域だけの発言・地域訂正だけの発言から正規表現の残骸を品目候補として拾い、
ベクトル検索で無関係な品目を見つけてしまうことを確認した(このファイルの
テストはロジックだけを検証し、実DBへの依存は `tests/agent_loop_test.py` 側に
留める)。
"""

from __future__ import annotations

from app.rag import router
from app.rag.router import _has_plausible_item_text


def test_single_leftover_kana_character_is_rejected():
    """「小倉北区浅野1丁目です」から regex の残骸として拾われる「す」。"""
    assert _has_plausible_item_text("す", None) is False
    assert _has_plausible_item_text("す", "小倉北区") is False


def test_real_short_hiragana_item_names_are_not_rejected():
    """「かご」「やかん」等、実在する2文字以上のひらがな品目まで誤って
    弾かないことを確認する(2026-09-10 レビュー指摘の回帰)。"""
    assert _has_plausible_item_text("かご", None) is True
    assert _has_plausible_item_text("やかん", None) is True


def test_real_item_names_with_kanji_or_katakana_are_not_rejected():
    assert _has_plausible_item_text("テレビ", None) is True
    assert _has_plausible_item_text("傘", None) is True
    assert _has_plausible_item_text("冷蔵庫", "小倉北区") is True


def test_filler_only_phrases_are_rejected_without_needing_district_data():
    """会話の相槌・丁寧表現だけの候補は、地区名グルーピングが引けなくても
    (ward が None でも)弾く。"""
    assert _has_plausible_item_text("すみません", None) is False
    assert _has_plausible_item_text("ください", None) is False


def test_bare_district_name_is_rejected_when_district_grouping_resolves(monkeypatch):
    """丁目省略の地区名(「黒崎」等)そのものは品目ではない。

    `_match_district` は実DBに依存する(`_district_groups` 参照)ので、ここでは
    「地区名として解決できた」場合の挙動だけを、モンキーパッチで固定して
    検証する。"""

    def fake_match_district(ward, prompt):
        if "黒崎" in prompt:
            return ("黒崎", ("黒崎1丁目", "黒崎2丁目"))
        return None

    monkeypatch.setattr(router, "_match_district", fake_match_district)

    assert _has_plausible_item_text("黒崎でした", "八幡西区") is False
    assert _has_plausible_item_text("すみません、やっぱり 黒崎でした", "八幡西区") is False


def test_real_item_survives_even_alongside_a_resolved_district_name(monkeypatch):
    """地区名が解決できても、それとは別に実質的な内容(かご等)が残っていれば
    弾かない。"""

    def fake_match_district(ward, prompt):
        if "黒崎" in prompt:
            return ("黒崎", ("黒崎1丁目", "黒崎2丁目"))
        return None

    monkeypatch.setattr(router, "_match_district", fake_match_district)

    assert _has_plausible_item_text("黒崎でかごを捨てたい", "八幡西区") is True
