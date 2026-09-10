"""`resolve_canonical_alias`(STAGE05 実採点の失敗題レビュー、2026-09-14)の
単体テスト。

B1実機の失敗トレース(ユーザー提供)で確認された、品名の言い換えが
部分一致・ベクトル検索の揺れで無関係な行に化ける不具合:
  - 5-1 #15: 「テレビ」→「テレビアンテナ」等(正しくは「テレビ(ブラウン管式・
    液晶式・プラズマ式)」)
  - 5-1 #17: 「お酒の空き瓶」→「酒パック」(正しくは「びん(飲料・食品の
    びん)」)
  - 5-2 #34: 「カップめんの容器」→「スープ袋」等(正しくは「カップめんなどの
    容器・フィルム」)

`_CANONICAL_ALIASES` はこのリポジトリの実DB(`var/kitakyushu_waste.db`)に
実在する行名で組んである。ここでは実DB越しに `_search_item`
(決定論的経路)と `execute_tool("search_waste_item", ...)`(agentic経路)の
両方から解決できることを確認する ― どちらも同じ `resolve_canonical_alias`
を経由するので、片方だけ直すという再発を防ぐ。
"""

from __future__ import annotations

from app.rag.router import (
    _resolve_embedded_alias,
    _search_item,
    resolve_canonical_alias,
)
from app.rag.tools import execute_tool


def test_tv_alias_resolves_to_the_actual_recycling_law_item():
    assert resolve_canonical_alias("テレビ") == "テレビ(ブラウン管式・液晶式・プラズマ式)"


def test_sake_bottle_alias_resolves_to_the_beverage_bottle_row():
    assert resolve_canonical_alias("お酒の空き瓶") == "びん(飲料・食品のびん)"
    assert resolve_canonical_alias("酒瓶") == "びん(飲料・食品のびん)"


def test_wine_bottle_alias_resolves_to_its_own_row():
    assert resolve_canonical_alias("ワインのびん") == "ワインびん"


def test_cup_noodle_container_alias_resolves_to_the_plastic_row():
    assert resolve_canonical_alias("カップめんの容器") == "カップめんなどの容器・フィルム"


def test_unknown_query_has_no_alias():
    assert resolve_canonical_alias("テレビ台") is None
    assert resolve_canonical_alias("傘") is None


def test_deterministic_path_uses_the_alias_before_partial_match_ordering():
    hits, stage = _search_item("テレビ")
    assert stage == "exact"
    assert len(hits) == 1
    assert hits[0].item_name == "テレビ(ブラウン管式・液晶式・プラズマ式)"
    assert hits[0].disposal_method == "市が収集しないもの"


def test_agentic_tool_uses_the_same_alias():
    result = execute_tool("search_waste_item", {"query": "お酒の空き瓶"})
    assert result["match_stage"] == "exact"
    assert result["items"][0]["item_name"] == "びん(飲料・食品のびん)"
    assert result["items"][0]["disposal_method"] == "かん・びん"


def test_cup_noodle_container_note_about_paper_containers_is_preserved():
    """5-2 #34 の遺漏(「紙製の容器は家庭ごみ」)は、正しい行にさえ解決できれば
    既存の備考にもう入っている ― 追加のテキストは要らない。"""
    result = execute_tool("search_waste_item", {"query": "カップめんの容器"})
    assert "紙製の容器は家庭ごみ" in result["items"][0]["notes"]


# --- 第6回実採点(2026-09-10)で追加した別名 ---------------------------------
# 化学物質名は早見表に品名として存在せず、包括行(「薬品(家庭医薬品でない
# もの)」等)にしか対応が無い。ベクトル類似に任せると「硫酸」が
# 「油(食用油)=家庭ごみ」に化けていた。


def test_hazard_substance_aliases_resolve_to_the_consult_only_row():
    for name in ("硫酸", "塩酸", "硝酸", "化学薬品", "劇物", "毒物"):
        assert resolve_canonical_alias(name) == "薬品(家庭医薬品でないもの)"


def test_syringe_and_pesticide_container_aliases():
    assert resolve_canonical_alias("注射器") == "注射筒"
    assert resolve_canonical_alias("農薬の空き容器") == "農薬びん(家庭用)(中身のないもの)"


def test_sofa_alias_covers_the_long_vowel_variant():
    assert resolve_canonical_alias("ソファ") == "ソファー"


def test_embedded_alias_finds_the_substance_name_inside_a_sentence():
    """候補抽出で切り出せない文でも、物質名が主辞として現れていれば解決する。"""
    assert (
        _resolve_embedded_alias("理科の実験で使った硫酸が残っているのですが")
        == "薬品(家庭医薬品でないもの)"
    )


def test_embedded_alias_respects_the_after_boundary():
    """「テレビ台」の中の「テレビ」は別品目なので拾わない。"""
    assert _resolve_embedded_alias("テレビ台を捨てたいです") is None
    assert _resolve_embedded_alias("ソファーベッドを捨てたいです") is None


def test_embedded_alias_accepts_a_part_suffix():
    """「の」+部位を表す語が続いても、指している品物は同じ。"""
    assert _resolve_embedded_alias("注射器の本体は燃えるごみですか") == "注射筒"


def test_deterministic_path_resolves_a_substance_through_the_embedded_alias():
    hits, stage = _search_item(
        "理科の実験で使った硫酸が残っているのですが、どう処分すればいいですか？"
    )
    assert stage == "exact"
    assert [h.item_name for h in hits] == ["薬品(家庭医薬品でないもの)"]
    assert hits[0].disposal_method == "市が収集しないもの"


def test_agentic_tool_resolves_the_same_embedded_alias():
    result = execute_tool(
        "search_waste_item",
        {"query": "理科の実験で使った硫酸が残っているのですが、どう処分すればいいですか？"},
    )
    assert result["match_stage"] == "exact"
    assert result["items"][0]["item_name"] == "薬品(家庭医薬品でないもの)"


def test_pesticide_container_resolves_to_the_household_waste_row():
    """「資源ごみにそのまま出していい？」に対し、正しい区分は家庭ごみ。"""
    result = execute_tool(
        "search_waste_item",
        {"query": "農薬の空き容器は、洗わずにそのまま資源ごみに出していいですか？"},
    )
    assert result["match_stage"] == "exact"
    assert result["items"][0]["item_name"] == "農薬びん(家庭用)(中身のないもの)"
    assert result["items"][0]["disposal_method"] == "家庭ごみ"
