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

from app.rag.router import _search_item, resolve_canonical_alias
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
