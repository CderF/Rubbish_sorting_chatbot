"""要件4-3: 早見表に個別記載のない品目は、類似品目に準じた案内 + 断り書き。

vector_search_items 自体(ベクトルRAGパス)は issue #7 で実装・テスト済み
(tests/vector_store_test.py)。ここで検証するのは STAGE04 が足した薄い層だけ:
_search_item がヒット段階を返すこと、build_system_prompt がそれに応じて
「類推である」規則を出すこと。LLM 出力そのものは対象外(決定的な部分のみ)。
"""

from __future__ import annotations

from types import SimpleNamespace

import app.rag.router as router
from app.rag.router import _search_item, build_system_prompt
from app.rag.types import RoutedAnswer


def _rec(name: str):
    # ItemRecord is a Protocol; the real impl returns a pydantic WasteItem.
    # Any object with the right attributes satisfies it structurally.
    return SimpleNamespace(
        item_name=name,
        disposal_method="家庭ごみ",
        notes="",
        source_url="https://www.city.kitakyushu.lg.jp/contents/x.html",
        source_page_title="早見表",
        score=0.9,
    )


def test_search_item_reports_the_tier(monkeypatch):
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: _rec(n))
    assert _search_item("テレビ") == ([_rec("テレビ")], "exact")

    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [_rec("テレビ台")])
    assert _search_item("テレビ")[1] == "partial"

    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [])
    monkeypatch.setattr(router, "vector_search_items", lambda q, top_k=3: [_rec("充電式電池")])
    hits, tier = _search_item("モバイルバッテリー")
    assert tier == "similar" and hits[0].item_name == "充電式電池"

    monkeypatch.setattr(router, "vector_search_items", lambda q, top_k=3: [])
    assert _search_item("宇宙服") == ([], "none")


def test_prompt_adds_the_analogy_caveat_only_for_similar():
    for tier in ("exact", "partial", "none"):
        ans = RoutedAnswer(found=True, item_match=tier)
        assert "類似する" not in build_system_prompt(ans)

    ans = RoutedAnswer(found=True, item_match="similar")
    prompt = build_system_prompt(ans)
    assert "類似する" in prompt
    assert "個別の記載がないため" in prompt
    assert "確認してください" in prompt


def test_blocking_route_sets_item_match(monkeypatch):
    monkeypatch.setattr(router, "lookup_item_exact", lambda n: None)
    monkeypatch.setattr(router, "lookup_item_partial", lambda n, limit=10: [])
    monkeypatch.setattr(router, "vector_search_items", lambda q, top_k=3: [_rec("充電式電池")])
    monkeypatch.setattr(router, "lookup_collection_schedule", lambda *a, **k: None)

    ans = router._blocking_route("モバイルバッテリーはどう捨てますか")
    assert ans.item_match == "similar"
    assert "類似する" in build_system_prompt(ans)
