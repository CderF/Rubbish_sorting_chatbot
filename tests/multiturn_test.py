"""`app/rag/multiturn.py`(STAGE05 5-1/5-2 の複数ターン一貫性)の単体テスト。

B1実機の生会話6本(2026-09-10)で再現した2つの不具合をそのまま回帰させる:
  - 5-2a: 「ソファを捨てたい」→(地区)→「冷蔵庫も捨てたいです」で、
    最後の回答からソファが消える。
  - 5-2c: 地域だけを訂正するターンで、同じ品目の類推対象がターンごとに
    揺れる(「ソファーベッド」→「つっぱり棒」)。

vllm/GPU に依存しない(`app/rag/agent.py` と同じ理由でこのテストは
このマシンでも走らせられる設計にしてある)。`execute_tool` は
`app.rag.multiturn` が束縛した参照を直接差し替える(`tests/agent_loop_test.py`
と同じ手法)。
"""

from __future__ import annotations

from types import SimpleNamespace

from app.rag import multiturn
from app.rag.multiturn import (
    SessionState,
    classify_turn,
    merge_with_state,
    update_state,
)
from app.rag.types import RoutedAnswer


def _item_hit(name: str, disposal_method: str = "粗大ごみ", source_url: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        item_name=name, disposal_method=disposal_method, notes="", source_url=source_url
    )


def _routed(*, item_hits=(), ward=None, town=None, detected_item=None, found=None) -> RoutedAnswer:
    ans = RoutedAnswer(
        found=found if found is not None else bool(item_hits),
        item_hits=list(item_hits),
        detected_item=detected_item,
        detected_ward=ward,
        detected_town=town,
    )
    ans.context = "stub"
    return ans


# --- classify_turn -----------------------------------------------------
def test_classify_turn_detects_correction_markers():
    assert classify_turn("冷蔵庫じゃなくて洗濯機でした") == "correction"
    assert classify_turn("すみません、やっぱり八幡西区黒崎でした") == "correction"


def test_classify_turn_defaults_to_keep():
    assert classify_turn("冷蔵庫も捨てたいです") == "keep"
    assert classify_turn("ソファを捨てたい") == "keep"


def test_classify_turn_does_not_treat_kanarazu_correction_words_as_correction():
    """「間違いなく」「間違いありません」は「確実に」の意味の慣用句で、
    前の品目を置き換える訂正ではない(2026-09-10 レビュー指摘の回帰)。"""
    assert classify_turn("冷蔵庫って間違いなく粗大ごみですか") == "keep"
    assert classify_turn("それは間違いありません") == "keep"


def test_classify_turn_still_detects_genuine_correction_with_machigai():
    assert classify_turn("冷蔵庫は間違いでした、洗濯機です") == "correction"
    assert classify_turn("さっきの品目、間違えました") == "correction"


# --- update_state --------------------------------------------------------
def test_update_state_first_item_is_recorded_by_resolved_name():
    state = SessionState()
    routed = _routed(item_hits=[_item_hit("ソファーベッド")], ward="小倉北区", town="浅野1丁目")
    update_state(state, "ソファを捨てたい", routed)
    assert state.items == ["ソファーベッド"]
    assert state.ward == "小倉北区"
    assert state.town == "浅野1丁目"


def test_update_state_addition_keeps_previous_item():
    state = SessionState(items=["ソファーベッド"], ward="小倉北区", town="浅野1丁目")
    routed = _routed(item_hits=[_item_hit("冷蔵庫(冷凍冷蔵庫)", "市が収集しないもの")])
    update_state(state, "冷蔵庫も捨てたいです", routed)
    assert state.items == ["ソファーベッド", "冷蔵庫(冷凍冷蔵庫)"]


def test_update_state_correction_replaces_previous_item():
    state = SessionState(items=["冷蔵庫(冷凍冷蔵庫)"])
    routed = _routed(item_hits=[_item_hit("洗濯機")])
    update_state(state, "冷蔵庫じゃなくて洗濯機でした", routed)
    assert state.items == ["洗濯機"]


def test_update_state_no_item_this_turn_leaves_items_untouched():
    state = SessionState(items=["ソファーベッド"], ward="小倉北区")
    routed = _routed(item_hits=[], ward=None, town="浅野1丁目")
    update_state(state, "小倉北区浅野1丁目です", routed)
    assert state.items == ["ソファーベッド"]
    assert state.town == "浅野1丁目"


def test_update_state_repeating_same_item_does_not_duplicate():
    state = SessionState(items=["ソファーベッド"])
    routed = _routed(item_hits=[_item_hit("ソファーベッド")])
    update_state(state, "ソファはどう捨てますか", routed)
    assert state.items == ["ソファーベッド"]


def test_update_state_records_every_item_hit_not_just_the_first(monkeypatch):
    """`item_hits[0]` だけでなく全件を見る(2026-09-10 レビュー指摘の回帰)。
    現行の絞り込み設計では1ターンに複数ヒットすることは稀だが、将来その
    絞り込みが緩んでも静かに取りこぼさないことを保証する。"""
    state = SessionState()
    routed = _routed(item_hits=[_item_hit("ソファーベッド"), _item_hit("冷蔵庫(冷凍冷蔵庫)")])
    update_state(state, "ソファと冷蔵庫を捨てたい", routed)
    assert state.items == ["ソファーベッド", "冷蔵庫(冷凍冷蔵庫)"]


# --- merge_with_state ------------------------------------------------------
def test_merge_is_a_no_op_for_the_common_single_item_case(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("execute_tool should not be called for the single-item case")

    monkeypatch.setattr(multiturn, "execute_tool", boom)

    state = SessionState(items=["ソファーベッド"], ward="小倉北区", town="浅野1丁目")
    routed = _routed(item_hits=[_item_hit("ソファーベッド")], ward="小倉北区", town="浅野1丁目")
    result = merge_with_state("ソファを捨てたい", state, routed)
    assert result is routed


def test_merge_skips_canned_response_and_safety_gate(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("execute_tool should not be called")

    monkeypatch.setattr(multiturn, "execute_tool", boom)

    state = SessionState(items=["ソファーベッド", "冷蔵庫"])
    routed = _routed()
    routed.canned_response = "固定回答"
    assert merge_with_state("何か", state, routed) is routed

    routed2 = _routed()
    routed2.safety_gate = "6-2"
    assert merge_with_state("何か", state, routed2) is routed2


def test_merge_adds_back_the_dropped_first_item_when_a_second_item_is_added(monkeypatch):
    """5-2a の回帰テスト: 「ソファを捨てたい」→(地区)→「冷蔵庫も捨てたいです」
    で、最終的な context・item_hits に両方の品目が残ること。"""

    def fake_execute_tool(name, arguments):
        if name == "search_waste_item":
            catalog = {
                "ソファーベッド": {
                    "item_name": "ソファーベッド",
                    "disposal_method": "粗大ごみ",
                    "notes": "事前申込制",
                    "source_url": "https://example.test/sofa",
                },
                "冷蔵庫(冷凍冷蔵庫)": {
                    "item_name": "冷蔵庫(冷凍冷蔵庫)",
                    "disposal_method": "市が収集しないもの",
                    "notes": "家電リサイクル法対象",
                    "source_url": "https://example.test/fridge",
                },
            }
            hit = catalog.get(arguments["query"])
            return {"match_stage": "exact" if hit else "none", "items": [hit] if hit else []}
        if name == "lookup_item_change_notice":
            return {"has_pending_change": False}
        if name == "lookup_collection_schedule":
            if arguments.get("waste_type") == "粗大ごみ":
                return {
                    "schedules": [
                        {
                            "ward": "小倉北区",
                            "town": "浅野1丁目",
                            "waste_type": "粗大ごみ",
                            "weekday": "第4月曜日",
                            "source_url": "https://example.test/schedule-sofa",
                        }
                    ]
                }
            return {"schedules": []}  # 市が収集しないものは収集曜日の概念が無い
        raise AssertionError(f"unexpected tool call: {name} {arguments}")

    monkeypatch.setattr(multiturn, "execute_tool", fake_execute_tool)

    state = SessionState(items=["ソファーベッド"], ward="小倉北区", town="浅野1丁目")
    routed = _routed(
        item_hits=[_item_hit("冷蔵庫(冷凍冷蔵庫)", "市が収集しないもの")],
        ward=None,
        town=None,
        detected_item="冷蔵庫",
    )

    # 実際の呼び出し順(app/server.py._blocking_merge)と揃える: 先に状態を
    # 更新してから合算する。
    update_state(state, "冷蔵庫も捨てたいです", routed)
    merged = merge_with_state("冷蔵庫も捨てたいです", state, routed)

    names = {getattr(h, "item_name", "") for h in merged.item_hits}
    assert names == {"ソファーベッド", "冷蔵庫(冷凍冷蔵庫)"}
    assert "ソファーベッド" in merged.context
    assert "冷蔵庫" in merged.context
    assert state.items == ["ソファーベッド", "冷蔵庫(冷凍冷蔵庫)"]


def test_merge_refetches_the_carried_item_on_a_region_only_turn(monkeypatch):
    """5-2c の回帰テスト: 地域だけを訂正するターンでも、前ターンの品目を
    コード側で確定名のまま再検索するので類推対象が揺れない。"""

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments):
        calls.append((name, dict(arguments)))
        if name == "search_waste_item":
            assert arguments["query"] == "ソファーベッド"
            return {
                "match_stage": "exact",
                "items": [
                    {
                        "item_name": "ソファーベッド",
                        "disposal_method": "粗大ごみ",
                        "notes": "事前申込制",
                        "source_url": "https://example.test/sofa",
                    }
                ],
            }
        if name == "lookup_item_change_notice":
            return {"has_pending_change": False}
        if name == "lookup_collection_schedule":
            assert arguments == {
                "ward": "八幡西区",
                "town": "黒崎",
                "waste_type": "粗大ごみ",
            }
            return {
                "schedules": [
                    {
                        "ward": "八幡西区",
                        "town": "黒崎1丁目",
                        "waste_type": "粗大ごみ",
                        "weekday": "第1月曜日",
                        "source_url": "https://example.test/schedule",
                    }
                ]
            }
        raise AssertionError(f"unexpected tool call: {name} {arguments}")

    monkeypatch.setattr(multiturn, "execute_tool", fake_execute_tool)

    state = SessionState(items=["ソファーベッド"], ward="小倉北区", town="浅野1丁目")
    # このターンの発言には品目名が一切出てこない(地域訂正のみ)。
    routed = _routed(item_hits=[], ward="八幡西区", town="黒崎", found=True)

    prompt = "すみません、やっぱり八幡西区黒崎でした"
    update_state(state, prompt, routed)
    merged = merge_with_state(prompt, state, routed)

    assert [h.item_name for h in merged.item_hits] == ["ソファーベッド"]
    assert merged.schedule_hits[0].weekday == "第1月曜日"
    assert ("search_waste_item", {"query": "ソファーベッド"}) in calls
    assert state.ward == "八幡西区"
    assert state.town == "黒崎"


def test_merge_falls_back_to_original_when_refetch_finds_nothing(monkeypatch):
    def fake_execute_tool(name, arguments):
        if name == "search_waste_item":
            return {"match_stage": "none", "items": []}
        raise AssertionError(f"unexpected tool call: {name}")

    monkeypatch.setattr(multiturn, "execute_tool", fake_execute_tool)

    state = SessionState(items=["ソファーベッド", "冷蔵庫(冷凍冷蔵庫)"])
    routed = _routed(item_hits=[_item_hit("冷蔵庫(冷凍冷蔵庫)")])
    result = merge_with_state("冷蔵庫も捨てたいです", state, routed)
    assert result is routed


def test_merge_carries_the_item_forward_on_a_followup_with_no_region_at_all(monkeypatch):
    """2026-09-10 レビュー指摘の回帰: 地域が一度もセッションに出てきていない
    ケースでも、品目だけの追問(「それは何色の袋に入れますか」)で前ターンの
    品目を毎回コード側で再検索して合流させる。以前は ward/town か
    `routed.found` のどちらかが立っていないと合流しなかったため、地域を
    一度も話していない純粋な品目追問だけの会話がこの経路から漏れていた。"""

    def fake_execute_tool(name, arguments):
        if name == "search_waste_item":
            assert arguments == {"query": "ペットボトル"}
            return {
                "match_stage": "exact",
                "items": [
                    {
                        "item_name": "ペットボトル",
                        "disposal_method": "ペットボトル",
                        "notes": "キャップとラベルは外して家庭ごみへ",
                        "source_url": "https://example.test/petbottle",
                    }
                ],
            }
        if name == "lookup_item_change_notice":
            return {"has_pending_change": False}
        raise AssertionError(f"unexpected tool call: {name} {arguments}")

    monkeypatch.setattr(multiturn, "execute_tool", fake_execute_tool)

    state = SessionState(items=["ペットボトル"])  # ward/town は一度も確立していない
    routed = _routed(item_hits=[], ward=None, town=None, found=False)

    merged = merge_with_state("それは何色の袋に入れますか", state, routed)

    assert [h.item_name for h in merged.item_hits] == ["ペットボトル"]
    assert "ペットボトル" in merged.context
    assert merged.schedule_hits == []


def test_merge_preserves_this_turns_own_general_info_alongside_the_carried_item(
    monkeypatch,
):
    """合流時に今回のターン自身が見つけていた内容(一般説明等、品目とは
    無関係)を消さない(2026-09-10 レビュー指摘)。"""

    def fake_execute_tool(name, arguments):
        if name == "search_waste_item":
            return {
                "match_stage": "exact",
                "items": [
                    {
                        "item_name": "ペットボトル",
                        "disposal_method": "ペットボトル",
                        "notes": "",
                        "source_url": "https://example.test/petbottle",
                    }
                ],
            }
        if name == "lookup_item_change_notice":
            return {"has_pending_change": False}
        raise AssertionError(f"unexpected tool call: {name}")

    monkeypatch.setattr(multiturn, "execute_tool", fake_execute_tool)

    state = SessionState(items=["ペットボトル"])
    routed = _routed(item_hits=[], found=True)
    routed.context = "## 一般説明の検索結果\n- 指定袋は透明または半透明であること"

    merged = merge_with_state("指定袋の色に決まりはありますか", state, routed)

    assert "指定袋は透明または半透明であること" in merged.context
    assert "ペットボトル" in merged.context
