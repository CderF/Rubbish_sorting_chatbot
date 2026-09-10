"""`app/rag/agent.py` の `_build_context_text`/`_filter_schedules_to_known_items`
(STAGE05 5-2 #32/#33/#37、2026-09-14)の単体テスト。

B1実機の失敗トレース(ユーザー提供)で確認された、複数品目を1つの context に
束ねるときの相互汚染:
  - 5-2 #32: 洗濯機(市が収集しないもの)なのに粗大ごみ収集日が付いてくる。
  - 5-2 #33: 毛布(家庭ごみ)が布団(粗大ごみ)の事前申込条件を引き継ぐ。
  - 5-2 #37: 自転車+マットレス(両方粗大ごみ、同じ収集日)なのに根拠なく
    「別々に申し込む必要がある」と示唆する条件が付く。

vllm/GPU に依存しない(このモジュールの既存の制約と同じ)。
"""

from __future__ import annotations

from types import SimpleNamespace

from app.rag.agent import _build_context_text, _filter_schedules_to_known_items


def _item(name: str, disposal_method: str, notes: str = "") -> SimpleNamespace:
    return SimpleNamespace(item_name=name, disposal_method=disposal_method, notes=notes)


def _sched(ward: str, town: str, waste_type: str, weekday: str) -> SimpleNamespace:
    return SimpleNamespace(ward=ward, town=town, waste_type=waste_type, weekday=weekday)


def test_schedule_for_a_waste_type_with_no_matching_item_is_dropped():
    """5-2 #32 の回帰: 洗濯機(市が収集しないもの)しか品目が無いのに、
    モデルが粗大ごみの収集日を(自発的または混乱して)呼んでいても、
    無関係な種別として context から落とす。"""
    items = [_item("洗濯機(乾燥機付きのものを含む)", "市が収集しないもの")]
    schedules = [_sched("小倉北区", "浅野1丁目", "粗大ごみ", "第4月曜日")]
    kept = _filter_schedules_to_known_items(items, schedules)
    assert kept == []


def test_matching_schedule_for_a_curbside_item_is_kept():
    items = [_item("布団", "粗大ごみ")]
    schedules = [_sched("小倉北区", "浅野1丁目", "粗大ごみ", "第2火曜日")]
    assert _filter_schedules_to_known_items(items, schedules) == schedules


def test_duplicate_schedules_are_deduped():
    items = [_item("自転車", "粗大ごみ"), _item("マットレス", "粗大ごみ")]
    same = _sched("小倉北区", "浅野1丁目", "粗大ごみ", "第4月曜日")
    kept = _filter_schedules_to_known_items(items, [same, same])
    assert len(kept) == 1


def test_advance_application_note_does_not_leak_to_a_household_waste_item():
    """5-2 #33 の回帰: 布団(粗大ごみ、事前申込制)+毛布(家庭ごみ)を一緒に
    案内するとき、事前申込制の注記が毛布の行にまで及んではならない。"""
    items = [_item("布団", "粗大ごみ", "事前申込制"), _item("毛布", "家庭ごみ")]
    schedules = [
        _sched("門司区", "老松町", "粗大ごみ", "第2火曜日"),
        _sched("門司区", "老松町", "家庭ごみ", "火曜日・金曜日"),
    ]
    context = _build_context_text(items, schedules, [], [])

    lines = context.splitlines()
    futon_idx = next(i for i, ln in enumerate(lines) if "粗大ごみ: 第2火曜日" in ln)
    moufu_idx = next(i for i, ln in enumerate(lines) if "家庭ごみ: 火曜日・金曜日" in ln)

    assert "事前申込制" in lines[futon_idx + 1]
    # 毛布の行の直後には事前申込制の注記が無い(布団の行にだけ付いている)。
    assert "事前申込制" not in lines[moufu_idx + 1] if moufu_idx + 1 < len(lines) else True
    assert context.count("事前申込制。上記の収集曜日") == 1


def test_multiple_bulky_items_get_no_application_count_inference_either_way():
    """2026-09-14 レビュー指摘: 「まとめて1回の申込で出せる」という指示も、
    検索結果に書かれていない申込回数を推測している点では「別々に申し込む
    必要がある」と同じ誤りだった。複数の粗大ごみ品目を束ねても、
    どちらの方向にも申込回数を推測する文言を context に足さない ―
    備考にそう書かれていない限り、一括か個別かは検索結果からは判断できない。"""
    items = [_item("自転車", "粗大ごみ", "事前予約制"), _item("マットレス", "粗大ごみ", "事前申込制")]
    schedules = [
        _sched("小倉北区", "浅野1丁目", "粗大ごみ", "第4月曜日"),
        _sched("小倉北区", "浅野1丁目", "粗大ごみ", "第4月曜日"),
    ]
    context = _build_context_text(items, schedules, [], [])
    assert "まとめて1回の申込で" not in context
    assert "別々に申し込む" not in context
