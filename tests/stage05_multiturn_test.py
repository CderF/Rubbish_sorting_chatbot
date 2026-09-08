"""STAGE05 マルチターン対応の結合テスト(`app/rag/router.py`)。

固定データを入れた一時DBに対して `_blocking_route` を会話の順に呼び、
前ターンのスロットが引き継がれること(5-1)と、追加・変更・訂正で組み立てが
変わること(5-2)を検証する。LLM もサーバーも使わない — 検索と context の
組み立てまでが検証範囲で、その先の文章化は採点・評価スクリプトの担当。

    python3 -m pytest tests/stage05_multiturn_test.py
"""

from __future__ import annotations

import datetime
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "stage05_test.db")

import pytest  # noqa: E402

from app.data import db  # noqa: E402
from app.data.schema import CollectionSchedule, WasteItem  # noqa: E402
from app.rag.router import _blocking_route, build_system_prompt  # noqa: E402

CRAWLED_AT = datetime.datetime(2026, 9, 8, 1, 0, 0)
HAYAMIHYO = "https://www.city.kitakyushu.lg.jp/contents/924_10041.html"
SCHEDULE_URL = "https://www.city.kitakyushu.lg.jp/contents/924_10102.html"


def _item(name: str, method: str, notes: str = "") -> WasteItem:
    return WasteItem(
        item_name=name,
        item_name_raw=name,
        disposal_method=method,
        notes=notes,
        source_url=HAYAMIHYO,
        source_page_title="ごみの分別早見表（さ行）",
        crawled_at=CRAWLED_AT,
    )


def _schedule(ward: str, town: str, waste_type: str, weekday: str) -> CollectionSchedule:
    return CollectionSchedule(
        ward=ward,
        town=town,
        waste_type=waste_type,
        weekday=weekday,
        source_url=SCHEDULE_URL,
        crawled_at=CRAWLED_AT,
    )


@pytest.fixture(scope="module", autouse=True)
def seeded_db():
    conn = db.connect()
    db.rebuild_db(
        conn,
        [
            _item("ソファ", "粗大ごみ", "事前に申し込みが必要です"),
            _item("冷蔵庫", "家電リサイクル法対象", "市では収集しません"),
            _item("洗濯機", "家電リサイクル法対象", "市では収集しません"),
            _item("ペットボトル", "資源化ごみ", "キャップとラベルを外す"),
        ],
        [
            # weekday は実データの表記そのまま(PR #20: ここに「曜日」を足さない)
            _schedule("小倉北区", "浅野1丁目", "家庭ごみ", "月曜日・木曜日"),
            _schedule("小倉北区", "浅野1丁目", "粗大ごみ", "第2水曜日"),
            _schedule("小倉北区", "篠崎1丁目", "家庭ごみ", "火曜日・金曜日"),
            _schedule("八幡西区", "黒崎", "家庭ごみ", "水曜日・土曜日"),
            _schedule("八幡西区", "黒崎", "粗大ごみ", "第4金曜日"),
        ],
    )
    conn.close()
    yield


# --- 5-1: 品名と地区が別のターンに分かれる ---------------------------------
def test_item_from_the_first_turn_is_still_searched_on_the_region_turn():
    first = _blocking_route("ソファを捨てたい")
    assert first.detected_item == "ソファ"
    assert "粗大ごみ" in first.context

    second = _blocking_route("小倉北区浅野1丁目です", state=first.state)

    # 前ターンの品目が context に残っている(履歴からの記憶に頼らない)
    assert "品名: ソファ" in second.context
    assert "事前に申し込みが必要です" in second.context
    # 今ターンで分かった地区の収集日も出ている
    assert "浅野1丁目" in second.context
    assert second.state.has_region


def test_a_region_only_turn_does_not_invent_an_item():
    """「…です」の語尾が品名として検索されない(修正前は品名「す」だった)。"""
    first = _blocking_route("ソファを捨てたい")
    second = _blocking_route("小倉北区浅野1丁目です", state=first.state)

    assert second.state.items == ("ソファ",)
    assert "該当する品目は見つかりませんでした" not in second.context


def test_bare_town_name_without_a_suffix_resolves():
    """「黒崎」のように町・丁目で終わらない町名も引ける(修正前は町名未検出)。"""
    ans = _blocking_route("八幡西区黒崎の家庭ごみの収集日は?")

    assert ans.detected_town == "黒崎"
    assert "水曜日・土曜日" in ans.context
    assert not ans.missing


def test_ward_without_a_town_still_asks_instead_of_guessing():
    ans = _blocking_route("小倉北区でソファを捨てたい")
    assert "町名" in ans.missing
    assert "町名が必要である" in build_system_prompt(ans)


def test_disposal_method_is_matched_to_the_collection_day_row():
    """5-1「組み合わせ」: どの収集日の行がその品物のものかを context で示す。"""
    first = _blocking_route("ソファを捨てたい")
    second = _blocking_route("小倉北区浅野1丁目です", state=first.state)

    assert "粗大ごみ: 第2水曜日 ← ソファはこの区分" in second.context


# --- 5-2: 追加 / 変更・訂正 -------------------------------------------------
def test_adding_a_second_item_keeps_the_first_in_the_answer():
    s = _blocking_route("ソファを捨てたい").state
    s = _blocking_route("小倉北区浅野1丁目です", state=s).state
    third = _blocking_route("冷蔵庫も追加でお願いします", state=s)

    assert third.state.items == ("ソファ", "冷蔵庫")
    assert "品名: ソファ" in third.context
    assert "品名: 冷蔵庫" in third.context
    # 地区は分かったままなので聞き返さない
    assert not third.missing
    assert "同じことを聞き返さない" in build_system_prompt(third)


def test_correcting_the_item_drops_the_old_premise():
    first = _blocking_route("冷蔵庫を捨てたい")
    second = _blocking_route("冷蔵庫じゃなくて洗濯機でした", state=first.state)

    assert second.state.items == ("洗濯機",)
    assert "品名: 冷蔵庫" not in second.context
    assert "品名: 洗濯機" in second.context
    assert "訂正前の品物・地区に基づく" in build_system_prompt(second)


def test_correcting_the_ward_rebuilds_the_schedule_from_the_new_premise():
    s = _blocking_route("ソファを捨てたい").state
    s = _blocking_route("小倉北区浅野1丁目です", state=s).state
    third = _blocking_route("すみません、やっぱり八幡西区黒崎でした", state=s)

    assert third.state.ward == "八幡西区"
    assert third.state.town == "黒崎"
    # 旧地区の収集日は消えている
    assert "浅野1丁目" not in third.context
    assert "第4金曜日" in third.context
    # 品目は保たれる
    assert third.state.items == ("ソファ",)


def test_multiple_items_are_announced_to_the_model():
    s = _blocking_route("ソファを捨てたい").state
    second = _blocking_route("冷蔵庫も", state=s)
    prompt = build_system_prompt(second)

    assert "相談に出ている品物は複数ある(ソファ・冷蔵庫)" in prompt


# --- 単発呼び出し(STAGE02/03)の互換 ---------------------------------------
def test_single_turn_calls_still_work_without_a_state():
    ans = _blocking_route("ペットボトルの捨て方")
    assert ans.found
    assert "キャップとラベルを外す" in ans.context
    assert ans.state is not None and ans.state.items == ("ペットボトル",)
