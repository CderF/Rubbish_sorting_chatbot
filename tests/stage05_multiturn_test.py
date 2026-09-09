"""STAGE05 マルチターン対応の結合テスト(`app/rag/router.py`)。

固定データを入れた一時DBに対して `_blocking_route` を会話の順に呼び、
前ターンのスロットが引き継がれること(5-1)と、追加・変更・訂正で組み立てが
変わること(5-2)を検証する。LLM もサーバーも使わない — 検索と context の
組み立てまでが検証範囲で、その先の文章化は採点・評価スクリプトの担当。

    python3 -m pytest tests/stage05_multiturn_test.py
"""

from __future__ import annotations

import datetime
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from app.data import db, lookup  # noqa: E402
from app.data.schema import CollectionSchedule, WasteItem  # noqa: E402
from app.rag import router  # noqa: E402

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


@pytest.fixture(autouse=True)
def seeded_db(tmp_path, monkeypatch):
    """テストごとに空のDBを作り直し、キャッシュ済みの接続を捨てる。

    `app/data/lookup.py` は最初の照会でスレッドローカルに接続を1つ抱えるので、
    DB_PATH を差し替えるだけでは前のテストのDBを見続ける。他のテストモジュール
    (`notices_integration_test.py` 等)も同じ手で DB を差し替えるため、この
    reload を省くとファイル単独では通るのに全体では落ちる。区ごとの町名一覧も
    `functools.lru_cache` に載るので明示的に捨てる。
    """
    path = tmp_path / "stage05.db"
    monkeypatch.setenv("DB_PATH", str(path))
    monkeypatch.setenv("RAG_DB_PATH", str(path))

    conn = db.connect()
    db.rebuild_db(
        conn,
        [
            _item("ソファ", "粗大ごみ", "事前に申し込みが必要です"),
            _item("冷蔵庫", "家電リサイクル法対象", "市では収集しません"),
            _item("洗濯機", "家電リサイクル法対象", "市では収集しません"),
            _item("ペットボトル", "資源化ごみ", "キャップとラベルを外す"),
            # 家庭ごみは collection_schedule に実在する5種別の1つ。区分が
            # 分かれる相談(粗大ごみ + 家庭ごみ)を組めるようにしておく。
            _item("歯ブラシ", "家庭ごみ", "そのまま出せます"),
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

    importlib.reload(lookup)
    importlib.reload(router)
    router._town_candidates.cache_clear()
    router._district_groups.cache_clear()
    yield


# --- 5-1: 品名と地区が別のターンに分かれる ---------------------------------
def test_item_from_the_first_turn_is_still_searched_on_the_region_turn():
    first = router._blocking_route("ソファを捨てたい")
    assert first.detected_item == "ソファ"
    assert "粗大ごみ" in first.context

    second = router._blocking_route("小倉北区浅野1丁目です", state=first.state)

    # 前ターンの品目が context に残っている(履歴からの記憶に頼らない)
    assert "品名: ソファ" in second.context
    assert "事前に申し込みが必要です" in second.context
    # 今ターンで分かった地区の収集日も出ている
    assert "浅野1丁目" in second.context
    assert second.state.has_region


def test_a_region_only_turn_does_not_invent_an_item():
    """「…です」の語尾が品名として検索されない(修正前は品名「す」だった)。"""
    first = router._blocking_route("ソファを捨てたい")
    second = router._blocking_route("小倉北区浅野1丁目です", state=first.state)

    assert second.state.items == ("ソファ",)
    assert "該当する品目は見つかりませんでした" not in second.context


def test_bare_town_name_without_a_suffix_resolves():
    """「黒崎」のように町・丁目で終わらない町名も引ける(修正前は町名未検出)。"""
    ans = router._blocking_route("八幡西区黒崎の家庭ごみの収集日は?")

    assert ans.detected_town == "黒崎"
    assert "水曜日・土曜日" in ans.context
    assert not ans.missing


def test_ward_without_a_town_still_asks_instead_of_guessing():
    ans = router._blocking_route("小倉北区でソファを捨てたい")
    assert "町名" in ans.missing
    assert "町名が必要である" in router.build_system_prompt(ans)


def test_disposal_method_narrows_the_collection_day_lookup():
    """5-1「組み合わせ」: 引き継いだ品目の区分で収集日を絞る。

    このテストは当初「その町の全種別を並べ、どれがその品物の行かをラベルで
    示す」ことを期待していた。STAGE04 で収集日パスが品目の `disposal_method`
    を種別として検索するようになったため、そもそも関係ない種別の行は返って
    こない。ラベルより絞り込みのほうが強い保証なので、そちらを検証する。
    """
    first = router._blocking_route("ソファを捨てたい")
    second = router._blocking_route("小倉北区浅野1丁目です", state=first.state)

    assert "粗大ごみ: 第2水曜日" in second.context
    # 同じ町には家庭ごみの行もあるが、ソファの区分ではないので出てこない。
    assert "家庭ごみ" not in second.context
    assert [s.waste_type for s in second.schedule_hits] == ["粗大ごみ"]


def test_multiple_items_label_which_collection_day_row_is_theirs():
    """品目が複数あるときは、どの行がどの品物のものかを context で示す(5-1)。

    区分が分かれる相談(ソファ=粗大ごみ / 歯ブラシ=家庭ごみ)では収集日が
    複数行になるため、絞り込みだけでは対応づけが残らない。
    """
    s = router._blocking_route("ソファを捨てたい").state
    s = router._blocking_route("小倉北区浅野1丁目です", state=s).state
    third = router._blocking_route("歯ブラシも", state=s)

    assert third.state.items == ("ソファ", "歯ブラシ")
    assert "粗大ごみ: 第2水曜日 ← ソファはこの区分" in third.context
    assert "家庭ごみ: 月曜日・木曜日 ← 歯ブラシはこの区分" in third.context


def test_schedule_phrasing_does_not_poison_the_carried_over_item_name():
    """収集日を尋ねる言い回しでも、スロットに入るのは品名だけ(STAGE05)。

    品名候補は長い順に並ぶため、先頭は文中の最長断片になる。検索の種としては
    それが正しいが、そのままスロットに入れると会話が続く限り残ってしまう。
    早見表と一致した名前まで戻してから畳み込んでいることを確認する。
    """
    for prompt in (
        "ソファの収集日を知りたい",
        "ソファの収集日はいつですか",
        "ソファを捨てたい",
    ):
        assert router._blocking_route(prompt).state.items == ("ソファ",), prompt


def test_items_in_different_categories_each_get_their_collection_day():
    """区分が分かれる相談では、区分ごとに収集日を引いて両方案内する(5-1)。

    STAGE04 は品目が1つだったので、その品目の区分で1回引くだけだった。
    2品目で区分が分かれると、片方の区分でしか引かない実装では、もう一方の
    収集日が黙って落ちる。
    """
    s = router._blocking_route("ソファの収集日を知りたい").state
    s = router._blocking_route("小倉北区浅野1丁目です", state=s).state
    third = router._blocking_route("歯ブラシも", state=s)

    got = {(h.waste_type, h.weekday) for h in third.schedule_hits}
    assert got == {("粗大ごみ", "第2水曜日"), ("家庭ごみ", "月曜日・木曜日")}


def test_an_item_without_a_collection_weekday_does_not_hide_the_others():
    """曜日を持たない区分の品目が混ざっても、他の品目の収集日は残る。

    ペットボトルの区分「資源化ごみ」は collection_schedule に実在しない
    (回収の枠が曜日ではない)。この品目だけを見て収集日検索を打ち切ると、
    同じ相談に出ているソファの粗大ごみの日まで失われる。
    """
    s = router._blocking_route("ソファの収集日を知りたい").state
    s = router._blocking_route("小倉北区浅野1丁目です", state=s).state
    third = router._blocking_route("ペットボトルも", state=s)

    assert [h.waste_type for h in third.schedule_hits] == ["粗大ごみ"]
    # 曜日を持たない側は、存在しない収集日を作らせないよう明示する。
    assert "「ペットボトル」は決まった収集曜日を持たない区分" in third.context


# --- 5-2: 追加 / 変更・訂正 -------------------------------------------------
def test_adding_a_second_item_keeps_the_first_in_the_answer():
    s = router._blocking_route("ソファを捨てたい").state
    s = router._blocking_route("小倉北区浅野1丁目です", state=s).state
    third = router._blocking_route("冷蔵庫も追加でお願いします", state=s)

    assert third.state.items == ("ソファ", "冷蔵庫")
    assert "品名: ソファ" in third.context
    assert "品名: 冷蔵庫" in third.context
    # 地区は分かったままなので聞き返さない
    assert not third.missing
    assert "同じことを聞き返さない" in router.build_system_prompt(third)


def test_correcting_the_item_drops_the_old_premise():
    first = router._blocking_route("冷蔵庫を捨てたい")
    second = router._blocking_route("冷蔵庫じゃなくて洗濯機でした", state=first.state)

    assert second.state.items == ("洗濯機",)
    assert "品名: 冷蔵庫" not in second.context
    assert "品名: 洗濯機" in second.context
    assert "訂正前の品物・地区に基づく" in router.build_system_prompt(second)


def test_correcting_the_ward_rebuilds_the_schedule_from_the_new_premise():
    s = router._blocking_route("ソファを捨てたい").state
    s = router._blocking_route("小倉北区浅野1丁目です", state=s).state
    third = router._blocking_route("すみません、やっぱり八幡西区黒崎でした", state=s)

    assert third.state.ward == "八幡西区"
    assert third.state.town == "黒崎"
    # 旧地区の収集日は消えている
    assert "浅野1丁目" not in third.context
    assert "第4金曜日" in third.context
    # 品目は保たれる
    assert third.state.items == ("ソファ",)


def test_multiple_items_are_announced_to_the_model():
    s = router._blocking_route("ソファを捨てたい").state
    second = router._blocking_route("冷蔵庫も", state=s)
    prompt = router.build_system_prompt(second)

    assert "相談に出ている品物は複数ある(ソファ・冷蔵庫)" in prompt


# --- 単発呼び出し(STAGE02/03)の互換 ---------------------------------------
def test_single_turn_calls_still_work_without_a_state():
    ans = router._blocking_route("ペットボトルの捨て方")
    assert ans.found
    assert "キャップとラベルを外す" in ans.context
    assert ans.state is not None and ans.state.items == ("ペットボトル",)
