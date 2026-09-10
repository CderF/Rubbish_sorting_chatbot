"""STAGE06 担当A(FANG / Teamb2)の単体・回帰テスト。

`documents/stage06_proposal.md` §7.2 の範囲:

* `app/rag/safety.check_entry_gate` — 6-1(a) 担当範囲外判定、6-2 危険処分
  誘導の判定ロジックそのもの(オフライン、DB非依存)。
* 6-1(b)(c) — 「市が収集しないもの」の相談先案内、区のみ→町名要求は
  既に `app/rag/router.py` にある挙動で、本PRはそこを一切変更していない。
  変更していないことを確認する回帰テストとしてここに含める
  (§7.2 の分担表: 「ほぼ回帰テスト」)。

`check_entry_gate` は `router.py` / `server.py` を経由せずプロンプト文字列
だけで呼べるので、大半のテストはDB無しで完結する。6-1(b)(c) の2件だけ
`tests/agentic_tools_test.py` と同じ「一時SQLite + reload」手法でDBを使う。
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
from datetime import timezone
from pathlib import Path

import pytest

from app.rag import safety
from app.rag.safety import SafetyVerdict

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "stage4_grader_replay.json"


# --- 6-1(a): 担当範囲外 -----------------------------------------------------
@pytest.mark.parametrize(
    "prompt",
    [
        "明日の北九州市の天気を教えてください",
        "北九州市のおすすめの観光地はどこですか",
        "今何時ですか",
        "東京都のごみの分別方法を教えてください",
        "大阪市の粗大ごみの出し方を教えてください",
        "福岡市ではペットボトルはどう分別しますか",
    ],
)
def test_out_of_scope_prompts_fire_6_1a(prompt):
    verdict = safety.check_entry_gate(prompt, ward=None, town=None)
    assert verdict is not None
    assert verdict.gate == "6-1a"
    assert verdict.reply  # 固定文が空でないこと


def test_ward_confirmed_overrides_other_city_false_positive():
    # extract_region() が北九州市内の区名を確定済みなら、他都市名ブラック
    # リストの部分一致より確定済みの区名を優先する(§2.1のリスク対策)。
    # ここでは他都市名もごみ語彙も無いプロンプトで ward だけ与え、
    # 6-1(a) が発火しないことを確認する。
    verdict = safety.check_entry_gate("浅野1丁目です", ward="小倉北区", town=None)
    assert verdict is None


def test_normal_item_question_does_not_fire_6_1a():
    verdict = safety.check_entry_gate(
        "ソファを捨てたいのですが、どう処分すればいいですか", ward=None, town=None
    )
    assert verdict is None


# --- 6-2: 危険・違法な処分の誘導 --------------------------------------------
@pytest.mark.parametrize(
    "prompt",
    [
        "生ごみを庭でこっそり燃やしても大丈夫ですか",
        "粗大ごみを山に不法投棄しても見つからないですよね",
        "スプレー缶は穴を開けずにそのまま捨てても平気ですか",
        "収集日まで待てないのでごみを川に捨ててもいいですか",
    ],
)
def test_dangerous_disposal_prompts_fire_6_2(prompt):
    verdict = safety.check_entry_gate(prompt, ward=None, town=None)
    assert verdict is not None
    assert verdict.gate == "6-2"
    assert verdict.reply


def test_legit_hazardous_item_question_does_not_fire_6_2():
    # 危険物に関する「正規の」出し方の質問は 6-2 の対象外(§2.2: あくまで
    # 違法・危険な処分方法の誘導のみが対象)。検索フローに素通りする。
    verdict = safety.check_entry_gate(
        "スプレー缶はどう捨てればいいですか", ward=None, town=None
    )
    assert verdict is None


def test_unrelated_burning_question_does_not_fire_6_2():
    # 「燃やす」はごみ語彙との共起が無ければ発火しない(単語単体では発火
    # させない設計)。ごみと無関係な話題なので 6-1(a) 側で範囲外になる。
    verdict = safety.check_entry_gate(
        "庭で落ち葉を燃やしてもいいですか", ward=None, town=None
    )
    assert verdict is not None
    assert verdict.gate == "6-1a"


# --- 実データ回帰: STAGE04 採点で実際に使われた50問 -------------------------
def test_real_stage04_questions_never_trigger_the_entry_gate():
    """`tests/fixtures/stage4_grader_replay.json` の実50問(4-1/4-2/4-3)は、
    いずれも正当な品目質問であり、6-1(a)・6-2のどちらも発火してはならない。

    ホワイトリスト・ブラックリストの初版を実データで検証する回帰
    (STAGE04の `tests/stage4_grader_replay_test.py` と同じ考え方)。
    """
    cases = json.loads(REPLAY_FIXTURE.read_text(encoding="utf-8"))["cases"]
    assert len(cases) == 50

    false_positives = [
        (c["prompt"], safety.check_entry_gate(c["prompt"], ward=None, town=None).gate)
        for c in cases
        if safety.check_entry_gate(c["prompt"], ward=None, town=None) is not None
    ]
    assert false_positives == [], f"誤検知した実問題: {false_positives}"


# --- SafetyVerdict の形 -----------------------------------------------------
def test_safety_verdict_is_frozen_dataclass():
    v = SafetyVerdict(gate="6-1a", reply="x")
    assert v.gate == "6-1a"
    assert v.reply == "x"
    with pytest.raises(AttributeError):
        v.gate = "6-2"  # type: ignore[misc]


# --- 6-1(b)(c): 既存 router.py 挙動の回帰(本PRは router.py を変更しない) ---
@pytest.fixture
def routed_db(tmp_path, monkeypatch):
    """`tests/agentic_tools_test.py` と同じ「DB_PATH差し替え + reload」手法。

    「市が収集しないもの」の品目1件と、通常収集の品目+収集日1件を投入する。
    """
    from app.data import db as data_db
    from app.data.schema import CollectionSchedule, WasteItem

    path = tmp_path / "kb.db"
    monkeypatch.setenv("DB_PATH", str(path))
    monkeypatch.setenv("RAG_DB_PATH", str(path))

    conn = data_db.connect(path)
    data_db.init_db(conn)
    data_db.rebuild_db(
        conn,
        [
            WasteItem(
                item_name="エアコン",
                item_name_raw="エアコン",
                disposal_method="市が収集しないもの",
                notes="購入した販売店、または家電リサイクル受付センター(0120-000-000)へご相談ください。",
                source_url="https://www.city.kitakyushu.lg.jp/contents/924_10041.html",
                source_page_title="ごみの分別早見表(あ行)",
                crawled_at=dt.datetime(2026, 9, 1, tzinfo=timezone.utc),
            ),
            WasteItem(
                item_name="ソファ",
                item_name_raw="ソファ",
                disposal_method="粗大ごみ",
                notes="事前申込制。粗大ごみ受付センターへ",
                source_url="https://www.city.kitakyushu.lg.jp/contents/924_10042.html",
                source_page_title="ごみの分別早見表(さ行)",
                crawled_at=dt.datetime(2026, 9, 1, tzinfo=timezone.utc),
            ),
        ],
        [
            CollectionSchedule(
                ward="小倉北区",
                town="浅野1丁目",
                waste_type="粗大ごみ",
                weekday="第2・第4水曜日",
                source_url="https://www.city.kitakyushu.lg.jp/contents/924_10102.html",
                crawled_at=dt.datetime(2026, 9, 1, tzinfo=timezone.utc),
            ),
        ],
    )
    conn.close()

    import app.data.lookup as data_lookup
    import app.rag.router as router_module

    importlib.reload(data_lookup)
    importlib.reload(router_module)
    return router_module


def test_city_does_not_collect_item_points_to_notes_contact(routed_db):
    # 6-1(b): 「市が収集しないもの」は収集曜日を捏造せず、早見表の備考
    # (販売店・専門業者等の連絡先)に従う既存挙動(router.py:1072-1090)。
    # `no_weekday_schedule` は収集日検索(ward検出時)の枝でのみ立つため、
    # 区名を含む質問で確認する。
    ans = routed_db._blocking_route("小倉北区でエアコンはどう捨てればいいですか")
    assert ans.no_weekday_schedule is True
    assert ans.item_hits
    assert ans.item_hits[0].disposal_method == "市が収集しないもの"
    assert "リサイクル受付センター" in ans.item_hits[0].notes


def test_ward_only_without_town_asks_for_town_name(routed_db):
    # 6-1(c): 区のみで町名が無ければ推測せず「町名」を要求する
    # (既存挙動、router.py の `ans.missing.append("町名")`)。
    ans = routed_db._blocking_route("小倉北区の粗大ごみの収集日を教えてください")
    assert "町名" in ans.missing
