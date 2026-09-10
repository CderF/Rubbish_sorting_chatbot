"""STAGE06: 6-3 完全空振りフォールバック(`safety.no_result_reply` の配線)と
6-4 トレース閲覧(`app/trace_view.py`)の単体テスト。

すべてオフライン。6-3 の配線だけ孤立 SQLite に対して `_blocking_route` を呼ぶ
(`tests/notices_integration_test.py` と同じ DB_PATH 差し替え + reload 手法)。
`app/server.py` のルートは fastapi 必須のため、ここではロジック層のみ検証する。
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
from datetime import timezone
from pathlib import Path

import pytest

from app import trace_view
from app.data.schema import WasteItem
from app.notices import store
from app.rag import safety


# --------------------------------------------------------------------------
# 6-3: safety.no_result_reply
# --------------------------------------------------------------------------
def test_no_result_reply_is_a_non_empty_safe_message():
    reply = safety.no_result_reply()
    assert reply
    assert "確認できませんでした" in reply
    assert ("公式" in reply) and ("窓口" in reply)
    # §5 未決事項が確定するまで電話番号は載せない方針。
    assert not any(ch.isdigit() for ch in reply)


# --------------------------------------------------------------------------
# 6-3: _blocking_route の配線
# --------------------------------------------------------------------------
@pytest.fixture
def empty_db(tmp_path, monkeypatch):
    """スキーマだけの空 DB を用意し、依存モジュールを reload して router を返す。

    ベクトル段(`var/chroma` の実インデックス)は無効化する — 6-3 は「早見表・
    収集日・類推のどれにもヒットしない」状態のテストで、リポジトリ同梱の
    chroma を引くとプロンプト中の「ごみ」に類似品目が当たってしまう。
    """
    path = tmp_path / "kb.db"
    monkeypatch.setenv("DB_PATH", str(path))
    monkeypatch.setenv("RAG_DB_PATH", str(path))
    conn = store.connect()
    store.init_schema(conn)
    conn.close()

    import app.data.lookup as data_lookup
    import app.notices.lookup as notices_lookup
    import app.rag.router as router

    importlib.reload(data_lookup)
    importlib.reload(notices_lookup)
    importlib.reload(router)
    monkeypatch.setattr(router, "vector_search_items", lambda *a, **k: [])
    yield router
    # reload 後の router が抱えた区名/町名の lru_cache は tmp DB の内容で埋まって
    # いる。後続テスト(collection_schedule 系)が本番DBの結果を期待して落ちない
    # よう、明示的に捨てる。
    for name in ("_town_candidates", "_district_groups"):
        fn = getattr(router, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
    importlib.reload(data_lookup)
    importlib.reload(notices_lookup)
    importlib.reload(router)


@pytest.fixture
def seeded_router(empty_db, tmp_path):
    """空 DB に1品目だけ入れて router を reload し直す。"""
    conn = store.connect()
    store.insert_waste_item(
        conn,
        WasteItem(
            item_name="ペットボトル",
            item_name_raw="ペットボトル",
            disposal_method="資源化ごみ",
            notes="キャップとラベルを外す",
            source_url="https://www.city.kitakyushu.lg.jp/contents/924_10041.html",
            source_page_title="ごみの分別早見表",
            crawled_at=dt.datetime(2026, 9, 1, tzinfo=timezone.utc),
        ),
    )
    conn.commit()
    conn.close()
    importlib.reload(empty_db)
    return empty_db


def test_total_miss_returns_the_6_3_canned_reply(empty_db):
    ans = empty_db._blocking_route("この文章はごみ分別と全く関係のない架空の相談です")

    assert ans.found is False
    assert ans.safety_gate == "6-3"
    assert ans.canned_response == safety.no_result_reply()


def test_a_hit_does_not_trigger_the_6_3_fallback(seeded_router):
    ans = seeded_router._blocking_route("ペットボトルはどう捨てますか")

    assert ans.found is True
    assert ans.safety_gate is None
    assert ans.canned_response is None


def test_missing_town_asks_for_it_instead_of_the_6_3_canned_reply(empty_db):
    # コードレビューで発見した回帰: 区名はあるが町名が無く、品目も収集日も
    # ヒットしない(＝ ans.found is False)場合、修正前は6-3が先に拾ってしまい
    # build_system_prompt() の「町名が必要、推測せず利用者に追加入力を促す」
    # という既存の指示(router.py の `ans.missing`)を握りつぶしていた。
    # 「あと一歩の情報が要る」ケースは6-3の対象外とし、従来どおり
    # missing 経由でLLMに追加入力を促させる。
    ans = empty_db._blocking_route("小倉北区の収集日を教えてください")

    assert ans.found is False
    assert "町名" in ans.missing
    assert ans.safety_gate is None
    assert ans.canned_response is None


def test_6_3_fallback_calls_safety_no_result_reply(empty_db, monkeypatch):
    """固定文をハードコードで比較していないこと(safety 側を差し替えると追従する)。"""
    monkeypatch.setattr(empty_db.safety, "no_result_reply", lambda: "＜差し替え文＞")

    ans = empty_db._blocking_route("ごみとは無関係の架空の相談です")

    assert ans.safety_gate == "6-3"
    assert ans.canned_response == "＜差し替え文＞"


def test_6_3_stays_off_when_no_result_reply_is_empty(empty_db, monkeypatch):
    """no_result_reply が空("")なら従来どおり LLM 生成に回す(足場の挙動)。"""
    monkeypatch.setattr(empty_db.safety, "no_result_reply", lambda: "")

    ans = empty_db._blocking_route("ごみとは無関係の架空の相談です")

    assert ans.found is False
    assert ans.safety_gate is None
    assert ans.canned_response is None


# --------------------------------------------------------------------------
# 6-4: trace_view.read_trace_tail
# --------------------------------------------------------------------------
def _write_jsonl(path: Path, records: list) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def test_read_trace_tail_returns_newest_first_within_limit(tmp_path):
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, [{"prompt": f"q{i}"} for i in range(10)])

    rows = trace_view.read_trace_tail(str(p), limit=3)

    assert [r["prompt"] for r in rows] == ["q9", "q8", "q7"]


def test_read_trace_tail_skips_blank_and_broken_lines(tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_text(
        '{"prompt": "ok1"}\n'
        "\n"
        "not json at all\n"
        '{"prompt": "ok2"}\n'
        "[1, 2, 3]\n",  # JSON だが dict ではない
        encoding="utf-8",
    )

    rows = trace_view.read_trace_tail(str(p), limit=100)

    assert [r["prompt"] for r in rows] == ["ok2", "ok1"]


def test_read_trace_tail_missing_file_or_zero_limit(tmp_path):
    assert trace_view.read_trace_tail(str(tmp_path / "nope.jsonl")) == []
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, [{"prompt": "x"}])
    assert trace_view.read_trace_tail(str(p), limit=0) == []


# --------------------------------------------------------------------------
# 6-4: trace_view.summarize_trace_rows
# --------------------------------------------------------------------------
def test_summarize_maps_labels_and_flags():
    rows = [
        {
            "session_id": "grade-abc",
            "prompt": "スプレー缶は?",
            "detected_ward": "小倉北区",
            "detected_town": "浅野1丁目",
            "detected_item": "スプレー缶",
            "item_candidates": ["スプレー缶"],
            "item_match_stage": "exact",
            "item_match": "exact",
            "timeline_change": True,
            "analogy_unreliable": False,
            "safety_gate": None,
            "canned_response": False,
            "schedule_hits": 2,
            "missing": [],
            "sources": ["knowledge://お知らせ-4-2.md#x"],
            "final_response": "…",
        }
    ]

    out = trace_view.summarize_trace_rows(rows)[0]

    assert out["match_stage_label"] == "完全一致"
    assert out["timeline_change"] is True
    assert out["analogy"] is False
    assert out["safety_gate_label"] is None
    assert out["ward"] == "小倉北区" and out["town"] == "浅野1丁目"


def test_summarize_passes_through_the_safety_gate():
    rows = [{"safety_gate": "6-3", "canned_response": True, "prompt": "?"}]
    out = trace_view.summarize_trace_rows(rows)[0]
    assert out["safety_gate"] == "6-3"
    assert out["safety_gate_label"] == "6-3 完全空振り"
    assert out["canned_response"] is True


def test_summarize_marks_analogy_from_item_match():
    rows = [{"item_match": "analogy", "analogy_unreliable": True, "prompt": "?"}]
    out = trace_view.summarize_trace_rows(rows)[0]
    assert out["analogy"] is True
    assert out["analogy_unreliable"] is True


def test_summarize_tolerates_old_rows_missing_new_keys():
    out = trace_view.summarize_trace_rows([{"prompt": "old"}])[0]
    assert out["prompt"] == "old"
    assert out["ward"] is None
    assert out["safety_gate"] is None
    assert out["match_stage_label"] == "該当なし"
