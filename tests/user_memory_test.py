"""STAGE07 7-4: `app/user_memory.py` の純粋な読み書き層の単体テスト。

DB・LLM・ネットワーク不要、すべて `:memory:` の孤立SQLiteで完結する。
配線(誰が利用者か・どこで自動的に呼ぶか)はこのPRのスコープ外なので、
ここではストア自体の契約だけを担保する。
"""

from __future__ import annotations

import pytest

from app import user_memory


@pytest.fixture
def conn():
    c = user_memory.connect(":memory:")
    user_memory.init_schema(c)
    yield c
    c.close()


def test_remember_then_recall_round_trips(conn):
    user_memory.remember(conn, "mentor-1", "ward", "小倉北区")
    assert user_memory.recall(conn, "mentor-1", "ward") == "小倉北区"


def test_recall_missing_key_is_none(conn):
    assert user_memory.recall(conn, "mentor-1", "nope") is None


def test_remembering_the_same_key_again_overwrites(conn):
    user_memory.remember(conn, "mentor-1", "ward", "小倉北区")
    user_memory.remember(conn, "mentor-1", "ward", "八幡西区")
    assert user_memory.recall(conn, "mentor-1", "ward") == "八幡西区"


def test_recall_all_returns_every_key_for_that_user_only(conn):
    user_memory.remember(conn, "mentor-1", "ward", "小倉北区")
    user_memory.remember(conn, "mentor-1", "town", "浅野1丁目")
    user_memory.remember(conn, "mentor-2", "ward", "門司区")

    assert user_memory.recall_all(conn, "mentor-1") == {
        "ward": "小倉北区",
        "town": "浅野1丁目",
    }
    assert user_memory.recall_all(conn, "mentor-2") == {"ward": "門司区"}


def test_recall_all_for_unknown_user_is_empty(conn):
    assert user_memory.recall_all(conn, "ghost") == {}


def test_forget_removes_one_key_and_reports_whether_it_existed(conn):
    user_memory.remember(conn, "mentor-1", "ward", "小倉北区")

    assert user_memory.forget(conn, "mentor-1", "ward") is True
    assert user_memory.recall(conn, "mentor-1", "ward") is None
    assert user_memory.forget(conn, "mentor-1", "ward") is False  # 二度目は False


def test_forget_all_clears_only_that_user_and_returns_the_count(conn):
    user_memory.remember(conn, "mentor-1", "ward", "小倉北区")
    user_memory.remember(conn, "mentor-1", "town", "浅野1丁目")
    user_memory.remember(conn, "mentor-2", "ward", "門司区")

    removed = user_memory.forget_all(conn, "mentor-1")

    assert removed == 2
    assert user_memory.recall_all(conn, "mentor-1") == {}
    assert user_memory.recall_all(conn, "mentor-2") == {"ward": "門司区"}


@pytest.mark.parametrize("user_id,key", [("", "ward"), ("mentor-1", "")])
def test_remember_rejects_empty_user_id_or_key(conn, user_id, key):
    with pytest.raises(ValueError):
        user_memory.remember(conn, user_id, key, "x")


def test_db_path_honours_the_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom.db"
    monkeypatch.setenv("USER_MEMORY_DB_PATH", str(custom))
    assert user_memory.db_path() == custom


def test_connect_creates_the_parent_directory(tmp_path):
    target = tmp_path / "nested" / "dir" / "mem.db"
    conn = user_memory.connect(target)
    try:
        user_memory.init_schema(conn)
        assert target.exists()
    finally:
        conn.close()
