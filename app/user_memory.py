"""STAGE07 7-4: 利用者ごとに、セッションをまたいで覚えておく記憶ストア。

STAGE05 の `app/rag/session_state.py` が持つスロットは**1会話(session_id)限り**
で、会話が終われば消える(`app/server.py` の `sessions` / `states` は
`MAX_SESSIONS` を超えると LRU で捨てる)。7-4 が求めるのはそれとは別の層 —
「この利用者は前も浅野1丁目の話をしていた」のように、**別の会話・別の日**でも
覚えている記憶。

このモジュールはストアそのものだけを提供する純粋な読み書き層で、
「誰が利用者か」「どこで自動的に読み書きするか」は決めない(1日でのSTAGE07
着手範囲としてスコープ外。将来 MCP ツール `remember`/`recall` や WebUI の
ログインユーザー単位から接続する想定)。`app/notices/store.py` と同じ方針:
DB・LLM 呼び出しを一切行わず、SQLite への素朴な読み書きだけを行う。

利用イメージ(将来の配線例。ここでは実装しない):

    conn = user_memory.connect()
    user_memory.remember(conn, "mentor-1", "ward", "小倉北区")
    user_memory.remember(conn, "mentor-1", "town", "浅野1丁目")
    ...
    facts = user_memory.recall_all(conn, "mentor-1")
    # => {"ward": "小倉北区", "town": "浅野1丁目"}
    # これを DialogState の初期値や検索コンテキストに足せる。
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# 早見表・収集日の DB(`app/data/db.py`)とは別ファイルにする。7-4 はそちらの
# スキーマ・クロール処理と無関係な独立した機能なので、同じ DB に間借りすると
# 将来のマイグレーション(クロール DB の再構築等)に巻き込まれる。
DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent / "var" / "user_memory.db"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_memory (
    user_id    TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);
"""


def db_path() -> Path:
    """`USER_MEMORY_DB_PATH` 環境変数で差し替え可能(既定は `var/user_memory.db`)。

    `app/data/db.py.db_path()` / `app/notices/admin.py` と同じ「env var が
    最優先」パターンに揃えている。
    """
    return Path(os.environ.get("USER_MEMORY_DB_PATH", DEFAULT_DB_PATH))


def connect(path: "str | Path | None" = None) -> sqlite3.Connection:
    resolved = Path(path) if path else db_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def remember(
    conn: sqlite3.Connection,
    user_id: str,
    key: str,
    value: str,
    *,
    now: "str | None" = None,
) -> None:
    """`user_id` の `key` に `value` を覚えさせる(既にあれば上書き)。"""
    if not user_id or not key:
        raise ValueError("user_id と key は空にできません")
    conn.execute(
        """
        INSERT INTO user_memory (user_id, key, value, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (user_id, key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        (user_id, key, value, now or _now_iso()),
    )
    conn.commit()


def recall(conn: sqlite3.Connection, user_id: str, key: str) -> "str | None":
    """1件だけ取り出す。無ければ `None`。"""
    row = conn.execute(
        "SELECT value FROM user_memory WHERE user_id = ? AND key = ?",
        (user_id, key),
    ).fetchone()
    return row["value"] if row else None


def recall_all(conn: sqlite3.Connection, user_id: str) -> "dict[str, str]":
    """その利用者が覚えられている全項目を `{key: value}` で返す。"""
    rows = conn.execute(
        "SELECT key, value FROM user_memory WHERE user_id = ? ORDER BY key",
        (user_id,),
    ).fetchall()
    return {r["key"]: r["value"] for r in rows}


def forget(conn: sqlite3.Connection, user_id: str, key: str) -> bool:
    """1件忘れさせる。実際に消えた行があれば `True`。"""
    cur = conn.execute(
        "DELETE FROM user_memory WHERE user_id = ? AND key = ?", (user_id, key)
    )
    conn.commit()
    return cur.rowcount > 0


def forget_all(conn: sqlite3.Connection, user_id: str) -> int:
    """その利用者の記憶を全部消す。消した件数を返す(利用者からの明示的な
    「忘れて」要求向け — 個人情報を溜め込むだけにしないための出口)。"""
    cur = conn.execute("DELETE FROM user_memory WHERE user_id = ?", (user_id,))
    conn.commit()
    return cur.rowcount
