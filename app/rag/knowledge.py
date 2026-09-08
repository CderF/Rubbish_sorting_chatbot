"""ナレッジソースの一覧と内容を公開する(STAGE02 2-3要件)。

「検索対象の外部データファイル一覧と内容をWebUIで確認できる」ための読み取り
専用モジュール。データ層が作る構造化DB(`documents/data_schema.md`)を参照元
ページ単位に集計して返す。

DBがまだ存在しない段階でも例外を投げず「0件」を返す。データ層の完成に伴って
自動的に中身が埋まるため、WebUI側の改修は不要。

STAGE04 で取り込んだ配布お知らせ(`knowledge_sources` テーブル)も、クロール
ページと並べて一覧に出す。早見表に混ぜて `waste_items` に入っているため、
そちらの集計からは `knowledge://` 出典を除外して二重表示を防ぐ。
"""

import hashlib
import os
import sqlite3

# データ層が入っていれば、DBの場所はデータ層の定義(app/data/db.py の
# DB_PATH 環境変数と既定パス)に従う。パスを二重管理するとナレッジ一覧だけ
# 空になる事故が起きるため、こちらでは決め打ちしない。
# RAG_DB_PATH を置くとそれが最優先(テスト・別配置用の逃げ道)。
_FALLBACK_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge.db"
)


def db_path() -> str:
    """参照するSQLiteのパス。呼ぶたびに解決するので環境変数の変更が効く。"""
    override = os.environ.get("RAG_DB_PATH")
    if override:
        return override
    try:
        from app.data import db  # type: ignore[import-not-found]

        return str(db.db_path())
    except Exception:
        # データ層が未実装、あるいは db_path() を持たない場合
        return _FALLBACK_DB_PATH

# 1ページあたりに返すレコード数の上限(WebUIのビューアが固まらないように)。
MAX_RECORDS = 500

# title_sql は一覧に出す見出しの作り方。collection_schedule には
# source_page_title が無い(documents/data_schema.md)ので、区名で代用する。
_KINDS = {
    "waste_items": {
        "label": "ごみの分別早見表",
        "title_sql": "MAX(source_page_title)",
        "columns": ["item_name", "disposal_method", "notes"],
        "headers": ["品名", "出し方", "備考"],
    },
    "collection_schedule": {
        "label": "資源とごみの収集日",
        "title_sql": "MAX(ward)",
        "columns": ["ward", "town", "waste_type", "weekday"],
        "headers": ["区", "町名", "ごみ種別", "収集曜日"],
    },
}


def _source_id(table: str, url: str) -> str:
    """参照元URLから安定したIDを作る(WebUIのリンク用)。"""
    return hashlib.sha1(f"{table}\n{url}".encode()).hexdigest()[:12]


def _connect() -> sqlite3.Connection | None:
    path = db_path()
    if not os.path.exists(path):
        return None
    # 読み取り専用で開く。データ層のクロール中に誤って書き込む事故を防ぐ。
    # ここは毎回接続を開いて閉じるので、スレッドを跨いでも問題にならない。
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r["name"] for r in rows}



def _notice_sources(conn: sqlite3.Connection) -> list[dict]:
    """STAGE04: one entry per ingested お知らせ file (`knowledge_sources`)."""
    if "knowledge_sources" not in _tables(conn):
        return []
    rows = conn.execute(
        """
        SELECT source_file,
               COUNT(*)        AS n,
               MAX(ingested_at) AS ingested_at
          FROM knowledge_sources
         GROUP BY source_file
         ORDER BY source_file
        """
    ).fetchall()
    return [
        {
            "id": _source_id("knowledge_sources", r["source_file"]),
            "kind": "knowledge_notice",
            "kind_label": "配布お知らせ(STAGE04)",
            "title": r["source_file"],
            "url": f"knowledge://{r['source_file']}",
            "record_count": r["n"],
            "crawled_at": r["ingested_at"],
        }
        for r in rows
    ]


def _notice_source_detail(conn: sqlite3.Connection, source_id: str) -> dict | None:
    if "knowledge_sources" not in _tables(conn):
        return None
    files = conn.execute(
        "SELECT DISTINCT source_file FROM knowledge_sources"
    ).fetchall()
    for row in files:
        source_file = row["source_file"]
        if _source_id("knowledge_sources", source_file) != source_id:
            continue
        recs = conn.execute(
            """
            SELECT ks.anchor, ks.title,
                   wi.item_name, wi.disposal_method, wi.effective_from
              FROM knowledge_sources ks
              LEFT JOIN waste_items wi ON wi.id = ks.produced_item_id
             WHERE ks.source_file = ?
             ORDER BY ks.id
             LIMIT ?
            """,
            (source_file, MAX_RECORDS + 1),
        ).fetchall()
        truncated = len(recs) > MAX_RECORDS
        return {
            "id": source_id,
            "kind": "knowledge_notice",
            "kind_label": "配布お知らせ(STAGE04)",
            "url": f"knowledge://{source_file}",
            "headers": ["品名", "お知らせ", "適用開始", "反映後の内容"],
            "rows": [
                [
                    r["item_name"] or r["anchor"],
                    r["title"],
                    r["effective_from"] or "",
                    r["disposal_method"] or "",
                ]
                for r in recs[:MAX_RECORDS]
            ],
            "truncated": truncated,
        }
    return None


def list_sources() -> dict:
    """取り込み済みの参照元ページを一覧で返す。

    戻り値の `available` が False のときは、DBがまだ無い(データ層が未着手)。
    WebUI はその旨を表示する。
    """
    conn = _connect()
    if conn is None:
        return {"available": False, "db_path": db_path(), "sources": [], "total_records": 0}

    try:
        present = _tables(conn)
        sources: list[dict] = []
        total = 0

        for src in _notice_sources(conn):
            total += src["record_count"]
            sources.append(src)

        for table, spec in _KINDS.items():
            if table not in present:
                continue
            rows = conn.execute(
                f"""
                SELECT source_url,
                       {spec["title_sql"]} AS page_title,
                       COUNT(*)         AS n,
                       MAX(crawled_at)  AS crawled_at
                  FROM {table}
                 WHERE is_current = 1
                   AND source_url NOT LIKE 'knowledge://%'
                 GROUP BY source_url
                 ORDER BY page_title, source_url
                """
            ).fetchall()
            for r in rows:
                total += r["n"]
                sources.append(
                    {
                        "id": _source_id(table, r["source_url"]),
                        "kind": table,
                        "kind_label": spec["label"],
                        "title": r["page_title"] or r["source_url"],
                        "url": r["source_url"],
                        "record_count": r["n"],
                        "crawled_at": r["crawled_at"],
                    }
                )

        return {
            "available": True,
            "db_path": db_path(),
            "sources": sources,
            "total_records": total,
        }
    finally:
        conn.close()


def get_source(source_id: str) -> dict | None:
    """1ページ分の取り込み内容を返す。見つからなければ None。"""
    conn = _connect()
    if conn is None:
        return None

    try:
        notice = _notice_source_detail(conn, source_id)
        if notice is not None:
            return notice

        present = _tables(conn)
        for table, spec in _KINDS.items():
            if table not in present:
                continue
            urls = conn.execute(
                f"SELECT DISTINCT source_url FROM {table} "
                f"WHERE is_current = 1 AND source_url NOT LIKE 'knowledge://%'"
            ).fetchall()
            for row in urls:
                url = row["source_url"]
                if _source_id(table, url) != source_id:
                    continue

                cols = ", ".join(spec["columns"])
                recs = conn.execute(
                    f"""
                    SELECT {cols} FROM {table}
                     WHERE is_current = 1 AND source_url = ?
                     ORDER BY id
                     LIMIT {MAX_RECORDS + 1}
                    """,
                    (url,),
                ).fetchall()

                truncated = len(recs) > MAX_RECORDS
                return {
                    "id": source_id,
                    "kind": table,
                    "kind_label": spec["label"],
                    "url": url,
                    "headers": spec["headers"],
                    "rows": [[r[c] for c in spec["columns"]] for r in recs[:MAX_RECORDS]],
                    "truncated": truncated,
                }
        return None
    finally:
        conn.close()
