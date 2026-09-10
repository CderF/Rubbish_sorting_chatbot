"""File + ingest management for distributed お知らせ knowledge files (WebUI).

The read path stays in `app/rag/knowledge.py`; this module is the *write* side
that the ナレッジ一覧 modal calls: stage an uploaded file on disk, ingest it on
demand, and delete it — undoing what its ingest did to `waste_items`.

Pure Python, no FastAPI import, so it is unit-testable offline the same way
`app/notices/ingest.py` is. `app/server.py` wraps each function in one
`require_login`-guarded endpoint.

Only `## お知らせ NN: 品目` markdown files are managed here. Crawled 早見表 /
収集日 sources stay read-only and are never touched.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.data import db as _db
from app.notices import store
from app.notices.ingest import ingest_notice_file, relink

# knowledge/incoming/ is the tracked ingest-input directory (see .gitignore:
# "knowledge/incoming/ itself IS tracked: the distributed お知らせ files are
# ingest inputs"). Uploads land next to the two distributed files.
INCOMING_DIR = (
    Path(__file__).resolve().parent.parent.parent / "knowledge" / "incoming"
)

# The two distributed files are ~7 KB. Cap well above that but low enough that an
# accidental binary upload is rejected rather than written.
MAX_NOTICE_BYTES = 512 * 1024
ALLOWED_SUFFIXES = (".md", ".txt")


class NoticeFileError(ValueError):
    """Rejected upload or bad file name — surfaced to the WebUI as a 400."""


def _safe_name(name: str) -> str:
    """Return a bare file name that stays inside INCOMING_DIR, or raise.

    Rejects path separators, `..`, empty names, unexpected suffixes and absurd
    lengths, so a name coming from the browser can never escape the directory.
    """
    raw = (name or "").strip()
    base = Path(raw).name
    if not base or base != raw or base in (".", "..") or "\\" in base:
        raise NoticeFileError("不正なファイル名です: " + repr(name))
    if len(base) > 120:
        raise NoticeFileError("ファイル名が長すぎます（120文字以内）")
    if Path(base).suffix.lower() not in ALLOWED_SUFFIXES:
        raise NoticeFileError(
            "対応する拡張子は " + ", ".join(ALLOWED_SUFFIXES) + " だけです: " + base
        )
    return base


def safe_name(name: str) -> str:
    """Public wrapper around `_safe_name` for `app/server.py` (header filename)."""
    return _safe_name(name)


def _path(name: str) -> Path:
    return INCOMING_DIR / _safe_name(name)


def save_notice_file(name: str, data: bytes) -> dict:
    """Write an uploaded file into INCOMING_DIR. Does NOT ingest it."""
    base = _safe_name(name)
    if len(data) > MAX_NOTICE_BYTES:
        raise NoticeFileError(
            "ファイルが大きすぎます（" + str(MAX_NOTICE_BYTES // 1024) + " KiB まで）"
        )
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NoticeFileError("UTF-8 テキストとして読めません") from exc

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    path = INCOMING_DIR / base
    replaced = path.exists()
    path.write_bytes(data)
    return {"name": base, "size": len(data), "replaced": replaced}


def _ingested_counts(conn: "sqlite3.Connection | None") -> "dict[str, dict]":
    if conn is None:
        return {}
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_sources'"
    ).fetchone()
    if not has_table:
        return {}
    counts: "dict[str, dict]" = {}
    for row in store.list_knowledge_sources(conn):
        agg = counts.setdefault(
            row["source_file"], {"record_count": 0, "change_notice_count": 0}
        )
        agg["record_count"] += 1
        if row.get("is_change_notice"):
            agg["change_notice_count"] += 1
    return counts


def _disk_files() -> "dict[str, Path]":
    if not INCOMING_DIR.is_dir():
        return {}
    return {
        p.name: p
        for p in sorted(INCOMING_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES
    }


def notice_file_status(conn: "sqlite3.Connection | None") -> "list[dict]":
    """One entry per managed お知らせ file: on-disk files ∪ ingested source_files.

    `conn=None` means the shared DB does not exist yet — nothing is ingested.
    `ingested=False` ⇒ uploaded but not yet taken in; `on_disk=False` ⇒ ingested
    earlier, file since removed (re-ingest is impossible until it is re-uploaded).
    """
    counts = _ingested_counts(conn)
    on_disk = _disk_files()

    result: "list[dict]" = []
    for name in sorted(set(on_disk) | set(counts)):
        p = on_disk.get(name)
        agg = counts.get(name)
        stat = p.stat() if p else None
        result.append(
            {
                "name": name,
                "on_disk": p is not None,
                "size": stat.st_size if stat else None,
                "mtime": (
                    datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                    if stat
                    else None
                ),
                "ingested": agg is not None,
                "record_count": agg["record_count"] if agg else 0,
                "change_notice_count": agg["change_notice_count"] if agg else 0,
            }
        )
    return result


def _one_status(conn: sqlite3.Connection, name: str) -> "dict | None":
    for entry in notice_file_status(conn):
        if entry["name"] == name:
            return entry
    return None


def ingest_saved_file(
    conn: sqlite3.Connection, name: str, *, now: "datetime | None" = None
) -> dict:
    """Ingest a staged file. If it was ingested before, its old rows are removed
    first so an edited file cannot leave stale sections behind."""
    base = _safe_name(name)
    path = INCOMING_DIR / base
    if not path.is_file():
        raise NoticeFileError("ファイルがありません: " + base)

    reingested = base in _ingested_counts(conn)
    if reingested:
        store.delete_knowledge_source_file(conn, base)

    result = ingest_notice_file(path, conn, now=now)
    link = relink(conn)
    return {
        "name": base,
        "reingested": reingested,
        "summary": result.summary(),
        "inserted": list(result.inserted),
        "skipped": list(result.skipped),
        "linked": len(result.linked),
        "unlinked": list(result.unlinked),
        "relink_summary": link.summary(),
        "status": _one_status(conn, base),
    }


def delete_notice_file(conn: sqlite3.Connection, name: str) -> dict:
    """Remove a file's ingested rows (restoring any predecessor) and the file."""
    base = _safe_name(name)
    undo = store.delete_knowledge_source_file(conn, base)
    path = INCOMING_DIR / base
    file_removed = path.is_file()
    path.unlink(missing_ok=True)
    return {"name": base, "file_removed": file_removed, **undo}


def raw_notice_text(name: str) -> str:
    path = _path(name)
    if not path.is_file():
        raise NoticeFileError("ファイルがありません: " + _safe_name(name))
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# connection helpers for app/server.py (kept here so server.py stays FastAPI-only)
# --------------------------------------------------------------------------
def with_rw_conn(fn):
    """Run ``fn(conn)`` against the shared DB, schema ensured, connection closed
    afterwards. Used by the ingest / delete endpoints, which need the DB to
    exist."""
    conn = store.connect()
    try:
        store.init_schema(conn)
        return fn(conn)
    finally:
        conn.close()


def status_payload() -> "list[dict]":
    """`notice_file_status` for the read-only ``GET /knowledge`` endpoint: opens
    the shared DB only if it already exists, so merely listing files never
    creates it."""
    path = _db.db_path()
    if not path.exists():
        return notice_file_status(None)
    conn = store.connect(path)
    try:
        return notice_file_status(conn)
    finally:
        conn.close()
