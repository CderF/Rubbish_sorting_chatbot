"""Tests for the WebUI knowledge-file management path (`app/notices/admin.py`).

Offline: in-memory SQLite + a temporary INCOMING_DIR, no server, no GPU, no
network. The 早見表 seed mirrors `tests/notices_ingest_test.py`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.data.schema import WasteItem
from app.notices import admin, store

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE = REPO_ROOT / "tests" / "fixtures" / "notices"
ADDITIONS = KNOWLEDGE / "お知らせ-4-1.md"
CHANGES = KNOWLEDGE / "お知らせ-4-2.md"
SEED = Path(__file__).resolve().parent / "fixtures" / "hayamihyo_seed.json"
INGEST_AT = datetime(2026, 9, 8, 3, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    c = store.connect(":memory:")
    store.init_schema(c)
    yield c
    c.close()


@pytest.fixture
def seeded_conn(conn):
    for row in json.loads(SEED.read_text(encoding="utf-8"))["rows"]:
        store.insert_waste_item(
            conn,
            WasteItem(
                item_name=row["item_name"],
                item_name_raw=row["item_name"],
                disposal_method=row["disposal_method"],
                notes="",
                source_url="fixture://test-only",
                source_page_title=row["source_page_title"],
                crawled_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                is_current=True,
            ),
        )
    conn.commit()
    return conn


@pytest.fixture
def incoming(tmp_path, monkeypatch):
    d = tmp_path / "incoming"
    d.mkdir()
    monkeypatch.setattr(admin, "INCOMING_DIR", d)
    return d


# --------------------------------------------------------------------------
# file-name safety
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad", ["../evil.md", "a/b.md", "notes.pdf", "", "   ", "..", "x" * 121 + ".md"]
)
def test_safe_name_rejects_traversal_bad_suffix_and_overlong(bad):
    with pytest.raises(admin.NoticeFileError):
        admin._safe_name(bad)


def test_safe_name_accepts_a_plain_markdown_name():
    assert admin._safe_name("  お知らせ-4-1.md  ") == "お知らせ-4-1.md"


# --------------------------------------------------------------------------
# upload — save only, no ingest
# --------------------------------------------------------------------------
def test_save_writes_the_file_and_status_shows_it_pending(incoming, conn):
    data = ADDITIONS.read_bytes()
    info = admin.save_notice_file("uploaded.md", data)

    assert info == {"name": "uploaded.md", "size": len(data), "replaced": False}
    assert (incoming / "uploaded.md").read_bytes() == data

    status = admin.notice_file_status(conn)
    assert len(status) == 1
    assert status[0]["name"] == "uploaded.md"
    assert status[0]["on_disk"] is True
    assert status[0]["ingested"] is False
    assert status[0]["record_count"] == 0


def test_save_rejects_oversize_and_non_utf8(incoming):
    with pytest.raises(admin.NoticeFileError):
        admin.save_notice_file("big.md", b"x" * (admin.MAX_NOTICE_BYTES + 1))
    with pytest.raises(admin.NoticeFileError):
        admin.save_notice_file("bin.md", b"\xff\xfe\x00\x01")


def test_save_flags_replacement(incoming):
    admin.save_notice_file("dup.md", b"# one\n")
    info = admin.save_notice_file("dup.md", b"# two\n")

    assert info["replaced"] is True
    assert (incoming / "dup.md").read_text(encoding="utf-8") == "# two\n"


# --------------------------------------------------------------------------
# ingest a staged file
# --------------------------------------------------------------------------
def test_ingest_saved_addition_file_inserts_rows(incoming, conn):
    admin.save_notice_file("adds.md", ADDITIONS.read_bytes())
    out = admin.ingest_saved_file(conn, "adds.md", now=INGEST_AT)

    assert out["reingested"] is False
    assert len(out["inserted"]) == 15
    assert conn.execute("SELECT COUNT(*) FROM waste_items").fetchone()[0] == 15
    assert out["status"]["ingested"] is True
    assert out["status"]["record_count"] == 15


def test_ingest_saved_change_file_supersedes_seed_rows(incoming, seeded_conn):
    admin.save_notice_file("chg.md", CHANGES.read_bytes())
    out = admin.ingest_saved_file(seeded_conn, "chg.md", now=INGEST_AT)

    assert out["linked"] == 15
    retired = seeded_conn.execute(
        "SELECT COUNT(*) FROM waste_items WHERE is_current = 0"
    ).fetchone()[0]
    assert retired == 15
    assert out["status"]["change_notice_count"] == 15


def test_ingest_missing_file_raises(incoming, conn):
    with pytest.raises(admin.NoticeFileError):
        admin.ingest_saved_file(conn, "nope.md")


def test_reingest_after_edit_drops_the_stale_sections(incoming, conn):
    admin.save_notice_file("adds.md", ADDITIONS.read_bytes())
    admin.ingest_saved_file(conn, "adds.md", now=INGEST_AT)
    assert conn.execute("SELECT COUNT(*) FROM waste_items").fetchone()[0] == 15

    trimmed = ADDITIONS.read_text(encoding="utf-8").split("## お知らせ 03:")[0]
    admin.save_notice_file("adds.md", trimmed.encode("utf-8"))
    out = admin.ingest_saved_file(conn, "adds.md", now=INGEST_AT)

    assert out["reingested"] is True
    assert conn.execute("SELECT COUNT(*) FROM waste_items").fetchone()[0] == 2
    assert out["status"]["record_count"] == 2


# --------------------------------------------------------------------------
# delete — undo the ingest and drop the file
# --------------------------------------------------------------------------
def test_delete_change_file_restores_the_predecessors(incoming, seeded_conn):
    admin.save_notice_file("chg.md", CHANGES.read_bytes())
    admin.ingest_saved_file(seeded_conn, "chg.md", now=INGEST_AT)

    out = admin.delete_notice_file(seeded_conn, "chg.md")

    assert out["removed_sources"] == 15
    assert out["removed_items"] == 15
    assert out["restored_predecessors"] == 15
    assert out["file_removed"] is True
    assert not (incoming / "chg.md").exists()

    rows = seeded_conn.execute("SELECT * FROM waste_items").fetchall()
    assert len(rows) == 15  # back to the seed alone
    assert all(r["is_current"] == 1 for r in rows)
    assert all(r["superseded_by"] is None for r in rows)
    assert (
        seeded_conn.execute("SELECT COUNT(*) FROM knowledge_sources").fetchone()[0] == 0
    )
    assert admin.notice_file_status(seeded_conn) == []


def test_delete_addition_file_removes_rows_without_restoring_anything(incoming, conn):
    admin.save_notice_file("adds.md", ADDITIONS.read_bytes())
    admin.ingest_saved_file(conn, "adds.md", now=INGEST_AT)

    out = admin.delete_notice_file(conn, "adds.md")

    assert out["removed_items"] == 15
    assert out["restored_predecessors"] == 0
    assert conn.execute("SELECT COUNT(*) FROM waste_items").fetchone()[0] == 0


def test_delete_without_prior_ingest_just_removes_the_file(incoming, conn):
    admin.save_notice_file("adds.md", ADDITIONS.read_bytes())
    out = admin.delete_notice_file(conn, "adds.md")

    assert out["removed_items"] == 0
    assert out["removed_sources"] == 0
    assert out["file_removed"] is True
    assert not (incoming / "adds.md").exists()


# --------------------------------------------------------------------------
# raw text
# --------------------------------------------------------------------------
def test_raw_notice_text_returns_file_contents(incoming):
    admin.save_notice_file("adds.md", b"# hi\n\n## body\n")
    assert admin.raw_notice_text("adds.md") == "# hi\n\n## body\n"


def test_raw_notice_text_missing_raises(incoming):
    with pytest.raises(admin.NoticeFileError):
        admin.raw_notice_text("ghost.md")
