"""Router-facing read API for STAGE 04, in the same shape as `app/data/lookup.py`.

Same contract: no `sqlite3.Connection` argument, a lazily-created thread-local
connection (`server.py` calls the data layer from a threadpool), `PRAGMA
query_only`, and nothing here raises for a missing item.

The one deliberate difference: this does **not** filter `is_current = 1`.
`app/data/lookup.py` does, so it can only ever return the newest rule — which is
exactly what requirement 4-2 forbids. These functions return the whole old→new
timeline so the router can put both the current and the upcoming rule in front of
the LLM.
"""

from __future__ import annotations

import threading
from datetime import date

from app.stage04 import store
from app.stage04.store import ItemTimeline

_local = threading.local()


def _conn():
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = store.connect()
        store.init_schema(conn)  # CREATE TABLE IF NOT EXISTS — must precede query_only
        conn.execute("PRAGMA query_only = ON")
        _local.conn = conn
    return conn


def item_timeline(item_name: str) -> ItemTimeline:
    """The old→new rule timeline for one item. `found=False` ⇒ not in the DB
    (caller falls back to the 4-3 similar-item path)."""
    return store.item_timeline(_conn(), item_name)


def pending_change(item_name: str, as_of: date) -> "ItemTimeline | None":
    """The timeline **iff** a rule switch-over falls on or after `as_of`
    (i.e. requirement 4-2 applies to this item right now); otherwise `None`.

    Use it as the router hook::

        tl = pending_change(item_name, date.today())
        if tl is not None:
            # add a "この品目は <tl.change_dates[-1]> に変わります" block to the
            # LLM context, listing both tl.segment_on(today) and the later
            # segment(s); do not answer with only one of them.
    """
    tl = store.item_timeline(_conn(), item_name)
    if not tl.found:
        return None
    return tl if tl.changes_after(as_of) else None
