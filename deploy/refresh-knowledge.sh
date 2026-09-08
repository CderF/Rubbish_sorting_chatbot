#!/usr/bin/env bash
# Bring the structured DB and the vector index up to date, in the order the
# STAGE 02-04 features need. Idempotent. Called by `make knowledge`, and safe to
# run from cron or by hand after `git pull`.
#
#   1. crawl        -> var/kitakyushu_waste.db (waste_items / collection_schedule)
#   2. ingest       -> distributed お知らせ rows into waste_items
#   3. relink       -> connect change-notices to their 早見表 predecessor
#   4. index        -> rebuild Chroma from the current rows
#
# Skipping 2+3 leaves 4-2 (出し方の変更) unanswered; skipping 4 hides 4-1
# additions from the 4-3 similar-item fallback.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$(command -v python)}"
for c in ../venv/bin/python .venv/bin/python; do
    [ -x "$c" ] && PYTHON="$c" && break
done

echo "[refresh] crawl"        ; "$PYTHON" -m app.data.db
echo "[refresh] ingest"       ; "$PYTHON" -m app.notices.ingest knowledge/incoming/*.md
echo "[refresh] relink"       ; "$PYTHON" -m app.notices.ingest --relink
echo "[refresh] vector index" ; "$PYTHON" -m app.rag.vector_store
echo "[refresh] done"
