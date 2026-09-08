# STAGE 02-04 data pipeline. Every target is idempotent — safe to re-run.
#
#   make knowledge   crawl + ingest お知らせ + link + rebuild the vector index
#   make deploy      git pull + make knowledge + restart the service   [B1 only]
#
# PYTHON defaults to the venv python if one is beside the repo, else `python`.

PYTHON ?= $(firstword $(wildcard ../venv/bin/python .venv/bin/python) python)
NOTICES := $(wildcard knowledge/incoming/*.md)

.PHONY: crawl ingest relink index knowledge test deploy

crawl: ## re-crawl the city site into var/kitakyushu_waste.db (slow, hits the site)
	$(PYTHON) -m app.data.db

ingest: ## ingest the distributed お知らせ files into waste_items
	@test -n "$(NOTICES)" && $(PYTHON) -m app.notices.ingest $(NOTICES) || echo "no knowledge/incoming/*.md"

relink: ## link change-notices to 早見表 rows that now exist
	$(PYTHON) -m app.notices.ingest --relink

index: ## rebuild the Chroma vector index from the current waste_items rows
	$(PYTHON) -m app.rag.vector_store

# Full refresh. `ingest` + `relink` must follow any DB rebuild or the STAGE 04
# 4-2 (出し方の変更) data is missing; `index` must follow or 4-1 additions and
# the 4-3 similar-item fallback can't see the new items.
knowledge: crawl ingest relink index

test:
	$(PYTHON) -m pytest tests/ --ignore=tests/spec_test.py -q

deploy: ## B1 only: pull, refresh all data, restart
	git pull --ff-only
	$(MAKE) knowledge
	sudo systemctl restart rag-chatbot.service
