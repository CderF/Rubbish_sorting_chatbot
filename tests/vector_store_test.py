"""Offline seam test for app/rag/vector_store.py (issue #7 Testing Decisions).

Fixture text through the real intfloat/multilingual-e5-small model into an
EphemeralClient (in-memory Chroma, no disk writes) — no live crawling, no
GPU. Asserts that semantically-close queries return the expected top hit,
not exact-string behavior (that's app/data/lookup.py's job).
"""

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

from app.rag.vector_store import (
    _rebuild_explanations_collection,
    _rebuild_items_collection,
    vector_search_explanations,
    vector_search_items,
)

results = []


def check(name, condition, detail=""):
    verdict = "PASS" if condition else "FAIL"
    print(f"[{verdict}] {name}" + (f" — {detail}" if detail and not condition else ""))
    results.append(condition)


@dataclass
class _FixtureItem:
    item_name: str
    disposal_method: str
    notes: str
    source_url: str


client = chromadb.EphemeralClient()

items = [
    _FixtureItem("テレビ(ブラウン管式・液晶式・プラズマ式)", "市が収集しないもの",
                 "販売店などに引き取りを依頼", "https://example.test/hayamihyo/ta"),
    _FixtureItem("テレビ台", "粗大ごみ", "事前申込制", "https://example.test/hayamihyo/ta"),
    _FixtureItem("ペットボトル", "資源化ごみ", "キャップとラベルを外して", "https://example.test/hayamihyo/a"),
    _FixtureItem("電子レンジ", "粗大ごみ", "事前申込制", "https://example.test/hayamihyo/a"),
]
n = _rebuild_items_collection(items, client=client)
check("index build: item count matches fixture", n == len(items), f"got {n}")

chunks = [
    {
        "text": "ごみステーションの防鳥ネットを無料で貸し出しています。カラスによる散乱対策に有効です。",
        "source_url": "https://example.test/station/net",
        "source_page_title": "ごみステーション対策",
    },
    {
        "text": "視覚障害のある方向けに、指定袋へ点字シールを貼付するサービスを行っています。",
        "source_url": "https://example.test/bag/braille",
        "source_page_title": "視覚障害者向け支援",
    },
]
n = _rebuild_explanations_collection(chunks, client=client)
check("index build: chunk count matches fixture", n == len(chunks), f"got {n}")

# 1. semantic fallback: query text doesn't exactly match any item_name, but a
#    close semantic match exists (mirrors the real "テレビ" case: bare name
#    absent, qualified compound names present).
hits = vector_search_items("テレビ", top_k=2, client=client)
check(
    "vector_search_items: 'テレビ' surfaces a テレビ* item as top hit",
    bool(hits) and "テレビ" in hits[0].item_name,
    f"got {[h.item_name for h in hits]}",
)
check(
    "vector_search_items: hits carry source_url",
    bool(hits) and all(h.source_url for h in hits),
)

# 2. unrelated query should not blow up, and should rank the unrelated fixture
#    item(s) below the closer one (loose check: still returns top_k, no crash).
hits2 = vector_search_items("ペットボトルの捨て方", top_k=1, client=client)
check(
    "vector_search_items: unrelated-domain query still returns closest fixture item",
    bool(hits2) and hits2[0].item_name == "ペットボトル",
    f"got {[h.item_name for h in hits2]}",
)

# 3. empty / whitespace query must not raise, per the "never raise" contract
#    shared with app/data/lookup.py.
check("vector_search_items: empty query returns []", vector_search_items("", client=client) == [])
check("vector_search_items: whitespace query returns []", vector_search_items("   ", client=client) == [])

# 4. explanation search
hits3 = vector_search_explanations("カラスにごみを荒らされる", top_k=1, client=client)
check(
    "vector_search_explanations: crow/scatter query hits the ネット chunk",
    bool(hits3) and hits3[0]["source_url"] == "https://example.test/station/net",
    f"got {[h['source_url'] for h in hits3]}",
)
check(
    "vector_search_explanations: hits carry source_url + source_page_title",
    bool(hits3) and hits3[0]["source_url"] and hits3[0]["source_page_title"],
)

hits4 = vector_search_explanations("目が見えない人への配慮", top_k=1, client=client)
check(
    "vector_search_explanations: visual-impairment query hits the braille chunk",
    bool(hits4) and hits4[0]["source_url"] == "https://example.test/bag/braille",
    f"got {[h['source_url'] for h in hits4]}",
)

check(
    "vector_search_explanations: empty query returns []",
    vector_search_explanations("", client=client) == [],
)

# 5. querying against a never-built collection must not raise — mirrors
#    STAGE02 running before any index has been built. Needs a genuinely
#    isolated client: two chromadb.EphemeralClient() instances in the same
#    process share their in-memory backend (verified separately), so a
#    PersistentClient at a throwaway temp dir is used instead.
_empty_client = chromadb.PersistentClient(path=tempfile.mkdtemp())
check(
    "vector_search_items: unbuilt collection returns [] without raising",
    vector_search_items("テレビ", client=_empty_client) == [],
)
check(
    "vector_search_explanations: unbuilt collection returns [] without raising",
    vector_search_explanations("ごみステーション", client=_empty_client) == [],
)

print()
print(f"=== {sum(1 for r in results if r)}/{len(results)} checks passed ===")
