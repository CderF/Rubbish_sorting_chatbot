"""ベクトルRAGパス(issue #7) — 早見表フォールバック検索 + 説明ページの意味検索。

早見表の完全一致・部分一致(`app/data/lookup.py`)がどちらも外れたときの3段目の
フォールバックとして `vector_search_items` を、早見表・収集日の対象外(ごみ
ステーション・指定ごみ袋・ごみ出し支援などの説明ページ)を検索する
`vector_search_explanations` を提供する。

Chromaコレクションの構築(`_rebuild_*_collection`)と検索(`vector_search_*`)を
分離しているのは、検索側をChromaクライアントを差し替えるだけでテストできる
ようにするため(固定fixtureをEphemeralClientに詰めてから検索を検証する)。

**並行アクセスについて(実装時に判明した重要事項)**: chromadb(内部のSQLite
バックエンド)と sentence-transformers の推論は、いずれも複数スレッドから
同時にアクセスするとプロセスごとクラッシュする(catchできる例外ではなく
SIGSEGV)ことを実測で確認した。`router.py` は `asyncio.to_thread` 経由の
スレッドプールから呼ぶため、本モジュールが公開する関数は内部で全て専用の
単一ワーカースレッド(`_executor`)に処理を委譲し、直列化する。データ層の
`app/data/lookup.py` がスレッドごとに接続を持つ方式で解決したのと同種の
問題だが、根本原因(sqlite3の「作成スレッド以外禁止」チェック vs
ネイティブ実装側の非スレッドセーフ)が異なるため、対策も異なる。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chromadb
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from app.rag.types import ChunkRecord, ItemRecord

# embedding/Chromaに触れる処理は全てこの1本のスレッドで実行する(モジュール
# docstring参照)。検索はインデックス済みの近傍探索なので、直列化しても
# 実質的なコストは小さい。
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-vector")


def _run(fn, *args, **kwargs):
    return _executor.submit(fn, *args, **kwargs).result()

BASE_URL = "https://www.city.kitakyushu.lg.jp"

# ごみステーション / 指定ごみ袋等 / ごみ出し支援(documents/stage02&03_proposal.md
# 3-2章。同ドメイン全体ではなく、この3索引ページ+直下リンクに範囲を限定する
# 判断は issue #7 の Implementation Decisions を参照)。
EXPLANATION_INDEX_PAGES = [
    f"{BASE_URL}/kurashi/menu01_00363.html",  # ごみステーション
    f"{BASE_URL}/kurashi/menu01_00364.html",  # 指定ごみ袋等
    f"{BASE_URL}/kurashi/menu01_00365.html",  # ごみ出し支援
]

_USER_AGENT = "Mozilla/5.0 (compatible; KitakyushuWasteBot/1.0)"

# 索引ページには本文と無関係な全サイト共通ナビが混在する(早見表・収集日の
# 索引ページにも同じリンクが出る)。プレフィックスで判別できないため、リンク
# テキストの完全一致denylistで除外する。
_GLOBAL_NAV_LINK_TEXTS = {
    "救急医療テレフォンセンター",
    "北九州市コールセンター",
    "サイト内検索方法",
    "ページ番号検索方法",
    "北九州市公式ホームページの使い方",
}

_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

_DEFAULT_CHROMA_PATH = Path(__file__).resolve().parent.parent.parent / "var" / "chroma"
_ITEMS_COLLECTION = "waste_items_v1"
_EXPLANATIONS_COLLECTION = "explanations_v1"

_CHUNK_SIZE = 300
_CHUNK_OVERLAP = 50


@dataclass
class _VectorItemMatch:
    """`app.rag.types.ItemRecord` を構造的に満たす最小限の型。

    Chromaのメタデータから復元するだけなので、`app.data.schema.WasteItem` の
    id/crawled_at等(埋め込みインデックスには保存していない)は持たない。
    """

    item_name: str
    disposal_method: str
    notes: str
    source_url: str


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _embedder


def _embed_passages(texts: list[str]) -> list[list[float]]:
    """格納側(インデックス構築時)のembedding。e5系モデルの規約で "passage: " を前置する。"""
    prefixed = [f"passage: {t}" for t in texts]
    return _get_embedder().encode(prefixed, normalize_embeddings=True).tolist()


def _embed_query(text: str) -> list[float]:
    """検索側のembedding。e5系モデルの規約で "query: " を前置する。"""
    return _get_embedder().encode([f"query: {text}"], normalize_embeddings=True).tolist()[0]


# --------------------------------------------------------------------------
# Chroma client
# --------------------------------------------------------------------------
_client: chromadb.ClientAPI | None = None


def _chroma_path() -> Path:
    return Path(os.environ.get("CHROMA_PATH", _DEFAULT_CHROMA_PATH))


def _default_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        path = _chroma_path()
        path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(path))
    return _client


def _get_or_create_collection(client: chromadb.ClientAPI, name: str, *, reset: bool):
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


# --------------------------------------------------------------------------
# 説明ページのクロール+チャンク化(索引ページ→本文ページ、深さ2)
# --------------------------------------------------------------------------
def _get(session: requests.Session, url: str) -> str:
    resp = session.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


def _discover_explanation_pages(index_html: str) -> list[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    seen: dict[str, None] = {}
    for a in soup.select("a[href]"):
        text = a.get_text(strip=True)
        href = a.get("href", "")
        if not text or text in _GLOBAL_NAV_LINK_TEXTS or "/contents/" not in href:
            continue
        url = href if href.startswith("http") else f"{BASE_URL}{href}"
        seen.setdefault(url, None)
    return list(seen)


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """行(ブロック要素境界)単位で貪欲に詰め、chunk_sizeを超えたら区切る。
    次のチャンクの先頭に直前チャンク末尾のoverlap文字を引き継ぐ。
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return []

    chunks: list[str] = []
    current = ""
    for line in lines:
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > chunk_size and current:
            chunks.append(current)
            current = current[-overlap:] + "\n" + line if overlap else line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _parse_explanation_page(html: str, source_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else None

    content = soup.find("div", id="mainContentsInner")
    if content is None:
        return []
    text = content.get_text("\n", strip=True)
    # 各ページ末尾に共通する問い合わせ先ブロックはチャンク化しない(全ページで
    # ほぼ同一の文言が重複し、検索結果を汚染するため)。
    text = text.split("このページの作成者")[0].strip()
    if not text:
        return []

    return [
        {"text": chunk, "source_url": source_url, "source_page_title": title}
        for chunk in _chunk_text(text)
    ]


def fetch_explanation_chunks(session: requests.Session | None = None) -> list[dict]:
    """3つの索引ページ+直下リンクをクロールし、チャンク化して返す(embedding前)。"""
    session = session or requests.Session()
    records: list[dict] = []
    for index_url in EXPLANATION_INDEX_PAGES:
        index_html = _get(session, index_url)
        for url in _discover_explanation_pages(index_html):
            records.extend(_parse_explanation_page(_get(session, url), url))
    return records


# --------------------------------------------------------------------------
# インデックス構築(fixture・本番データの両方から呼べるよう2層に分離)
#
# `_*_impl` が実際にembedding/Chromaへ触れる部分(常に _executor 上で実行)。
# 公開関数はその薄いラッパーで、呼び出し元のスレッドに関わらず処理を
# _executor に委譲してから結果を返す。
# --------------------------------------------------------------------------
def _rebuild_items_impl(items: Sequence[ItemRecord], client: chromadb.ClientAPI | None) -> int:
    client = client or _default_client()
    collection = _get_or_create_collection(client, _ITEMS_COLLECTION, reset=True)
    items = list(items)
    if not items:
        return 0

    embeddings = _embed_passages([it.item_name for it in items])
    collection.add(
        ids=[str(i) for i in range(len(items))],
        embeddings=embeddings,
        documents=[it.item_name for it in items],
        metadatas=[
            {
                "item_name": it.item_name,
                "disposal_method": it.disposal_method,
                "notes": it.notes or "",
                "source_url": it.source_url,
            }
            for it in items
        ],
    )
    return len(items)


def _rebuild_items_collection(
    items: Sequence[ItemRecord], client: chromadb.ClientAPI | None = None
) -> int:
    """`items`(item_name/disposal_method/notes/source_urlを持つオブジェクトなら
    何でもよい — WasteItemでもfixture用dataclassでも可)から早見表用コレクションを
    全件洗い替えで再構築する。"""
    return _run(_rebuild_items_impl, items, client)


def _rebuild_explanations_impl(chunks: Sequence[dict], client: chromadb.ClientAPI | None) -> int:
    client = client or _default_client()
    collection = _get_or_create_collection(client, _EXPLANATIONS_COLLECTION, reset=True)
    chunks = list(chunks)
    if not chunks:
        return 0

    embeddings = _embed_passages([c["text"] for c in chunks])
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "source_url": c["source_url"],
                "source_page_title": c.get("source_page_title") or "",
            }
            for c in chunks
        ],
    )
    return len(chunks)


def _rebuild_explanations_collection(
    chunks: Sequence[dict], client: chromadb.ClientAPI | None = None
) -> int:
    """`chunks` は {"text", "source_url", "source_page_title"} を持つdictのリスト
    (`fetch_explanation_chunks()` の戻り値、またはfixture)。"""
    return _run(_rebuild_explanations_impl, chunks, client)


def build_item_index(client: chromadb.ClientAPI | None = None) -> int:
    """本番用: 構造化DBの現行早見表アイテム全件からコレクションを再構築する。"""
    from app.data import db

    conn = db.connect()
    items = db.list_current_waste_items(conn)
    return _rebuild_items_collection(items, client)


def build_explanation_index(client: chromadb.ClientAPI | None = None) -> int:
    """本番用: 説明ページを実際にクロールしてコレクションを再構築する。"""
    chunks = fetch_explanation_chunks()
    return _rebuild_explanations_collection(chunks, client)


# --------------------------------------------------------------------------
# 検索(router.py が呼ぶ公開関数)
# --------------------------------------------------------------------------
# Chroma の hnsw:space="cosine" では距離 = 1 - コサイン類似度(範囲 [0, 2]、
# 0 が完全一致)。これより遠いヒットは「関連なし」として返さない。無関係な
# 品名クエリでも top_k 件が必ず返り found=True になってしまう問題(PR #14
# レビュー #1)への対策。実データでの調整が必要な暫定値。
_ITEM_MATCH_MAX_DISTANCE = 0.55


def _filter_by_distance(
    metadatas: list, distances: list, max_distance: float
) -> list:
    """距離しきい値で足切りする。距離が取れないときは従来どおり素通し。"""
    if not distances or len(distances) != len(metadatas):
        return metadatas
    return [
        m
        for m, d in zip(metadatas, distances)
        if d is not None and d <= max_distance
    ]


def _search_items_impl(
    query: str, top_k: int, client: chromadb.ClientAPI | None
) -> list[ItemRecord]:
    client = client or _default_client()
    try:
        collection = client.get_collection(_ITEMS_COLLECTION)
    except Exception:
        return []
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[_embed_query(query)], n_results=min(top_k, collection.count())
    )
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    metadatas = _filter_by_distance(metadatas, distances, _ITEM_MATCH_MAX_DISTANCE)
    return [
        _VectorItemMatch(
            item_name=m["item_name"],
            disposal_method=m["disposal_method"],
            notes=m["notes"],
            source_url=m["source_url"],
        )
        for m in metadatas
    ]


def vector_search_items(
    query: str, top_k: int = 3, client: chromadb.ClientAPI | None = None
) -> list[ItemRecord]:
    """早見表パスの完全一致・部分一致がどちらも外れた場合の3段目フォールバック。
    ヒットなし・コレクション未構築のいずれも例外を投げず空リストを返す。"""
    if not query or not query.strip():
        return []
    return _run(_search_items_impl, query, top_k, client)


def _search_explanations_impl(
    query: str, top_k: int, client: chromadb.ClientAPI | None
) -> list[ChunkRecord]:
    client = client or _default_client()
    try:
        collection = client.get_collection(_EXPLANATIONS_COLLECTION)
    except Exception:
        return []
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[_embed_query(query)], n_results=min(top_k, collection.count())
    )
    docs = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [
        ChunkRecord(
            text=doc,
            source_url=m["source_url"],
            source_page_title=m["source_page_title"],
            score=1.0 - dist,  # cosine距離 → 類似度寄りのスコアに変換(厳密な確率ではない)
        )
        for doc, m, dist in zip(docs, metadatas, distances)
    ]


def vector_search_explanations(
    query: str, top_k: int = 3, client: chromadb.ClientAPI | None = None
) -> list[ChunkRecord]:
    """説明ページ(早見表・収集日の対象外)に対する意味検索。"""
    if not query or not query.strip():
        return []
    return _run(_search_explanations_impl, query, top_k, client)


if __name__ == "__main__":
    n_items = build_item_index()
    n_chunks = build_explanation_index()
    print(f"waste_items collection: {n_items} 件")
    print(f"explanations collection: {n_chunks} チャンク")
