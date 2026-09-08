# STAGE 02 & 03 — 開発分担計画書

Team B (Teamb2 FANG XIUWEN ／ Teamb3 藤井誠人)
作成日: 2026-09-08
関連: [[STAGE02&03設計提案書]](stage02&03_proposal.md参照)、[[ドメイン用語集]](CONTEXT.md参照)

---

## 1. 役割分担

| レイヤー | 担当 | 範囲 |
|---|---|---|
| **データ層** | Teamb2 FANG XIUWEN | クローリング、早見表パス、収集日パスの構造化DB・検索関数 |
| **応用層** | Teamb3 藤井誠人 | ベクトルRAGパス、画像分類パス、LLM意図解析・ルーティング、WebUI、`/chat`統合 |

両者は最初から並行着手する。分担の境界線は「構造化DB(SQLite等)に対する検索関数」と「それを呼び出して自然文回答を組み立てる層」であり、詳細は3章のインターフェース契約を参照。

---

## 2. モジュール/ディレクトリ構成

既存の `app/server.py`(単一ファイル、STAGE1-1時点の `/chat` エンドポイント・VLM呼び出し・SSE実装済み)を土台に、以下を追加する。

```
app/
├── server.py              # 既存。/chat 内でrag/router.pyを呼ぶよう変更(藤井)
├── webui.html              # 既存。ナレッジ一覧・画像アップロード欄を追加(藤井)
├── data/                   # ★データ層(FANG)
│   ├── crawler.py          # 早見表8ページ + 収集日7区ページのクローリング(再実行可能)
│   ├── db.py                # SQLiteスキーマ定義・初期化(スキーマ詳細は documents/data_schema.md 参照)
│   └── lookup.py            # lookup_item_exact / lookup_item_partial / lookup_collection_schedule
└── rag/                     # ★応用層(藤井)
    ├── vector_store.py      # Chroma構築、embedding、説明ページ・早見表フォールバック用のベクトル検索
    ├── image_classify.py    # VLMへ画像を渡し品目名相当の文字列を抽出
    └── router.py            # 質問/画像をどのパスに振り分けるか判定し、統合レスポンスを組み立てる
```

---

## 3. インターフェース契約

両者はこの3章の内容を実装前に合意し、以後は互いの内部実装を待たずに**ダミーデータ・スタブ関数**でモックしながら並行開発できるようにする。

### 3-1. データ層が公開する関数(`app/data/lookup.py`)

```python
def lookup_item_exact(item_name: str) -> ItemRecord | None:
    """完全一致検索。呼び出し側は正規化不要(内部でNFKC正規化・全角半角統一を行う)。"""

def lookup_item_partial(item_name: str) -> list[ItemRecord]:
    """部分一致(LIKE)検索。0件ならから空リスト。例外は投げない。"""

def lookup_collection_schedule(
    ward: str, town: str | None, waste_type: str | None = None
) -> list[ScheduleRecord] | None:
    """
    区×町名×ごみ種別で収集日を検索。
    - town が None、または該当データが見つからない場合は None を返す
      (応用層はこれを「町名の追加入力を促す」トリガーとして扱う)
    - waste_type が None の場合はその区・町のごみ種別すべてを返す
    """
```

```python
# ItemRecord: 早見表1件分
{
    "item_name": str,       # 品名
    "disposal_method": str,  # 出し方
    "notes": str,            # 備考
    "source_url": str,       # 引用用の参照ページURL(必須)
}

# ScheduleRecord: 収集日1件分
{
    "ward": str,
    "town": str,
    "waste_type": str,
    "weekday": str,
    "source_url": str,       # 必須
}
```

**共通ルール**: 見つからない場合は `None` または空リストを返し、例外を投げない。「該当情報なし」を自然文に変換するのは応用層の責務(3-7の未識別時の原則を参照)。`source_url` は両関数とも必須で返す。

### 3-2. 応用層が公開する関数

```python
# app/rag/vector_store.py
def vector_search_items(query: str, top_k: int = 3) -> list[ItemRecord]:
    """早見表の完全一致・部分一致がヒットしなかった場合のフォールバック。"""

def vector_search_explanations(query: str, top_k: int = 3) -> list[ChunkRecord]:
    """説明・注意事項ページ(ごみステーション等)に対するベクトル検索。"""

# app/rag/image_classify.py
def classify_image_to_item_name(image_base64: str, engine, processor) -> str | None:
    """VLMに画像を渡し、品目名相当の文字列を抽出する。特定できない場合は None。
    engine/processor は app/server.py が起動時に初期化する共有vLLMエンジンを
    そのまま受け取る(L4 1枚のVRAMで2つ目のモデルをロードしないため。詳細は
    GitHub issue #7 のImplementation Decisions参照)。"""

# app/rag/router.py
def handle_query(prompt: str, img_b64: str | None) -> RoutedAnswer:
    """
    /chat から呼ばれるエントリポイント。
    1. img_b64があれば classify_image_to_item_name() で品名を得てからテキストパスに合流
    2. プロンプトから品名・地区(区/町名)を抽出
    3. lookup_item_exact -> lookup_item_partial -> vector_search_items の順にフォールバック
    4. 地区が含まれていれば lookup_collection_schedule も並行して呼ぶ
    5. 結果を統合し、source_url一覧を添えた RoutedAnswer を返す
    """
```

呼び出し方向は一方向: **`app/rag/router.py` が `app/data/lookup.py` をimportして呼ぶ**。データ層は応用層のモジュールに依存しない。同一プロセス内呼び出しとし、API化はしない(L4サーバー1台内で完結するため)。

### 3-3. `/chat` との接続(`app/server.py`、藤井担当)

既存の `_build_messages` / `_sse` の前段に `app.rag.router.handle_query()` を呼び出し、検索結果をLLMへのプロンプトに組み込んだ上でSSEストリームを開始する。`img` フィールドの扱いは既存実装(`_decode_image`)を流用可能。

---

## 4. タスクリスト

### 4-1. データ層(Teamb2 FANG XIUWEN)

**STAGE02**
- [x] 早見表8ページ(あ行〜わ行)のクローリングスクリプト作成(`app/data/crawler.py`)
- [x] 早見表HTML表パース → `WasteItem` 構造化(品名/出し方/備考/source_url)
- [x] SQLiteスキーマ設計・初期化(`app/data/db.py`)
- [x] `lookup_item_exact` / `lookup_item_partial` 実装(`app/data/lookup.py`) ※**最優先、藤井のSTAGE02統合作業のブロッカーになるため**
- [x] 取り込み済みファイル一覧のエクスポート(WebUIの2-3要件用、`db.list_sources()`)

**STAGE03**
- [x] 収集日データ(7区×町名/丁目)のクローリングスクリプト作成
- [x] 収集日データパース → `CollectionSchedule` 構造化(区/町名/ごみ種別/曜日/source_url)
- [x] `lookup_collection_schedule` 実装(区のみ指定時は `None` を返す仕様を厳守)
- [x] 差分再取得ロジック(STAGE04前方互換。`db.rebuild_db()` が「全件再取得+上書き」を実装)

実装メモ(GitHub issue #4のspecからの変更点・実装時に判明した事項):
- 3公開関数は `ItemRecord`/`ScheduleRecord` という素のdictではなく、`app/data/schema.py` の `WasteItem`/`CollectionSchedule` pydanticモデルのインスタンスを返す(issue #4のImplementation Decisionsで確定済み)
- 早見表8ページはページごとにHTML構造が3パターンある(`<table summary="分別早見表">` / `<caption>分別早見表</caption>` / どちらも無く5つの独立tableが並ぶだけ)。`crawler.py` の `_is_waste_item_table` はこの3パターン全てに対応
- 収集日7区のページもthead/tbodyの有無が区によって異なる(小倉北区・若松区はthead/tbodyあり、残り5区はヘッダー行`<tr><th>...</th></tr>`が五十音グループごとにデータ行の間に直接繰り返される)。`parse_schedule_page` はthead/tbodyに依存せず`<th>`の有無で行を判定する
- `CollectionSchedule` の `ward`/`town`/`waste_type` にもNFKC正規化のバリデータが必要だった(`WasteItem.item_name` と同様)。当初town側だけ正規化していたら全角チルダ「～」を含む町名(例: 青葉1～2丁目)がクエリ側でNFKC正規化されて半角「~」になり、DB格納値(非正規化)と一致せず検索がヒットしなかった
- 実測データ量: `waste_items` 864件、`collection_schedule` 2472件(7区・778町名)

### 4-2. 応用層(Teamb3 藤井誠人)

**STAGE02**
- [ ] 説明ページのクローリング+チャンク化
- [ ] `intfloat/multilingual-e5-small` セットアップ + Chroma構築(`app/rag/vector_store.py`)
- [ ] `vector_search_items` / `vector_search_explanations` 実装
- [ ] 意図解析・ルーティングロジック初版(`app/rag/router.py`、当初はデータ層のスタブ関数でモック可)
- [ ] `/chat` への統合(`app/server.py` 改修)、SSEレスポンスへの引用付与
- [ ] WebUI: ナレッジ一覧・内容ビューア(2-3要件)

**STAGE03**
- [ ] `classify_image_to_item_name` 実装(`app/rag/image_classify.py`)
- [ ] `router.py` に3-1(品名+地区統合)・3-2(画像分岐)ロジック追加
- [ ] WebUI: 画像アップロード入力欄の追加
- [ ] 未識別時のフォールバック文言実装(3-7の原則、`documents/stage02&03_proposal.md` 3-7章参照)

### 4-3. 共同作業

- [ ] 本ドキュメント3章のインターフェース契約レビュー・合意(着手前に実施)
- [ ] `lookup_item_exact` 完成後、結合テスト(藤井のrouter.pyから実呼び出しに切り替え)
- [ ] STAGE02自動採点 → 合格後にメンターへSlack連絡
- [ ] STAGE03自動採点 → 合格後にメンターへSlack連絡

---

## 5. 依存関係とタイムライン

```
FANG: 早見表クローラ → lookup_item_exact/partial ─┬─→ 藤井: router.py実結合 → STAGE02自動採点
                                                    │
藤井: ベクトルRAG基盤 + WebUI雛形(並行着手可) ──────┘

FANG: 収集日クローラ → lookup_collection_schedule ─→ 藤井: 3-1統合 ─┐
藤井: 画像分類パス(並行着手可) ──────────────────────────────────┴─→ STAGE03自動採点
```

- **本日〜明日**: 両者並行着手。FANGは早見表パスを最優先で完成させ関数を確定させる(藤井が待たされないよう、境界のインターフェースだけ先に固定しダミーデータで統合を進めてよい)。藤井はベクトルRAG基盤とWebUI雛形を並行して進める
- **STAGE02自動採点合格後**: FANGは収集日パスに着手、藤井は画像分類パスと3-1/3-2のrouter統合に着手
- **STAGE03**: STAGE02合格後、間を空けずに着手(`stage02&03_proposal.md` 7章のスケジュールと同一)

---

## 6. Git運用

- ブランチ命名: `feature/stage02-data`(FANG)、`feature/stage02-rag`(藤井)、STAGE03は `feature/stage03-data`、`feature/stage03-rag`
- `app/data/lookup.py` の関数シグネチャ(3-1章)を変更する場合は、変更前に相手に一声かける(応用層側のimportが壊れるため)
- 依存パッケージ追加(LangChain, chromadb, sentence-transformers等)は各自 `pip install` 後、`requirements.txt` に追記してコミットする(現状 `requirements.txt` にはvLLM/FastAPI/Pillow等のみで、RAG関連は未追加)
