# STAGE 04 — お知らせ取り込み仕様

Team B / Teamb4 (周 楚涵)
作成日: 2026-09-08

関連: [[data_schema.md]](データスキーマ定義書) / [[STAGE04_分担提案.md]](STAGE04 分担) / [[CONTEXT.md]](ドメイン用語集)

配布された「お知らせ」ファイルを構造化DBへ取り込み、要件 4-1・4-2 を満たすためのデータ側の実装仕様。
回答生成側(本田さん担当)への引き渡し契約は「5. 引き渡し契約」を参照。

---

## 1. 対象ファイルと2種類のお知らせ

| ファイル | 件数 | 種別 | 対応要件 |
|---|---|---|---|
| `knowledge/incoming/お知らせ-4-1.md` | 15 | **追加通知** — 早見表に無い品目(ホバーボード、ドローン、VRゴーグル等)の回収案内 | 4-1 |
| `knowledge/incoming/お知らせ-4-2.md` | 15 | **変更通知** — 早見表にある品目(スプレー缶、蛍光管、乾電池等)の出し方が2026年10月から変わる | 4-2 |

**判別はファイル名ではなく本文で行う。** 本文に「現行（早見表）の…扱いは◯年◯月まで有効です」節があれば変更通知、
無ければ追加通知として扱う。将来1ファイルに両方が混在しても正しく振り分けられる。

## 2. モジュール構成

| ファイル | 役割 |
|---|---|
| `app/stage04/notice_parser.py` | Markdown解析のみ。I/O・DB・ネットワークに触れない |
| `app/stage04/store.py` | SQLite接続、DDL、`waste_items` の読み書き、`knowledge_sources` 管理 |
| `app/stage04/ingest.py` | `Notice` → `WasteItem` 変換とバージョン列の接続、CLIエントリポイント |
| `tests/stage04_ingest_test.py` | オフラインテスト20件(GPU・サーバー不要) |
| `tests/fixtures/hayamihyo_seed.json` | **テスト専用**の擬似早見表15件。本番投入禁止 |

`app/data/` 配下(Teamb2担当)のファイルは一切変更していない。`app/data/schema.py` の `WasteItem` を
そのまま再利用しており、独自モデルは作っていない。

`store.py` の `waste_items` / `collection_schedule` のDDLは `data_schema.md` の転記で、
`CREATE TABLE IF NOT EXISTS` のため、issue #4 の `app/data/db.py` が入っても衝突せず先に作った方が有効になる。

## 3. フィールドマッピング

| `WasteItem` | 取り込み時の値 |
|---|---|
| `item_name` | 見出しの品名(4-1は末尾の「回収」を除去)。モデルのvalidatorがNFKC正規化する |
| `item_name_raw` | 見出しの原文(例: `ホバーボード回収`) |
| `disposal_method` | 本文の指示部分。「現行（早見表）の…」節は含めない |
| `notes` | タイトル / (変更通知のみ)現行の扱いと有効期限 / 発行部署 |
| `source_url` | `knowledge://<ファイル名>#<アンカー>`(「6. 出典の表記」参照) |
| `source_page_title` | `【市からのお知らせ】…` の見出し |
| `crawled_at` | 取り込み実行時刻(UTC) |
| `is_current` | `1` |
| `effective_from` | 「2026年10月から」→ `2026-10-01`。追加通知は `NULL` |
| `superseded_by` | `NULL`(新行側) |

## 4. 新旧バージョン列の作り方

`data_schema.md` §4 の運用ルールに従う。変更通知1件につき:

1. 同一 `item_name` かつ `is_current = 1` の既存行(＝前身)を**INSERT前に**引く
2. 新行を `INSERT`(`is_current = 1`、`effective_from` に適用開始日)
3. 前身を `UPDATE is_current = 0, superseded_by = <新行id>`

### `effective_from` をどちらの行に書くかの解釈

§4 は「`effective_from` に新情報の適用開始日を設定する」とだけ書かれており、
新旧どちらの行を指すか明示されていない。**本実装では新行にのみ設定した。**
`effective_from` の定義が「この情報が有効になった日付」である以上、
引退した旧行に後継の開始日を書くと定義と矛盾するため。旧行の `effective_from` はクロール由来の値のまま(現状 `NULL`)。

→ **FANGさん、この解釈で問題ないか確認をお願いします。**

### 旧規則の終了日

`valid_until` カラムは追加していない。**後継行の `effective_from` − 1日** で導出できるため、
スキーマ変更なしで済む。`store.version_window(versions, i)` がこれを返す。

### 前身が見つからない場合

早見表クローラー(issue #4)が未完成の現時点では、変更通知15件すべてが前身なしになる。
その場合も新行は通常どおり `INSERT` し、`IngestResult.unlinked` に品名を列挙して可視化する。
クローラー完成後に同じファイルを再取り込みすれば、そのときリンクが張られる。

## 5. 引き渡し契約(本田さん向け)

```python
from app.stage04 import store

conn = store.connect("data/knowledge.db")
versions = store.lookup_item_versions(conn, "スプレー缶")   # 古い順、is_current で絞らない
valid_from, valid_until = store.version_window(versions, 0)  # その版の有効期間
```

- `lookup_item_versions(conn, item_name) -> list[WasteItem]`
  同一品名の**全バージョン**を古い順(`effective_from` が `NULL` の早見表ベースライン → 日付昇順)で返す。
  該当なしは空リスト、例外は投げない。クエリ側もNFKC正規化する。
- `version_window(versions, index) -> (valid_from, valid_until)`
  `versions[index]` の有効期間。最新版の `valid_until` は `None`。

**なぜ別関数が必要か**: `app/data/lookup.py` の `lookup_item_exact` は `WHERE is_current = 1` を必ず付ける取り決め
(`data_schema.md` §3)なので、構造上いつでも新規則しか見えない。これは要件4-2が禁じている
「どちらか一方で断定する」状態そのものなので、旧行も返す読み取り口を分けて用意した。
**Teamb2/Teamb3の既存契約は変更していない。**

現在日付が2026-09-08、切替が2026-10-01なので、スプレー缶なら
「今は燃えないごみ、2026年10月1日からは『かん・びん』の収集日」という2区間の案内になる。
時系列の判定文言・プロンプトは本田さん担当。

## 6. 出典の表記

配布ファイルはWebページではないためhttp URLが存在しない。`source_url` は NOT NULL であり
要件2-2は出典表示を求めるので、`knowledge://お知らせ-4-2.md#お知らせ-01-スプレー缶` の形式を使う。
市のサイト由来(`https://www.city.kitakyushu.lg.jp/...`)と配布ファイル由来をスキーム名だけで区別できる。

→ **回答本文での出典の見せ方は、Teamb2/Teamb3と合わせて決めたい(STAGE02の未決事項)。**

## 7. `knowledge_sources` テーブル

STAGE04で追加した3つ目のテーブル。共有2テーブルには無い項目(`content_hash` / `source_type` / 取り込み日時)を
ここに持たせることで、**Teamb2のスキーマを変更せずに**配布ナレッジを扱えるようにしている。

| カラム | 用途 |
|---|---|
| `source_file` / `anchor` / `title` | どのファイルのどのお知らせか |
| `source_type` | 現状 `notice` のみ |
| `content_hash` | お知らせ1件の本文SHA-256。**UNIQUE制約で冪等性を担保** |
| `ingested_at` / `record_count` | 取り込み日時と件数 |

`store.list_knowledge_sources(conn)` はWebUIの「検索対象にした外部データのファイル一覧」(STAGE02 2-3)に
そのまま渡せる。Teamb2がissue #4で作るクロール済みソース一覧と統合して表示する想定。

## 8. 実行方法

```bash
python -m app.stage04.ingest knowledge/incoming/お知らせ-4-1.md --db data/knowledge.db
python -m app.stage04.ingest knowledge/incoming/お知らせ-4-2.md --db data/knowledge.db
```

同じファイルを二度流しても新規行は増えない(`content_hash` で判定)。

```bash
python -m pytest tests/stage04_ingest_test.py -v
```

テストはすべてインメモリSQLite上で完結し、GPU・L4サーバー・ネットワークを必要としない。

## 9. まだやっていないこと(担当境界)

- **`app/server.py` は変更していない。`/chat` はまだこのデータを見ない。**
  取り込んだ内容が回答に現れるのは、router(`app/rag/`、藤井さん)が `lookup_item_versions` を呼び、
  本田さんの時系列判定ロジックが文面を組み立ててからになる。これは分担上の境界であり、実装漏れではない。
- 4-3(早見表に無い品目の類似品物フォールバック) — 本田さん担当
- 早見表・収集日クローラー(issue #4) — Teamb2担当

## 10. チームへの確認事項

1. **FANGさん**: `data_schema.md` §4 の `effective_from` は新行に書く解釈でよいか(「4.」参照)
2. **FANGさん / 藤井さん**: `lookup_item_versions` を追加したこと。既存の `lookup_item_exact` 等の契約には手を触れていない
3. **全員**: 配布ファイル由来の出典 `knowledge://…` の表記と、回答本文での見せ方
