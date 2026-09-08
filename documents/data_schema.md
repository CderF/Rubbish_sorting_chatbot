# データスキーマ定義書

Team B (データ層: Teamb2 FANG XIUWEN)
作成日: 2026-09-08
関連: [[STAGE02&03開発分担計画書]](stage02&03_dev_plan.md参照)、[[ドメイン用語集]](CONTEXT.md参照)

STAGE02/03で構築する構造化DB(SQLite)の2テーブルを定義する。STAGE04(動的ナレッジ拡張、新旧情報並存)の担当者が土台として使えるよう、バージョン管理用のカラムをあらかじめ用意しているが、**STAGE02/03の実装ではこれらのカラムは初期値のまま(常に最新1件)で運用してよい**。

---

## 1. `waste_items`(早見表パス)

ごみ分別早見表(あ行〜わ行、8ページ)由来の品目データ。

| カラム名 | 型 | NULL | 説明 |
|---|---|---|---|
| `id` | INTEGER | NOT NULL | 主キー、自動採番 |
| `item_name` | TEXT | NOT NULL | 検索キーとなる品名(NFKC正規化・全角半角統一済み) |
| `item_name_raw` | TEXT | NOT NULL | クロール時点のページ表記そのまま(表記ゆれ確認・デバッグ用) |
| `disposal_method` | TEXT | NOT NULL | 出し方 |
| `notes` | TEXT | NULL可 | 備考(早見表に記載がなければ空文字) |
| `source_url` | TEXT | NOT NULL | 参照元ページURL(引用に必須) |
| `source_page_title` | TEXT | NULL可 | ページ見出し(WebUIのファイル一覧表示用、STAGE02 2-3対応) |
| `crawled_at` | TEXT | NOT NULL | このレコードを取得したUTC日時、ISO8601形式(例: `2026-09-08T01:23:45Z`) |
| `is_current` | INTEGER(BOOL) | NOT NULL | 最新版フラグ。STAGE02/03では常に `1`。STAGE04で新しいクロール結果が古い内容と食い違った場合、古い行を `0` にして残しつつ新しい行を追加する運用を想定 |
| `effective_from` | TEXT | NULL可 | この情報が有効になった日付。STAGE02/03では未使用(`NULL`のまま)。STAGE04で「いつから何が変わるか」を案内する際に使う |
| `superseded_by` | INTEGER | NULL可 | このレコードを置き換えた新レコードの `id`(自己参照)。STAGE02/03では未使用(`NULL`のまま) |

**インデックス**: `item_name` に対して(完全一致・部分一致検索用)、`(item_name, is_current)` に対して(最新版の一意性を保つ想定の複合インデックス)

**サンプル行**:

```json
{
  "id": 1,
  "item_name": "テレビ",
  "item_name_raw": "テレビ（ブラウン管式を除く）",
  "disposal_method": "粗大ごみ",
  "notes": "収集は事前申込制。詳細は粗大ごみ受付センターへ",
  "source_url": "https://www.city.kitakyushu.lg.jp/contents/924_10041.html",
  "source_page_title": "ごみの分別早見表(た行)",
  "crawled_at": "2026-09-08T01:00:00Z",
  "is_current": 1,
  "effective_from": null,
  "superseded_by": null
}
```

---

## 2. `collection_schedule`(収集日パス)

「資源とごみの収集日」(7区、区内は町名・丁目単位)由来の収集曜日データ。

| カラム名 | 型 | NULL | 説明 |
|---|---|---|---|
| `id` | INTEGER | NOT NULL | 主キー、自動採番 |
| `ward` | TEXT | NOT NULL | 区名(例: "門司区") |
| `town` | TEXT | NOT NULL | 町名・丁目(例: "中町1丁目") |
| `waste_type` | TEXT | NOT NULL | ごみ種別(家庭ごみ/プラスチック/資源ごみ/粗大ごみ 等、ページに準拠した表記) |
| `weekday` | TEXT | NOT NULL | 収集曜日。ページの表記をそのまま格納した完全な文字列(例: `"火曜日・金曜日"`、月次の粗大ごみは `"第2木曜日"` 等)。カンマ区切りの裸の曜日ではない — 利用側で「曜日」を追加で足さないこと(実際の不具合: STAGE03採点で二重表記"第2木曜日曜日"になっていた) |
| `source_url` | TEXT | NOT NULL | 参照元ページURL |
| `crawled_at` | TEXT | NOT NULL | 取得日時(ISO8601、`waste_items`と同形式) |
| `is_current` | INTEGER(BOOL) | NOT NULL | 最新版フラグ。STAGE02/03では常に `1` |
| `effective_from` | TEXT | NULL可 | STAGE02/03では未使用。収集曜日の改定日案内用(STAGE04) |
| `superseded_by` | INTEGER | NULL可 | STAGE02/03では未使用 |

**インデックス**: `(ward, town, waste_type)` の複合インデックス(収集日パスの主要な検索条件)

**サンプル行**:

```json
{
  "id": 1,
  "ward": "門司区",
  "town": "中町1丁目",
  "waste_type": "家庭ごみ",
  "weekday": "火曜日・金曜日",
  "source_url": "https://www.city.kitakyushu.lg.jp/contents/924_10102.html",
  "crawled_at": "2026-09-08T01:10:00Z",
  "is_current": 1,
  "effective_from": null,
  "superseded_by": null
}
```

---

## 3. STAGE02/03時点の運用ルール

- 両テーブルとも `is_current` は常に `1`、`effective_from` / `superseded_by` は常に `NULL` のまま運用する
- 再クロール時は既存データを `DELETE` してから全件 `INSERT` し直す、単純な「全件洗い替え」でよい(`documents/stage02&03_proposal.md` 4章の前方互換方針を参照)
- `app/data/lookup.py` の `lookup_item_exact` / `lookup_item_partial` / `lookup_collection_schedule` は、内部で `WHERE is_current = 1` を必ず付与してクエリする(STAGE04導入後に旧データが残っていても誤って返さないため、STAGE02/03の時点からこの条件を入れておく)

## 4. STAGE04担当者向けメモ

- 新旧情報を並存させたい場合、行ごとの操作は次の通り(**`effective_from` を書き込むのは常に新規行側**。カラムの定義(1〜2章)が「この情報が有効になった日付」である以上、退役する旧行に新情報の適用日を書くのは意味が反転してしまうため):
  - **新規行(新情報)**: `INSERT`。`is_current = 1`、`effective_from` = 新情報の適用開始日
  - **旧行(現在 `is_current = 1` の行)**: `UPDATE`。`is_current = 0` に変更、`superseded_by` に新規行の `id` を設定。この行の `effective_from` は触らない(その行が過去に新規行として作られた際に設定された値、または初回クロール由来で `NULL` のまま — どちらであっても上書きしない)
- 「いつから何がどう変わるか」を案内する要件(4-2)は、同一 `item_name`(または同一 `ward`+`town`+`waste_type`)に対して `is_current = 0` の行と `is_current = 1` の行を両方取得し、新行の `effective_from` を「切替日」として提示する(旧行の `effective_from` は — 存在すれば — 「その情報がいつから有効だったか」の参考情報になる)
- 本設計はテーブル定義のみの前方互換であり、更新検知・差分抽出のロジックはSTAGE04側で新規実装が必要(STAGE02/03では単純な全件洗い替えのみ)
