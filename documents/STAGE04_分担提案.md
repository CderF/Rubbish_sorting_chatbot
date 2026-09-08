# STAGE 04 — 分担と連携ポイント(動的なナレッジ拡張)

Team B / Teamb4 (周 楚涵)
作成日: 2026-09-08
更新: 2026-09-08 — 確定した `documents/data_schema.md` / `app/data/*.py`(PR #6)に合わせて全面改訂(FANG さんの PR #3 レビュー反映)

関連: [[data_schema.md]](DBスキーマ) / [[STAGE04_取り込み仕様.md]](取り込み側の実装仕様・引き渡し契約) / [[stage02&03_dev_plan.md]](分担・インターフェース契約) / [[CONTEXT.md]](用語集)

---

## 0. この文書の目的

STAGE 04(動的なナレッジ拡張)の担当分けと、STAGE 02/03 の実装との連携ポイントを1枚にまとめる。
データ側の具体的な実装仕様と本田さんへの引き渡し契約は [[STAGE04_取り込み仕様.md]] を参照。

- STAGE 02/03 は「分別・出し方・収集日・画像判定そのもの」= FANG さん(データ層)・藤井さん(応用層)
- **STAGE 04 は「ナレッジが変わること」= 周(取り込み・データ側)+ 本田さん(回答生成側)**
  - 周: 配布お知らせの取り込み、新旧バージョン列の構築、差分/relink、バージョン列の読み取り関数
  - 本田さん: バージョン列 → 時系列回答、4-2 の文面/プロンプト、4-3 類似品物フォールバック、評価

---

## 1. STAGE 04 が扱う 3 つのサブ要件

### 4-1 新ナレッジを取り込むと、回答に反映される

- 採点中に配布されるお知らせファイル(今回は Markdown 2 本、各 15 件)を解析し、`waste_items` に取り込む。
- 早見表に無い品目(ホバーボード等)は単独レコードとして `INSERT`。
- 実装: `app/notices/{notice_parser,ingest,store}.py`(PR #5)。

### 4-2 旧根拠と新お知らせが併存する場合、断定せず「いつから・何が・どう変わるか」を示す

- 早見表にある品目の規則変更(スプレー缶等、2026-10-01 適用)は、置き換える行と**並存**させる。
- `documents/data_schema.md` §4 の運用ルール通り:
  - 新規行を `INSERT`(`is_current = 1`、`effective_from` = 適用開始日)
  - 旧行を `UPDATE`(`is_current = 0`、`superseded_by` = 新規行の `id`。旧行の `effective_from` は触らない)
- 回答側は同一 `item_name` の全バージョンを取得し、`effective_from` を「切替日」として時系列で案内する。片方だけで断定しない。

### 4-3 早見表にない品物でも、類似品物の情報から案内する

- 構造化の完全一致・部分一致が両方 miss → 類似度検索 → top-k → 「類似する『◯◯』に準じると…。ただし個別記載がないため確実な分別は市に確認を」。
- **回答に「これは類推であり原文ではない」と明記**(STAGE 06 と整合)。
- 藤井さんの `vector_search_items`(issue #7)を再利用する想定。

---

## 2. STAGE 02/03 のマージ前でも進められること

1. データ側(周): クローラー(FANG さん PR #6)の上に **差分検知 + relink** を実装する。`data_schema.md` §4 が「更新検知・差分抽出のロジックは STAGE04 側で新規実装」と明記している通り。
2. 回答側(本田さん): `store.item_timeline()` / `app/notices/lookup.pending_change()` の戻り値に対する **時系列回答関数を純関数として** 書ける。DB・LLM・サーバー不要。詳細は [[STAGE04_取り込み仕様.md]] の「引き渡し契約」。
3. 評価ケース(質問 → 期待回答)は実装非依存でいま書ける。

---

## 3. 周(Teamb4)の担当タスク

| # | タスク | 状態 |
|---|---|---|
| 1 | お知らせ解析 + 取り込み(4-1) + 新旧バージョン列(4-2 データ側) | **PR #5 実装済み** |
| 2 | `relink` — クローラーを後から実行した場合に前身へ接続し直す | **PR #5 実装済み** |
| 3 | 再クロール時の差分検知(行単位の追加/変更/削除抽出) | 未着手(FANG さん PR #6 の全件洗い替えの上に追加) |
| 4 | `collection_schedule` のバージョン列対応(収集日改定お知らせが来た場合) | 未着手(テーブルは対応済み、ロジック未) |
| 5 | STAGE 04 評価ケース | 未着手 |

---

## 4. STAGE 02/03 実装との連携ポイント(確定分)

STAGE 04 は STAGE 02/03 の検索層の**外側**に足す。当初の「4 点お願い」は、確定した schema・実装に対して以下の通り整理された。

### ① レコードのスキーマ

**共有テーブル(`app/data/schema.py`)は変更しない。** 当初案からの差分:

| 当初案(PR #3 旧) | 確定した実装 | 補足 |
|---|---|---|
| `fetched_at` | `crawled_at` | 単純な改名 |
| `effective_date` | `effective_from` | 単純な改名。**書き込むのは常に新規行側**(`data_schema.md` §4、FANG さん確認済み) |
| `supersedes`(新→旧を指す) | `superseded_by`(**旧→新を指す。向きが逆**) | バージョン列の辿り方はこの向きに依存。`is_current = 0` の行の `superseded_by` を辿ると後継 `id` が得られる |
| `valid_until`(カラム) | **カラムなし。導出**する | 後継行の `effective_from` − 1 日。`store.version_window()` が返す |
| `item_key`(新カラム) | **不要**。`item_name` 自体が NFKC 正規化済みの検索キー | |
| `source_type`(全ナレッジ1テーブル + 列) | **なし。テーブルは分割**(`waste_items` / `collection_schedule`、キー構造が別物) | 配布お知らせは `waste_items` に入れ、`source_url` を `knowledge://<file>#<anchor>` にしてサイト由来と区別する |
| `content_hash` | 共有 schema には無し。**STAGE 04 独自の `knowledge_sources` テーブルに持たせる**(PR #5、FANG さん了承済み) | 冪等性キー。共有カラムと独立 |
| `published_at` | 現状不要(配布お知らせに公示日が明記されていない)。必要になれば `knowledge_sources` に独立追加 | |

STAGE 04 が追加するのは `knowledge_sources` テーブル1つだけ(`app/notices/store.py`、PR #5)。

### ② 検索関数

- **全バージョン取得は別関数**。`lookup_item_exact` / `lookup_item_partial`(issue #4)は藤井さんが依存する契約なのでシグネチャを変えない。
  → `app/notices/store.py` に `lookup_item_history(conn, item_name) -> list[WasteItem]`(FANG さん提案の名前)として実装済み(PR #5)。`is_current` で絞らず全バージョンを古い順で返す。
  回答側は通常これを直接使わず、平坦化した `store.item_timeline()` / `app/notices/lookup.pending_change()` を使う。
- **説明ページ検索**は当初 `search_descriptions` としていたが、issue #7 で `vector_search_explanations(query, top_k)` として仕様化済み(藤井さん担当、未実装)。**名前を合わせて再利用**する。
- **ナレッジソース一覧(manifest)**: `app/data/db.py` の `list_sources()`(クロール済みページ)が近い役割を持つ。STAGE 04 は取り込んだお知らせ用に `store.list_knowledge_sources()` を持つ。統一ビューが必要なら `list_sources()` を拡張する形で FANG さんと相談。

### ③ ナレッジファイルの配置とインデックス — 確定

- 構造化 DB: `app/data/db.py` の `var/kitakyushu_waste.db`(`DB_PATH` で上書き可)。**STAGE 04 の取り込みも同じ DB に書く**(`store.py` が `app/data/db.py` を土台にする)。
- 取り込み順序: クローラーより先にお知らせを取り込んでも `INSERT` はされ、前身リンクだけ保留になる。クローラー実行後に `python -m app.notices.ingest --relink` で接続。
- ベクトルストアへの新規お知らせの反映(再 embedding)は、藤井さんの `app/rag/vector_store.py`(issue #7)完成後に連携方法を決める。

### ④ 回答内の出典フォーマット — 未決(STAGE 02 と共通の宿題)

配布お知らせは http URL が無いため `source_url` を `knowledge://<file>#<anchor>` にしている。
回答本文での出典の見せ方(SSE 本文に含める / 別フィールド / WebUI のみ)はチームで決める。

---

## 5. 中文サマリー(周・FANG 向け)

FANG 的 PR #3 反馈已全部吸收,本文档按落地的 schema 重写:

- **改名**: `fetched_at`→`crawled_at`、`effective_date`→`effective_from`(只写新行,FANG 已在 `data_schema.md` §4 确认)
- **`supersedes`→`superseded_by`,方向相反**: 旧行指向新行。版本链靠这个方向遍历
- **`valid_until` 不设列**,由后继行 `effective_from` − 1 日导出(`store.version_window()`)
- **`item_key` 不要**: `item_name` 本身就是 NFKC 归一化后的检索键
- **`source_type` 统一表不要**: `waste_items` / `collection_schedule` 是分表。配布お知らせ进 `waste_items`,`source_url` 用 `knowledge://` 前缀区分
- **`content_hash`** 放 STAGE 04 自己的 `knowledge_sources` 表(PR #5,FANG 已同意)
- **全版本取得是独立函数**: `lookup_item_exact/partial`(藤井依赖)不动;`store.lookup_item_history()`(采用 FANG 建议名)已在 PR #5 实现。回答侧一般用平坦化的 `store.item_timeline()` / `lookup.pending_change()`
- **`search_descriptions`** → 用 issue #7 的 `vector_search_explanations`(藤井担当)
- **manifest** → `app/data/db.py` 的 `list_sources()` 已近似;需要统一视图就扩展它

数据侧实现细节和给本田的交接契约见 [[STAGE04_取り込み仕様.md]]。
