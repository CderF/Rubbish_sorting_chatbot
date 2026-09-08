# STAGE 04 — お知らせ取り込み仕様

Team B / Teamb4 (周 楚涵)
作成日: 2026-09-08（PR #6 のデータ層マージを受けて更新）

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
| `app/stage04/store.py` | `app/data/db.py` の上に構築。`knowledge_sources` テーブルと、バージョン列の読み取り(`lookup_item_history` / `version_window`)だけを追加する |
| `app/stage04/ingest.py` | `Notice` → `WasteItem` 変換、新旧リンク、`relink`、CLIエントリポイント |
| `tests/stage04_ingest_test.py` | 単体テスト(パーサ・store・ingest・relink) |
| `tests/stage04_eval.py` + `tests/fixtures/stage04_eval_cases.json` | 『質問 → 期待DB状態』評価(4-1/4-2、データ側)。`python -m tests.stage04_eval` でグレーダー風の表を出力 |
| `tests/fixtures/hayamihyo_seed.json` | **テスト専用**の擬似早見表15件。本番投入禁止 |

`app/data/` 配下(Teamb2担当)のファイルは一切変更していない。`app/data/schema.py` の `WasteItem` を
そのまま再利用しており、独自モデルは作っていない。

`store.py` は `app/data/db.py`(PR #6 / issue #4、Teamb2)の `connect()` / `init_db()` / スキーマ / DBパスを
そのまま使う。取り込み先はクローラーが作る DB と同一(既定 `var/kitakyushu_waste.db`、`DB_PATH` で上書き可)。
独自に持つのは `knowledge_sources` テーブルだけ。`waste_items` / `collection_schedule` のDDLは重複させていない。

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

### 前身が見つからない場合と `relink`

早見表クローラーより先に取り込んだ場合、前身が無いので新行を `INSERT` するだけになり、
`IngestResult.unlinked` に品名を列挙する。クローラー実行後に:

```bash
python -m app.stage04.ingest --relink
```

を実行すると、`knowledge_sources` に記録した変更通知のうち前身が未接続のものを、
いま存在する早見表レコード(`source_url` が `knowledge://` でない `is_current = 1` 行)に接続する。
`relink` は何度実行しても安全(接続済みはスキップ)。再取り込み(ファイル自体は冪等スキップ)でも同じ効果。

### 早見表の「現行」表記がお知らせと食い違う場合

実サイトをクロールした早見表(864件)と、`お知らせ-4-2.md` が「現行（早見表）の…扱い」として
述べる分別が一致しないことがある(例: スプレー缶 → クロールは「家庭ごみ」、お知らせは「燃えないごみ」)。
また7品目(蛍光管・保冷剤・使い捨てライター・カセットテープ・使い捨てカイロ・使用済みの花火・乾電池)は
現在のクロール結果に存在しない。

対応:
- リンクは**品名(item_name)の一致で張る**。分別表記の一致は要求しない
- 食い違いは `IngestResult.disputed` / `RelinkResult.disputed` に列挙して可視化する(リンクはブロックしない)
- お知らせが述べる旧分別は新行の `notes` と `knowledge_sources.prior_disposal_method` に保存するので、
  前身行がクロールに無くても、回答側は新行だけから「旧: 〜、新: 〜」を組み立てられる
- クローラー側(PR #6、Teamb2)の取りこぼし/表記は別途 FANG さんと突き合わせる。STAGE04 の取り込みは
  「欠落」も「不一致」も破綻せず処理する

## 5. 引き渡し契約(本田さん向け)

```python
from datetime import date
from app.stage04 import store

conn = store.connect()                       # 共用DB（var/kitakyushu_waste.db、DB_PATH で上書き可）
tl = store.item_timeline(conn, "スプレー缶")   # ← これ1本で足りる想定
```

### `item_timeline(conn, item_name) -> ItemTimeline` （推奨。本田さんはこれだけ使えばよい）

`lookup_item_history` + `version_window` を平坦化し、`notes` のパースや `is_current` /
`superseded_by` の解釈を一切せずに「旧ルール → 新ルール」の時系列を扱えるようにしたもの。

```python
@dataclass(frozen=True)
class TimelineSegment:
    disposal_method: str
    valid_from: date | None      # None = ベースライン（開始日不明）
    valid_until: date | None      # None = 現在も有効
    source_url: str
    from_notice: bool             # 配布お知らせ由来か
    synthesized: bool             # 行が無く、お知らせの記述から復元した区間か
    disputed: bool                # 早見表の行とお知らせの言う現行が食い違う
    hayamihyo_disposal_method: str | None  # disputed のとき、早見表側の値

@dataclass(frozen=True)
class ItemTimeline:
    item_name: str
    found: bool                    # False ⇒ DBに無い ⇒ 呼び出し側は 4-3 へ
    segments: tuple[TimelineSegment, ...]   # 時系列順

    current_segment          # valid_until が None の区間（最新ルール）
    change_dates             # 切替日のタプル
    segment_on(when)         # その日に適用される区間
    changes_after(when)      # when より後の切替日。空でなければ 4-2 が該当
```

パターン別の戻り:

| ケース | `segments` |
|---|---|
| 4-1 追加(ホバーボード等) | 1区間、`valid_from = valid_until = None` |
| 4-2 前身あり(スプレー缶等) | 2区間以上、両側とも実レコード |
| 4-2 前身なし(乾電池・蛍光管等) | 2区間。古い方は `synthesized = True`、分別は `knowledge_sources.prior_disposal_method` から復元 |
| 4-2 かつ早見表と食い違い(スプレー缶等) | 古い区間の `disposal_method` はお知らせの言う現行(`disputed = True`、早見表側の値は `hayamihyo_disposal_method`)。**お知らせを正とする** |
| DBに無い | `found = False`、`segments = ()` |

### 低レベル関数（`item_timeline` の内部。直接使ってもよい）

- `lookup_item_history(conn, item_name) -> list[WasteItem]` — 全バージョンを古い順。該当なしは空、例外なし。NFKC正規化済み
- `version_window(versions, index) -> (valid_from, valid_until)` — その版の有効期間。最新版の `valid_until` は `None`

**なぜ別関数が必要か**: `app/data/lookup.py` の `lookup_item_exact` は `WHERE is_current = 1` を必ず付ける取り決め
(`data_schema.md` §3)なので、構造上いつでも新規則しか見えない。これは要件4-2が禁じている
「どちらか一方で断定する」状態そのものなので、旧行も返す読み取り口を分けて用意した。
**Teamb2/Teamb3の既存契約は変更していない。**

現在日付が2026-09-08、切替が2026-10-01なので、スプレー缶なら
「今は燃えないごみ、2026年10月1日からは『かん・びん』の収集日」という2区間の案内になる。
時系列の判定文言・プロンプトは本田さん担当。

前身行がクロールに無くて `lookup_item_history` が新行1件しか返さない品目でも、
その行の `notes` に「現行（早見表）の「◯◯」としての扱いは◯年◯月◯日まで有効。」が入っているので、
旧分別と切替日は新行だけから取れる。

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
python -m app.data.db                                          # 先にクローラーで早見表DBを構築
python -m app.stage04.ingest knowledge/incoming/お知らせ-4-1.md knowledge/incoming/お知らせ-4-2.md
python -m app.stage04.ingest --relink                          # 早見表を後から入れた場合
```

DBパスは `app/data/db.py` と共通(既定 `var/kitakyushu_waste.db`、`--db` か `DB_PATH` で上書き可)。
同じファイルを二度流しても新規行は増えない(`content_hash` で判定)。

```bash
python -m pytest tests/stage04_ingest_test.py -v
```

テストはすべてインメモリSQLite上で完結し、GPU・L4サーバー・ネットワークを必要としない。

## 8-2. 応用層(`app/rag/*`)への接続点

PR #6〜#10 がマージされ、`app/rag/router.py`(藤井さん)・`app/rag/knowledge.py`
(WebUI 2-3)が main にある。STAGE04 側はそこに次の形で噛み合う。

### router 向けの読み取り API — `app/stage04/lookup.py`

`app/data/lookup.py` と同じ作法(接続引数なし・スレッドローカル接続・`query_only`・
未検出でも例外なし)。違いは **`is_current = 1` で絞らない**こと(4-2 は旧ルールも要る)。

```python
from datetime import date
from app.stage04 import lookup

tl = lookup.item_timeline(item_name)              # ItemTimeline（found=False なら 4-3 へ）
tl = lookup.pending_change(item_name, date.today())  # 切替が今日以降にあるときだけ ItemTimeline、なければ None
```

### `router.py` への 4-2 反映 — **PR #5 で実装済み**

`_blocking_route` の末尾で `_apply_timeline(ans, item_name)` を呼び、`pending_change` が
タイムラインを返したら context に「現在: 旧 / YYYY-MM-DD から: 新」ブロックを足し、
`ans.timeline_change` を立て、`build_system_prompt` が時系列ルールを1行追加する。
参考(実装の要点):

```python
from app.stage04.lookup import pending_change

tl = pending_change(item_name, date.today())
if tl is not None:
    now_seg = tl.segment_on(date.today())
    later = [s for s in tl.segments if s.valid_from and s.valid_from > date.today()]
    ans.context += (
        "\n\n## この品目は出し方が変わります（断定せず両方を案内する）\n"
        f"- 現在（〜{now_seg.valid_until}）: {now_seg.disposal_method}\n"
        + "".join(f"- {s.valid_from} から: {s.disposal_method}\n" for s in later)
    )
    ans.sources = _dedupe(ans.sources + [s.source_url for s in tl.segments])
```

回答文言のトーン調整(自然な言い回し)は本田さんが `_apply_timeline` の出力文と
system prompt ルールを見て詰める。

### WebUI ナレッジ一覧 — 対応済み

`app/rag/knowledge.py` を更新し、取り込んだお知らせを **ファイル単位**で一覧に出す
(`knowledge_sources` を参照)。早見表に混ざった `knowledge://` 出典行は
クロールページ側の集計から除外して二重表示を防いでいる。

## 9. まだやっていないこと(担当境界)

- **4-3(類似品物の但し書き)は未対応。** `_search_item` がヒット段階(exact/partial/similar)を
  `RoutedAnswer` に立て、`build_system_prompt` に「similar のときは『類似する◯◯に準じると…。
  確実な分別は市に確認を』と書く」規則を足す。`_search_item` の1行変更が要るので藤井さんと相談。
- 4-3(早見表に無い品目の類似品物フォールバック) — 本田さん担当
- 早見表・収集日クローラー(issue #4) — Teamb2担当

## 10. チームへの確認事項

1. **FANGさん**: `data_schema.md` §4 の `effective_from` は新行に書く解釈でよいか(「4.」参照)
2. **FANGさん / 藤井さん**: 全バージョン取得を `lookup_item_history`（FANGさん提案の名前）として追加。既存の `lookup_item_exact` 等の契約は不変
3. **全員**: 配布ファイル由来の出典 `knowledge://…` の表記と、回答本文での見せ方
4. **FANGさん**: クローラー結果と `お知らせ-4-2.md` の「現行」表記の食い違い(8件不一致・7件欠落)。クローラー側の問題か、お知らせが意図的にずらした採点データかを確認したい
