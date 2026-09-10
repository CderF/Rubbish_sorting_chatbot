# STAGE 07 — 挑戦枠 実装状況

Team B / 周楚涵(Teamb4)
作成日: 2026-09-10(最終評価前日、残り1日での着手)

関連: [[CONTEXT.md]](ドメイン用語集) / `app/rag/tools.py`(Agentic RAG、PR #40) / `app/mcp_server.py` / `app/user_memory.py`

---

## 0. この文書の目的

課題PDF(`北九州ゴミ分類Chatbot課題.pdf`, p.56)が定めるSTAGE07「挑戦枠」4項目(7-1〜7-4)について、
残り1日で何を作ったか・何を対象外にしたかを記録する。STAGE07は**自動採点が無く、個別にメンター確認**
してもらう形式なので、STAGE04〜06のような「要件分析書を先に固めてから実装」ではなく、
**1日で確実に動く部分を先に作り切り、この文書で状況を報告する**形にした。

STAGE05(マルチターン、PR #24)・STAGE06(安全対応、PR #43/#45)は本書の対象外。両方とも
`_blocking_route` / `_route_safely` の中の話で、本書が扱う MCP サーバ・記憶ストアはどちらも
そのレイヤーの外(データ層への薄いクライアント)にあるため、依存しない。

---

## 1. 前提: 現行コードの再利用可能な部分

| 項目 | 既存の地盤 | 場所 |
|---|---|---|
| 7-1 の検索ロジック | Agentic RAG(PR #40)が既に `TOOL_SCHEMAS` + `execute_tool()` として整備済み。早見表・収集日・お知らせ・ベクトルRAGの4パスを薄くラップしただけで、検索ロジック自体の再実装は無い | [app/rag/tools.py](../app/rag/tools.py) |
| 7-1 の「1発の相談」用ルート | 決定論ルータ `_blocking_route(prompt)` は `state` 省略で単発呼び出しとして動く(STAGE02〜04と同じ契約) | [app/rag/router.py](../app/rag/router.py) |
| 7-3 の「表」 | `/knowledge`(STAGE02 2-3)・`/trace`(STAGE06 6-4、PR #45)が既に表形式のWebUIページを持つ | [app/webui.html](../app/webui.html), [app/trace.html](../app/trace.html) |
| 7-3 の「ファイル編集」 | `feat/webui-knowledge-manage`(PR #38、マージ済み)が お知らせ.md の アップロード・取り込み・削除を実装済み | [app/notices/admin.py](../app/notices/admin.py) |

---

## 2. 各要件の状況

### 2.1 7-1 — MCPサーバとして公開する【実装済み】

`app/mcp_server.py`(`FastMCP`、公式 `mcp` SDK v1)。**8ツール + 1静的リソース + 2リソーステンプレート**
を公開:

| ツール | 中身 |
|---|---|
| `search_waste_item(query)` | `tools.execute_tool("search_waste_item", …)` にそのまま委譲(早見表パス) |
| `lookup_collection_schedule(ward, town, waste_type)` | 同上(収集日パス。丁目省略の地区名解決込み) |
| `lookup_item_change_notice(item_name)` | 同上(お知らせパス、STAGE04 4-2) |
| `search_general_info(query)` | 同上(ベクトルRAGパス) |
| `get_disposal_guidance(prompt, user_id=None)` | 1発の相談文を `router._blocking_route(prompt)` に通し、`context`・`sources`・マッチ段階・`missing`・表(markdown)を返す。**LLM生成は通さない**。`user_id` を渡すと7-4の記憶で地区を補う(§2.4) |
| `remember_fact` / `recall_facts` / `forget_facts(user_id, …)` | 7-4 の記憶ストアへの明示的な読み書き |

| リソース | 中身 |
|---|---|
| `knowledge://sources` | 取り込み済みの参照元ページ一覧(`/knowledge` と同じデータ、`knowledge.list_sources()`) |
| `knowledge://{source_id}` | 1ページ分の内容(`knowledge.get_source()`) |
| `schedule-calendar://{ward}/{town}` | 収集日を1週間カレンダーのSVG画像にしたもの(7-3、§2.3) |

会話状態(STAGE05のスロット)は公開しない — マルチターンは接続元のクライアントが管理するものなので、
「その1発で何が引けるか」だけを返す設計にした。7-4の記憶はそれとは別の永続層。

**「自分たちで実装した」の解釈**: プロトコル実装(JSON-RPCフレーミング・stdioトランスポート)は
公式SDK(`mcp` パッケージ)に委ね、ツール・リソースのハンドラ・検索ロジックへの接続を自分たちで書いた。
FastAPIを使って `/chat` を自分たちで実装したのと同じ扱いとした。

### 2.2 7-2 — 外部クライアントから接続して呼び出せる状態にする【実装済み・stdio】

**接続方式は stdio**(SSE/HTTPのリモート公開は不採用 — 認証・TLS・B1へのデプロイが1日に収まらないため。
§4 未決事項参照)。

**検証方法**: 公式 `mcp` クライアントSDK(`mcp.client.stdio.stdio_client` + `ClientSession`)で
サブプロセスとして起動し、`initialize` ハンドシェイク→ `list_tools`/`list_resources`/
`list_resource_templates` → `call_tool`/`read_resource` を実行(Claude Desktop が使うのと
同じプロトコル経路)。以下を確認済み:

- 8ツールすべてが正しい名前で列挙される
- `remember_fact` → `recall_facts` が実プロセス間で往復する(7-4)
- `knowledge://sources` が読める(1静的リソース)
- `knowledge://{source_id}` / `schedule-calendar://{ward}/{town}` の2リソーステンプレートが
  列挙され、**日本語パスセグメント**(`小倉北区`・`浅野1丁目` をURLエンコードしたもの)を渡した
  `schedule-calendar://` の読み取りが `mimeType: image/svg+xml` で正しいSVGを返す

**Claude Desktop からの接続手順**(`claude_desktop_config.json` に追記):

```json
{
  "mcpServers": {
    "kitakyushu-waste": {
      "command": "/absolute/path/to/Rubbish_sorting_chatbot/.venv/bin/python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "/absolute/path/to/Rubbish_sorting_chatbot"
    }
  }
}
```

事前準備:

```bash
pip install -r requirements.txt -r requirements-mcp.txt
```

`var/kitakyushu_waste.db`(構造化DB)と `var/chroma/`(ベクトル索引)が要る。GPU・vLLMは不要
(`get_disposal_guidance` はLLM生成をしない設計にしたのはこのため — メンターの手元で軽量に動く)。

### 2.3 7-3 — リッチUI【実装済み】

- **画像**: `schedule-calendar://{ward}/{town}`(MCPリソース、`mimeType: image/svg+xml`)。収集日を
  1週間カレンダーのSVGにし、該当する曜日を塗る。PIL・matplotlib等の追加依存は使わず文字列組み立てのみ
  (`app/mcp_server.py` の `_render_week_svg`)。実クライアントSDKでの読み取り・mimeType確認済み(§2.2)。
- **表**: `get_disposal_guidance` が早見表・収集日の結果を GFM markdown の表にして `markdown` フィールドで
  返す。加えて WebUI 側は `/knowledge`・`/trace`(STAGE06、PR #45)が既に表形式。
- **ファイル編集**: PR #38(マージ済み)でお知らせファイルのアップロード・取り込み・削除を実装済み。

### 2.4 7-4 — パーソナライズ【実装済み・MCP接続単位】

`app/user_memory.py`: 利用者ID単位の永続 key-value ストア(`remember` / `recall` / `recall_all` /
`forget` / `forget_all`、独立した `var/user_memory.db`)。STAGE05の `session_state.py` は**1会話限り**の
スロットなので、7-4が求める「セッションを跨いだ記憶」とは別の層として独立させた。

**配線した内容**:

1. **利用者の識別軸は MCP 接続単位に決定**。WebUI(`require_login` が全員共通パスワードで個人を
   識別しない)ではなく、`get_disposal_guidance` / `remember_fact` 等の `user_id` 引数として接続元の
   クライアントに渡してもらう形にした。WebUIのログイン単位化は対象外(§4)。
2. **`get_disposal_guidance` に自動読み書きを実装**:
   - `user_id` を渡すと、まず素の発話で引く。地区が足りなければ(`missing` に「町名」等)、記憶に
     ある区・町名を文末に補ってもう一度引く(`used_memory` で分かる)。
   - 今回の発話から区・町名が読めていれば、次回のために記憶へ書き戻す(訂正すれば上書きされる)。
   - `extract_region()` は文中のどこからでも区名・町名を拾う既存ロジックなので、記憶を「文末に
     こっそり足す」だけで専用の合成ロジックを持たずに済んだ。
3. `remember_fact` / `recall_facts` / `forget_facts` で、接続元のクライアントが任意のキーを
   明示的に管理することもできる(区・町名以外を覚えさせたい場合)。

デモの流れ: 「ソファの収集日は?」(`user_id="mentor-1"`)→ 地区不明 → 「小倉北区浅野1丁目です、ソファの
収集日は?」→ 引けて、区・町名を記憶 → **別セッション**で「冷蔵庫は?」(同じ `user_id`)→ 記憶から
地区を補って引き直し、聞き返さない。

---

## 3. 検証

```bash
python -m pytest tests/mcp_server_test.py tests/user_memory_test.py -q
```

`app/rag/tools.py` への委譲・`get_disposal_guidance` の材料整形と7-4連携・`user_memory` /
`remember_fact`・`recall_facts`・`forget_facts` の読み書き契約・`knowledge://` リソース・
収集日SVGの曜日抽出とレンダリングを、オフラインで検証(`tests/agentic_tools_test.py` と同じ
孤立SQLiteフィクスチャ)。29 + 11 件。

実プロトコル経由の疎通確認は上記(§2.2)のとおり、公式クライアントSDKでの手動実行で確認済み
(自動テストには含めていない — サブプロセス起動を挟むため遅く、CIが無い現状のリポジトリでは
`tests/spec_test.py` と同じ「手元で確認する」対象とした)。

---

## 4. 未決事項

| 項目 | 内容 | いつまでに |
|---|---|---|
| SSE/HTTPでの遠隔公開(7-2) | B1にホストしてメンターが直接接続できるようにするか。認証・TLS・デプロイが必要 | 時間があれば |
| WebUIログイン単位の記憶(7-4) | 現状は MCP 接続(`user_id` 引数)単位のみ。WebUI から使うなら `require_login` を個人単位に拡張する必要がある | 余裕があれば |
| `get_disposal_guidance` の記憶利用範囲 | 現状は区・町名だけを自動で読み書き。品目の好み等まで広げるかは未定 | 余裕があれば |
| `var/user_memory.db` の運用 | 個人情報(区・町名)を平文でSQLiteに残す。`forget_facts` で消せるが、保持期間・アクセス制御の方針は未整理 | メンター確認時に相談 |
