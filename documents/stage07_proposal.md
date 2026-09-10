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

`app/mcp_server.py`(`FastMCP`、公式 `mcp` SDK v1)。5ツールを公開:

| ツール | 中身 |
|---|---|
| `search_waste_item(query)` | `tools.execute_tool("search_waste_item", …)` にそのまま委譲(早見表パス) |
| `lookup_collection_schedule(ward, town, waste_type)` | 同上(収集日パス。丁目省略の地区名解決込み) |
| `lookup_item_change_notice(item_name)` | 同上(お知らせパス、STAGE04 4-2) |
| `search_general_info(query)` | 同上(ベクトルRAGパス) |
| `get_disposal_guidance(prompt)` | **新規**。1発の相談文を `router._blocking_route(prompt)` に通し、ルータが組んだ `context`・`sources`・マッチ段階・`missing` をそのまま返す。**LLM生成は通さない**(このプロセスに vLLM は無い)。接続元のクライアント(Claude Desktop 等)がこの材料を使って文章を書く |

会話状態(STAGE05のスロット)は公開しない — マルチターンは接続元のクライアントが管理するものなので、
「その1発で何が引けるか」だけを返す設計にした。

**「自分たちで実装した」の解釈**: プロトコル実装(JSON-RPCフレーミング・stdioトランスポート)は
公式SDK(`mcp` パッケージ)に委ね、ツールのハンドラ・スキーマ・検索ロジックへの接続を自分たちで書いた。
FastAPIを使って `/chat` を自分たちで実装したのと同じ扱いとした。

### 2.2 7-2 — 外部クライアントから接続して呼び出せる状態にする【実装済み・stdio】

**接続方式は stdio**(SSE/HTTPのリモート公開は不採用 — 認証・TLS・B1へのデプロイが1日に収まらないため。
§4 未決事項参照)。

**検証方法**: 公式 `mcp` クライアントSDK(`mcp.client.stdio.stdio_client` + `ClientSession`)で
サブプロセスとして起動し、`initialize` ハンドシェイク→ `list_tools` → `call_tool` を実行。
5ツールすべてが正しい名前で列挙され、`search_waste_item` の呼び出しが正しいJSONを返すことを確認済み
(Claude Desktop が使うのと同じプロトコル経路)。

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

### 2.3 7-3 — リッチUI【部分的に実装済み】

- **表**: `get_disposal_guidance` が早見表・収集日の結果を GFM markdown の表にして `markdown` フィールドで
  返す(MCPクライアント側でそのままレンダリングされる)。加えて WebUI 側は `/knowledge`・`/trace`
  (STAGE06、PR #45)が既に表形式。
- **ファイル編集**: PR #38(マージ済み)でお知らせファイルのアップロード・取り込み・削除を実装済み。
- **画像出力**: 対象外(スコープカット)。品目の分類元画像をMCPレスポンスに `ImageContent` として
  エコーする、程度なら小さいが、1日の残り時間を7-1/7-2の実接続確認に使うことを優先した。

### 2.4 7-4 — パーソナライズ【ストアのみ実装、配線は未着手】

`app/user_memory.py`: 利用者ID単位の永続 key-value ストア(`remember` / `recall` / `recall_all` /
`forget` / `forget_all`、独立した `var/user_memory.db`)。STAGE05の `session_state.py` は**1会話限り**の
スロットなので、7-4が求める「セッションを跨いだ記憶」とは別の層として独立させた。

**未着手(配線)**:

1. **「利用者」の身元をどこから取るか** — MCP接続なら接続(クライアントインスタンス)単位、WebUIなら
   `require_login` は全員共通パスワードで個人を識別しない。どちらを軸にするか要決定(§4)。
2. **どこで自動的に読み書きするか** — 例えば `get_disposal_guidance` の冒頭で `recall_all` した地区を
   `_blocking_route` に渡す候補地区として使う、MCPに `remember`/`recall` ツールを足して接続元のClaudeに
   明示的に管理させる、等。

ストアの契約(何を保存でき、何を返すか)は固まっているので、配線は接続方式が決まり次第すぐ着手できる。

---

## 3. 検証

```bash
python -m pytest tests/mcp_server_test.py tests/user_memory_test.py -q
```

`app/rag/tools.py` への委譲・`get_disposal_guidance` の材料整形・`user_memory` の読み書き契約を
オフラインで検証(`tests/agentic_tools_test.py` と同じ孤立SQLiteフィクスチャ)。

実プロトコル経由の疎通確認は上記(§2.2)のとおり、公式クライアントSDKでの手動実行で確認済み
(自動テストには含めていない — サブプロセス起動を挟むため遅く、CIが無い現状のリポジトリでは
`tests/spec_test.py` と同じ「手元で確認する」対象とした)。

---

## 4. 未決事項

| 項目 | 内容 | いつまでに |
|---|---|---|
| SSE/HTTPでの遠隔公開(7-2) | B1にホストしてメンターが直接接続できるようにするか。認証・TLS・デプロイが必要 | 時間があれば |
| 7-4 の利用者識別の軸 | MCP接続単位 or WebUIログイン単位 | 配線着手時 |
| 7-4 の自動読み書きの接続先 | `_blocking_route` への地区プリフィルか、MCPの明示的な `remember`/`recall` ツールか、両方か | 配線着手時 |
| 7-3 画像出力 | `ImageContent` でのエコー等、対象外としたもの | 余裕があれば |
