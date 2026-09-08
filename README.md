# 北九州市 ごみ分別 RAG チャットボット

GMO データ&AI 5Days インターン課題 / Team B。
北九州市の「ごみ分別・捨て方」を案内するチャットボット。

現状: **STAGE 1 完了**(LLMホスト + Streaming API + 速度計測 + VRAM WebUI)。
STAGE 2 以降の RAG は未実装のため、回答内容はまだ北九州市の実データに基づいていない。

---

## 構成

```
採点サーバー ──POST /chat──> B1 160.251.238.168:8080
メンター    ──ブラウザ────>  同ポート / (要ログイン)
                                 │
                                 ├─ systemd: rag-chatbot.service
                                 └─ uvicorn (FastAPI)
                                      └─ vLLM AsyncLLMEngine
                                           └─ Qwen3-VL-8B-Instruct (FP8)
                                                └─ NVIDIA L4 24GB
```

| 項目 | 値 |
|---|---|
| モデル | Qwen3-VL-8B-Instruct (Apache 2.0、テキスト+画像) |
| 量子化 | FP8 オンライン量子化(重み ~9GB) |
| 推論エンジン | vLLM 0.28.0(Dify / Agent SDK / Ollama は不使用) |
| サーバー | Ubuntu 22.04 / NVIDIA L4 24GB / driver 595.71.05 / CUDA 13.3 |

## リポジトリ構成

```
app/server.py              FastAPI + vLLM 本体
app/webui.html             WebUI(チャット・画像添付・VRAM/速度表示)
deploy/rag-chatbot.service systemd ユニット
tests/spec_test.py         採点サーバー仕様への適合テスト
documents/                 各STAGEの調査・構築記録
requirements.txt           直接依存(バージョン固定)
.env.example               必要な環境変数のテンプレート
```

**リポジトリに入っていないもの**: モデル重み(17GB)、`venv`(7.8GB)、`/etc/rag-chatbot.env`(パスワード)、
ログ・コンパイルキャッシュ、課題PDF(社外秘のため意図的に除外)。

---

## API

### `POST /chat` — 採点サーバー用
認証なし(採点サーバーは認証ヘッダを送らないため)。代わりに**送信元IPで制限**しており、
`GRADER_IPS` 以外からは 403。

```json
{ "session_id": "grade-xxxx", "prompt": "...", "img": null }
```

応答は `Content-Type: text/event-stream`:

```
data: {"delta": "北九"}

data: {"delta": "州市の"}

data: [DONE]
```

仕様上の注意:
- **4xx を返してはいけない** — 採点側は4xxの本文を「回答」として記録するため、不正解が静かに積み上がる。
  そのためリクエストは寛容にパースし、`prompt` 欠落や `img` の型不正でも 200 + SSE を返す
- 本文が空だとその問がエラーになるため、生成が空ならフォールバック文言を返す
- 同じ `session_id` は複数ターンで継続する(段階5・6-3)。`session_id` をキーに会話を保持している
- 同時最大4リクエスト

### `POST /ui/chat` — WebUI 用
生成処理は `/chat` と同一。パスワード認証あり。

### `GET /` `/login` `/vram` `/info`
WebUI と監視。パスワード認証あり。

---

## サーバー構築手順

すでに B1(`160.251.238.168`)では構築済み。新しいマシンに載せる場合のみ以下。

### 1. NVIDIA ドライバ + CUDA

```bash
sudo apt-get update
sudo apt-get install -y build-essential dkms "linux-headers-$(uname -r)" python3-dev ninja-build
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y nvidia-driver-595-server cuda-toolkit
sudo reboot
```

再起動後 `nvidia-smi` で L4 が見えることを確認する。

> `python3-dev` と `ninja-build` は必須。無いと Triton / FlashInfer が起動時の
> JIT ビルドに失敗し、`Python.h: No such file` や `FileNotFoundError: 'ninja'` で落ちる。

### 2. Python 環境とモデル

```bash
git clone https://github.com/CderF/Rubbish_sorting_chatbot.git ~/app
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -r ~/app/requirements.txt
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir ~/models/Qwen3-VL-8B-Instruct
```

重みは約17GB。ディスクの空きを確認しておくこと。

### 3. 環境変数とサービス登録

`.env.example` を見て `/etc/rag-chatbot.env` を作成(root所有・600)したうえで:

```bash
sudo cp ~/app/deploy/rag-chatbot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rag-chatbot.service
```

ユニット内の `User` / `WorkingDirectory` / `ExecStart` はマシンに合わせて調整する。

### 4. ファイアウォール

```bash
sudo ufw allow from 160.251.239.96 to any port 8080 proto tcp comment "grading server"
sudo ufw allow 8080/tcp comment "WebUI (password protected)"
```

> `ufw limit` は使わないこと。チャット1通ごとに SSE 接続が張られるため、通常利用で誤って
> 遮断される。ブルートフォース対策はアプリ側(同一IPから5分間に8回失敗で429)で行っている。

---

## 運用

```bash
systemctl status rag-chatbot.service     # 状態確認
sudo systemctl restart rag-chatbot.service
tail -50 ~/server.log                    # 起動失敗時
```

- **起動には約60秒かかる**。重みのロードに加え、起動時にテキスト・画像それぞれ1回ずつ
  ダミー生成してJITコンパイルを済ませているため。ポートはウォームアップ完了後に開くので、
  **接続できた時点で必ずウォーム状態**(初回リクエストが13秒かかる問題を回避している)
- `webui.html` だけの変更は再起動不要(静的ファイル)。`server.py` を変えたら再起動する

### 更新の流れ

```bash
cd ~/app && make deploy        # git pull → make knowledge → restart
```

`make deploy` は `make knowledge`(クロール → お知らせ取り込み → relink → ベクトル
インデックス再構築)を挟む。**データ層(`app/data/`)や `knowledge/incoming/` を変えた
場合は必ずこれを通す。** コードだけの変更なら `git pull && sudo systemctl restart
rag-chatbot.service` でよいが、その場合も systemd の `ExecStartPre` が起動時に
お知らせの取り込み(冪等)だけは流し直す。

個別ターゲット:

```bash
make crawl      # 早見表・収集日を再クロール(遅い・サイトに負荷)
make ingest     # 配布お知らせを waste_items に取り込み
make relink     # 変更通知を早見表レコードにリンク
make index      # Chroma ベクトルインデックスを現在の行から再構築
make knowledge  # 上記4つ全部
```

---

## チーム開発

GPUは1台に1枚しかないため、全員がサーバー上で開発することはできない。

**SSHポートフォワードで手元から使う**(ファイアウォールを触らずに済む):

```bash
ssh -L 8080:localhost:8080 intern2026-B1-teamb2
```

ブラウザで `http://localhost:8080/` を開く。プロンプト調整やWebUI開発はこれで足りる。

**GPUが要らない作業**: 北九州市ページの取得・パース、早見表/収集日の構造化、チャンク分割、
embedding とベクトル検索の実装(小さいモデルならCPUで動く)。STAGE 2 の大部分はローカルで進められる。

**B2(`160.251.238.57`)**: ドライバとCUDAは導入済みでモデル・サービスは未展開。
B1を採点用に安定させたまま実験したい場合はこちらを使える。

### 適合テスト

```bash
python tests/spec_test.py                       # サーバー上で
CHAT_URL=http://localhost:8080/chat python tests/spec_test.py   # トンネル経由
```

採点サーバーと同じ方法でSSEをパースし、`img: null` の受理、4xxを返さないこと、
複数ターンの文脈保持、4並列などを検証する。

---

## 既知の課題

- **回答が冗長** — 単純な質問に765トークン/30.7秒かかることがある。1問合計180秒の
  制限に対して余裕が少ない。`MAX_TOKENS` の抑制と簡潔化プロンプトが必要
- **回答内容がまだ正しくない** — RAG未実装のため一般論を答える。STAGE 2 で解消予定
