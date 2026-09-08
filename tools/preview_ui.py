#!/usr/bin/env python3
"""WebUI をローカルで実機確認するための開発用サーバー。

本番(`app/server.py` + vLLM)はGPUを17GB占有しており、L4 1枚では2つ目の
エンジンを起動できない。このスクリプトは vLLM と GPU 監視だけを差し替えて
`app/server.py` をそのまま起動するので、UI・`/knowledge`・引用付与・SSEの
描画を本物と同じコードで確認できる。回答文だけは固定のダミーが流れる。

    python3 tools/preview_ui.py            # 127.0.0.1:8090 で起動
    python3 tools/preview_ui.py --port 9000

VS Code Remote-SSH で接続している場合、PORTSパネルに転送が現れるので
手元のブラウザから http://localhost:8090 で開ける。SSH直結なら
    ssh -L 8090:localhost:8090 <このホスト>
でトンネルを張る。ログインパスワードは起動時に表示される。

データ層(`app/data/`)がある場合はデモデータを流し込んだ一時DBを使い、
無い場合は `app/rag/_stubs.py` のダミーで動く。
"""

import argparse
import asyncio
import os
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "app"))

PASSWORD = "preview"
# リポジトリの外に置く。`var/` は app/data/db.py の既定の置き場でもあり、
# プレビュー用のダミーDBを混ぜると紛らわしいため。
DEMO_DB = Path(tempfile.gettempdir()) / "rag_preview_ui.db"

DEMO_ITEMS = [
    ("アイロン", "アイロン", "家庭ごみ", "", "924_10038", "ごみの分別早見表(あ行)"),
    ("空き缶", "空き缶(飲料用)", "資源化ごみ", "中をすすいで出す", "924_10038", "ごみの分別早見表(あ行)"),
    ("テレビ", "テレビ(ブラウン管式を除く)", "家電リサイクル対象",
     "家電リサイクル法の対象。販売店へ引取りを依頼する", "924_10041", "ごみの分別早見表(た行)"),
    ("たんす", "たんす", "粗大ごみ", "収集は事前申込制。粗大ごみ受付センターへ",
     "924_10041", "ごみの分別早見表(た行)"),
    ("体温計(水銀)", "体温計(水銀式)", "特定ごみ", "破損しないよう包んで出す",
     "924_10041", "ごみの分別早見表(た行)"),
]
DEMO_SCHEDULES = [
    ("小倉北区", "浅野1丁目", "家庭ごみ", "火,金"),
    ("小倉北区", "浅野1丁目", "資源化ごみ", "木"),
    ("小倉北区", "浅野1丁目", "プラスチック", "水"),
    ("小倉北区", "大手町", "家庭ごみ", "月,木"),
]
URL = "https://www.city.kitakyushu.lg.jp/contents/{}.html"

DUMMY_REPLY = (
    "**テレビ**は「家電リサイクル対象」です。\n\n"
    "- 家電リサイクル法の対象のため、通常のごみとしては出せません。\n"
    "- 購入した販売店、または買い替え先の販売店へ引取りを依頼してください。\n\n"
    "(これはプレビュー用の固定応答です。実際の生成は行われません)"
)


CRAWLED_AT = "2026-09-08T01:00:00Z"

# データ層が未マージのときに使う、documents/data_schema.md どおりの最小スキーマ。
# app/data/db.py が入ればそちらが正となり、この定義は使われない。
_FALLBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS waste_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT, item_name TEXT NOT NULL,
    item_name_raw TEXT NOT NULL, disposal_method TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '', source_url TEXT NOT NULL,
    source_page_title TEXT, crawled_at TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 1, effective_from TEXT, superseded_by INTEGER);
CREATE TABLE IF NOT EXISTS collection_schedule (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ward TEXT NOT NULL, town TEXT NOT NULL,
    waste_type TEXT NOT NULL, weekday TEXT NOT NULL, source_url TEXT NOT NULL,
    crawled_at TEXT NOT NULL, is_current INTEGER NOT NULL DEFAULT 1,
    effective_from TEXT, superseded_by INTEGER);
"""


def seed_demo_db() -> str:
    """デモデータのDBを作る。戻り値は使った経路の説明。

    データ層(`app/data/db.py`)があればそれを使い、無ければ同じスキーマを
    直接作る。どちらでもナレッジ一覧の表示を確認できるようにするため。
    """
    import datetime

    DEMO_DB.parent.mkdir(parents=True, exist_ok=True)
    os.environ["DB_PATH"] = str(DEMO_DB)
    os.environ["RAG_DB_PATH"] = str(DEMO_DB)
    DEMO_DB.unlink(missing_ok=True)

    try:
        from app.data import db
        from app.data.schema import CollectionSchedule, WasteItem
    except ImportError:
        import sqlite3

        conn = sqlite3.connect(DEMO_DB)
        conn.executescript(_FALLBACK_SCHEMA)
        conn.executemany(
            "INSERT INTO waste_items(item_name, item_name_raw, disposal_method, notes,"
            " source_url, source_page_title, crawled_at, is_current)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            [(n, raw, m, note, URL.format(page), title, CRAWLED_AT)
             for n, raw, m, note, page, title in DEMO_ITEMS],
        )
        conn.executemany(
            "INSERT INTO collection_schedule(ward, town, waste_type, weekday,"
            " source_url, crawled_at, is_current) VALUES (?, ?, ?, ?, ?, ?, 1)",
            [(w, t, k, d, URL.format("924_10102"), CRAWLED_AT)
             for w, t, k, d in DEMO_SCHEDULES],
        )
        conn.commit()
        conn.close()
        return f"デモDB {DEMO_DB} (データ層は未導入のため、検索自体はダミー)"

    now = datetime.datetime.fromisoformat(CRAWLED_AT.replace("Z", "+00:00"))
    conn = db.connect(DEMO_DB)
    db.rebuild_db(
        conn,
        [
            WasteItem(item_name=n, item_name_raw=raw, disposal_method=m, notes=note,
                      source_url=URL.format(page), source_page_title=title, crawled_at=now)
            for n, raw, m, note, page, title in DEMO_ITEMS
        ],
        [
            CollectionSchedule(ward=w, town=t, waste_type=k, weekday=d,
                               source_url=URL.format("924_10102"), crawled_at=now)
            for w, t, k, d in DEMO_SCHEDULES
        ],
    )
    conn.close()
    return f"デモDB {DEMO_DB} (データ層あり)"


def install_stubs():
    """vLLM と NVML を差し替える。GPUもモデルも使わずに server.py を起動するため。"""
    vllm = types.ModuleType("vllm")

    class _Stub:
        def __init__(self, *a, **k):
            pass

        @classmethod
        def from_engine_args(cls, *a, **k):
            return cls()

    vllm.AsyncEngineArgs = vllm.AsyncLLMEngine = vllm.SamplingParams = _Stub
    sys.modules["vllm"] = vllm

    class _Mem:
        used, total = 17167 * 1024 ** 2, 23034 * 1024 ** 2

    class _Util:
        gpu = 37

    nvml = types.ModuleType("pynvml")
    nvml.nvmlInit = lambda: None
    nvml.nvmlDeviceGetHandleByIndex = lambda i: object()
    nvml.nvmlDeviceGetMemoryInfo = lambda h: _Mem()
    nvml.nvmlDeviceGetUtilizationRates = lambda h: _Util()
    nvml.nvmlDeviceGetName = lambda h: "NVIDIA L4 (preview)"
    sys.modules["pynvml"] = nvml


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--password", default=PASSWORD)
    args = ap.parse_args()

    os.environ["WEBUI_PASSWORD"] = args.password
    data_source = seed_demo_db()
    install_stubs()

    import server

    # モデルのロードは行わない
    server.app.router.on_startup.clear()

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "<preview>"

    class FakeOutput:
        def __init__(self, text):
            self.outputs = [types.SimpleNamespace(text=text)]

    class FakeEngine:
        async def generate(self, prompt_input, params, request_id):
            acc = ""
            for ch in DUMMY_REPLY:
                acc += ch
                await asyncio.sleep(0.006)   # ストリーミングの見た目を再現する
                yield FakeOutput(acc)

    server.processor = FakeProcessor()
    server.engine = FakeEngine()

    print("=" * 64)
    print("  WebUI プレビュー")
    print(f"  URL       : http://localhost:{args.port}")
    print(f"  パスワード: {args.password}")
    print(f"  検索データ: {data_source}")
    print("  回答生成  : 固定のダミー応答(vLLMは起動しません)")
    print()
    print("  VS Code Remote-SSH なら PORTS パネルに転送が現れます。")
    print(f"  SSH直結なら: ssh -L {args.port}:localhost:{args.port} <このホスト>")
    print("=" * 64, flush=True)

    import uvicorn
    uvicorn.run(server.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
