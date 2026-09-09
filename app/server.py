"""
STAGE 1-1 chat server (v2 — aligned with the grading server's documented contract).

    POST /chat   Content-Type: application/json
    { "session_id": "grade-xxxx", "prompt": "...", "img": null | "<base64>" }

    -> HTTP 200, Content-Type: text/event-stream
       data: {"delta": "..."}\n\n
       ...
       data: [DONE]\n\n

Contract notes that shaped this file (from the grading spec):
  * 4xx is NOT treated as an error by the grader — the response body is recorded
    as the *answer*. A FastAPI/Pydantic 422 would therefore be scored as a wrong
    answer, silently. So the body is parsed leniently and never rejected.
  * 5xx aborts the whole stage, but is the preferred signal for a genuine fault
    (it is visible). So real engine failures are deliberately NOT swallowed.
  * An empty body fails the question, so generation always emits something.
  * The same session_id is reused across turns of one question (stages 5 / 6-3);
    conversation state must be kept per session_id.
  * Up to 4 concurrent requests per team.
  * PIJ(プロンプトインジェクション)対策として、生成は全量バッファリングして
    から1フレームで送出する。SSE は一度送った delta を取り消せないため、
    逐次送出のままではシステムプロンプト漏洩・反復ループを検知しても本文に
    混ざったあとになる。フレーム形式は仕様どおり data: {"delta": ...} のまま。

Model: Qwen3-VL-8B-Instruct, served in-process via vLLM AsyncLLMEngine.
"""

import asyncio
import base64
import json
import os
import secrets
import sys
import time
import uuid
from collections import OrderedDict
from io import BytesIO

import pynvml
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

# systemd は作業ディレクトリ app/ で `uvicorn server:app` を起動するため、
# そのままでは app パッケージが解決できない。リポジトリルートを通しておくと
# `uvicorn server:app`(app/ から) でも `uvicorn app.server:app`(ルートから) でも
# 同じ import が通る。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from app.rag import guard, knowledge, router  # noqa: E402  (sys.path 調整後に import)

MODEL_NAME = "Qwen3-VL-8B-Instruct"
# Where the downloaded weights live. Override per machine via MODEL_PATH so the
# repo can be deployed by any team member / on either server.
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.expanduser(f"~/models/{MODEL_NAME}")
)
# webui.html sits next to this file in the repo.
WEBUI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui.html")

# --- access control -------------------------------------------------------
# /chat must stay unauthenticated (the grading server sends no auth headers),
# so it is restricted by source IP instead. The human-facing routes are behind
# a password so mentors can reach the WebUI from anywhere.
GRADER_IPS = {ip.strip() for ip in os.environ.get(
    "GRADER_IPS", "160.251.239.96,127.0.0.1").split(",") if ip.strip()}
WEBUI_USER = os.environ.get("WEBUI_USER", "mentor")
WEBUI_PASSWORD = os.environ.get("WEBUI_PASSWORD")  # required; never hardcoded

MAX_MODEL_LEN = 8192
# The grader allows 120s per turn; stay well inside it.
MAX_TOKENS = 1024
# Keep at most this many sessions in memory (grader issues a new id per question).
MAX_SESSIONS = 256
# 生成の打ち切り時刻(秒)。採点は1ターン120秒なので、その内側で必ず返す。
# 実測は765トークンで約30秒(≒24tok/s)、MAX_TOKENS まで回しても約42秒なので、
# 通常の生成がここに当たることはない。反復ループ等で生成が伸びたときに、
# 全量バッファリングのまま時間切れになるのを防ぐための上限。
GENERATION_DEADLINE_S = 90
# Never return an empty body — that fails the question outright.
EMPTY_FALLBACK = "申し訳ありません。回答を生成できませんでした。もう一度お試しください。"

app = FastAPI()
engine: AsyncLLMEngine | None = None
processor = None  # AutoProcessor, loaded once at startup

pynvml.nvmlInit()
_gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# session_id -> list of chat messages (multi-turn state, stages 5 / 6-3)
sessions: "OrderedDict[str, list]" = OrderedDict()
_sessions_lock = asyncio.Lock()

# Most recently completed request, shown on the WebUI (STAGE 1-2).
last_timing = {"first_token_s": None, "total_s": None, "tokens": None,
               "session_id": None, "retrieval": None}


# --------------------------------------------------------------------------
# startup
# --------------------------------------------------------------------------
@app.on_event("startup")
async def load_model():
    global engine, processor
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        # weight-only FP8 quantization to keep VRAM headroom on a single L4 (24GB)
        quantization="fp8",
        # the grader sends up to 4 concurrent requests
        max_num_seqs=8,
        trust_remote_code=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    await _warmup()


async def _warmup():
    """Pay the one-time Triton/JIT compilation cost here, not on the grader's
    first request.

    Cold, the first text request costs ~13s to first token and the first image
    request another ~12s (the vision encoder compiles its own kernels); warm,
    both are ~0.05s. The port only starts listening once this returns, so any
    client that can connect is talking to a warm engine.
    """
    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(buf, format="PNG")
    dummy_b64 = base64.b64encode(buf.getvalue()).decode()

    params = SamplingParams(temperature=0.0, max_tokens=4)
    for i, (text, img) in enumerate([("こんにちは", None), ("これは何ですか", dummy_b64)]):
        prompt_input = _build_prompt(_build_messages(text, img))
        async for _ in engine.generate(prompt_input, params, f"warmup-{i}"):
            pass


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------
def _decode_image(img_b64: str) -> Image.Image:
    raw = base64.b64decode(img_b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _build_messages(prompt: str, img_b64: str | None) -> list:
    """One user message, with an image part when the request carries one."""
    if img_b64:
        return [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                "_image": _decode_image(img_b64),
            }
        ]
    return [{"role": "user", "content": prompt}]


def _build_prompt(messages: list) -> dict:
    """Render the chat template and collect any images in the conversation."""
    images = [m["_image"] for m in messages if "_image" in m]
    clean = [{k: v for k, v in m.items() if k != "_image"} for m in messages]

    text_prompt = processor.apply_chat_template(
        clean, tokenize=False, add_generation_prompt=True
    )
    if images:
        return {"prompt": text_prompt, "multi_modal_data": {"image": images}}
    return {"prompt": text_prompt}


# --------------------------------------------------------------------------
# auth for the human-facing routes
# --------------------------------------------------------------------------
_basic = HTTPBasic(auto_error=False)
COOKIE_NAME = "rag_session"
_valid_tokens: set[str] = set()


def _password_ok(password: str) -> bool:
    return bool(WEBUI_PASSWORD) and secrets.compare_digest(password, WEBUI_PASSWORD)


def _authed(request: Request, creds: HTTPBasicCredentials | None) -> bool:
    """Either a login-cookie (browser) or HTTP Basic (curl / scripts)."""
    token = request.cookies.get(COOKIE_NAME)
    if token and token in _valid_tokens:
        return True
    return (
        creds is not None
        and secrets.compare_digest(creds.username, WEBUI_USER)
        and _password_ok(creds.password)
    )


def require_login(request: Request, creds: HTTPBasicCredentials | None = Depends(_basic)):
    if not WEBUI_PASSWORD:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "WEBUI_PASSWORD is not configured on the server",
        )
    if not _authed(request, creds):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unauthorized")
    return True


# --------------------------------------------------------------------------
# request parsing — deliberately lenient, must never produce a 4xx
# --------------------------------------------------------------------------
async def _parse_request(request: Request) -> tuple[str, str, str | None]:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    session_id = body.get("session_id") or f"anon-{uuid.uuid4().hex[:8]}"
    # PIJ対策の入力正規化。ゼロ幅文字による難読化を潰し、chat template の
    # 特殊トークン(<|im_start|> 等)を剥がしてから先へ渡す。`apply_chat_template`
    # は平文をそのまま埋め込むため、剥がさないとユーザー入力から偽の system
    # ターンを合成できてしまう(役割偽装のうち唯一、本当に役割境界を越えられる
    # 経路)。長さ上限は MAX_MODEL_LEN 超過で engine.generate が例外になり
    # SSEが途中で切れるのも同時に防ぐ。
    prompt = guard.normalize_input(body.get("prompt"))

    img = body.get("img")
    if not isinstance(img, str) or not img.strip():
        img = None

    return str(session_id), prompt, img


# --------------------------------------------------------------------------
# retrieval (STAGE 02 / 03)
# --------------------------------------------------------------------------
# STAGE04 採点デバッグ用。挙動は一切変えず、1リクエストのルーティング判定と
# 最終回答を1行 JSON で追記するだけ。書き込み失敗は握り潰す(生成を止めない)。
_TRACE_PATH = os.environ.get(
    "STAGE4_TRACE_PATH",
    os.path.join(_REPO_ROOT, "var", "stage4_trace.jsonl"),
)
_TRACE_ENABLED = os.environ.get("STAGE4_TRACE", "1") != "0"


def _write_trace(session_id: str, prompt: str, routed, final_text: str) -> None:
    if not (_TRACE_ENABLED and routed is not None):
        return
    try:
        rec = router.build_trace_record(prompt, routed)
        rec["session_id"] = session_id
        rec["final_response"] = final_text
        os.makedirs(os.path.dirname(_TRACE_PATH), exist_ok=True)
        with open(_TRACE_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 — trace はベストエフォート
        print(f"[trace] write failed: {exc!r}", flush=True)


async def _route_safely(prompt: str, img: str | None):
    """検索パスを呼ぶ。失敗しても生成は止めない。

    RAG側の例外で 5xx を返すとステージ全体が中断してしまう(冒頭の契約メモ参照)
    ため、ここだけは意図的に握り潰し、素の生成にフォールバックする。
    """
    try:
        return await router.handle_query(prompt, img, engine=engine, processor=processor)
    except Exception as exc:  # noqa: BLE001 — 生成の継続を最優先する
        print(f"[rag] handle_query failed, falling back to plain generation: {exc!r}",
              flush=True)
        return None


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------
async def _stream_chat(
    session_id: str,
    prompt: str,
    img: str | None,
    with_sources: bool = False,
    client: str = "",
):
    request_id = f"{session_id}-{uuid.uuid4().hex[:8]}"
    sampling_params = SamplingParams(temperature=0.3, max_tokens=MAX_TOKENS)

    # 会話履歴のキーは session_id 単体ではなく送信元IPとの組にする。
    # session_id はリクエスト本文でクライアントが自由に指定できるので、単体を
    # キーにすると、他人の session_id を当てるだけでその会話履歴が次の応答の
    # 文脈として読み出せてしまう(PIJ「情報抽出」)。採点サーバーは単一IPから
    # 来るため、仕様どおりの複数ターン継続はこの変更の影響を受けない。
    session_key = f"{client}|{session_id}"

    turn = _build_messages(prompt, img)
    async with _sessions_lock:
        history = sessions.get(session_key, [])
        messages = history + turn

    # 検索結果はシステムメッセージとして毎ターン先頭に差し込む。会話履歴には
    # 残さない(ターンごとに検索結果が変わるうえ、積み重なると文脈を圧迫するため)。
    routed = await _route_safely(prompt, img)

    # 確定回答(4-3 確度ゲート等): LLM に言い換え・脚色・補足をさせず、
    # そのまま1フレームで返す。engine.generate は通さない。
    if routed is not None and routed.canned_response:
        text = routed.canned_response
        yield f"data: {json.dumps({'delta': text}, ensure_ascii=False)}\n\n"
        async with _sessions_lock:
            sessions[session_key] = messages + [
                {"role": "assistant", "content": text}
            ]
            sessions.move_to_end(session_key)
            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
        last_timing.update(
            first_token_s=0.0, total_s=0.0, tokens=0, session_id=session_id,
            retrieval={
                "item": routed.detected_item,
                "canned": True,
                # PIJ検知で止めた場合は分類名を残す。採点中に /vram を見て
                # 「今どの攻撃が何件来ているか」を追えるようにするため。
                "attack": routed.attack_detected,
            },
        )
        if with_sources:
            payload = {"sources": router.describe_sources(routed)}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        _write_trace(session_id, prompt, routed, text)
        yield "data: [DONE]\n\n"
        return

    if routed is not None:
        prompt_input = _build_prompt(
            [{"role": "system", "content": router.build_system_prompt(routed)}] + messages
        )
    else:
        prompt_input = _build_prompt(messages)

    start = time.perf_counter()
    first_token_at = None
    prev_text = ""
    n_tokens = 0
    deadline_hit = False

    # PIJ対策: 生成が終わるまで1文字も送出しない(全量バッファリング)。
    #
    # SSE は一度 yield した delta を取り消せず、採点側は delta を連結した
    # 文字列をそのまま回答として読む。逐次送出のままでは、システムプロンプト
    # の漏洩や反復ループを検知できても「検知した時点で既に本文に混ざっている」
    # ため、出力側の検査が意味を持たない。全部生成してから `guard.scrub_output`
    # を通すことで、送出前に必ず1回検査されることを構造的に保証する。
    #
    # 代償は「最初の1文字がクライアントに届くまでの時間」で、生成そのものの
    # 所要時間は変わらない(warm状態で765トークン約30秒、1ターン120秒の制限に
    # 対して余裕がある)。ただし逐次送出と違い、遅い生成では最後まで何も届か
    # ないので、`GENERATION_DEADLINE_S` で必ず制限内に打ち切る。
    # `first_token_s` は引き続きモデルの初トークン時刻を測る。
    async for request_output in engine.generate(prompt_input, sampling_params, request_id):
        text = request_output.outputs[0].text
        if len(text) > len(prev_text):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            n_tokens += 1
        prev_text = text
        if time.perf_counter() - start > GENERATION_DEADLINE_S:
            # ここで抜けると async generator が閉じられ、vLLM 側で当該
            # リクエストが中断される。ここまでに生成された分を回答として使う
            # (全量バッファリングなので、打ち切っても本文は必ず1回検査される)。
            deadline_hit = True
            break

    prev_text, blocked = guard.scrub_output(prev_text)

    # An empty body fails the question outright, so always say something.
    if not prev_text.strip():
        prev_text = EMPTY_FALLBACK
    yield f"data: {json.dumps({'delta': prev_text}, ensure_ascii=False)}\n\n"

    # 引用はLLMに書かせず、検索結果から確定したURLをここで本文に足す
    # (STAGE02 2-2「根拠を参照ページまで示せる」。採点側は delta を連結した
    #  文字列だけを見るため、WebUI表示のみに留めず本文に含める)。
    # 出力ゲートが発火した回答には付けない — 本文を固定文に差し替えている
    # ので、退けた内容の参照だけが残るのは誤解を招く(4-3 の canned と同じ判断)。
    citations = (
        router.format_citations(routed) if routed is not None and not blocked else ""
    )
    if citations:
        prev_text += citations
        yield f"data: {json.dumps({'delta': citations}, ensure_ascii=False)}\n\n"

    # 根拠の詳細(どのレコードを、どの検索段で引いたか)は WebUI だけに流す。
    # 採点サーバーは delta を連結した文字列で採点するため、/chat のストリームに
    # 余計なイベントを混ぜない(仕様は data: {"delta": ...} のみ)。
    if with_sources and routed is not None:
        payload = {"sources": router.describe_sources(routed)}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    end = time.perf_counter()
    last_timing.update(
        first_token_s=round((first_token_at or end) - start, 3),
        total_s=round(end - start, 3),
        tokens=n_tokens,
        session_id=session_id,
        retrieval=None if routed is None else {
            "item": routed.detected_item,
            "output_blocked": blocked,
            "deadline_hit": deadline_hit,
            "ward": routed.detected_ward,
            "town": routed.detected_town,
            "item_hits": len(routed.item_hits),
            "schedule_hits": len(routed.schedule_hits),
            "missing": routed.missing,
            "sources": routed.sources,
        },
    )

    # Persist the turn so the next request with this session_id continues it.
    async with _sessions_lock:
        sessions[session_key] = messages + [{"role": "assistant", "content": prev_text}]
        sessions.move_to_end(session_key)
        while len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)

    _write_trace(session_id, prompt, routed, prev_text)
    yield "data: [DONE]\n\n"


def _sse(
    session_id: str,
    prompt: str,
    img: str | None,
    with_sources: bool = False,
    client: str = "",
) -> StreamingResponse:
    return StreamingResponse(
        _stream_chat(session_id, prompt, img, with_sources, client),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat")
async def chat(request: Request):
    """Grading endpoint. No auth (per spec), restricted by source IP."""
    client = request.client.host if request.client else ""
    if client not in GRADER_IPS:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "forbidden")
    session_id, prompt, img = await _parse_request(request)
    return _sse(session_id, prompt, img, client=client)


@app.post("/ui/chat")
async def ui_chat(request: Request, _=Depends(require_login)):
    """Same generation path as /chat, for the password-protected WebUI.

    WebUI にだけ、回答の根拠(引いたレコードと検索経路)を追加のSSEイベントで
    返す。採点用の /chat は仕様どおり delta のみ。
    """
    session_id, prompt, img = await _parse_request(request)
    client = request.client.host if request.client else ""
    return _sse(session_id, prompt, img, with_sources=True, client=client)


# --------------------------------------------------------------------------
# monitoring (STAGE 1-2 / 1-3)
# --------------------------------------------------------------------------
@app.get("/vram")
async def vram(_=Depends(require_login)):
    info = pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(_gpu_handle)
    return {
        "used_mb": info.used // 1024 // 1024,
        "total_mb": info.total // 1024 // 1024,
        "gpu_util_pct": util.gpu,
        "last_timing": last_timing,
        "active_sessions": len(sessions),
    }


@app.get("/info")
async def info(_=Depends(require_login)):
    return {
        "model": MODEL_NAME,
        "quantization": "FP8 (online, per-tensor)",
        "gpu": pynvml.nvmlDeviceGetName(_gpu_handle),
        "max_model_len": MAX_MODEL_LEN,
        "max_tokens": MAX_TOKENS,
    }


# --------------------------------------------------------------------------
# knowledge browser (STAGE 02 2-3)
#   「検索対象の外部データファイル一覧と内容をWebUIで確認できる」
# --------------------------------------------------------------------------
@app.get("/knowledge")
async def knowledge_list(_=Depends(require_login)):
    """取り込み済みの参照元ページ一覧。DB未作成なら available=false を返す。"""
    data = await asyncio.to_thread(knowledge.list_sources)
    data["using_stubs"] = router.using_stubs()
    return data


@app.get("/knowledge/{source_id}")
async def knowledge_detail(source_id: str, _=Depends(require_login)):
    """1ページ分の取り込み内容(表形式)。"""
    doc = await asyncio.to_thread(knowledge.get_source, source_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown source id")
    return doc


LOGIN_PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ログイン — ごみ分別チャットボット</title><style>
:root{--bg:#f4f6f8;--panel:#fff;--border:#dfe4ea;--text:#1a2027;--muted:#6b7684;--accent:#1a56b8;--err:#c0392b}
@media(prefers-color-scheme:dark){:root{--bg:#14181d;--panel:#1c2127;--border:#2b323b;--text:#e4e8ed;--muted:#8b95a3;--accent:#5a9bf5;--err:#e05c4b}}
*{box-sizing:border-box}
body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:var(--bg);
color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans JP",sans-serif}
form{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:32px;width:340px}
h1{margin:0 0 6px;font-size:17px}p{margin:0 0 22px;color:var(--muted);font-size:13px}
input{width:100%;padding:11px 13px;border:1px solid var(--border);border-radius:9px;font-size:14px;
background:var(--bg);color:var(--text)}
input:focus{outline:none;border-color:var(--accent)}
button{width:100%;margin-top:14px;padding:11px;border:none;border-radius:9px;background:var(--accent);
color:#fff;font-size:14px;font-weight:600;cursor:pointer}button:hover{opacity:.9}
.err{color:var(--err);font-size:13px;margin-top:12px;display:none}
</style></head><body>
<form method="post" action="/login">
  <h1>北九州市 ごみ分別チャットボット</h1>
  <p>閲覧にはパスワードが必要です。</p>
  <input type="password" name="password" placeholder="パスワード" autofocus required>
  <button type="submit">ログイン</button>
  <div class="err" id="err">パスワードが正しくありません。</div>
</form>
<script>
if (new URLSearchParams(location.search).has('e')) document.getElementById('err').style.display='block';
</script>
</body></html>"""


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return LOGIN_PAGE


# Brute-force protection lives here rather than in the firewall: a connection
# limit on the port would also throttle ordinary WebUI use (each chat message
# opens its own SSE connection).
LOGIN_MAX_FAILURES = 8
LOGIN_WINDOW_S = 300
_login_failures: "OrderedDict[str, list[float]]" = OrderedDict()


def _login_blocked(ip: str) -> bool:
    now = time.time()
    hits = [t for t in _login_failures.get(ip, []) if now - t < LOGIN_WINDOW_S]
    _login_failures[ip] = hits
    return len(hits) >= LOGIN_MAX_FAILURES


def _record_login_failure(ip: str):
    _login_failures.setdefault(ip, []).append(time.time())
    _login_failures.move_to_end(ip)
    while len(_login_failures) > 1024:
        _login_failures.popitem(last=False)


@app.post("/login")
async def login_submit(request: Request):
    ip = request.client.host if request.client else "?"
    if _login_blocked(ip):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "ログイン試行が多すぎます。しばらく待ってから再度お試しください。",
        )

    form = await request.form()
    if not _password_ok(str(form.get("password", ""))):
        _record_login_failure(ip)
        return RedirectResponse("/login?e=1", status_code=status.HTTP_303_SEE_OTHER)

    token = secrets.token_urlsafe(32)
    _valid_tokens.add(token)
    resp = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", max_age=86400)
    return resp


@app.get("/", response_class=HTMLResponse)
async def webui(request: Request, creds: HTTPBasicCredentials | None = Depends(_basic)):
    # Browsers get redirected to the login form rather than a bare 401.
    if not _authed(request, creds):
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
    return FileResponse(WEBUI_PATH)
