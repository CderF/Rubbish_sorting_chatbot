"""
STAGE 01 - 1-1: /chat adapter in front of a vLLM server.

Model: Qwen/Qwen3-VL-8B-Instruct  (Apache 2.0, vision-language)
  -> chosen per team decision (see documents/STAGE1-0_LLM調査報告.md):
     only ONE server is actually usable for this team, and STAGE 3-2 needs
     image classification on that same server, so a single VL model handles
     both text QA (now) and image input (later) instead of two models.
  Fallback if this one has issues: Qwen/Qwen2.5-VL-7B-Instruct (also Apache 2.0)

Architecture
    grading server / browser
        --> POST /chat        (THIS FastAPI adapter, port 8000)
              --> POST /v1/chat/completions   (vLLM OpenAI API, port 8001)
                    --> Qwen3-VL-8B-Instruct on the L4 GPU

Why an adapter?
    vLLM speaks the OpenAI API format. The assignment requires a specific
    format:  POST /chat  {"session_id","prompt","img"}  ->  data: {"delta": "..."}
    This adapter only translates between the two formats. Because the model
    is vision-capable, "img" (base64) is already wired through here, even
    though STAGE 1-1 itself only requires text.

--- how to run (two terminals on the L4 server) ---

0) needs vLLM >= 0.11.0 for Qwen3-VL support:
     pip install "vllm>=0.11.0" httpx fastapi uvicorn

1) start vLLM (does the heavy GPU work, item 1-2):
     vllm serve Qwen/Qwen3-VL-8B-Instruct \
        --port 8001 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --dtype bfloat16

   --max-model-len 8192  : kept small for now (text-only STAGE 1-1 testing).
       VL models spend a lot of tokens encoding an image, so revisit this
       number upward when STAGE 3-2 (image input) is implemented, and
       re-check it still fits 24GB together with the model weights + KV cache.
   --gpu-memory-utilization 0.9 : let vLLM use up to 90% of the L4's VRAM

2) start this adapter:
     uvicorn server:app --host 0.0.0.0 --port 8000

The grading server then calls:  POST http://<your-team-ip>:8000/chat
"""

import json
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

VLLM_URL = "http://localhost:8001/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

app = FastAPI()


def build_messages(prompt: str, img_b64: str | None):
    """Build an OpenAI-format 'messages' list. If an image is supplied,
    use multimodal content blocks (this is what the VL model expects)."""
    if not img_b64:
        return [{"role": "user", "content": prompt}]

    # Multimodal message: text + image together.
    # img_b64 is expected to be a raw base64 string (no data: prefix) per
    # the assignment's request format ("img": base64); we add the prefix
    # vLLM/OpenAI-style image_url expects.
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ],
        }
    ]


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    img_b64 = body.get("img")  # None for STAGE 1-1's text-only questions
    # session_id is received but not used yet (multi-turn state comes in STAGE 5).

    payload = {
        "model": MODEL_NAME,
        "messages": build_messages(prompt, img_b64),
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0.0,   # deterministic -> more stable for auto-grading
    }

    async def event_stream():
        # --- speed measurement (helps item 1-3) ---
        t_start = time.time()
        t_first = None
        n = 0

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", VLLM_URL, json=payload) as resp:
                async for line in resp.aiter_lines():
                    # vLLM sends lines like:  data: {...}
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
                        break

                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if not delta:
                        continue

                    if t_first is None:
                        t_first = time.time()   # time to first token (TTFT)
                    n += 1

                    # RE-EMIT in the format the assignment requires
                    yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

        t_end = time.time()
        if t_first:
            print(
                f"[speed] TTFT={t_first - t_start:.2f}s  "
                f"total={t_end - t_start:.2f}s  "
                f"chunks={n}  "
                f"speed={n / (t_end - t_first + 1e-9):.1f}/s"
            )

    # media_type MUST be text/event-stream, or the grader can't judge (required!)
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}
