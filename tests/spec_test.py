"""Conformance test against the grading server's documented /chat contract."""
import base64
import concurrent.futures
import io
import json
import os
import time
import urllib.request

# Point this at the /chat endpoint under test. From a developer machine, open an
# SSH tunnel first:  ssh -L 8080:localhost:8080 intern2026-B1-teamb2
URL = os.environ.get("CHAT_URL", "http://localhost:8080/chat")


def call(body: dict, timeout=120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status = r.status
            ctype = r.headers.get("content-type", "")
            # Parse the SSE stream exactly the way the grader says it does.
            answer = ""
            for raw in r:
                line = raw.decode("utf-8").rstrip("\n")
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                    for k in ("delta", "content", "text", "token", "answer",
                              "response", "message", "reply"):
                        if k in obj:
                            answer += obj[k]
                            break
                except json.JSONDecodeError:
                    answer += payload
            return status, ctype, answer, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("content-type", ""), e.read().decode()[:200], time.perf_counter() - t0


def show(name, status, ctype, answer, dt, expect_ok=True):
    sse = "text/event-stream" in ctype
    body_ok = bool(answer.strip())
    verdict = "PASS" if (status == 200 and sse and body_ok) else "FAIL"
    if not expect_ok:
        verdict = "PASS" if status == 200 and body_ok else "FAIL"
    print(f"[{verdict}] {name}")
    print(f"        status={status} sse={sse} {dt:.2f}s body={answer[:70]!r}")
    return verdict == "PASS"


results = []

# 1. img explicitly null (the grader always sends this key)
s, c, a, d = call({"session_id": "grade-t1", "prompt": "ペットボトルは何ごみ?一文で。", "img": None})
results.append(show("img: null is accepted", s, c, a, d))

# 2. malformed bodies must never yield 4xx (a 4xx body is recorded as the answer)
s, c, a, d = call({"session_id": "grade-t2"})  # no prompt
results.append(show("missing prompt -> not 4xx", s, c, a, d, expect_ok=False))

s, c, a, d = call({"prompt": "こんにちは"})  # no session_id
results.append(show("missing session_id -> not 4xx", s, c, a, d, expect_ok=False))

s, c, a, d = call({"session_id": "grade-t4", "prompt": "テスト", "img": 12345})  # wrong type
results.append(show("img wrong type -> not 4xx", s, c, a, d, expect_ok=False))

# 3. multi-turn continuity on one session_id (stages 5 / 6-3)
sid = "grade-multiturn"
s, c, a1, d = call({"session_id": sid, "prompt": "ペットボトルの分別を教えて。", "img": None})
print(f"        turn1: {a1[:60]!r}")
s, c, a2, d = call({"session_id": sid, "prompt": "それは何色の袋に入れますか?", "img": None})
carried = show("multi-turn keeps context", s, c, a2, d)
print(f"        turn2: {a2[:90]!r}")
results.append(carried)

# 4. a different session_id must NOT inherit the previous context
s, c, a3, d = call({"session_id": "grade-fresh", "prompt": "それは何色の袋に入れますか?", "img": None})
print(f"[INFO ] fresh session reply: {a3[:90]!r}")

# 5. image input (the test image is generated here so the suite is self-contained)
from PIL import Image, ImageDraw

_img = Image.new("RGB", (200, 300), color=(255, 255, 255))
_d = ImageDraw.Draw(_img)
_d.rectangle([50, 50, 150, 250], fill=(0, 150, 255))
_d.ellipse([70, 20, 130, 70], fill=(0, 120, 200))
_buf = io.BytesIO()
_img.save(_buf, format="PNG")
img = base64.b64encode(_buf.getvalue()).decode()
s, c, a, d = call({"session_id": "grade-img", "prompt": "この画像の物体の色と形は?", "img": img})
results.append(show("image input", s, c, a, d))

# 6. four concurrent requests (the grader's stated maximum)
def one(i):
    return call({"session_id": f"grade-conc-{i}",
                 "prompt": f"{i}番目の質問です。ごみ出しのマナーを一文で。", "img": None})

t0 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    conc = list(ex.map(one, range(4)))
wall = time.perf_counter() - t0
ok = all(s == 200 and a.strip() for s, c, a, d in conc)
print(f"[{'PASS' if ok else 'FAIL'}] 4 concurrent requests")
for i, (s, c, a, d) in enumerate(conc):
    print(f"        #{i} status={s} {d:.2f}s {a[:45]!r}")
print(f"        wall clock for all 4: {wall:.2f}s")
results.append(ok)

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
