"""画像分類パス(issue #7) — 品物の画像から早見表パスの検索キーとなる品目名を抽出する。

`app/server.py` が起動時に初期化する共有vLLMエンジンをそのまま受け取る設計で、
独自にモデルをロードしない。L4 1枚のVRAMをserver.pyの1モデルでほぼ使い切る
構成のため(`gpu_memory_utilization=0.85`)、2つ目のモデルインスタンスを持つ
余地がない(issue #7 Implementation Decisions参照)。

`vllm` はこのモジュールのimport時ではなく `classify_image_to_item_name` の
呼び出し時に遅延importする。`vllm` はLinux/CUDA向けのwheelしか配布されて
おらず、開発機(macOS)にはインストールできないため、モジュール読み込み自体は
`vllm` 無しでも通るようにし、`parse_classification_output`(純粋関数)だけは
どの環境でもオフラインテストできるようにするための構成(issue #7 Testing
Decisions参照)。`classify_image_to_item_name` 本体の実機検証はL4サーバー上の
`tests/spec_test.py` に委ねる。
"""

from __future__ import annotations

import base64
import uuid
from io import BytesIO

from PIL import Image

# モデルへの指示文。品目名だけを1行で返させ、判別できない場合のsentinelを
# 明示することで、パース側(parse_classification_output)が機械的に判定できる
# ようにする。
_INSTRUCTION = (
    "この画像に写っているのは、家庭で不要になった1つの物品です。\n"
    "その物品の一般的な名称だけを、日本語で1行で答えてください。\n"
    "余計な説明・敬語・句読点は付けないでください。\n"
    "物品が何か判別できない場合は「不明」とだけ答えてください。"
)

# モデルがsentinel通り「不明」と答えなかった場合の保険。表記ゆれを吸収する。
_SENTINEL_MARKERS = ("不明", "わかりません", "分かりません", "判別できません", "特定できません", "判断できません")

_MAX_TOKENS = 32


def _decode_image(image_base64: str) -> Image.Image:
    raw = base64.b64decode(image_base64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _build_prompt_input(processor, image: Image.Image) -> dict:
    """app/server.py の `_build_messages`/`_build_prompt` と同型のプロンプト構築。
    分類専用のインストラクションを使う点のみが異なる。"""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": _INSTRUCTION}],
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": text_prompt, "multi_modal_data": {"image": [image]}}


def parse_classification_output(raw_text: str) -> str | None:
    """VLMの生出力 → 品目名文字列、または特定できなかった場合はNone。

    純粋関数(GPU不要)。以下のいずれかでNoneを返す:
    - 空文字列・空白のみ
    - sentinel文字列(「不明」等)を含む
    """
    text = (raw_text or "").strip()
    if not text:
        return None

    # 複数行・文末の句読点が付くことがあるため、先頭の1行・1文だけを見る。
    first_line = text.splitlines()[0].strip()
    first_line = first_line.rstrip("。.!！?？ 　")
    if not first_line:
        return None
    if any(marker in first_line for marker in _SENTINEL_MARKERS):
        return None
    return first_line


async def classify_image_to_item_name(image_base64: str, engine, processor) -> str | None:
    """画像から品目名相当の文字列を抽出する。判別できない・デコード失敗の
    いずれの場合もNoneを返し、例外は投げない。

    早見表パスの検索自体はここでは行わない。戻り値をそのまま
    `lookup_item_exact` → `lookup_item_partial` → `vector_search_items` の順に
    渡すのは呼び出し側(router.py)の責務(documents/CONTEXT.md「画像分類パス」
    の定義通り)。
    """
    try:
        image = _decode_image(image_base64)
    except Exception:
        return None

    from vllm import SamplingParams  # 遅延import。モジュール docstring 参照

    prompt_input = _build_prompt_input(processor, image)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS)
    request_id = f"classify-{uuid.uuid4().hex[:8]}"

    final_text = ""
    async for request_output in engine.generate(prompt_input, sampling_params, request_id):
        final_text = request_output.outputs[0].text

    return parse_classification_output(final_text)
