"""ルール抽出が空振りしたターンだけをLLMで補うスロット抽出(STAGE05)。

`router.extract_turn()` の文字列ヒューリスティクスは、早見表に載っている品名を
そのまま含む発話には強いが、次のような言い方では何も取れない。

  - 「これって何ゴミ？」のように品名が指示語に置き換わっている
  - 「この前買ったやつのでっかい箱、どうすればいい」のような口語で、
    `is_plausible_item_name` の下限に引っかかる断片しか残らない
  - 「引っ越しで出た古い机とイス」のように品名が助詞で連なっていて、
    最長断片が品名として使えない

そこで**ルールが1つもスロットを取れなかったターンに限って**、共有vLLMエンジンに
同じ発話を投げ、JSONで品名・区・町名を出させる。ルールが1つでも取れていれば
LLMは呼ばない。決定的に動く経路を先に置き、確率的な経路を後ろに置くことで、
既存の挙動は一切変えずに取りこぼしだけを拾う。

出力は「JSONで返してください」と頼むのではなく、vLLM の structured outputs
(制約付きデコード)でスキーマに合うトークンしか選べないようにする。バックエンド
(xgrammar)がスキーマを状態機械に落とし、違反するトークンの logit を落とすため、
`json.loads` が失敗する余地が構造的に無い。区名は `enum` で実在の7区に縛るので、
存在しない区が返ることもない。

構成は `image_classify.py` に合わせてある。`vllm` は Linux/CUDA 向けの wheel しか
無く開発機には入らないため、モジュール読み込み時ではなく呼び出し時に遅延import
し、パース側(`parse_slot_output`)は純粋関数としてGPU無しでも動くようにする。
`router.py` を import しないのは、`router` 側からこのモジュールを呼ぶため
(循環参照になる)。区名リストはスキーマを組むときに引数で受け取る。
"""

from __future__ import annotations

import json
import uuid

# 生成させるのは短いJSON1個だけ。制約付きデコードなので閉じ括弧まで必ず出る。
_MAX_TOKENS = 128

_INSTRUCTION = """あなたは北九州市のごみ分別相談の受付です。
利用者の発言から、次の3つだけを読み取ってJSONで出力してください。

- item: 捨てたい品物の一般的な名称。1つだけ。発言に品物が出てこない、または
  「これ」「それ」のような指示語しか無い場合は null。
- ward: 北九州市の区名。発言に出てこなければ null。
- town: 区より下の町名・地区名。発言に出てこなければ null。

推測で埋めないでください。発言に書かれていないものは null にします。
品名は「大きいソファ」ではなく「ソファ」のように、修飾語を落とした
一般的な名称にしてください。"""


def slot_schema(wards: "tuple[str, ...]") -> dict:
    """制約付きデコードに渡すJSONスキーマを組む。

    `wards` は `router.WARDS`。区名を `enum` に畳んでおくと、モデルは実在する
    7区か null しか選べなくなる。ここを自由文字列にすると「北九州市」や
    「小倉」のような、`extract_region` の下流が扱えない値が入りうる。
    """
    return {
        "type": "object",
        "properties": {
            "item": {"type": ["string", "null"]},
            "ward": {"enum": [*wards, None]},
            "town": {"type": ["string", "null"]},
        },
        "required": ["item", "ward", "town"],
        "additionalProperties": False,
    }


def needs_fallback(turn) -> bool:
    """このターンでLLM抽出を走らせるべきか。

    ルールが品名・区・町名のどれか1つでも取れていれば False。決定的な抽出が
    効いているターンに確率的な経路を混ぜない、という切り分けをここ1箇所に
    閉じ込める。

    画像から品名が来ているターン(`from_image`)も対象外。画像分類が既に
    エンジンを1回使っており、そこで取れた品名のほうが確度が高い。
    """
    if turn.from_image:
        return False
    if turn.item or turn.ward or turn.town:
        return False
    # 「はい」「ありがとう」のような相槌でエンジンを回さないための下限。
    # 指示語だけの質問(「これって何ゴミ？」)は8文字あるので通る。
    return len(turn.text.strip()) >= 6


def parse_slot_output(raw_text: str) -> dict | None:
    """モデルの生出力 → {"item":…, "ward":…, "town":…}、読めなければ None。

    純粋関数(GPU不要)。制約付きデコードが効いていれば必ずスキーマ通りのJSONが
    来るが、`max_tokens` 打ち切りや将来のバックエンド差し替えで壊れた文字列が
    来る可能性は残るので、ここで防ぐ。読めなければ None を返し、呼び出し側は
    ルール抽出の結果(=何も無し)のまま進む。
    """
    text = (raw_text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    out = {}
    for key in ("item", "ward", "town"):
        value = data.get(key)
        if isinstance(value, str):
            value = value.strip()
            out[key] = value or None
        else:
            # 数値・配列・真偽値が来たら採用しない(null と同じ扱い)。
            out[key] = None
    return out


def _build_prompt_input(processor, text: str) -> dict:
    """`app/server.py` の `_build_messages`/`_build_prompt` と同型のプロンプト構築。

    抽出専用のインストラクションを使う点と、画像を載せない点だけが異なる。
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": text_prompt}


def sampling_params(wards: "tuple[str, ...]"):
    """スキーマ制約付きの `SamplingParams` を作る(`vllm` を遅延import)。

    vLLM 0.28 では制約付きデコードの入口が `SamplingParams.structured_outputs`
    (`StructuredOutputsParams`)。0.11 以前の `guided_decoding` /
    `GuidedDecodingParams` から改名されたもので、requirements.txt が
    `vllm==0.28.0` を固定しているため新しい名前だけを見る。
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=_MAX_TOKENS,
        structured_outputs=StructuredOutputsParams(json=slot_schema(wards)),
    )


async def extract_slots(text: str, wards: "tuple[str, ...]", engine, processor) -> dict | None:
    """発話1つをLLMに投げ、スロットのJSONを受け取る。失敗時は None。

    `engine`/`processor` は `server.py` が起動時に初期化する共有vLLMエンジン。
    `image_classify` と同じく、このモジュールがモデルを持つことはしない
    (L4 1枚を1モデルでほぼ使い切っているため)。

    例外は握り潰して None にする。抽出はあくまで補助であり、ここで落ちると
    ルールだけなら答えられたはずのターンまで 500 になるため。
    """
    try:
        params = sampling_params(wards)
        prompt_input = _build_prompt_input(processor, text)
        request_id = f"slots-{uuid.uuid4().hex[:8]}"

        final = None
        async for request_output in engine.generate(prompt_input, params, request_id):
            final = request_output
        if final is None or not final.outputs:
            return None
        return parse_slot_output(final.outputs[0].text)
    except Exception:
        return None


async def warmup(wards: "tuple[str, ...]", engine, processor) -> None:
    """スキーマの文法コンパイルを起動時に先払いする。

    制約付きデコードは、スキーマを受け取った最初の1回でバックエンド(xgrammar)が
    文法を状態機械にコンパイルする。`server.py` の `_warmup()` が Triton/JIT の
    コンパイルを先に済ませているのと同じ理由で、これも起動時に通しておく —
    そうしないと、ルール抽出が空振りした最初の1問だけが遅くなる。
    """
    await extract_slots("これって何ゴミですか", wards, engine, processor)
