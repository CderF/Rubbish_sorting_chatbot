"""STAGE06 安全ゲート(要件6-1〜6-3)の検出ロジック。

`documents/stage06_proposal.md` §7 の分担に沿った足場モジュール。検出そのものは
2名で並行して埋める:

* `check_entry_gate` … 担当A(FANG / Teamb2)。6-1(a) 担当範囲外 / 6-2 危険処分誘導。
  検索処理の前に評価し、発火したら固定回答で即返す(§2.1 / §2.2 / §4)。
* `no_result_reply` … 担当B(周 / Teamb4)。6-3 完全空振り時の固定文言(§2.3)。

依存を持たない葉モジュールにしてある(`router.py` / `server.py` の双方から
参照されるため、循環 import を避ける)。プロンプト文字列と、`extract_region()`
が既に返した区名・町名だけを入力に取る — この中で DB・LLM・ネットワークには
触れない(`app/rag/session_state.py` と同じ方針で、全分岐をオフラインで
単体テストできるようにする)。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyVerdict:
    """安全ゲートが発火したときの結果。

    `gate` は `RoutedAnswer.safety_gate` にそのまま載る識別子("6-1a" / "6-2" /
    "6-3")。`reply` は `engine.generate` を通さずそのまま返す固定回答。
    """

    gate: str
    reply: str


def check_entry_gate(
    prompt: str, ward: "str | None", town: "str | None"
) -> "SafetyVerdict | None":
    """検索処理より前に評価する入口ゲート(6-1a / 6-2)。担当A が実装する。

    `prompt` は生の発話。`ward` / `town` は `router.extract_region(prompt)` の
    結果 — 6-1(a) の他都市名ブラックリストは、北九州市内の区名/町名が確定して
    いれば誤検出を上書き解除する必要があるため受け取る(§2.1 の注意点)。

    戻り値:
      * `SafetyVerdict` … ゲート発火。呼び出し側は `canned_response` /
        `safety_gate` を立てて即返す。
      * `None` … 未発火。通常の検索フローへ進む。

    現状は足場のため常に `None`(＝挙動変化なし)。
    """
    return None


def no_result_reply() -> str:
    """6-3: どのゲートにも該当せず、品目・収集日のどちらもヒットしなかった
    ときの固定文言。担当B が実装する(窓口番号は §5 未決事項で確定)。

    現状は足場のため空文字(＝呼び出し側は従来どおり LLM 生成に回す)。
    """
    return ""
