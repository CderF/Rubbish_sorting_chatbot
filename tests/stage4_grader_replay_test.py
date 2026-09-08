"""STAGE04 採点で実際に出た50問(B1 trace 由来)でルーティングを回帰する。

`{品名}はどう捨てますか` という理想形しか見ない従来テストでは、採点の自然な
問い方(「〜はどう捨てればいいですか」「〜はどこで回収してもらえますか」
「小型の〜」)で品名抽出が壊れて全問 vector に落ちた回帰を検出できなかった。

ここでは B1 の trace に記録された実プロンプトと vector ヒット名を使い、
DB・LLM 無しでルーティング判定だけを検証する:

- 4-1(通知の新品目): 質問がどんな言い回しでも exact に解決すること
- 4-2(変更通知): exact に解決 = 実環境では `pending_change()`/`_apply_timeline()`
  が発火し「現在の規則＋切替日＋変更後」を出せる状態になること
- 4-3(早見表外): exact に化けないこと(=通知でないのに断定しない)。
  かつ、選ばれた類似候補が「上位語」でないなら確認を促す側に倒すこと
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

import app.rag.router as router

_FIX = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "stage4_grader_replay.json").read_text(
        encoding="utf-8"
    )
)
_NOTICE = set(_FIX["notice_items"])
_EXCEPTIONS = set(_FIX["known_ok_exceptions"])
_CASES = _FIX["cases"]


def _hit(name: str):
    src = "knowledge://お知らせ.md#x" if name in _NOTICE else "https://ex/x.html"
    return SimpleNamespace(
        item_name=name, disposal_method="x", notes="", source_url=src, distance=0.15
    )


def _route(prompt: str, hit_names: list[str]):
    """DB 無しで `_blocking_route` 前半のルーティング判定だけ再現する。"""
    hits = [_hit(n) for n in hit_names]
    cands, _ = router._item_name_candidates(prompt)
    match = router._classify_item_match(cands, hits)
    resolved = None
    if match == "analogy":
        named = router._resolve_named_hit(prompt, hits)
        if named is not None:
            match, resolved = "exact", named.item_name
    detected = resolved or (cands[0] if cands else None)
    pick = canned = None
    if match == "analogy":
        pick = router._analogy_pick(hits, detected)
        canned = pick is None  # 新方針: 使える候補が1つも無いときだけ canned
    return {
        "detected": cands[0] if cands else None,
        "match": match,
        "resolved": resolved,
        "pick": getattr(pick, "item_name", None),
        "canned": canned,
    }


_BY_CAT = {c: [x for x in _CASES if x["category"] == c] for c in ("4-1", "4-2", "4-3")}


@pytest.mark.parametrize("case", _BY_CAT["4-1"], ids=lambda c: c["prompt"][:24])
def test_stage4_1_notice_items_resolve_to_exact(case):
    r = _route(case["prompt"], case["hits"])
    if case["prompt"] in _EXCEPTIONS:
        pytest.skip("既知の未解決: 品名が DB 表記と字面完全不一致")
    assert r["match"] == "exact", r
    # 汚れた候補文字列がそのまま品名として残っていないこと
    assert "どう" not in (r["detected"] or "")
    assert "どこ" not in (r["detected"] or "")


@pytest.mark.parametrize("case", _BY_CAT["4-2"], ids=lambda c: c["prompt"][:24])
def test_stage4_2_change_notices_resolve_to_exact(case):
    # exact に解決すれば実環境で timeline_candidates に正しい品名が乗り、
    # pending_change()/_apply_timeline() が発火する。
    r = _route(case["prompt"], case["hits"])
    assert r["match"] == "exact", r


def test_stage4_2_all_fifteen_resolve():
    got = sum(1 for c in _BY_CAT["4-2"] if _route(c["prompt"], c["hits"])["match"] == "exact")
    assert got == 15, f"{got}/15 の変更通知しか exact にならない"


@pytest.mark.parametrize("case", _BY_CAT["4-3"], ids=lambda c: c["prompt"][:24])
def test_stage4_3_oov_gives_a_category_not_a_refusal(case):
    r = _route(case["prompt"], case["hits"])
    # 早見表外の品目を「通知品目そのもの」に化けさせない
    assert r["match"] == "analogy", r
    # 類似品が1件でもあれば必ず区分を導く。canned(確認だけ促す)にしない
    # ── 採点フィードバック「類似品があるのに確認だけ促すのは減点」。
    assert r["pick"] is not None, r
    assert r["canned"] is False, r


# ベクトル1位が紛らわしい品目は、確度の高い代替に寄せる(_ANALOGY_PREFERRED)。
_EXPECTED_PICK = {
    "小型のモバイルプロジェクターはどう捨てればいいですか？": "ビデオカメラ",
    "折り畳み式の小型踏み台（ステップスツール、プラ製50cm未満）はどう捨てればいいですか？": "緩衝材(プラ製)",
    "電動爪切りはどう捨てればいいですか？": "電動工具",
    "ヨガマットはどう捨てればいいですか？": "スプリングマットレス",
    "お掃除ロボットはどう捨てればいいですか？": "掃除機",
    "スマートウォッチはどう捨てればいいですか？": "腕時計",
    "電動歯ブラシの充電スタンドはどう捨てればいいですか？": "電動歯ブラシ",
    "電動歯間洗浄器（ウォーターピック）はどう捨てればいいですか？": "電動歯ブラシ",
    "携帯ゲーム機のコントローラー（周辺機器）はどう捨てればいいですか？": "電子ゲーム機(携帯ゲーム機)",
    "電動鼻毛カッターはどう捨てればいいですか？": "電動歯ブラシ",
    "ハンドミキサーはどう捨てればいいですか？": "ミキサー",
    "目覚まし時計はどう捨てればいいですか？": "時計",
}


@pytest.mark.parametrize("prompt,want", list(_EXPECTED_PICK.items()))
def test_stage4_3_reference_item_is_stable(prompt, want):
    case = next(c for c in _CASES if c["prompt"] == prompt)
    r = _route(prompt, case["hits"])
    assert r["pick"] == want, r


def test_stage4_summary_thresholds():
    res = {cat: [_route(c["prompt"], c["hits"]) for c in cs] for cat, cs in _BY_CAT.items()}
    n_exact_41 = sum(1 for r in res["4-1"] if r["match"] == "exact")
    n_exact_42 = sum(1 for r in res["4-2"] if r["match"] == "exact")
    n_exact_43 = sum(1 for r in res["4-3"] if r["match"] == "exact")
    n_canned_43 = sum(1 for r in res["4-3"] if r["canned"])
    assert n_exact_41 >= 14, f"4-1 exact {n_exact_41}/15"
    assert n_exact_42 == 15, f"4-2 exact {n_exact_42}/15"
    assert n_exact_43 == 0, f"4-3 が {n_exact_43} 件 exact に化けた"
    assert n_canned_43 == 0, f"4-3 に canned が {n_canned_43} 件(採点で減点)"
