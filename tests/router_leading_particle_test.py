"""`app/rag/router._trim_particles` / `_QUESTION_TAIL` の回帰テスト。

`_LEADING`(文頭の助詞除去)は文字クラスの貪欲マッチなので、「やかん」
「ねじ回し」「かばん」「のこぎり」のように内容語自体が助詞と同じ仮名
(や・ね・か・の等)で始まる品名の頭まで削ってしまっていた(実測:
「やかんを処分したい」→ 候補「んを処分したい」、「ねじ回しはどう捨てますか」
→ 検出品名「じ回し」、「小倉北区でかばんを捨てたい」→「ばんを捨てたい」)。

`_trim_particles` は断片の「位置」で剥がし方を変える:
  - 発話の真の冒頭 / 区切り記号の直後  → 内容語の頭。閉じたつなぎ言葉だけ剥がす。
  - 接缝(区名・町名・ごみ種別語を除去した直後) → つなぎの助詞1文字だけ剥がす。
  - どちらでもない(語の途中で切れた残骸) → 従来どおり `_LEADING` を貪欲に。

区名を含むプロンプトは `lookup_towns` を差し替えてオフラインで検証する
(2026-09-14 レビュー #48 指摘: 実アプリで最も多い「品名+地区」の形が
未 cover だった)。
"""

from __future__ import annotations

import pytest

from app.rag import router


@pytest.fixture
def offline_region(monkeypatch):
    """`extract_region` の町名解決(データ層)を固定リストに差し替える。"""
    towns = ("浅野1丁目", "浅野2丁目", "城内")
    monkeypatch.setattr(router, "lookup_towns", lambda ward: list(towns))
    router._town_candidates.cache_clear()
    yield
    router._town_candidates.cache_clear()


# --- 地区名なし(素の文頭処理) -------------------------------------------
def test_item_starting_with_a_particle_kana_is_not_mangled():
    # 「や」「ね」「か」等は _PARTICLES に含まれる仮名なので、素朴な文頭除去
    # だと内容語の1文字目・2文字目まで削られていた。主候補がそのまま
    # 「やかん」になること(以前は文全体が返り、`"やかん" in ...` が空振り
    # で通っていた ― レビュー #48-3)。
    assert router.extract_item_name("やかんを処分したい") == "やかん"


def test_leading_particle_is_stripped_from_the_start_of_the_utterance():
    assert router.extract_item_name("ねじ回しはどう捨てますか") == "ねじ回し"


def test_real_leading_filler_at_the_start_is_still_stripped():
    """レビュー #48-2 回帰: 文頭の本物のつなぎ言葉は従来どおり落とす。
    `_written_from_the_start` で `text` 全体を保護してしまい、main では
    「傘」に解決できていた質問が「でも傘」に化けていた。"""
    assert router.extract_item_name("でも傘はどう捨てますか") == "傘"
    assert router.extract_item_name("で、ソファを捨てたい") == "ソファ"


def test_desire_form_question_tail_is_trimmed():
    """「〜を捨てたい」「〜を処分したい」も語尾として落とす(サ変の
    「処分し+たい」も含む)。これが無いと「かばんを捨てたい」がそのまま
    候補になり、部分一致も外れる。"""
    assert router.extract_item_name("かばんを捨てたい") == "かばん"
    assert router.extract_item_name("やかんを処分したいのですが") == "やかん"


def test_leading_particle_is_stripped_from_non_first_fragments_too():
    """句読点で区切った2番目以降の断片も、断片自身の書き出しとして保護される。"""
    candidates, _ = router._item_name_candidates(
        "ドライバー、ねじ回し、はさみを捨てたい"
    )
    assert "ねじ回し" in candidates
    assert "はさみ" in candidates
    assert "じ回し" not in candidates  # 修正前の壊れた形が残っていないこと
    assert "さみ" not in candidates


def test_single_kanji_item_starting_with_a_kana_particle_word_is_unaffected():
    assert router.extract_item_name("傘はどう捨てますか") == "傘"


def test_noun_that_looks_like_a_particle_run_survives():
    """「のこぎり」「のり」は文頭でも `の` を助詞として削ってはいけない。"""
    assert router.extract_item_name("のこぎりを捨てたい") == "のこぎり"
    assert router.extract_item_name("のりの捨て方") == "のり"


# --- 地区名あり(#48-1: 実アプリで最も多い「品名+地区」の形) -----------
@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("小倉北区浅野1丁目でやかんはどう捨てますか", "やかん"),
        ("小倉北区でかばんを捨てたい", "かばん"),
        ("小倉北区浅野1丁目のねじ回しの捨て方", "ねじ回し"),
        ("小倉北区浅野1丁目で傘はどう捨てますか", "傘"),
    ],
)
def test_leading_particle_after_a_removed_region_does_not_eat_the_item_head(
    offline_region, prompt, expected
):
    """区名・町名を空白化した「跡」に残る助詞は落とす。ただし貪欲に落とすと
    「でかばん」→「ばん」のように内容語まで削れるので、接缝では助詞1文字だけ。
    以前は判定を『元のプロンプト』に対して行っていたため、区名がある質問では
    ガードが一切発火しなかった(レビュー #48-1)。"""
    assert router.extract_item_name(prompt) == expected


def test_removed_region_seam_still_trims_the_connective_particle(offline_region):
    """接缝の「で」「の」自体はつなぎなので、内容語が漢字なら普通に落ちる。"""
    cands, _ = router._item_name_candidates("小倉北区城内でテレビはどう捨てますか")
    assert cands[0] == "テレビ"
