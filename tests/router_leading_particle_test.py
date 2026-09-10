"""`app/rag/router._trim_particles` の回帰テスト。

`_LEADING`(文頭の助詞除去)は文字クラスによる素朴な除去なので、「やかん」
「ねじ回し」のように内容語自体が助詞と同じ仮名(や・ね・か等)で始まる品名まで
削ってしまっていた(実測: 「やかんを処分したい」→ 候補「んを処分したい」、
「ねじ回しはどう捨てますか」→ 検出品名「じ回し」)。`_trim_particles` は、
断片が発話中で「語の書き出し」の位置(先頭、または断片分割に使う区切りの
直後)に現れているときだけ `_LEADING` を当てないことでこれを防ぐ。

すべて区名を含まないプロンプトを使い、DB接続(`extract_region` → データ層)を
経由せずオフラインで検証する。
"""

from __future__ import annotations

from app.rag import router


def test_item_starting_with_a_particle_kana_is_not_mangled():
    # 「や」「ね」「か」等は _PARTICLES に含まれる仮名なので、素朴な文頭除去
    # だと内容語の1文字目・2文字目まで削られていた。
    assert router.extract_item_name("やかんを処分したい") is not None
    assert "やかん" in router.extract_item_name("やかんを処分したい")


def test_leading_particle_is_stripped_from_the_start_of_the_utterance():
    assert router.extract_item_name("ねじ回しはどう捨てますか") == "ねじ回し"


def test_leading_particle_is_stripped_from_non_first_fragments_too():
    """句読点で区切った2番目以降の断片も、断片自身の書き出しとして保護される。"""
    candidates, _ = router._item_name_candidates(
        "ドライバー、ねじ回し、はさみを捨てたい"
    )
    assert "ねじ回し" in candidates
    assert "じ回し" not in candidates  # 修正前の壊れた形が残っていないこと


def test_leading_particle_after_a_removed_region_is_still_stripped():
    """区名・町名を除去した「跡」に残る助詞は、従来どおり落とす。

    「小倉北区浅野1丁目で傘はどう捨てますか」→ 区・町名除去後は「で傘は
    どう捨てますか」となり、この「で」は語の書き出しではなく助詞の残骸
    なので削ってよい(`_QUESTION_TAIL` が末尾の「はどう捨てますか」も
    落とすので、region除去後の先頭処理だけをきれいに検証できる)。
    """
    assert router.extract_item_name("小倉北区浅野1丁目で傘はどう捨てますか") == "傘"


def test_single_kanji_item_starting_with_a_kana_particle_word_is_unaffected():
    # 「かさ(傘)」のように漢字の品名はそもそも _PARTICLES に引っかからない
    # ため、region除去が絡まない単純なケースでも回帰していないことを確認する。
    assert router.extract_item_name("傘はどう捨てますか") == "傘"
