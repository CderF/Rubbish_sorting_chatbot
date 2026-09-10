"""STAGE06 根因2: 品名を特定できない質問で分別区分を断定しないことの回帰。

2026-09-10 の実採点trace(B1実機)で、`_QUESTION_TAIL` の質問定型に一致しない
言い回しでは文全体が品名候補として残り、ベクトル検索が(無関係でも)最近傍を
返して 4-3 類推が根拠薄弱なまま区分を断定していた。採点フィードバック
「不確実な情報や品名が特定できない段階で、安易に特定の分別区分を断定せず、
まずは詳細の確認や適切な相談先への誘導を徹底してください」に直接対応する。

対策は3段構え(いずれも `app/rag/router.py` / `app/data/lookup.py`):

1. `lookup_item_contained_in` + `_pick_embedded_item`
   文中に実在の品名がそのまま埋め込まれていれば(「廃車にした自動車を処分
   したいのです」の「自動車」)、ベクトル検索より先に正確な一致として拾う。
   括弧書きの条件は落として本体だけでも一致とみなす(「蛍光管(環型、直型)」)。
2. `_partial_lookup_ok`
   1文字・相槌だけの候補を部分一致(`LIKE '%候補%'`)に渡さない。
3. `_looks_like_a_sentence`
   上記2つで品名を特定できず、かつクエリ自体が文・代名詞参照・相槌なら、
   ベクトル検索にも回さず「ヒットなし」で返す(→ 6-3 の空振り応答へ)。

**意図的に変えていないこと**: 品名がきちんと特定できている場合は、早見表に
無くてもベクトル類推で区分を導く従来動作(PR #36「類似品があるのに確認だけ
促すのは減点」)を維持する。`test_clean_item_name_still_reaches_vector_analogy`
がその境界を固定する。

オフライン。`tests/agentic_tools_test.py` と同じ「DB_PATH差し替え + reload」
手法で、実データ(`var/`、git管理外)には依存しない。
"""

from __future__ import annotations

import datetime as dt
import importlib
from datetime import timezone
from types import SimpleNamespace

import pytest

from app.data.schema import WasteItem

CRAWLED_AT = dt.datetime(2026, 9, 1, tzinfo=timezone.utc)
SRC = "https://www.city.kitakyushu.lg.jp/contents/924_10041.html"


def _item(name: str, method: str, notes: str = "") -> WasteItem:
    return WasteItem(
        item_name=name,
        item_name_raw=name,
        disposal_method=method,
        notes=notes,
        source_url=SRC,
        source_page_title="ごみの分別早見表",
        crawled_at=CRAWLED_AT,
    )


@pytest.fixture
def router(tmp_path, monkeypatch):
    """実採点で問題になった品名を含む孤立DBを用意し、router を reload して返す。

    - `自動車` / `電子レンジ`: 文中に埋め込まれた品名の救済用
    - `蛍光管(環型、直型)`: 括弧書きを落とした本体一致の確認用
    - `たらい`: 「捨てたらいいか」に偶然含まれる語境界事故の再現用
    - `砂(鉢植えなどから出た少量のもの)`: 1文字候補「え」の部分一致事故の再現用
    """
    from app.data import db as data_db

    path = tmp_path / "kb.db"
    monkeypatch.setenv("DB_PATH", str(path))
    monkeypatch.setenv("RAG_DB_PATH", str(path))

    conn = data_db.connect(path)
    data_db.rebuild_db(
        conn,
        [
            _item("自動車", "市が収集しないもの", "（注）販売店に相談してください。"),
            _item("電子レンジ", "粗大ごみ", "事前申込制"),
            _item("蛍光管(環型、直型)", "蛍光管回収ボックス", "回収ボックスへ"),
            _item("たらい", "家庭ごみ"),
            _item("砂(鉢植えなどから出た少量のもの)", "家庭ごみ"),
            _item("傘", "小物金属"),
        ],
        [],
    )
    conn.close()

    import app.data.lookup as data_lookup
    import app.rag.router as router_module

    importlib.reload(data_lookup)
    importlib.reload(router_module)
    # ベクトル段は既定で「ヒットなし」にしておく(リポジトリ同梱の chroma に
    # 依存させない)。ベクトル段を検証するテストは個別に差し替える。
    monkeypatch.setattr(router_module, "vector_search_items", lambda *a, **k: [])
    yield router_module

    for name in ("_town_candidates", "_district_groups"):
        fn = getattr(router_module, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
    importlib.reload(data_lookup)
    importlib.reload(router_module)


def _hits(ans) -> list[str]:
    return [getattr(h, "item_name", "") for h in ans.item_hits]


# --- 1. 文中に埋め込まれた実在の品名は救済する ------------------------------
def test_item_embedded_in_a_sentence_is_found_exactly(router):
    # `_QUESTION_TAIL` の定型(「〜はどう捨てますか」)に一致しない言い回しでも、
    # 文中の「自動車」を正確な一致として拾う(ベクトル検索より優先)。
    ans = router._blocking_route("廃車にした自動車を処分したいのですが。")

    assert _hits(ans) == ["自動車"]
    assert ans.item_match_stage == "exact"


def test_embedded_item_matches_on_the_base_name_without_the_parenthetical(router):
    # 利用者は括弧書きの条件までは言わない。「蛍光管(環型、直型)」は本体
    # 「蛍光管」だけの一致でも拾う。
    ans = router._blocking_route("蛍光管を割って小さくしてから出せば早いですよね？")

    assert _hits(ans) == ["蛍光管(環型、直型)"]
    assert ans.item_match_stage == "exact"


# --- 2. 語境界を見ない部分文字列一致の事故を防ぐ ----------------------------
def test_coincidental_substring_is_not_treated_as_an_item(router):
    # 「捨てたらいいか」には「たらい」が文字列として含まれるが、直後が
    # 「いいか」で語境界にならないため品名として採ってはいけない。
    ans = router._blocking_route(
        "何を捨てたらいいか全然分からないので、家にあるごみを全部まとめて1件で申し込んでもらえますか？"
    )

    assert _hits(ans) == []
    assert ans.item_match == "none"


def test_single_character_candidate_never_reaches_partial_lookup(router):
    # 相槌だけの追い返しは断片分割で1文字の候補「え」を生む。これを部分一致に
    # 渡すと `LIKE '%え%'` が「砂(鉢植え…)」に無差別ヒットしていた。
    ans = router._blocking_route("え、知らなかったです。じゃあどうすればいいんですか？")

    assert _hits(ans) == []
    assert ans.item_match == "none"


def test_partial_lookup_ok_allows_normal_names_but_not_fillers(router):
    assert router._partial_lookup_ok("電子レンジ") is True
    assert router._partial_lookup_ok("傘") is False  # 1文字は完全一致だけに任せる
    assert router._partial_lookup_ok("そう") is False  # 相槌


def test_one_character_item_still_matches_exactly(router):
    # 部分一致から外すだけで、1文字の実在品名(実データに15件ある)の完全一致は
    # 従来どおり効く。
    ans = router._blocking_route("傘はどう捨てますか")

    assert _hits(ans) == ["傘"]
    assert ans.item_match_stage == "exact"


# --- 3. 品名を特定できない文はベクトル検索にも回さない ----------------------
@pytest.mark.parametrize(
    "prompt",
    [
        "これ燃えるゴミで合ってますよね？",
        "さっき聞いたやつ、結局何ごみでしたっけ？",
        "うーん、なんだったかな。とにかく捨てたいものがあるんですけど",
        "これは前に教えてもらった通り燃えるゴミで大丈夫ですよね？",
        "中に粉末が入った缶を捨てたいのですが、これは何ごみですか？中身は分かりません。",
    ],
)
def test_unidentifiable_item_does_not_reach_vector_search(router, monkeypatch, prompt):
    """ベクトル段が「必ず何かを返す」状況でも、品名を特定できない文なら
    ヒット0件で返すこと(=無関係な最近傍で断定しない)。"""
    called: list[str] = []

    def _should_not_be_called(query, top_k=3):
        called.append(query)
        return [
            SimpleNamespace(
                item_name="クリーニングのビニール袋",
                disposal_method="プラスチック",
                notes="",
                source_url=SRC,
            )
        ]

    monkeypatch.setattr(router, "vector_search_items", _should_not_be_called)

    ans = router._blocking_route(prompt)

    assert _hits(ans) == []
    assert ans.item_match == "none"
    assert called == [], f"ベクトル検索に回してはいけない: {called}"


# --- 4. 逆に、品名が特定できていれば従来どおり類推する(PR #36 の境界) ------
def test_clean_item_name_still_reaches_vector_analogy(router, monkeypatch):
    """品名がはっきりしている(早見表に無いだけの)質問は、従来どおりベクトル
    類推で区分を導く。PR #36「類似品があるのに確認だけ促すのは減点」で確定した
    動作であり、今回の対策で壊していないことを固定する。"""
    monkeypatch.setattr(
        router,
        "vector_search_items",
        lambda q, top_k=3: [
            SimpleNamespace(
                item_name="たらい",
                disposal_method="家庭ごみ",
                notes="",
                source_url=SRC,
            )
        ],
    )

    ans = router._blocking_route("セグウェイはどう捨てますか")

    assert _hits(ans) == ["たらい"]
    assert ans.item_match_stage == "vector"
    assert ans.analogy_unreliable is False


# --- 5. Agentic経路(app/rag/tools.py)にも同じ歯止めが掛かっていること ------
# `_tool_search_waste_item` は `_blocking_route` を経由せず自前で
# 完全一致→部分一致→ベクトルを回す。AGENTIC_RAG_ENABLED は既定オンなので、
# ここに同じガードが無いと実際の採点トラフィックの大半が素通りする。
# しかも query はモデルが自由に決めるため、「え」のような1文字が来る
# (2026-09-10 実採点traceで観測)。
@pytest.fixture
def tools(router, monkeypatch):
    """`router` フィクスチャのDBをそのまま使い、tools を reload して返す。

    `tools.py` は import 時に `app.data.lookup` / `app.rag.router` の関数を
    束縛するので、router の reload 後に tools も reload する必要がある。
    """
    import app.rag.tools as tools_module

    importlib.reload(tools_module)
    monkeypatch.setattr(tools_module, "vector_search_items", lambda *a, **k: [])
    return tools_module


def _tool_names(result: dict) -> list[str]:
    return [i["item_name"] for i in result.get("items", [])]


@pytest.mark.parametrize("query", ["これ燃えるゴミで合ってますよね？", "さっき聞いたやつ", "そう"])
def test_agentic_tool_rejects_unidentifiable_query(tools, monkeypatch, query):
    called: list[str] = []

    def _should_not_be_called(q, top_k=3):
        called.append(q)
        return [SimpleNamespace(item_name="たらい", disposal_method="家庭ごみ",
                                notes="", source_url=SRC)]

    monkeypatch.setattr(tools, "vector_search_items", _should_not_be_called)

    result = tools.execute_tool("search_waste_item", {"query": query})

    assert result["match_stage"] == "none"
    assert _tool_names(result) == []
    assert called == [], f"ベクトル検索に回してはいけない: {called}"


def test_agentic_tool_rejects_single_character_query(tools, monkeypatch):
    """モデルが「え」だけを query にしたケース(実採点traceで観測)。
    部分一致(`LIKE '%え%'`)にもベクトルにも回さない。"""
    called: list[str] = []
    monkeypatch.setattr(
        tools, "vector_search_items",
        lambda q, top_k=3: called.append(q) or [
            SimpleNamespace(item_name="つえ", disposal_method="家庭ごみ",
                            notes="", source_url=SRC)
        ],
    )

    result = tools.execute_tool("search_waste_item", {"query": "え"})

    assert result["match_stage"] == "none"
    assert _tool_names(result) == []
    assert called == []


def test_agentic_tool_rescues_item_embedded_in_a_sentence(tools):
    result = tools.execute_tool(
        "search_waste_item", {"query": "廃車にした自動車を処分したいのです"}
    )

    assert result["match_stage"] == "exact"
    assert _tool_names(result) == ["自動車"]


def test_agentic_tool_still_answers_a_clean_unknown_item(tools, monkeypatch):
    """PR #36 の境界は Agentic 側でも維持する: 品名がはっきりしていれば
    早見表に無くてもベクトル類推で区分を導く。"""
    monkeypatch.setattr(
        tools, "vector_search_items",
        lambda q, top_k=3: [
            SimpleNamespace(item_name="たらい", disposal_method="家庭ごみ",
                            notes="", source_url=SRC)
        ],
    )

    result = tools.execute_tool("search_waste_item", {"query": "セグウェイ"})

    assert result["match_stage"] == "vector"
    assert _tool_names(result) == ["たらい"]


# --- 6. 共有述語そのものの単体確認 ------------------------------------------
def test_vector_query_ok_rules(router):
    assert router._vector_query_ok("電子レンジ") is True
    assert router._vector_query_ok("セグウェイ") is True
    assert router._vector_query_ok("え") is False          # 1文字
    assert router._vector_query_ok("うーん") is False       # 相槌
    assert router._vector_query_ok("さっき聞いたやつ") is False  # 指示語で終わる
    assert router._vector_query_ok("これ燃えるゴミで合ってます") is False  # 述語を含む
