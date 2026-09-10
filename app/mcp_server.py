"""STAGE07: 段階5までに作った分別判定・収集日参照を MCP サーバとして公開する。

外部の AI クライアント(Claude Desktop など)から stdio で接続し、以下を使える。
検索ロジックは一切再実装しておらず、Agentic RAG(#40)が既に定義済みの
`app/rag/tools.py`(データ層・ベクトル RAG パス・お知らせパスの薄いラッパー)、
決定論ルータ `app/rag/router._blocking_route`、ナレッジ一覧 `app/rag/knowledge.py`
をそのまま呼ぶ。

ツール(7-1):

  - search_waste_item(query)            早見表パス(完全一致→部分一致→ベクトル)
  - lookup_collection_schedule(ward,town,waste_type)  収集日パス(丁目省略の地区名も解決)
  - lookup_item_change_notice(item_name)  お知らせパス(STAGE04 4-2 の変更予定)
  - search_general_info(query)          ベクトル RAG パス(分別以外の一般案内)
  - get_disposal_guidance(prompt, user_id=None)  上記を組み合わせた「1発の相談」。
    ルータが組んだ根拠・表(markdown, 7-3)を返す(LLM 生成は通さない)。
    `user_id` を渡すと 7-4 の記憶(区・町名)を読み書きする
  - remember_fact / recall_facts / forget_facts(user_id, …)  7-4 の記憶ストアの
    直接操作。接続元のクライアントが明示的に管理する経路

リソース(7-1 / 7-3):

  - knowledge://sources        取り込み済みの参照元ページ一覧(/knowledge と同じ)
  - knowledge://{source_id}    1ページ分の内容
  - schedule-calendar://{ward}/{town}  収集日を1週間カレンダーのSVG画像にしたもの
    (テキスト以外の出力、7-3)。PIL・vLLM 等は使わず文字列組み立てのみ

会話状態(STAGE05 のマルチターン・スロット)は公開しない。マルチターンは接続元の
クライアントが管理するものなので、こちらは「その 1 発で何が引けるか」だけを返す。
7-4 の記憶はそれとは別の層(セッションをまたぐ永続ストア、`app/user_memory.py`)。

起動:

    python -m app.mcp_server          # stdio(Claude Desktop 用)

依存: `var/kitakyushu_waste.db`(クローラの構造化 DB)と `var/chroma/`(ベクトル
索引)。`DB_PATH` / `RAG_DB_PATH` / `USER_MEMORY_DB_PATH` 環境変数で差し替え可。
GPU・vLLM は不要。
"""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape as xml_escape

from mcp.server.fastmcp import FastMCP

from app import user_memory
from app.rag import knowledge, router, tools

mcp = FastMCP("kitakyushu-waste")


# --------------------------------------------------------------------------
# 原子ツール — app/rag/tools.execute_tool にそのまま委譲する
# --------------------------------------------------------------------------
@mcp.tool()
def search_waste_item(query: str) -> dict[str, Any]:
    """ごみの品目名から分別区分(出し方)と備考を引く。

    完全一致 → 部分一致 → 意味の近い品目のベクトル検索、の順で試し、`match_stage`
    にどの段で当たったかを返す。早見表に無い品物は `match_stage="none"`。
    """
    return tools.execute_tool("search_waste_item", {"query": query})


@mcp.tool()
def lookup_collection_schedule(
    ward: str, town: str, waste_type: str | None = None
) -> dict[str, Any]:
    """区・町名(丁目まで)から収集日を引く。`waste_type` を渡すとその区分に絞る。

    「黒崎」のように丁目を省略した地区名でも、地区内の全丁目で収集日が一致する
    場合はそれを返し、`district_note` にその旨を添える。
    """
    return tools.execute_tool(
        "lookup_collection_schedule",
        {"ward": ward, "town": town, "waste_type": waste_type},
    )


@mcp.tool()
def lookup_item_change_notice(item_name: str) -> dict[str, Any]:
    """その品目に「これから出し方が変わる」予定(STAGE04 4-2)があるかを引く。

    予定があれば現行ルール(`current`)と変更後(`future`、切替日つき)を両方返す。
    """
    return tools.execute_tool("lookup_item_change_notice", {"item_name": item_name})


@mcp.tool()
def search_general_info(query: str) -> dict[str, Any]:
    """分別区分そのものではない一般的な案内(ごみステーションの使い方・粗大ごみの
    申し込み方法など)を、クロールした本文からベクトル検索で引く。
    """
    return tools.execute_tool("search_general_info", {"query": query})


# --------------------------------------------------------------------------
# 統合ツール — ルータが 1 発の相談に対して組んだ根拠をそのまま返す
# --------------------------------------------------------------------------
def _markdown_cell(value: object) -> str:
    """Return a value safe to place in a GFM table cell.

    Knowledge records are normally controlled by the crawler, but notices and
    user-provided values can contain pipes or line breaks.  Without escaping,
    one record can break the table layout returned by the MCP tool.
    """
    text = "" if value is None else str(value)
    return (
        text.replace("|", r"\|")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
    )


def _schedule_table(schedules: list[dict]) -> str:
    if not schedules:
        return ""
    lines = ["| 区・町名 | ごみ種別 | 収集日 |", "|---|---|---|"]
    for s in schedules:
        lines.append(
            f"| {_markdown_cell(s.get('ward', ''))}{_markdown_cell(s.get('town', ''))} "
            f"| {_markdown_cell(s.get('waste_type', ''))} "
            f"| {_markdown_cell(s.get('weekday', ''))} |"
        )
    return "\n".join(lines)


def _item_table(items: list[dict]) -> str:
    if not items:
        return ""
    lines = ["| 品名 | 出し方 | 備考 |", "|---|---|---|"]
    for i in items:
        lines.append(
            f"| {_markdown_cell(i.get('item_name', ''))} "
            f"| {_markdown_cell(i.get('disposal_method', ''))} "
            f"| {_markdown_cell(i.get('notes', ''))} |"
        )
    return "\n".join(lines)


def _needs_region(ans) -> bool:
    return any(("町名" in m) or ("区名" in m) for m in ans.missing)


@mcp.tool()
def get_disposal_guidance(prompt: str, user_id: str | None = None) -> dict[str, Any]:
    """1 発の相談文を決定論ルータに通し、回答の「材料」をそのまま返す。

    LLM による文章生成は通さない(このサーバに vLLM は無い)。返すのは:

      - `context`      : ルータが組んだ根拠テキスト(そのまま案内文の材料になる)
      - `sources`      : 採用した参照 URL
      - `detected`     : 抽出した品名・地区・画像由来かどうか
      - `match_stage`  : 早見表のヒット段(exact / partial / vector / analogy / none)
      - `found`        : 品目か収集日のどちらかが引けたか
      - `missing`      : 回答に足りない項目(例: 町名)
      - `canned_response`: 確度ゲート(4-3)・類別質問・安全拒否などでルータが
        確定させた固定文。`None` でなければクライアントは自由生成せず**これを
        そのまま**利用者に出す(`found=False`・`context=""` のときに本当に
        見せるべき文がここに入る)
      - `markdown`     : 早見表・収集日の結果を表(GFM)にしたもの(STAGE07 7-3)
      - `used_memory`  : 記憶していた区・町名で補って引き直したか(7-4)
      - `remembered`   : `user_id` を渡したときの、この時点での記憶内容

    `user_id` を渡すと 7-4 のパーソナライズが働く:
      1. まず素の `prompt` で引く。
      2. 地区が足りず(`missing` に「町名」等)、かつ記憶に区・町名があれば、
         それを文末に補ってもう一度引く(例: 「ソファの収集日は?」+
         「(小倉北区浅野1丁目)」)。`extract_region()` は文中のどこからでも
         区名・町名を拾うので、素の文字列抽出のまま働く — 記憶側を特別扱いする
         専用の結合ロジックは持たない。
      3. 今回の発話から区・町名が読めていれば(初めて言った、あるいは記憶を
         訂正した)、次回のために記憶へ書き戻す。

    接続元のクライアントは、この材料を使って利用者向けの文章を書けばよい。
    """
    remembered: dict[str, str] = {}
    if user_id:
        conn = user_memory.connect()
        user_memory.init_schema(conn)
        remembered = user_memory.recall_all(conn, user_id)
        conn.close()

    ans = router._blocking_route(prompt)

    # 7-4: 記憶で補うのは「今回の発話に区も町名も一切書かれていない」ときだけ。
    # 発話に区名があるのに記憶の区を足すと、`extract_region` は文中の出現位置では
    # なく WARDS の並び順で1つ選ぶため記憶側が勝ち、ユーザーが書いた区を黙って
    # 上書きしてしまう(レビュー #47-1)。記憶は区・町名が両方そろっているときだけ
    # 使う — 町名まで無ければどのみち収集日は引けない。
    used_memory = False
    can_fill = bool(
        user_id
        and not ans.detected_ward
        and not ans.detected_town
        and _needs_region(ans)
        and remembered.get("ward")
        and remembered.get("town")
    )
    if can_fill:
        augmented = f"{prompt}({remembered['ward']}{remembered['town']})"
        retried = router._blocking_route(augmented)
        # 補ったうえで実際に引けたときだけ採用する。空振りの再検索を採用すると、
        # ユーザーが一度も言っていない住所について「データがありません」と
        # 答えてしまう(レビュー #47-2)。
        if retried.found:
            ans = retried
            used_memory = True

    # 書き戻すのは区・町名がそろった「完全な住所」を言ったターンだけ。区だけの
    # ターンで区を上書きすると、記憶が「同時に存在したことのない区/町名の組」に
    # 壊れ、`recall_facts` が実在しない住所を読み上げる(レビュー #47-3)。
    if user_id and ans.detected_ward and ans.detected_town:
        conn = user_memory.connect()
        user_memory.init_schema(conn)
        user_memory.remember(conn, user_id, "ward", ans.detected_ward)
        user_memory.remember(conn, user_id, "town", ans.detected_town)
        conn.close()
        remembered = {
            **remembered,
            "ward": ans.detected_ward,
            "town": ans.detected_town,
        }

    src = router.describe_sources(ans)
    parts: list[str] = []
    item_md = _item_table(src.get("items", []))
    sched_md = _schedule_table(src.get("schedules", []))
    if item_md:
        parts.append("### 分別早見表\n" + item_md)
    if sched_md:
        parts.append("### 収集日\n" + sched_md)

    return {
        "context": ans.context,
        "sources": list(ans.sources),
        "detected": src.get("detected", {}),
        "match_stage": ans.item_match_stage,
        "item_match": ans.item_match,
        "found": ans.found,
        "missing": list(ans.missing),
        "canned_response": ans.canned_response,
        "markdown": "\n\n".join(parts),
        "used_memory": used_memory,
        "remembered": remembered if user_id else None,
    }


# --------------------------------------------------------------------------
# パーソナライズ(STAGE07 7-4) — app/user_memory.py への薄い委譲
# --------------------------------------------------------------------------
@mcp.tool()
def remember_fact(user_id: str, key: str, value: str) -> dict[str, Any]:
    """利用者(`user_id`)の記憶に1件書く(セッションをまたいで残る)。

    `get_disposal_guidance` は区・町名だけを自動的に読み書きするが、これは
    任意の key で明示的に覚えさせたいとき用(接続元のクライアントが判断して呼ぶ)。
    """
    try:
        conn = user_memory.connect()
        user_memory.init_schema(conn)
        user_memory.remember(conn, user_id, key, value)
        conn.close()
        return {"ok": True}
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def recall_facts(user_id: str) -> dict[str, Any]:
    """利用者(`user_id`)について覚えている全項目を返す。"""
    conn = user_memory.connect()
    user_memory.init_schema(conn)
    facts = user_memory.recall_all(conn, user_id)
    conn.close()
    return {"facts": facts}


@mcp.tool()
def forget_facts(user_id: str, key: str | None = None) -> dict[str, Any]:
    """利用者の記憶を消す。`key` を渡せば1件だけ、省略すれば全部。

    `key` に空文字を渡すのは入力ミスとして扱い、全削除にはフォールバックしない。

    利用者からの明示的な「忘れて」要求に応える出口(個人情報を溜め込むだけに
    しないため)。
    """
    if key is not None and not key:
        return {"error": "key は空にできません"}

    conn = user_memory.connect()
    user_memory.init_schema(conn)
    removed = (
        (1 if user_memory.forget(conn, user_id, key) else 0)
        if key is not None
        else user_memory.forget_all(conn, user_id)
    )
    conn.close()
    return {"removed": removed}


# --------------------------------------------------------------------------
# リソース(STAGE07 7-1) — ナレッジ一覧を MCP resource として公開
# --------------------------------------------------------------------------
@mcp.resource("knowledge://sources")
def knowledge_sources() -> dict[str, Any]:
    """取り込み済みの参照元ページ一覧(STAGE02 2-3、WebUI の `/knowledge` と同じデータ)。"""
    return knowledge.list_sources()


@mcp.resource("knowledge://{source_id}")
def knowledge_source(source_id: str) -> dict[str, Any]:
    """1ページ分の取り込み内容(表形式)。`knowledge://sources` の `id` を渡す。"""
    doc = knowledge.get_source(source_id)
    return doc if doc is not None else {"error": f"unknown source id: {source_id}"}


# --------------------------------------------------------------------------
# 画像出力(STAGE07 7-3) — 収集日を1週間カレンダーのSVGにする
# --------------------------------------------------------------------------
_WEEKDAY_KANJI = ("月", "火", "水", "木", "金", "土", "日")


def _marked_weekdays(schedules: list[dict]) -> "list[str]":
    """`weekday` 文字列(「月曜日・木曜日」「第2水曜日」等)に出てくる曜日を拾う。

    `k in weekday` のような素朴な部分一致だと、どの曜日の行にも付く接尾辞
    「曜日」の「日」が日曜日と誤判定される(「水曜日」に「日」が含まれてしまう)。
    「{曜日漢字}曜」の並びで見ることで、実際に「日曜」と書かれた行だけを拾う。
    """
    text = " ".join(s.get("weekday") or "" for s in schedules)
    return [k for k in _WEEKDAY_KANJI if f"{k}曜" in text]  # 月〜日の順に揃える


def _render_week_svg(marked: "list[str]", title: str) -> str:
    """PIL・matplotlib 等は使わず、SVGをそのまま文字列で組み立てる(依存ゼロ)。"""
    cell, gap, x0, y0 = 40, 4, 10, 34
    width = x0 * 2 + len(_WEEKDAY_KANJI) * cell
    height = y0 + cell + 10
    safe_title = xml_escape(title)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="sans-serif">',
        f'<text x="{x0}" y="18" font-size="13" fill="#1a2027">{safe_title}</text>',
    ]
    for i, k in enumerate(_WEEKDAY_KANJI):
        x = x0 + i * cell
        hit = k in marked
        fill = "#1a56b8" if hit else "#e8edf3"
        text_fill = "#ffffff" if hit else "#5a6472"
        size = cell - gap
        parts.append(f'<rect x="{x}" y="{y0}" width="{size}" height="{size}" rx="6" fill="{fill}"/>')
        parts.append(
            f'<text x="{x + size / 2}" y="{y0 + size / 2 + 5}" font-size="16" '
            f'text-anchor="middle" fill="{text_fill}">{k}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


@mcp.resource(
    "schedule-calendar://{ward}/{town}",
    mime_type="image/svg+xml",
    name="収集日カレンダー(SVG)",
    description="区・町名の収集日を1週間分のカレンダー画像にする(STAGE07 7-3)",
)
def schedule_calendar_svg(ward: str, town: str) -> str:
    """`lookup_collection_schedule` と同じ地区解決で曜日を引き、月〜日のうち
    収集日に当たる曜日を塗ったSVGを返す。ヒットが無ければ強調無しの週表を返す
    (存在しないSVGを返すより「何も無い」ことが見える形にする)。
    """
    result = tools.execute_tool(
        "lookup_collection_schedule", {"ward": ward, "town": town, "waste_type": None}
    )
    if "error" in result:
        # 「この地区に収集日が無い」ときの素の週表と見分けが付くよう、エラーは
        # タイトルに出す(空 town・DB 破損などを「収集日ゼロ」と混同しない
        # ためのレビュー #47-5 対応)。タイトルは `_render_week_svg` 側で
        # XML エスケープされる。
        return _render_week_svg([], f"{ward}{town}: {result['error']}")
    schedules = result.get("schedules", [])
    title = f"{ward}{town} の収集日"
    return _render_week_svg(_marked_weekdays(schedules), title)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
