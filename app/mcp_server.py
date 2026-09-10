"""STAGE07 7-1 / 7-2: 段階5までに作った分別判定・収集日参照を MCP サーバとして公開する。

外部の AI クライアント(Claude Desktop など)から stdio で接続し、以下のツールを
呼び出せる。検索ロジックは一切再実装しておらず、Agentic RAG(#40)が既に定義済みの
`app/rag/tools.py`(データ層・ベクトル RAG パス・お知らせパスの薄いラッパー)と、
決定論ルータ `app/rag/router._blocking_route` をそのまま呼ぶ。

公開ツール:

  - search_waste_item(query)            早見表パス(完全一致→部分一致→ベクトル)
  - lookup_collection_schedule(ward,town,waste_type)  収集日パス(丁目省略の地区名も解決)
  - lookup_item_change_notice(item_name)  お知らせパス(STAGE04 4-2 の変更予定)
  - search_general_info(query)          ベクトル RAG パス(分別以外の一般案内)
  - get_disposal_guidance(prompt)       上記を組み合わせた「1発の相談」。ルータが
                                        組んだ根拠 context・参照 URL・マッチ段階を
                                        そのまま返す(LLM 生成は通さない)。表形式
                                        (markdown)も同時に返す(STAGE07 7-3)。

会話状態(STAGE05 のマルチターン・スロット)は公開しない。マルチターンは接続元の
クライアントが管理するものなので、こちらは「その 1 発で何が引けるか」だけを返す。

起動:

    python -m app.mcp_server          # stdio(Claude Desktop 用)

依存: `var/kitakyushu_waste.db`(クローラの構造化 DB)と `var/chroma/`(ベクトル
索引)。`DB_PATH` / `RAG_DB_PATH` 環境変数で差し替え可。GPU・vLLM は不要。
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from app.rag import router, tools

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
def _schedule_table(schedules: list[dict]) -> str:
    if not schedules:
        return ""
    lines = ["| 区・町名 | ごみ種別 | 収集日 |", "|---|---|---|"]
    for s in schedules:
        lines.append(
            f"| {s.get('ward', '')}{s.get('town', '')} "
            f"| {s.get('waste_type', '')} | {s.get('weekday', '')} |"
        )
    return "\n".join(lines)


def _item_table(items: list[dict]) -> str:
    if not items:
        return ""
    lines = ["| 品名 | 出し方 | 備考 |", "|---|---|---|"]
    for i in items:
        notes = (i.get("notes") or "").replace("\n", " ")
        lines.append(
            f"| {i.get('item_name', '')} | {i.get('disposal_method', '')} | {notes} |"
        )
    return "\n".join(lines)


@mcp.tool()
def get_disposal_guidance(prompt: str) -> dict[str, Any]:
    """1 発の相談文を決定論ルータに通し、回答の「材料」をそのまま返す。

    LLM による文章生成は通さない(このサーバに vLLM は無い)。返すのは:

      - `context`      : ルータが組んだ根拠テキスト(そのまま案内文の材料になる)
      - `sources`      : 採用した参照 URL
      - `detected`     : 抽出した品名・地区・画像由来かどうか
      - `match_stage`  : 早見表のヒット段(exact / partial / vector / analogy / none)
      - `found`        : 品目か収集日のどちらかが引けたか
      - `missing`      : 回答に足りない項目(例: 町名)
      - `markdown`     : 早見表・収集日の結果を表(GFM)にしたもの(STAGE07 7-3)

    接続元のクライアントは、この材料を使って利用者向けの文章を書けばよい。
    """
    ans = router._blocking_route(prompt)
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
        "markdown": "\n\n".join(parts),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
