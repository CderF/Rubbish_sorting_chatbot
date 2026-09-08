"""レイヤー間で受け渡す型の定義。

`ItemRecord` / `ScheduleRecord` は開発分担計画書 3-1 でデータ層と合意済みの契約。
`ChunkRecord` / `RoutedAnswer` は応用層が定義する(計画書では未定義だったため、
本ファイルを正とする)。
"""

from dataclasses import dataclass, field
from typing import Protocol, TypedDict


class ItemRecord(Protocol):
    """早見表1件分。データ層 `lookup_item_exact` / `lookup_item_partial` の戻り値。

    計画書 3-1 の初版では素のdictだったが、データ層は `app/data/schema.py` の
    Pydanticモデル `WasteItem` を返す実装になった(issue #4 の Implementation
    Decisions で確定)。ここでは属性名だけを定めた Protocol とし、Pydanticモデル・
    スタブのdataclassのどちらも構造的に満たせるようにしている。
    応用層は属性アクセス(`rec.item_name`)で読むこと。添字アクセスは不可。
    """

    item_name: str
    disposal_method: str
    notes: str
    source_url: str


class ScheduleRecord(Protocol):
    """収集日1件分。データ層 `lookup_collection_schedule` の戻り値。

    `ItemRecord` と同じく、実体は `app/data/schema.py` の `CollectionSchedule`。
    """

    ward: str
    town: str
    waste_type: str
    weekday: str
    source_url: str


class ChunkRecord(TypedDict):
    """説明ページのチャンク1件分。ベクトルRAGパスの戻り値。

    こちらは応用層(`vector_store.py`)が生成して応用層が消費する型で、
    レイヤー境界を跨がないため dict のままでよい。
    """

    text: str
    source_url: str
    source_page_title: str
    score: float


@dataclass
class RoutedAnswer:
    """`handle_query` の戻り値。LLMに渡す材料と、回答の組み立て方の指示をまとめる。

    LLMには `context` のみを事実として渡し、`sources` は生成後に本文へ付与する
    (LLMにURLを書かせるとハルシネートするため、引用はコード側で確定させる)。
    """

    context: str = ""
    """LLMに渡す整形済みの検索結果。空文字なら検索ヒットなし。"""

    sources: list[str] = field(default_factory=list)
    """引用URL。重複排除済み、ヒット順を保持。"""

    found: bool = False
    """いずれかの検索パスがヒットしたか。False なら憶測を禁じる指示を出す。"""

    timeline_change: bool = False
    """STAGE04 4-2: この品目は将来出し方が変わる。旧・新・切替日を時系列で
    案内させ、どちらか一方だけで断定させない。"""

    item_match: str = "none"
    """品名検索の結果が、質問された品名そのものか否か:
      "exact"   — 返ってきた早見表行の item_name が質問の品名と一致
      "analogy" — ヒットはあるが、いずれも別の(多くはより具体的な)品目
      "none"    — ヒットなし
    どの段(exact/partial/vector)でヒットしたかではなく、名前が一致したかで
    判定する(PR #14 レビュー)。"analogy" のとき要件4-3 の断り書きを出す。"""

    canned_response: str | None = None
    """設定されている場合、サーバーはこの文字列をそのまま唯一の回答として
    ストリームし、LLM生成をスキップする。

    小型モデルに言い換え・脚色・補足をさせてはならない確定回答用。現状の
    唯一の利用箇所は 4-3 確度ゲート(`analogy_unreliable`): 根拠にできる
    類似品が無いのに LLM に自由生成させると「電動自転車かもしれない」等の
    検索結果に無い推測を書いてしまうため(採点フィードバックの実例)。
    引用URL(`format_citations`)も付けない — 断定を避ける回答に、退けた
    候補ページへの参照を添えるのは誤解を招くため。"""

    analogy_unreliable: bool = False
    """要件4-3: 類似候補(`item_match == "analogy"`)はあるが、質問品と語の上でも
    十分に関連づかず、区分の根拠として信頼できないと判定した状態。

    ベクトル類似度だけで拾った候補(例: セグウェイ→かご、スマートウォッチ→
    湿気取りの容器)は、item_name のみを埋め込んだ弱いベクトルでは誤ヒットでも
    距離が近く、しきい値では良し悪しを分離できない(採点フィードバックの実例)。
    そこで「候補が質問品と語彙的に関連するか」で確度を判定し、関連しなければ
    このフラグを立てて、類推で断定させず『類似品でも十分な根拠を確認できない
    ため断定できない。北九州市に確認を』と案内させる。WebUIの根拠表示のため
    `item_hits` 自体は残す。"""

    missing: list[str] = field(default_factory=list)
    """回答に不足している入力項目(例: ["町名"])。

    設計提案書 3-7 の「推測で答えず再入力を促す」を構造的に表現する。
    空でなければ、LLMには不足項目の再入力を促す回答を書かせる。
    """

    item_hits: list[ItemRecord] = field(default_factory=list)
    schedule_hits: list[ScheduleRecord] = field(default_factory=list)

    detected_item: str | None = None
    """プロンプト(または画像)から抽出した品名。デバッグ・WebUI表示用。"""

    detected_ward: str | None = None
    detected_town: str | None = None

    item_match_stage: str | None = None
    """`item_hits` がどの検索段で得られたか。

    早見表パスは 完全一致 -> 部分一致 -> ベクトル類似度 の3段フォールバック
    (設計提案書 3-3)。どの段まで落ちたかで結果の確からしさが変わる:
    完全一致は質問された品目そのもの、ベクトル類似度は「意味が近い別の品目」
    が混ざりうる。WebUIの根拠表示でこれを区別するために保持する。

    値は "exact" / "partial" / "vector" のいずれか。ヒット0件なら None。
    """

    item_from_image: bool = False
    """`detected_item` が画像分類パス由来かどうか(STAGE03 3-2)。"""

    district_schedule: bool = False
    """`schedule_hits` が「丁目を省略した地区名」から解決されたものかどうか。

    例:「黒崎」は町名としては存在せず「黒崎1〜5丁目」「黒崎城石」の6町が
    実在するが、この6町の収集日は全て一致する(実測: 全7区344地区中277地区
    (92%)が地区内で完全一致)。一致する場合に限り、代表として1町分の
    `schedule_hits` をそのまま使う。`district_towns` に地区内の全町名を
    残し、LLMには「代表の1町だけを見た結果ではなく地区内で確認済み」と
    伝えられるようにする(丁目を勝手に決めつけて答えたわけではないことを
    明示するため)。"""

    district_towns: list[str] = field(default_factory=list)
    """`district_schedule` が True のとき、収集日が一致すると確認した地区内
    の全町名(丁目単位)。"""

    no_weekday_schedule: bool = False
    """この品目は決まった収集曜日を持たない区分(小物金属・小型電子機器・
    電池類・古着・紙パック・蛍光管等の「回収ボックスへ」方式)だった。

    このとき `schedule_hits` は意図的に空のまま(collection_schedule には
    元々該当データが無い)。地区が分かっているのに収集日情報が見つからな
    かった、という `missing` 案内('町名'等の追加入力を促す文言)を出すのは
    誤りなので、代わりにこのフラグを見て「決まった収集曜日はない」と明示
    させる(STAGE03採点フィードバック: 品目ごとの「収集枠」の違いを反映
    できていなかった実例が根拠)。"""
