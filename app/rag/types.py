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
    """品名検索がどの段でヒットしたか: "exact" / "partial" / "similar" / "none"。
    "similar"(ベクトル類似のみ)のとき、要件4-3 として「早見表に個別記載が
    ないので類似品目に準じた案内である」旨を明示させる。"""

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
