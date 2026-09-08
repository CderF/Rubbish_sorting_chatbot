"""レイヤー間で受け渡す型の定義。

`ItemRecord` / `ScheduleRecord` は開発分担計画書 3-1 でデータ層と合意済みの契約。
`ChunkRecord` / `RoutedAnswer` は応用層が定義する(計画書では未定義だったため、
本ファイルを正とする)。
"""

from dataclasses import dataclass, field
from typing import Protocol, TypedDict

from app.rag.session_state import DialogState


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
class ItemResult:
    """1つの品目に対する早見表パスの検索結果(STAGE05)。

    STAGE02/03 は1リクエスト1品目だったので `RoutedAnswer` が品目の情報を
    直接持っていたが、5-2「追加なら前の案内を保ったまま足す」で1回の回答が
    複数品目を扱うようになったため、品目単位に切り出した。
    """

    item_name: str
    hits: list[ItemRecord] = field(default_factory=list)
    match_stage: str | None = None
    from_image: bool = False
    timeline_change: bool = False
    timeline_lines: list[str] = field(default_factory=list)
    """STAGE04 4-2 の「いつから何がどう変わるか」。品目ごとに持つ。"""


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

    missing: list[str] = field(default_factory=list)
    """回答に不足している入力項目(例: ["町名"])。

    設計提案書 3-7 の「推測で答えず再入力を促す」を構造的に表現する。
    空でなければ、LLMには不足項目の再入力を促す回答を書かせる。
    """

    item_results: list[ItemResult] = field(default_factory=list)
    """相談中の品目それぞれの検索結果(STAGE05)。順序は登場順、末尾が最新。"""

    item_hits: list[ItemRecord] = field(default_factory=list)
    """`item_results` の全ヒットを平坦化したもの。

    STAGE02/03 からの互換フィールド。WebUIの根拠パネルと `format_citations`
    が単一リストを前提にしているため残している。"""

    schedule_hits: list[ScheduleRecord] = field(default_factory=list)

    state: "DialogState | None" = None
    """このターン終了時点の会話スロット(STAGE05)。`server.py` がセッションに
    保存し、次のターンで `handle_query` に渡し直す。"""

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
