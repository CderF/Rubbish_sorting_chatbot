"""データ層(`app/data/lookup.py`)が未実装の間だけ使うダミー実装。

開発分担計画書 3-1 のシグネチャをそのまま満たす。`app/data/lookup.py` が
現れた時点で `router.py` の import が本物に切り替わるため、このファイルは
削除・変更不要(結合テスト時の対照データとして残す)。

**このデータは本物ではない。** 採点・デモに使ってはならない。
`router.using_stubs()` が True を返す間は WebUI にもその旨を表示する。
"""

import unicodedata
from dataclasses import dataclass

_STUB_URL = "https://www.city.kitakyushu.lg.jp/contents/924_10041.html"
_SCHED_URL = "https://www.city.kitakyushu.lg.jp/contents/924_10102.html"


# 本物のデータ層は app/data/schema.py の Pydanticモデルを返す。ここでは
# 同じ属性名を持つ dataclass で代用し、応用層から見て区別がつかないようにする。
@dataclass
class _StubItem:
    item_name: str
    disposal_method: str
    notes: str
    source_url: str


@dataclass
class _StubSchedule:
    ward: str
    town: str
    waste_type: str
    weekday: str
    source_url: str


_ITEMS = [
    _StubItem("ペットボトル", "資源化ごみ", "キャップとラベルを外し、中をすすいで出す", _STUB_URL),
    _StubItem("電子レンジ", "粗大ごみ", "収集は事前申込制。粗大ごみ受付センターへ申し込む", _STUB_URL),
    _StubItem("乾電池", "特定ごみ", "透明な袋に入れ、他のごみと混ぜずに出す", _STUB_URL),
]

_SCHEDULES = [
    _StubSchedule("小倉北区", "浅野1丁目", "家庭ごみ", "火,金", _SCHED_URL),
    _StubSchedule("小倉北区", "浅野1丁目", "資源化ごみ", "木", _SCHED_URL),
]


def _norm(s: str) -> str:
    """データ層と同じ正規化を模す(NFKC + 前後空白除去)。"""
    return unicodedata.normalize("NFKC", s).strip()


def lookup_item_exact(item_name: str) -> _StubItem | None:
    key = _norm(item_name)
    for rec in _ITEMS:
        if _norm(rec.item_name) == key:
            return rec
    return None


def lookup_item_partial(item_name: str, limit: int = 10) -> list[_StubItem]:
    key = _norm(item_name)
    if not key:
        return []
    hits = [
        rec
        for rec in _ITEMS
        if key in _norm(rec.item_name) or _norm(rec.item_name) in key
    ]
    return hits[:limit]


def lookup_towns(ward: str) -> list[str]:
    key = _norm(ward)
    return sorted({_norm(rec.town) for rec in _SCHEDULES if _norm(rec.ward) == key})


def lookup_collection_schedule(
    ward: str, town: str | None, waste_type: str | None = None
) -> list[_StubSchedule] | None:
    # 契約: 町名が無ければ推測せず None(応用層が町名の追加入力を促すトリガー)
    if not town:
        return None
    hits = [
        rec
        for rec in _SCHEDULES
        if _norm(rec.ward) == _norm(ward) and _norm(rec.town) == _norm(town)
    ]
    if waste_type:
        hits = [r for r in hits if _norm(r.waste_type) == _norm(waste_type)]
    return hits or None
