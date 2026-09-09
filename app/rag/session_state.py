"""会話をまたぐスロット状態(STAGE05 マルチターン対応)。

STAGE02/03 の検索は「1リクエスト内で完結する」設計だった
(`documents/stage02&03_proposal.md` 4章)。STAGE05 の 5-1「品名・地区が数回に
分かれて出てくる相談でも前のやりとりを引き継ぐ」は、その前提を破らずに
**検索の入力を1文からセッション単位のスロットに差し替える**ことで満たす。

つまり本モジュールが持つのは「今この会話で分かっていること」だけで、回答の
組み立ても検索も一切行わない。`app/stage04/store.py` の `ItemTimeline` と同じ
形(ロード済みデータ + そのデータに対する判断メソッド)にしてあるので、DB・
GPU・サーバーのいずれも無しで全分岐をテストできる。

依存を持たない葉モジュールにしているのは意図的で、`app/server.py`(状態の寿命
管理)と `app/rag/router.py`(検索)の双方から参照されるため。どちらかの中に
`DialogState` を置くと import が双方向になり循環する — `server.py` は
トップレベルで `pynvml.nvmlInit()` と vLLM のロードを行うので、そちらに寄せると
router のテストにGPUが要るようになる。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

# 1つの相談で同時に追いかける品目の上限。超えたら古いものから落とす。
# 上限を設けるのは、品目ごとに早見表の検索結果を context に載せるため、
# 際限なく増やすと MAX_MODEL_LEN(8192)を圧迫するから。
MAX_ITEMS = 4

# 6-4「どう調べて・どう判断したかを後から確認できる」用の記録。直近のみ保持。
MAX_TURN_LOG = 20

# --- 発話の分類に使う語 -----------------------------------------------------
# 5-2「追加なら前の案内を保ったまま足し、変更・訂正なら新しい前提で組み直す」を
# 判別するための手がかり。ここは決定的に動くヒューリスティクスに留める
# (LLMに判定させるとテストが書けず、6-3「想定外でも破綻しない」の下限も
#  確保できないため)。
_CORRECTION_WORDS = (
    "やっぱり",
    "やっぱ",
    "じゃなくて",
    "じゃなく",
    "ではなくて",
    "ではなく",
    "でなくて",
    "間違え",
    "間違い",
    "訂正",
    "に変更",
    "取り消",
    "キャンセル",
)

CORRECTION_WORDS = _CORRECTION_WORDS
"""訂正マーカー。品名抽出側(`router.extract_item_name`)も、訂正より後ろの
部分だけを品名候補にするために参照する。"""

_ADD_WORDS = (
    "追加",
    "あと",
    "それと",
    "他に",
    "ほかに",
    "ついでに",
    "一緒に",
)

ADD_WORDS = _ADD_WORDS

# 品名として採用してはいけない語。助詞・丁寧語・指示語が品名候補として
# 残ると `LIKE '%す%'` のような検索になり、無関係な早見表の行が根拠として
# 混入する(STAGE05 の実測: 「小倉北区浅野1丁目です」→ 品名「す」)。
_STOPWORD_ITEMS = frozenset(
    {
        "です", "でした", "ます", "ました", "する", "します", "ください",
        "お願い", "おねがい", "よろしく", "はい", "いいえ", "うん",
        "そう", "どう", "なに", "何", "もの", "こと", "ところ", "とき",
        "ごみ", "ゴミ", "住んで", "住んでいます", "在住",
        # 5-2 の追加・訂正マーカー自体が品名候補として残ることがある
        # (「やっぱり八幡西区黒崎です」→ 最長断片が「やっぱり」)。
        "やっぱり", "やっぱ", "訂正", "追加", "それと", "ついでに", "一緒に",
    }
)

_DEMONSTRATIVES = frozenset(
    {"それ", "これ", "あれ", "その", "この", "あの", "さっき", "先ほど", "同じ"}
)

_KANA_ONLY = re.compile(r"^[ぁ-んァ-ヶーｦ-ﾟ]+$")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


def is_demonstrative(candidate: str) -> bool:
    """「それ」「さっきの」など、前ターンの品目を指す語か。

    True なら品名として検索してはならず、状態に既にある品目を指しているものと
    して扱う(`router.extract_item_name` がここを見て None を返す)。
    """
    c = normalize(candidate)
    return bool(c) and (c in _DEMONSTRATIVES or any(c.startswith(d) for d in _DEMONSTRATIVES))


def is_plausible_item_name(candidate: str) -> bool:
    """早見表の検索キーとして投げてよい品名候補か。

    「傘」のような1文字の品目が実在するため長さだけでは弾けない。仮名1文字
    (助詞・語尾の残骸)と、機能語・指示語だけを落とす。
    """
    c = normalize(candidate)
    if not c:
        return False
    if c in _STOPWORD_ITEMS or is_demonstrative(c):
        return False
    # 仮名1文字は品名になり得ない(漢字1文字の「傘」「本」は許す)。
    if len(c) == 1 and _KANA_ONLY.match(c):
        return False
    return True


def has_correction_marker(text: str) -> bool:
    t = normalize(text)
    return any(w in t for w in _CORRECTION_WORDS)


def has_add_marker(text: str, item: str | None = None) -> bool:
    """「も」「追加」など、前の案内に足す意図を示す語があるか。

    裸の「も」は「もう捨てた」等で誤爆するため、品名に直接付いた
    「<品名>も」の形だけを見る。
    """
    t = normalize(text)
    if any(w in t for w in _ADD_WORDS):
        return True
    if item and f"{normalize(item)}も" in t:
        return True
    return False


@dataclass(frozen=True)
class TurnInput:
    """1ターンぶんの抽出結果。`router.extract_turn()` が組み立てる。

    抽出そのもの(正規表現・区名リスト)は router 側の責務で、ここは
    「抽出済みの値をどう畳み込むか」だけを扱う。
    """

    text: str = ""
    item: str | None = None
    ward: str | None = None
    town: str | None = None
    waste_type: str | None = None
    asks_schedule: bool = False
    from_image: bool = False


@dataclass(frozen=True)
class DialogState:
    """あるセッションで「今わかっていること」。

    不変オブジェクトとして扱い、更新は `merge()` が新しいインスタンスを返す
    (`ItemTimeline` と同じ方針)。同時に4リクエストが走るサーバー上で、
    途中まで書き換わった状態が他のターンから見えることを防ぐ。
    """

    items: tuple[str, ...] = ()
    ward: str | None = None
    town: str | None = None
    waste_type: str | None = None
    wants_schedule: bool = False
    """一度でも収集日を尋ねられたか。5-1 では「粗大ごみの日は?」→「小倉北区
    浅野1丁目です」のように、質問と地区が別ターンに分かれる。"""

    last_item_update: str = "none"
    """直近ターンで品目スロットに起きたこと。

    取りうる値は "add" / "replace" / "refocus" / "none" の4つ。"refocus" は
    既に追っている品目がもう一度名指しされ、`items` の末尾(=いま話題の中心)へ
    並べ替えただけの場合で、品目の増減は無い。`corrected` は "replace" だけを
    訂正とみなすので、"refocus" は 5-2 の「変更・訂正」には当たらない。
    """

    last_region_update: str = "none"
    """直近ターンで地区スロットに起きたこと。"set" / "replace" / "none"。"""

    turn_log: tuple[dict, ...] = ()

    # --- 参照系 ------------------------------------------------------------
    @property
    def primary_item(self) -> str | None:
        """いま話題の中心にある品目(最後に出てきたもの)。"""
        return self.items[-1] if self.items else None

    @property
    def has_region(self) -> bool:
        """収集日を引くのに十分な地区情報があるか(区だけでは足りない)。"""
        return bool(self.ward and self.town)

    @property
    def corrected(self) -> bool:
        """直近ターンが訂正・変更だったか。回答の組み立て方が変わる(5-2)。"""
        return self.last_item_update == "replace" or self.last_region_update == "replace"

    def describe(self) -> dict:
        """WebUIの根拠パネル・`/vram` の表示用。"""
        return {
            "items": list(self.items),
            "ward": self.ward,
            "town": self.town,
            "waste_type": self.waste_type,
            "wants_schedule": self.wants_schedule,
            "last_item_update": self.last_item_update,
            "last_region_update": self.last_region_update,
        }


def _merge_items(state: DialogState, turn: TurnInput) -> tuple[tuple[str, ...], str]:
    """品目スロットの畳み込み。戻り値は (新しいitems, 何をしたか)。

    方針:
      - 訂正マーカーがあれば直近の品目を差し替える(「ソファじゃなくて冷蔵庫」)
      - それ以外の新しい品名は**足す**。5-2 は「追加なら前の案内を保ったまま
        足す」を求めており、マーカーの無い言い換え(「冷蔵庫は?」)も追加として
        扱うほうが、前ターンの案内を失わない分だけ安全側に倒れる
      - 既に追っている品目の再登場では品目の増減は起こさず、`items` の末尾へ
        並べ替えて「いま話題の中心」にするだけ("refocus"。順序が既に末尾なら
        "none")
    """
    item = turn.item
    if not item:
        return state.items, "none"

    item = normalize(item)
    if item in state.items:
        # 既出。順序だけ最後に寄せて「いま話題の中心」にする。
        reordered = tuple(i for i in state.items if i != item) + (item,)
        return reordered, "none" if reordered == state.items else "refocus"

    if state.items and has_correction_marker(turn.text) and not has_add_marker(turn.text, item):
        return state.items[:-1] + (item,), "replace"

    return (state.items + (item,))[-MAX_ITEMS:], "add"


def _merge_region(state: DialogState, turn: TurnInput) -> tuple[str | None, str | None, str]:
    """地区スロットの畳み込み。戻り値は (区, 町名, 何をしたか)。

    区が変わったら町名は**必ず捨てる**。「小倉北区浅野1丁目」の後に
    「やっぱり八幡西区」と言われて `八幡西区 + 浅野1丁目` を引くと、存在しない
    地区の収集日を答えるか、別人の地区を答えることになる。
    """
    if turn.ward and turn.ward != state.ward:
        return turn.ward, turn.town, "replace" if state.ward else "set"

    ward = state.ward or turn.ward
    if turn.town and turn.town != state.town:
        return ward, turn.town, "replace" if state.town else "set"

    return ward, state.town or turn.town, "none"


def merge(state: DialogState | None, turn: TurnInput) -> DialogState:
    """1ターンぶんの抽出結果を状態に畳み込み、新しい状態を返す。

    ここが STAGE05 の中心。副作用も I/O も無く、入力から出力が一意に決まる。
    """
    state = state or DialogState()

    items, item_update = _merge_items(state, turn)
    ward, town, region_update = _merge_region(state, turn)

    log = state.turn_log + (
        {
            "turn": len(state.turn_log) + 1,
            "text": normalize(turn.text)[:120],
            "extracted": {
                "item": turn.item,
                "ward": turn.ward,
                "town": turn.town,
                "waste_type": turn.waste_type,
                "from_image": turn.from_image,
            },
            "item_update": item_update,
            "region_update": region_update,
            "items_after": list(items),
            "region_after": {"ward": ward, "town": town},
        },
    )

    return replace(
        state,
        items=items,
        ward=ward,
        town=town,
        waste_type=turn.waste_type or state.waste_type,
        wants_schedule=state.wants_schedule or turn.asks_schedule,
        last_item_update=item_update,
        last_region_update=region_update,
        turn_log=log[-MAX_TURN_LOG:],
    )
