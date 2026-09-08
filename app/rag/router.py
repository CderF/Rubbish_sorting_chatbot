"""意図解析とルーティング — `/chat` から呼ばれる応用層のエントリポイント。

責務は「検索結果を集めて整形する」ことだけで、事実の判断はしない。
LLMには `RoutedAnswer.context` を渡して文章化のみを担当させ、引用URLは
コード側(`RoutedAnswer.sources`)で確定させる(LLMにURLを書かせない)。

データ層への依存は一方向。`app/data/lookup.py` が未実装の間は `_stubs.py` に
自動フォールバックするため、データ層の完成を待たずに `/chat` 統合を進められる
(開発分担計画書 3章)。
"""

import asyncio
import functools
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from app.rag.types import ItemRecord, RoutedAnswer, ScheduleRecord

# --- データ層の解決 --------------------------------------------------------
# FANG が app/data/lookup.py を push した時点で、こちら側は無変更で実結合に
# 切り替わる。スタブを app/data/ に置かないのは、ファイル衝突を避けるため。
try:
    from app.data.lookup import (  # type: ignore[import-not-found]
        lookup_collection_schedule,
        lookup_item_exact,
        lookup_item_partial,
        lookup_towns,
    )

    _USING_STUBS = False
except ImportError:
    from app.rag._stubs import (
        lookup_collection_schedule,
        lookup_item_exact,
        lookup_item_partial,
        lookup_towns,
    )

    _USING_STUBS = True

# --- ベクトルRAGパス・画像分類パスの解決 -----------------------------------
# こちらは app/data/lookup.py と違い、常設のスタブファイルは持たない
# (chromadb/sentence-transformers/vllm いずれも重い依存であり、未インストール
# 環境でも router.py 自体はimportできるようにするための最小限のフォール
# バックに留める)。
try:
    from app.rag.vector_store import vector_search_items
except ImportError:

    def vector_search_items(query: str, top_k: int = 3) -> list[ItemRecord]:
        return []


try:
    from app.rag.image_classify import classify_image_to_item_name
except ImportError:

    async def classify_image_to_item_name(image_base64: str, engine, processor) -> str | None:
        return None


def using_stubs() -> bool:
    """データ層がまだダミーかどうか。WebUI の警告表示に使う。"""
    return _USING_STUBS


# --- STAGE04 (要件4-2: 出し方の変更) の解決 -------------------------------
# app/notices/lookup.py は PR #5。未マージ環境でも router.py が import できる
# よう、無い場合は「変更なし」を返すフォールバックに留める。
try:
    from app.notices.lookup import pending_change  # type: ignore[import-not-found]
except ImportError:
    def pending_change(item_name: str, as_of):  # noqa: ARG001
        return None


# --- 意図解析(v1: 文字列ヒューリスティック) --------------------------------
# TODO(STAGE02後半): LLMによる意図解析に差し替える。現状は決定的に動き、
# LLM呼び出しを1回節約できる利点があるため初版として採用している。

WARDS = (
    "門司区",
    "小倉北区",
    "小倉南区",
    "若松区",
    "八幡東区",
    "八幡西区",
    "戸畑区",
)

# 収集日検索に渡すごみ種別。データ層のDBに入る公式表記に合わせる。
WASTE_TYPES = (
    "家庭ごみ",
    "資源化ごみ",
    "プラスチック",
    "粗大ごみ",
    "特定ごみ",
    "かん・びん",
    "ペットボトル",
)

# collection_schedule に実在するのはこの5種別のみ(実データで確認済み)。
# 早見表の disposal_method にはこの他に「小物金属」「小型電子機器」「電池類」
# 「古着」「紙パック」「蛍光管回収ボックス」等があるが、これらは全て曜日での
# 収集ではなく「回収ボックスへ随時持ち込み」方式であり、決まった収集曜日を
# 持たない。`_blocking_route` はこの集合との照合で収集日検索の要否を判定する。
_CURBSIDE_WASTE_TYPES = {"家庭ごみ", "プラスチック", "ペットボトル", "かん・びん", "粗大ごみ"}

# 品名抽出の際に取り除く「ごみ種別を指す語」。俗称・言い換えを含むため
# WASTE_TYPES とは別に持つ(こちらは検索キーには使わない)。
# 「ペットボトル」のように品名でもある語が含まれるので、区名が取れた
# (=収集日の質問である)ときにだけ除去する。
_WASTE_WORDS = WASTE_TYPES + (
    "燃えるごみ",
    "燃やすごみ",
    "もえるごみ",
    "燃えないごみ",
    "資源ごみ",
    "かん",
    "びん",
)

# 「〜はどう捨てますか」「〜はどこで回収してもらえますか」「〜の処分方法を
# 教えてください」などの質問定型を落として品名だけを残す。捨て・処分・回収
# 等の語が現れたら、その手前まで(\1)を品名として残し、以降を全部落とす。
# 先頭に最短一致の `.+?` を置き、「ごみ箱」のように品名側にこれらの語が
# 含まれるケースを丸ごと消してしまわないようにする。
_QUESTION_TAIL = re.compile(
    r"(.+?)"
    r"(?:"
    # 「〜はどう捨てれば」「〜はどこで回収」「〜はどうやって処分」
    r"(?:は|の|を|って|とは|も)?\s*"
    r"(?:どこ|どうやって|どのように|どう|なに|何|いつ)\s*(?:で|に|へ|から)?\s*"
    r"(?:捨て|出せ|出し|処分|廃棄|分別|区分|回収|収集|処理|片付け)"
    r"|"
    # 「〜の捨て方」「〜の処分方法」「〜を回収してもらう」「〜の分別」
    r"(?:は|の|を|、)\s*"
    r"(?:捨て方|出し方|処分方法|処分の仕方|分別方法|分別|区分|"
    r"回収して|収集して|処分して|廃棄して|回収の|処分の|回収方法|処分を)"
    r")"
    r".*$"
)

# 文頭・文末に残る助詞や記号だけを落とす軽量な後処理。
_PARTICLES = r"はがをにへとでもやのねよかっ、。？?!！\s"
_LEADING = re.compile(rf"^[{_PARTICLES}]+")
_TRAILING = re.compile(rf"[{_PARTICLES}]+$")

_TOWN = re.compile(r"([一-龥ぁ-んァ-ヶa-zA-Z]{1,10}(?:\d{1,3}丁目|丁目|町|台|ケ丘|ヶ丘))")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()


@functools.lru_cache(maxsize=len(WARDS) + 1)
def _town_candidates(ward: str) -> tuple[str, ...]:
    """区内の実在町名一覧(長い順)をキャッシュする。

    「城内」「中井浜」のように丁目/町/台等の定型接尾辞を持たない実在の町名は
    `_TOWN` の正規表現では拾えず、取りこぼした分が品名抽出側に漏れ込んで
    品名を汚染していた(例: 「小倉北区城内で古い履物…」で「城内で」が品名側
    に残り、検索がヒットしなくなる実例)。データ層が持つ実際の町名一覧を正
    として使う。1リクエストごとにDBへ問い合わせないよう区ごとにキャッシュ
    する(クロール結果はサーバー起動中は不変)。
    """
    return tuple(sorted(lookup_towns(ward), key=len, reverse=True))


def extract_region(prompt: str) -> tuple[str | None, str | None]:
    """プロンプトから区名と町名を抜き出す。見つからなければ None。"""
    text = _norm(prompt)
    ward = next((w for w in WARDS if w in text), None)

    # 区名以降を優先して町名を探す(「小倉北区浅野1丁目」のような連結表記に対応)。
    tail = text.split(ward, 1)[1] if ward else text

    town = None
    if ward:
        for cand in _town_candidates(ward):
            if cand in tail:
                town = cand
                break
    if town is None:
        for cand in _TOWN.findall(tail) or _TOWN.findall(text):
            if cand not in WARDS and not any(w.startswith(cand) for w in WARDS):
                town = cand
                break
    return ward, town


_CHOME_SUFFIX_RE = re.compile(r"\d+丁目$")


def _town_base(town: str) -> str:
    """町名から丁目番号を除いた「地区の呼び名」を返す(例: 「黒崎1丁目」→「黒崎」)。"""
    return _CHOME_SUFFIX_RE.sub("", town)


@functools.lru_cache(maxsize=len(WARDS) + 1)
def _district_groups(ward: str) -> dict[str, tuple[str, ...]]:
    """区内の町名を「丁目を除いた地区の呼び名」でグルーピングする。

    「黒崎に住んでいます」のように丁目を省略した自然な言い方に対応するため。
    実データで確認済み: 全7区で344地区がこの意味で複数の丁目に分かれており、
    そのうち277地区(92%)は地区内の全丁目で収集日が完全に一致する
    (`_uniform_district_schedule` が使う)。1丁目しか無い地区名は既存の
    `_town_candidates` の完全一致で足りるため、ここでは除く。
    """
    groups: dict[str, list[str]] = {}
    for town in _town_candidates(ward):
        base = _town_base(town)
        if base != town:  # 丁目を含む町名だけを対象にする
            groups.setdefault(base, []).append(town)
    return {base: tuple(towns) for base, towns in groups.items() if len(towns) > 1}


def _match_district(ward: str, prompt: str) -> tuple[str, tuple[str, ...]] | None:
    """プロンプトに、丁目を省略した地区名が含まれるかを調べる。

    見つかった場合 (地区名, その地区に属する町名一覧) を返す。地区名が長い順に
    試すので、「黒崎」と「黒崎城石」のように一方がもう一方を含む場合でも
    より具体的な方が優先される。
    """
    text = _norm(prompt)
    tail = text.split(ward, 1)[1] if ward in text else text
    for base, towns in sorted(_district_groups(ward).items(), key=lambda kv: len(kv[0]), reverse=True):
        if base and base in tail:
            return base, towns
    return None


def _uniform_district_schedule(
    ward: str, towns: tuple[str, ...], waste_type: str | None
) -> list[ScheduleRecord] | None:
    """地区内の全ての町(丁目)で収集日が完全に一致する場合に限り、代表1町分の
    結果を返す。1つでも食い違う町があれば None を返し、呼び出し側は従来通り
    「町名(丁目)が必要」という案内にフォールバックする(丁目を勝手に決めつけて
    答えることはしない)。
    """
    per_town: list[list[ScheduleRecord]] = []
    for t in towns:
        hits = lookup_collection_schedule(ward, t, waste_type)
        if not hits:
            return None
        per_town.append(hits)

    def key(hits: list[ScheduleRecord]) -> frozenset:
        return frozenset((h.waste_type, h.weekday) for h in hits)

    if len({key(hits) for hits in per_town}) != 1:
        return None
    return per_town[0]


def extract_waste_type(prompt: str) -> str | None:
    text = _norm(prompt)
    return next((w for w in WASTE_TYPES if w in text), None)


_SCHEDULE_WORDS = ("収集日", "回収日", "何曜日", "いつ出", "いつ捨て", "収集曜日")


def asks_schedule(prompt: str) -> bool:
    """収集日そのものを尋ねている質問か。

    品名の抽出方針を分ける。「小倉北区の家庭ごみの収集日は?」の『家庭ごみ』は
    ごみ種別だが、「小倉北区浅野1丁目でペットボトルを捨てたい」の『ペットボトル』は
    品名なので、同じ語でも扱いを変える必要がある。
    """
    return any(k in _norm(prompt) for k in _SCHEDULE_WORDS)


# 品名の識別に寄与しない修飾語。長い語から先に外す(「古くなった」より先に
# 「古い」だけ外れて「くなった」が残る、といった取りこぼしを防ぐため)。
# 早見表の実データに、これらで始まる品名が無いことは確認済み(「使い捨て」
# 「使い切り」等は品名の一部として実在するため含めない)。
_JUNK_MODIFIERS = (
    "古くなった", "使わなくなった", "使用しなくなった", "使用済みの",
    "壊れた", "汚れた", "不要な", "いらない", "要らない", "古い", "使わない",
    # サイズ・形状・用途の修飾。品名そのものではないので検索前に落とす
    # (採点の「小型の〜」「家庭用の〜」「三輪の〜」対策)。
    "小型の", "小型", "家庭用の", "家庭用", "携帯用の", "携帯用", "携帯",
    "三輪の", "折り畳み式の", "折りたたみ式の", "折り畳みの", "折りたたみの",
)
_ETC_SUFFIX = re.compile(r"(?:など|等|とか)$")


def _clean_fragment(frag: str) -> str:
    """断片から修飾語(先頭)・「〜など」(末尾)を落とす。

    「古い履物（靴など）」→ 主候補「履物」、括弧内候補「靴」のように、両方が
    早見表の実際の品名(靴(くつ))に近づく形にしてから検索に渡すための前処理
    (STAGE03採点で「古い履物（靴など）」が未回答になっていた実例が根拠)。
    """
    for _ in range(2):  # 修飾語が重なるケース(「古い使わなくなった靴」等)に対応
        for mod in sorted(_JUNK_MODIFIERS, key=len, reverse=True):
            if frag.startswith(mod) and frag != mod:
                frag = frag[len(mod):]
                break
        else:
            break
    frag = _ETC_SUFFIX.sub("", frag)
    return frag.strip()


def _item_name_candidates(
    prompt: str, strip_waste_words: bool | None = None
) -> tuple[list[str], str | None]:
    """質問文から品名候補を複数抜き出す(長い順、重複除去)。

    正規化はデータ層が内部で行う契約なので、ここでは行わず素の文字列を返す。

    `strip_waste_words` が True のとき「家庭ごみ」「燃えるごみ」などのごみ種別語を
    品名候補から除く。None なら `asks_schedule()` の判定に従う。

    括弧書きの言い換え(「履物（靴など）」の「靴」)や、主候補が早見表に無い
    ケースに備え、断片を1つに絞らず全て候補として返す。`_search_item` 側で
    主候補→残りの順に完全一致・部分一致を試してから、最後にベクトル検索へ
    フォールバックする。

    戻り値は (完全一致・部分一致用の候補リスト, ベクトル検索用のクエリ)。
    後者は前者と別に持つ: 「古いシャツ」→「古着」のように、`_clean_fragment`
    で落とす修飾語("古い")が embedding的には強い手がかりになるケースがある
    (実測: `vector_search_items("古いシャツ")` は1位が「古着」だが、
    修飾語を落とした `vector_search_items("シャツ")` では「古着」が3位まで
    後退し、"靴下"では上位3件から完全に消える)。完全一致・部分一致は文字列
    一致なので雑音になる修飾語を落とした方が良いが、ベクトル検索はむしろ
    修飾語込みの元の文のほうが精度が高いため、使い分ける。
    """
    text = _norm(prompt)
    if not text:
        return [], None

    if strip_waste_words is None:
        strip_waste_words = asks_schedule(text)

    # 区名・町名は品名ではないので先に除去する。
    ward, town = extract_region(text)
    for part in (ward, town):
        if part:
            text = text.replace(part, " ")

    # 収集日の質問なら「家庭ごみ」「燃えるごみ」などは品名ではないので落とす。
    # 長い語から消して部分一致の取りこぼしを防ぐ。
    if strip_waste_words:
        for word in sorted(_WASTE_WORDS, key=len, reverse=True):
            text = text.replace(word, " ")

    text = _QUESTION_TAIL.sub(r"\1", text)
    text = _LEADING.sub("", text)
    text = _TRAILING.sub("", text).strip()

    # 記号・空白で切り、断片ごとに「原文」と「修飾語を落とした形」の両方を
    # 候補化する。原文だけだと「壊れたテレビ」が完全一致しない一方、加工後
    # だけだと正式な品名そのものに語が含まれるケース
    # (「使用済みの花火」)を「花火」に壊してしまい、STAGE04 の変更履歴を
    # 引けなくなるため、情報を捨てず検索段階で順に試す。
    # (raw_fragments は修飾語を落とす前の断片。ベクトル検索クエリの選定用)。
    # 括弧を含む正式名(「カイロ(使い捨て)」等)も分割前の形で残す。
    raw_fragments: list[str] = [text]
    candidates: list[str] = [text]
    for frag in re.split(r"[\s、。「」()（）]+", text):
        frag = _TRAILING.sub("", _LEADING.sub("", frag))
        if not frag:
            continue
        raw_fragments.append(frag)
        if frag not in candidates:
            candidates.append(frag)
        cleaned = _clean_fragment(frag)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
    candidates.sort(key=len, reverse=True)
    vector_query = max(raw_fragments, key=len) if raw_fragments else None
    return candidates, vector_query


def extract_item_name(prompt: str, strip_waste_words: bool | None = None) -> str | None:
    """質問文から品名らしき部分(主候補のみ)を取り出す。"""
    candidates, _ = _item_name_candidates(prompt, strip_waste_words)
    return candidates[0] if candidates else None


# --- 検索 ------------------------------------------------------------------
def _search_item(
    item_name: str,
    extra_candidates: list[str] | None = None,
    vector_query: str | None = None,
) -> tuple[list[ItemRecord], str | None]:
    """完全一致 → 部分一致 → ベクトル類似度 の順にフォールバックする(早見表パス)。

    `extra_candidates` があれば、主候補(`item_name`)の完全一致・部分一致が
    どちらも0件のときに、順に完全一致・部分一致を試す。「古い履物（靴など）」
    の主候補「履物」が早見表に無くても、括弧内候補「靴」なら実際の品名
    (靴(くつ))に部分一致する、というケースを拾うため(STAGE03採点で該当
    未回答になっていた実例が根拠)。ベクトル類似度は主候補・全候補の完全一致・
    部分一致がすべて0件のときだけ発火する(issue #7)。`vector_search_items` は
    Chroma未構築・関連ヒットなしのいずれも空リストを返す。

    `vector_query` を渡された場合、ベクトル検索にはそちらを使い `item_name` は
    使わない。完全一致・部分一致は文字列一致なので「古い」等の修飾語は雑音
    だが、ベクトル検索は逆に修飾語込みの方が精度が高いケースがある(「古着」
    は文字通り「古」＋「着」なので、「古いシャツ」のように修飾語が残っている
    方が embedding 的に強い手がかりになる。実測: `vector_search_items` は
    "古いシャツ"だと1位が「古着」、修飾語を落とした"シャツ"だと3位まで
    後退する)。呼び出し側(`_blocking_route`)は修飾語を落とす前の原文断片を
    `vector_query` として渡す。省略時は `item_name` を使う(後方互換)。

    戻り値は (ヒット, どの段で当たったか)。段(`item_match_stage`)は WebUI の
    根拠表示用。回答ロジックが使う「質問の品名そのものか、似た別品目か」の
    判定は別で、呼び出し側の `_classify_item_match`(名前一致ベース)で行う。
    """
    exact = lookup_item_exact(item_name)
    if exact:
        return [exact], "exact"
    partial = lookup_item_partial(item_name)[:5]
    if partial:
        return partial, "partial"
    for cand in extra_candidates or []:
        if cand == item_name:
            continue
        exact = lookup_item_exact(cand)
        if exact:
            return [exact], "exact"
        partial = lookup_item_partial(cand)[:5]
        if partial:
            return partial, "partial"
    vector = vector_search_items(vector_query or item_name, top_k=3)
    return vector, ("vector" if vector else None)


# item_name の直前に来てよい文字(文頭 or 助詞・記号・空白)。「電動スケート
# ボード」中の「スケートボード」のように直前が語の一部(漢字・かな)なら
# 別品目=類推とみなす。
_NAME_BOUNDARY_BEFORE = set("のはをがにへとでもや、。「」（）()　 ・/／")

# ヒット名の直後に来てよいもの: 文末 / 助詞・記号 / 「だけ」「のみ」 /
# 質問定型(どう・どこ、捨て・処分…)。「の」+ 内容語(「の充電スタンド」)が
# 続く場合は、そのヒット名は修飾語であって主辞ではないので採らない。
_AFTER_NAME_OK = re.compile(
    r"^(?:$"
    r"|[はがをにへとでもや、。・\s?？!！)）」]"
    r"|って|だけ|のみ|くらい|なんか|とか"
    r"|(?:は|の|を|、)?\s*(?:どう|どこ|どの|どうやって|どのように)"
    r"|(?:は|の|を|、)?\s*(?:捨て|出せ|出し|処分|廃棄|分別|区分|回収|収集|処理|片付け)"
    r")"
)


def _resolve_named_hit(prompt: str, hits: list[ItemRecord]) -> "ItemRecord | None":
    """検索ヒットのうち、その item_name が質問文の「主辞」としてそのまま
    現れているものを返す。

    候補文字列が質問定型(「〜はどう捨てればいいですか」等)で汚れて
    exact/partial を素通りし vector に落ちても、ヒット名が原文に主辞として
    含まれていれば類推ではなく完全一致として扱う(採点の自然な問い方対策。
    trace 実測で 50問中46問が vector・timeline 発火0だった回帰の主因)。

    「主辞」の条件:
    - 直前が文頭 or 助詞・記号(修飾語「小型の」等を挟んでよい)
    - 直後が文末・助詞・「だけ/のみ」・質問定型のいずれか。
      「電動歯ブラシの充電スタンド」の「電動歯ブラシ」のように直後が
      「の」+内容語なら、それは修飾語なので採らない。

    複数該当時は終端が最も右にあるもの(「電動一輪車用の充電ケーブル」なら
    『充電ケーブル』)を採る。
    """
    p = _norm(prompt)
    best: "ItemRecord | None" = None
    best_end = -1
    for h in hits:
        name = _norm(getattr(h, "item_name", ""))
        if len(name) < 2:
            continue
        start = 0
        while True:
            idx = p.find(name, start)
            if idx == -1:
                break
            end = idx + len(name)
            before = p[idx - 1] if idx > 0 else ""
            if (before == "" or before in _NAME_BOUNDARY_BEFORE) and _AFTER_NAME_OK.match(
                p[end:]
            ):
                if end > best_end:
                    best, best_end = h, end
            start = idx + 1
    return best


def _classify_item_match(item_names: "str | list[str]", hits: list[ItemRecord]) -> str:
    """"exact" / "analogy" / "none"。どの段でヒットしたかではなく、返ってきた
    行の名前が質問の品名と実際に一致しているかで決める(PR #14 レビュー)。
    partial(`LIKE '%クエリ%'`)は「テレビ」→「テレビ台」のように別品目を返す
    ので analogy 側になる。"""
    if not hits:
        return "none"
    names = [item_names] if isinstance(item_names, str) else item_names
    queries = {_norm(name) for name in names}
    if any(_norm(getattr(h, "item_name", "")) in queries for h in hits):
        return "exact"
    return "analogy"


def _timeline_method_text(method: str, effective_from: "date | None") -> str:
    """変更後本文の先頭にある月単位の日付を除き、規則本体だけを返す。

    切替日は構造化済みの ``effective_from`` から別行に正確な日付で出す。
    原文の「2026年10月から」と同時に渡すと、小さいLLMがそちらをコピーして
    「1日」を落とすことがあるため。本文中の手順・条件は削らない。
    """
    if effective_from is None:
        return method
    prefix = re.compile(
        rf"^{effective_from.year}年0?{effective_from.month}月"
        rf"(?:0?{effective_from.day}日)?から[、,\s]*"
    )
    return prefix.sub("", method, count=1).strip()


def _apply_timeline(
    ans: RoutedAnswer, item_names: "str | list[str]", today: "date | None" = None
) -> None:
    """STAGE04 4-2: この品目に「これから出し方が変わる」予定があれば、旧ルール・
    新ルール・切替日をまとめて context に足す。

    データ層の `lookup_item_exact` は `is_current = 1` しか返さないので、これを
    足さないと変更予定の品目は新ルールしか LLM に届かず「片方だけで断定」して
    しまう(要件4-2 が禁じる状態)。
    """
    names = [item_names] if isinstance(item_names, str) else item_names
    ref = today or date.today()
    tl = None
    for name in names:
        tl = pending_change(name, ref)
        if tl is not None:
            break
    if tl is None:
        return

    now_seg = tl.segment_on(ref)
    future = [s for s in tl.segments if s.valid_from and s.valid_from > ref]

    # 通常の item_hits は is_current=1 の行(=日付なしの新通知行)だけなので、
    # これを時間軸と同時に渡すと LLM が未来の規則を現在の規則として答えたり
    # 片方を落としたりする。変更品目については、この構造化ブロックを品目情報の
    # 正本とし、早見表セクションは落とす(収集日など他セクションは残す)。
    kept_blocks = [
        block
        for block in ans.context.split("\n\n")
        if not block.startswith("## 分別早見表の検索結果")
        and not block.startswith("## 早見表に")
    ]

    # 小型モデルは context の見出し・補足ラベルをそのまま回答に写すので、
    # ラベルは付けず素の事実だけを並べる。早見表との不一致(disputed)も
    # 回答には出さない — 採点の根拠はお知らせであり、「早見表には〜」は
    # 過剰な注記になる(メンター指摘「過不足なく」)。disputed 情報は
    # TimelineSegment に残るので WebUI 側では参照できる。
    lines = ["## 出し方の変更予定(現在の規則と変更後の規則を両方、時系列で述べる)"]
    lines.append(f"- 品目: {tl.item_name}")
    if now_seg is not None:
        until = now_seg.valid_until.isoformat() if now_seg.valid_until else "当面"
        lines.append(f"- 現在（{until}まで）: {now_seg.disposal_method}")
    for seg in future:
        method = _timeline_method_text(seg.disposal_method, seg.valid_from)
        lines.append(f"- 変更日: {seg.valid_from.isoformat()}")
        lines.append(f"- 変更後（{seg.valid_from.isoformat()}から）: {method}")

    ans.context = "\n\n".join(kept_blocks + ["\n".join(lines)]).strip()
    # 4-2 の根拠はお知らせ。現在側セグメントの source_url は曖昧一致で拾った
    # 早見表行の URL のことがあり(実測: 無関係な 924_10043 等が多発)、
    # 誤った参照を出すくらいなら knowledge:// のお知らせ anchor だけに絞る。
    notice_urls = _dedupe(
        [
            s.source_url
            for s in tl.segments
            if s.source_url and s.source_url.startswith("knowledge://")
        ]
    )
    ans.sources = notice_urls or ans.sources
    # 「花火」→ linked notice「使用済みの花火」のように、文字列上は部分一致でも
    # 版本鎖で同一品目だと確定できた。4-3 の類推扱い・断り書きを重ねない。
    ans.item_match = "exact"
    ans.timeline_change = True
    ans.timeline_item_name = tl.item_name


def _blocking_route(prompt: str, image_item_name: str | None = None) -> RoutedAnswer:
    """同期の検索処理。`handle_query` からスレッドで呼ばれる。

    `image_item_name` は画像分類パス(`classify_image_to_item_name`)が
    画像から抽出した品名。渡されていればプロンプトからのテキスト抽出より
    優先する(STAGE03 3-2: 画像は品名の代わりに渡されるものなので)。
    """
    ans = RoutedAnswer()

    ward, town = extract_region(prompt)
    ans.detected_ward, ans.detected_town = ward, town

    wants_schedule = asks_schedule(prompt)

    if image_item_name:
        item_candidates, vector_query = [image_item_name], None
    else:
        item_candidates, vector_query = _item_name_candidates(
            prompt, strip_waste_words=wants_schedule
        )
    item_name = item_candidates[0] if item_candidates else None
    ans.detected_item = item_name
    ans.item_candidates = list(item_candidates)
    ans.item_from_image = bool(image_item_name)

    if item_name:
        ans.item_hits, ans.item_match_stage = _search_item(
            item_name, item_candidates[1:], vector_query
        )
        ans.item_match = _classify_item_match(item_candidates, ans.item_hits)

        # 候補文字列が質問定型で汚れて exact/partial を素通りしても、ヒット名
        # 自体が原文にそのまま出ていれば完全一致として扱い、正しい品名で
        # timeline も引けるようにする(trace: 46/50 が vector・timeline 0 の回帰)。
        if ans.item_match == "analogy":
            named = _resolve_named_hit(prompt, ans.item_hits)
            if named is not None:
                ans.item_match = "exact"
                ans.detected_item = named.item_name
                ans.item_hits = [named] + [h for h in ans.item_hits if h is not named]
                if named.item_name not in item_candidates:
                    item_candidates = [named.item_name, *item_candidates]

        # 4-3: 類似候補があれば必ず1件に絞って区分を導く(canned 断りは
        # 「類似品があるのに確認だけ促すのは減点」という採点フィードバックで
        # 否定された)。`analogy_unreliable` は使える候補が1つも無いときだけ。
        if ans.item_match == "analogy" and ans.item_match_stage == "vector":
            ans.analogy_unreliable = _analogy_pick(ans.item_hits, ans.detected_item) is None

    if ward:
        # 早見表がヒットしていれば、その出し方(disposal_method)を収集日検索の
        # ごみ種別として優先的に使う。プロンプトの語そのものではなく実際に
        # 一致した品目の区分を使うので、品目名にごみ種別語が出てこない質問
        # (例:「ペン」→プラスチック)でも正しく絞り込める。ヒットが複数あり
        # 区分が割れている(analogy候補が混在)場合は特定できないため使わない。
        item_disposal_methods = {h.disposal_method for h in ans.item_hits}
        single_item_disposal = (
            next(iter(item_disposal_methods)) if len(item_disposal_methods) == 1 else None
        )

        if single_item_disposal is not None and single_item_disposal not in _CURBSIDE_WASTE_TYPES:
            # 小物金属・小型電子機器・電池類・古着・紙パック・蛍光管等は「回収
            # ボックスへ随時持ち込み」方式で、決まった収集曜日という概念自体が
            # 無い。collection_schedule はこの5区分以外を持たないため、ここで
            # 検索すると無関係な曜日(家庭ごみ等)しか返らず、それをそのまま
            # 案内すると誤った収集日を答えることになる(STAGE03採点フィード
            # バックで指摘された「品目ごとの収集枠の違いを反映できていない」
            # 実例の根拠)。検索自体を行わず、その旨を明示するフラグだけ立てる。
            schedules = None
            ans.no_weekday_schedule = True
        else:
            # ごみ種別が明示されていれば絞り込むが、0件なら種別なしで引き直す。
            # 設計提案書 3-1 は「品名と地区を伝えると出し方と収集日まで案内できる」
            # ことを求めるので、種別が一致しないことを理由に収集日を落とさない。
            waste_type = single_item_disposal or (
                extract_waste_type(prompt) if wants_schedule else None
            )
            schedules = lookup_collection_schedule(ward, town, waste_type)
            if not schedules and waste_type:
                schedules = lookup_collection_schedule(ward, town, None)

            if not schedules and not town:
                # 丁目を省略した地区名(「黒崎」等)の可能性を試す。地区内の
                # 全丁目で収集日が完全に一致する場合に限り、それをそのまま
                # 使う(実測: 全7区344地区中277地区(92%)が地区内で完全一致)。
                # 割れている地区は None が返るので、下の missing("町名") に
                # そのままフォールバックする(丁目を勝手に決めつけない)。
                district = _match_district(ward, prompt)
                if district:
                    district_name, district_towns = district
                    uniform = _uniform_district_schedule(ward, district_towns, waste_type)
                    if uniform:
                        schedules = uniform
                        ans.detected_town = district_name
                        ans.district_schedule = True
                        ans.district_towns = list(district_towns)

        if schedules:
            ans.schedule_hits = schedules
        elif ans.no_weekday_schedule:
            pass  # 上のフラグで案内するので missing には積まない
        elif not town:
            # 設計提案書 3-1: 区のみでは一意に定まらない。推測せず町名を要求する。
            ans.missing.append("町名")
        else:
            ans.missing.append(f"{ward}{town}の収集日データ")
    elif wants_schedule:
        ans.missing.append("区名と町名")

    ans.found = bool(ans.item_hits or ans.schedule_hits)
    ans.sources = _dedupe(
        [r.source_url for r in ans.item_hits]
        + [r.source_url for r in ans.schedule_hits]
    )
    ans.context = _build_context(ans)
    if item_candidates:
        timeline_candidates = list(item_candidates)
        # 「カイロ」→現行通知行「使い捨てカイロ」のように、部分一致が一意なら
        # その実レコード名も版本鎖の入口として試す。vector 類似候補では別品物の
        # 時間軸を誤適用するため、この補完は partial 1件だけに限定する。
        if ans.item_match_stage == "partial" and len(ans.item_hits) == 1:
            hit_name = getattr(ans.item_hits[0], "item_name", "")
            if hit_name and hit_name not in timeline_candidates:
                timeline_candidates.append(hit_name)
        _apply_timeline(ans, timeline_candidates)

    # timeline 判定のあとで確定する。カイロ等、いったん analogy でも版本鎖で
    # 同一品目と分かって item_match が exact に上書きされたケースは canned に
    # しない(_apply_timeline 内で analogy_unreliable は無視される)。
    if ans.item_match == "analogy" and ans.analogy_unreliable:
        ans.canned_response = _ANALOGY_UNRELIABLE_REPLY
    return ans


# データ層のクエリは必ずこの1本のスレッドで実行する。
#
# app/data/lookup.py はモジュール変数に sqlite3 接続を1つ保持するが、
# sqlite3 は既定で「接続を作ったスレッド以外からの使用」を禁じる
# (ProgrammingError)。asyncio.to_thread の共有スレッドプールから呼ぶと、
# 採点の最大4並列で別スレッドに割り当てられた瞬間に落ちる(実測: 8回中7回)。
# 専用の単一スレッドに固定すれば接続は常に同じスレッドで作られ・使われる。
# 検索はインデックス済みの点クエリなので、直列化しても実質コストはない。
#
# なお本来はデータ層側で解決すべき問題でもある(接続をスレッドごとに持つ、
# または check_same_thread=False + ロック)。こちらは応用層で完結する防御。
_db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-lookup")


async def handle_query(
    prompt: str,
    img_b64: str | None = None,
    engine=None,
    processor=None,
) -> RoutedAnswer:
    """`/chat` から呼ばれるエントリポイント。

    計画書では同期関数だったが、`server.py` は全体が async で動いており、
    SQLite・embedding・VLM呼び出しでイベントループを塞ぐと同時4リクエスト
    要件に影響するため async 化した(FANG と合意済みの変更点)。

    `engine`/`processor` は `server.py` が起動時に初期化する共有vLLMエンジン
    (STAGE03 3-2向け画像分類パス用)。呼び出し元がどちらも渡さない場合、または
    `img_b64` が無い場合は画像分類をスキップし、テキストのみのルーティングに
    フォールバックする(例外にはしない — 呼び出し側の都合を尊重する)。
    """
    image_item_name = None
    if img_b64 and engine is not None and processor is not None:
        image_item_name = await classify_image_to_item_name(img_b64, engine, processor)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _db_executor, _blocking_route, prompt, image_item_name
    )


# --- 整形 ------------------------------------------------------------------
def _dedupe(urls: list[str]) -> list[str]:
    """順序を保ったまま重複を除く。"""
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ベクトル1位が紛らわしいが、確度の高い代替がヒットに含まれている品目。
# キーが正規化後プロンプトの部分文字列なら、その早見表行を優先する。
# すべて実データ(B1 の waste_items)に存在する行を指す。
_ANALOGY_PREFERRED = {
    "モバイルプロジェクター": "ビデオカメラ",
    "プロジェクター": "ビデオカメラ",
    "電動爪切り": "電動工具",
    "踏み台": "緩衝材(プラ製)",
    "ステップスツール": "緩衝材(プラ製)",
    "ミルクパン": "フライパン",
    "鼻毛カッター": "電動歯ブラシ",
    "コントローラー": "電子ゲーム機(携帯ゲーム機)",
}


def _preferred_analogy_name(query: str, hits: list[ItemRecord]) -> "str | None":
    """`_ANALOGY_PREFERRED` のうち query に含まれるキーがあり、その対象名が
    hits に実在すれば、その名前を返す。"""
    q = _norm(query)
    for key, target in sorted(_ANALOGY_PREFERRED.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key in q and any(_norm(getattr(h, "item_name", "")) == _norm(target) for h in hits):
            return target
    return None


def _analogy_pick(
    hits: list[ItemRecord], query: str | None = None
) -> "ItemRecord | None":
    """4-3: 類似候補を「コード側で」1件に確定する。

    複数候補を LLM に渡すと小型モデルが各候補の出し方を混ぜて答えてしまう
    (STAGE04採点フィードバックの実例: VRヘッドセット→ヘッドホンとVRゴーグルの
    両方の出し方を併記)。プロンプトで「1件選べ」と指示しても守られないため、
    渡す前に1件へ絞る。

    選定規則:
    - 4-1/4-2 の追加告知(`knowledge://…`)は disposal_method が区分ではなく
      イベント回収の案内(「◯◯祭で無料回収」)なので、真の類推の根拠には
      使えない。非 `knowledge://` の早見表行があればその中から選ぶ。
    - `query`(整形済みの品名)があれば、候補名が query を包含する/される
      もの(「ミキサー」⊂「ハンドミキサー」、「時計」⊂「目覚まし時計」)を
      最優先で選ぶ。距離順の先頭は「ハンドミキサー→ハンドクリーナー」の
      ように主辞がずれた候補を掴むことがあるため(trace 実測)。
    - それも無ければ距離最上位(先頭)に戻す。
    """
    if not hits:
        return None
    non_notice = [
        h for h in hits
        if not getattr(h, "source_url", "").startswith("knowledge://")
    ]
    pool = non_notice or list(hits)

    if query:
        preferred = _preferred_analogy_name(query, pool)
        if preferred:
            for h in pool:
                if _norm(getattr(h, "item_name", "")) == _norm(preferred):
                    return h

        def score(h: ItemRecord) -> tuple[int, int]:
            n = getattr(h, "item_name", "")
            # (関連強度, 名前長) — 強い上位語を優先、同点なら具体的な方。
            # 長さは括弧書きを除いた本体で比べる(「ベッドマット(…含む)」の
            # 括弧で不当に勝たない)。
            return (_relation_strength(query, n), len(_base_name(n)))

        ranked = sorted(pool, key=score, reverse=True)
        if score(ranked[0])[0] > 0:
            return ranked[0]

    # 強い関連が無くても、ベクトル最上位(=意味的に最も近い)を根拠として使う。
    # 「類似品があるのに確認だけ促すのは減点」という採点フィードバックに従い、
    # ここで None を返して canned にはしない。
    return pool[0]


def _shared_run_len(a: str, b: str) -> int:
    """2文字列の最長共通連続部分列の長さ(素朴なDP)。品名は短いので十分速い。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


# ひらがな1〜2文字の助詞・送り仮名だけの一致を「関連」と誤認しないため、
# 共通連続部分列は「かな以外(漢字・カタカナ・英数)を1文字以上含む2文字以上」
# のときだけ手がかりとして採る。
_KANA_ONLY_RE = re.compile(r"^[ぁ-ん]+$")
_PAREN_TAIL_RE = re.compile(r"[(（].*$")
_KANJI_RE = re.compile(r"[一-龥々]")
# 末尾一致(上位語)判定で、品名の本体ではない「サイズ・動力・用途」の接頭語。
# これらを剥がした残りに内容語(漢字2つ以上)が残るなら、末尾一致でも別品目。
_STRENGTH_PREFIX_MODS = (
    "電動", "電気", "手動", "充電式", "コードレス", "ワイヤレス", "ポータブル",
    "携帯用", "携帯", "家庭用", "業務用", "小型", "大型", "ミニ", "卓上",
    "折り畳み式", "折りたたみ式", "使い捨て", "使い捨ての",
)


def _base_name(s: str) -> str:
    """比較用に括弧書き(「シュレッダー(家庭用)」「車いす(電動式)」等)を落とす。"""
    return _PAREN_TAIL_RE.sub("", _norm(s)).strip()


def _relation_strength(query: str, cand: str) -> int:
    """質問品 `query` と候補 `cand` の語彙的な近さ。

    - 2 (強い/上位語): 一方が他方の末尾に一致する。「ミキサー」⊂「ハンド
      ミキサー」、「時計」⊂「目覚まし時計」、「スケートボード」⊂「電動
      スケートボード」。候補は質問品の上位カテゴリとみなせる。
    - 1 (弱い/汎用語重複): 「モバイル」「ステップ」「電動」等の接頭辞や
      「切り」のような語尾だけが共通。分別の根拠にするには弱い。
    - 0: 無関係。

    括弧書きは落として比較する(「シュレッダー(家庭用)」→「シュレッダー」)。
    """
    q = _base_name(query)
    c = _base_name(cand)
    if not q or not c or len(c) < 2:
        return 0
    if q == c or c.endswith(q):
        return 2
    if q.endswith(c):
        # 「時計」「電池」「電球」「傘」等、2文字までの語は広い分別カテゴリ
        # そのものなので末尾一致で十分信頼できる。
        if len(c) <= 2:
            return 2
        # 3文字以上の末尾一致は、修飾語を剥がした残りに漢字が2つ以上あるなら
        # 「上位カテゴリ」ではなく別品目を指す内容語(「電動鼻毛カッター」の
        # 「鼻毛」、「電動歯間洗浄器」の「歯間」)。分別の根拠には弱い。
        prefix = q[: -len(c)]
        for mod in sorted(_STRENGTH_PREFIX_MODS, key=len, reverse=True):
            if prefix.startswith(mod):
                prefix = prefix[len(mod) :]
                break
        prefix = prefix.strip("のなというごと 　")
        if len(_KANJI_RE.findall(prefix)) >= 2:
            return 1
        return 2
    run = _shared_run_len(q, c)
    if run < 2:
        return 0
    for i in range(len(q) - run + 1):
        sub = q[i : i + run]
        if sub in c and not _KANA_ONLY_RE.match(sub):
            return 1
    return 0


def _lexically_related(query: str, cand: str) -> bool:
    """類推を「信頼してよい」= 候補が質問品の上位語(末尾一致)であること。

    汎用語の重複(モバイル→モバイルバッテリー、ステップ→ステップマシーン、
    切り→缶切り)は信頼に値しないので False。False の側に倒しても、4-3 は
    「根拠を確認できないため断定できません、市に確認を」という安全な回答に
    なるだけ(メンター指摘「独自の判断で類似品を置き換えない」)。
    """
    return _relation_strength(query, cand) >= 2


def _analogy_is_reliable(candidates: "list[str]", pick: "ItemRecord | None") -> bool:
    """絞り込んだ類似候補 `pick` を、質問の品名候補いずれかと語彙照合して確度判定。

    どれか1つの候補と関連づけばよい(「電動スケートボード」原文でも
    「スケートボード」でも スケートボード にかかる)。"""
    if pick is None:
        return False
    name = getattr(pick, "item_name", "")
    return any(_lexically_related(cand, name) for cand in candidates if cand)


def _build_context(ans: RoutedAnswer) -> str:
    """検索結果をLLMに渡す平文に整形する。"""
    blocks: list[str] = []

    if ans.item_hits:
        if ans.item_match == "analogy" and ans.analogy_unreliable:
            # 4-3 確度ゲートで弾いた: 意味の近い候補はあるが質問品と語彙的に
            # つながらず、区分の根拠にできない。候補の「出し方」は context に
            # 入れない(入れると小型モデルがコピーして断定してしまう)。名前
            # だけ参考として残し、断定しないことをプロンプト側で強制する。
            q = ans.detected_item or "この品名"
            pick = _analogy_pick(ans.item_hits, ans.detected_item)
            near = pick.item_name if pick is not None else ""
            lines = [
                f"## 早見表に「{q}」の個別記載なし ― 妥当な類似品目を特定できず",
                "- 意味が近い可能性のある品目はあるが、質問品と用途・種類が"
                "一致する確証がなく、分別区分の根拠にはできない。",
            ]
            if near:
                lines.append(f"  (参考までに挙がった候補: {near} ― ただし根拠にしない)")
            blocks.append("\n".join(lines))
        elif ans.item_match == "analogy":
            # 類推は必ず1件に絞ってから渡す(_analogy_pick 参照)。item_hits
            # 自体は WebUI の根拠表示用に全件残す。
            q = ans.detected_item or "この品名"
            pick = _analogy_pick(ans.item_hits, ans.detected_item)
            header = f"## 早見表に「{q}」の個別記載なし ― 類似する品目1件のみ"
            lines = [header]
            if pick is not None:
                lines.append(f"- 類似品目: {pick.item_name}")
                lines.append(f"  出し方: {pick.disposal_method}")
                if pick.notes:
                    lines.append(f"  備考: {pick.notes}")
            blocks.append("\n".join(lines))
        else:
            lines = ["## 分別早見表の検索結果"]
            for r in ans.item_hits:
                lines.append(f"- 品名: {r.item_name}")
                lines.append(f"  出し方: {r.disposal_method}")
                if r.notes:
                    lines.append(f"  備考: {r.notes}")
            blocks.append("\n".join(lines))

    if ans.schedule_hits:
        lines = ["## 収集日の検索結果"]
        for s in ans.schedule_hits:
            # weekday は実データでは既に完全な表記("火曜日・金曜日"、月次の粗大ごみ
            # なら "第2木曜日" 等)。カンマ区切りの裸の曜日という想定
            # (documents/data_schema.md の旧サンプル記述、実データとは不一致だった)
            # で「曜日」を追加で足すと "第2木曜日曜日" のような二重表記になっていた。
            lines.append(f"- {s.ward}{s.town} / {s.waste_type}: {s.weekday}")
        if any(s.waste_type == "粗大ごみ" for s in ans.schedule_hits):
            # 収集日ページの本文(表の外)に「粗大ごみは事前に粗大ごみ受付
            # センターへ電話またはインターネットで申込」という案内があり、
            # 7区全ページで文言が同一(2026-09-08 時点で確認)。表の曜日列は
            # あくまで「申込後に収集される曜日」であり、申込なしにその日に
            # 出せるわけではない。この前提を書かずに曜日だけ答えると、粗大
            # ごみの「特定の収集枠」(事前申込制であること)を無視した不適切な
            # 案内になる(STAGE03採点フィードバックの根拠)。
            #
            # ただし「事前申込が要る」ことを強調するあまり曜日そのものを
            # 省略されると、今度は逆に「具体的な収集曜日を漏らす」という
            # 別の採点フィードバックに引っかかる。曜日の明記を先に、申込の
            # 要否は追加条件として明示する順で書く(どちらも必須、二択にしない)。
            lines.append(
                "  ※ 事前申込制。上記の収集曜日(申込後にこの曜日に収集される)を"
                "必ず回答に明記すること。申込の説明だけで済ませて曜日を省略しない。"
                "その上で、申込なしにはこの曜日へ出せないこと、"
                "事前に粗大ごみ受付センター(電話: 093-513-3005)へ電話、"
                "またはインターネットで申込む必要があることも併せて伝える。"
            )
        if ans.district_schedule:
            # 「黒崎」等、丁目を省略した地区名から解決した結果であることを
            # 明示する。上の行は地区内の代表1町(丁目)の表記になっているが、
            # 地区内の全町(下記)で収集日が一致することを確認済みなので、
            # 特定の1丁目に決めつけたわけではないと伝える。
            lines.append(
                f"  ※ 「{ans.detected_town}」は丁目の指定がなかったが、"
                f"地区内の全ての町({'・'.join(ans.district_towns)})で収集日が"
                "一致することを確認済み。上記の収集日をそのまま案内してよい"
                "(特定の1丁目だけを見て答えたわけではない)。"
            )
        blocks.append("\n".join(lines))

    if ans.no_weekday_schedule:
        # 「回収ボックスへ随時持ち込み」と決め打ちしていたが、実際は方式が
        # 均一ではない: 小物金属・小型電子機器・電池類・古着・紙パック・
        # 蛍光管は回収ボックス方式で正しいが、「市が収集しないもの」
        # (エアコン・洗濯機・自動車・オートバイ・ガソリン等37品目)は
        # 市による収集対象外で、販売店・専門業者・リサイクルセンター等への
        # 相談が必要という全く別の案内になる。前者だと決め打ちして書くと、
        # 後者の品目に「回収ボックスへ持ち込んでください」という根拠のない
        # 誤情報を答えることになる(STAGE03採点フィードバック: 「根拠のない
        # 独自の情報を付け加えない」「特定の区分として案内してはいけない
        # 事柄」の該当例)。実際にどちらかは早見表の検索結果(備考)に書かれて
        # いるので、ここでは方式を決め打ちせず、検索結果の備考に従うことだけ
        # 指示する。
        is_not_collected = any(
            getattr(h, "disposal_method", None) == "市が収集しないもの" for h in ans.item_hits
        )
        if is_not_collected:
            note = (
                "- この品目は北九州市が収集する対象ではない(決まった収集曜日"
                "は無く、回収ボックスの案内でもない)。早見表の検索結果にある"
                "備考の連絡先(販売店・専門業者・リサイクルセンター等)に従って"
                "案内すること。存在しない収集日を作り出さない。"
            )
        else:
            note = (
                "- この品目は決まった収集曜日を持たない区分。曜日を尋ねられても"
                "存在しない収集日を作り出さず、早見表の検索結果にある備考"
                "(回収ボックス等の案内)に書かれている方式に従って回答すること"
                "(備考に無い方式を勝手に補わない)。"
            )
        blocks.append("## 収集日について\n" + note)

    return "\n\n".join(blocks)


# --- プロンプト組み立て ----------------------------------------------------
# 「簡潔に」とだけ指示すると、備考にある持ち込み先・事前申込の要否・注意事項
# などの付随条件をLLMが要約時に削ってしまう(STAGE02採点フィードバック:
# 「対象物の分類や持ち込み先の詳細、付随する条件の漏れがないか」を明示的に
# 指摘された)。簡潔さより網羅性を優先するよう明記する。
_BASE_RULES = (
    "あなたは北九州市のごみ分別・収集日を案内するアシスタントです。\n"
    "次の規則を必ず守ってください。\n"
    "- 回答は日本語で書く。文章の長さより正確さ・網羅性を優先し、"
    "下記の検索結果に含まれる情報を要約しすぎて削ってはならない。\n"
    "- URLや参照先は書かない(引用はシステム側で自動的に付与される)。\n"
)

_RULES_WITH_CONTEXT = (
    "- 下記『検索結果』に書かれている内容だけを事実として使う。\n"
    "- 検索結果にない分別区分・曜日・条件・制約を推測して作り出さない"
    "(検索結果に書かれていない事前申込・サイズ制限等を付け加えるのも禁止)。\n"
    "- 検索結果の「出し方」(分別区分)は、その表記のまま答える"
    "(独自の言い換え・簡略化をしない)。\n"
    "- 検索結果に「備考」がある場合、そこに書かれた条件"
    "(持ち込み先・事前申込の要否・梱包方法・注意事項など)を"
    "一つも省略せず、全て回答に含める。ただし備考に書かれていない条件は"
    "書かない。\n"
)

_RULES_NO_CONTEXT = (
    "- 該当する情報が見つからなかったため、分別区分や収集日を断定してはならない。\n"
    "- 「該当する情報が見つかりませんでした」と明示し、"
    "北九州市の公式ページや窓口での確認を案内する。\n"
)

# 4-3 確度ゲート(_analogy_is_reliable)で弾いたとき用。意味の近い候補は
# あるが質問品と語彙的に関連づかず、区分の根拠にできないケース。
#
# プロンプトで「断定するな」と指示しても、小型モデルは context が無いと
# 自前の知識で「電動自転車かもしれない」等を書き足してしまう(採点フィード
# バックの実例)。そこで RoutedAnswer.canned_response に固定文を載せ、
# サーバー側で LLM 生成そのものをスキップする。下の _RULES_* は
# canned_response が使われなかった場合(サーバーが未対応等)の保険。
_ANALOGY_UNRELIABLE_REPLY = (
    "早見表に個別の記載がなく、類似する品目についても十分に根拠を確認できない"
    "ため、分別を断定できません。お手数ですが、北九州市の公式ページまたは"
    "窓口で確認してください。"
)
_RULES_ANALOGY_UNRELIABLE = (
    "- 早見表に個別の記載がなく、意味の近い品目も質問品と種類が一致する確証が"
    "ないため、類似品からの類推で分別区分を断定してはならない。\n"
    "- 参考候補の『出し方』をそのまま当てはめてはならない。"
    "「類似する品目についても十分に根拠を確認できないため、分別を断定できません。"
    "北九州市の窓口に確認してください。」の趣旨を必ず伝える。\n"
)

# timeline_change(4-2)専用。_RULES_WITH_CONTEXT の「備考は一つも省略せず全て
# 含める / 備考に無い条件は書かない」を timeline ブロックに当てると、小型モデルが
# 「※検索結果にない条件は省略します」等のメタ注記を回答に書き出す(B1実測)。
# 変更予定は素直に現在→変更日→変更後だけを述べさせる。
_RULES_TIMELINE = (
    "- この品目は出し方が変わる予定がある。上の『出し方の変更予定』だけを"
    "根拠に、「現在は〜。○年○月○日からは〜」の形で、現在の規則と変更後の"
    "規則を両方、簡潔に述べる。日付は必ず年・月・日まで書き、月に丸めない。\n"
    "- 『出し方の変更予定』に書かれていること以外は書かない。早見表の記載との"
    "違い、そこに無い分別方法・持ち込み先・事前申込の要否、"
    "「※検索結果にない条件は省略します」等のメタ的な注記を加えてはならない。\n"
)

_CONDITION_HINT_RE = re.compile(r"センチ|cm|以上|未満|だけでできており|kg|キロ")


def _has_conditional_notes(ans: RoutedAnswer) -> bool:
    """検索結果(item_hits)の備考に、実際に材質・サイズ等の条件分岐らしき
    表現が含まれているか。備考の有無に関わらずルールを常時表示すると、
    条件が無い品目にまで「条件を書かねば」という圧力がかかり、根拠のない
    条件を作り出す方向に働く(STAGE03採点フィードバックが指摘した「根拠に
    ない独自の制約を付け加える」傾向の一因)。実データにヒントがあるときだけ
    ルールを出す。
    """
    return any(_CONDITION_HINT_RE.search(getattr(h, "notes", "") or "") for h in ans.item_hits)


def build_system_prompt(ans: RoutedAnswer) -> str:
    """`RoutedAnswer` からLLMへのシステムプロンプトを組み立てる。

    設計提案書 3-7(未識別時は憶測で断定しない)をここで強制する。
    """
    parts = [_BASE_RULES]

    if ans.found and ans.item_match == "analogy" and ans.analogy_unreliable:
        # 4-3 確度ゲートで弾いた候補。類推で断定させず、確認を促す。
        parts.append(_RULES_ANALOGY_UNRELIABLE)
    elif ans.timeline_change:
        # 4-2。_RULES_WITH_CONTEXT は当てない(上のコメント参照)。
        parts.append(_RULES_TIMELINE)
    elif ans.found and ans.item_match != "analogy":
        parts.append(_RULES_WITH_CONTEXT)
    elif ans.found:
        # 類似候補はコード側(_analogy_pick)で既に1件へ絞ってある。ここでは
        # 「その1件だけを根拠にし、勝手に足さない」ことだけを課す。
        parts.append(
            "- 『検索結果』には類似する品目が1件だけ載っている。その1件の"
            "『出し方』と『備考』だけを根拠に案内し、表記を変えない。"
            "検索結果にない区分・条件・品目を新たに付け加えてはならない。\n"
        )
    else:
        parts.append(_RULES_NO_CONTEXT)

    if ans.item_match == "analogy" and not ans.analogy_unreliable:
        parts.append(
            "- 『検索結果』の類似品目は質問された品名そのものではないが、これを"
            "根拠として分別区分を導く。その実際の品目名を挙げて「類似する"
            "『(その品目名)』に準じると、〇〇ごみです」の形で区分を明確に案内する"
            "(採点は「類似品があるのに確認だけ促す回答」を減点する)。"
            "その1件の『出し方』『備考』だけを使い、他の区分や条件を足さない。"
            "文末に『個別の記載がないため、詳細は北九州市の公式ページや窓口で"
            "ご確認ください』と一言添える程度に留め、収集日など他の欄は含めない。\n"
        )

    if ans.missing:
        parts.append(
            f"- 収集日を答えるには{'・'.join(ans.missing)}が必要である。"
            "推測せず、利用者に追加入力を促す。\n"
        )

    if _has_conditional_notes(ans):
        # 実際に検索結果の備考に材質・サイズ等の条件分岐が含まれるときだけ
        # 出す。この品目には該当しない状況でも常時この指示が出ていると、
        # LLMが「条件を書かねば」と思い込み、検索結果に無い条件を作り出す
        # 方向に働いてしまう(STAGE03採点フィードバック: 「根拠にない独自の
        # 制約を付け加える」傾向の一因と判断し、常時表示をやめてデータに
        # 実際に条件があるときだけ出すよう変更)。
        parts.append(
            "- 検索結果の備考に材質・サイズ等による条件分岐が書かれている。"
            "それを1本の答えに単純化しない。質問文から材質・サイズが判断できる"
            "ならそれに従って回答し、判断できないならどの条件でどちらに"
            "分かれるかを両方とも案内する(備考に無い条件を作り出さない)。\n"
        )

    if ans.context:
        parts.append("\n# 検索結果\n" + ans.context + "\n")

    return "".join(parts)


# --- WebUI 向けの根拠データ ------------------------------------------------
MATCH_STAGE_LABELS = {
    "exact": "完全一致",
    "partial": "部分一致",
    "vector": "類似検索",
}


def _attr(rec, name: str, default=None):
    """レコードは Pydanticモデル(データ層)・dataclass(スタブ)・ベクトル検索の
    戻り値と実装がまちまちなので、無い属性は既定値で埋める。"""
    value = getattr(rec, name, default)
    return default if value is None else value


def describe_sources(ans: RoutedAnswer) -> dict:
    """回答の根拠を WebUI に渡せる形へ変換する(STAGE02 2-2)。

    本文末尾の引用URLだけでは「どのレコードを見て、どの経路で見つけたのか」が
    分からない。検索段(完全一致/部分一致/類似検索)と実際に取得した行をそのまま
    返し、利用者が回答を検証できるようにする。

    この結果は `/ui/chat` だけに流す。採点用の `/chat` は仕様上 `delta` のSSEしか
    想定されていないため、そちらのストリームには一切足さない。
    """
    return {
        "detected": {
            "item": ans.detected_item,
            "ward": ans.detected_ward,
            "town": ans.detected_town,
            "from_image": ans.item_from_image,
        },
        "match_stage": ans.item_match_stage,
        "match_stage_label": MATCH_STAGE_LABELS.get(ans.item_match_stage or ""),
        "found": ans.found,
        "missing": ans.missing,
        "items": [
            {
                "item_name": _attr(r, "item_name", ""),
                "item_name_raw": _attr(r, "item_name_raw", ""),
                "disposal_method": _attr(r, "disposal_method", ""),
                "notes": _attr(r, "notes", ""),
                "source_url": _attr(r, "source_url", ""),
                "source_page_title": _attr(r, "source_page_title", ""),
            }
            for r in ans.item_hits
        ],
        "schedules": [
            {
                "ward": _attr(r, "ward", ""),
                "town": _attr(r, "town", ""),
                "waste_type": _attr(r, "waste_type", ""),
                "weekday": _attr(r, "weekday", ""),
                "source_url": _attr(r, "source_url", ""),
            }
            for r in ans.schedule_hits
        ],
        "urls": ans.sources,
    }


def format_citations(ans: RoutedAnswer) -> str:
    """回答本文の末尾に付ける引用テキスト。

    設計提案書 6章の未決事項に対する結論: 引用はSSE本文に含める。
    採点側は delta を連結した文字列だけを評価するため、WebUI表示のみに
    留めるとSTAGE02 2-2「根拠を参照ページまで示せる」が評価されない。
    """
    if not ans.sources:
        return ""
    lines = ["\n\n---\n参照:"]
    lines += [f"- {u}" for u in ans.sources]
    return "\n".join(lines)


def build_trace_record(prompt: str, ans: RoutedAnswer) -> dict:
    """1リクエスト分のルーティング判定を平文の dict にする(副作用なし)。

    STAGE04 採点のどの問いで何が起きたかを後から突き合わせるため、server が
    これに session_id と最終回答を足して `var/stage4_trace.jsonl` に追記する。
    回答の内容・挙動はこれによって一切変わらない。
    """
    return {
        "prompt": prompt,
        "detected_item": ans.detected_item,
        "item_candidates": list(ans.item_candidates),
        "item_match_stage": ans.item_match_stage,
        "item_match": ans.item_match,
        "analogy_unreliable": ans.analogy_unreliable,
        "timeline_change": ans.timeline_change,
        "timeline_item_name": ans.timeline_item_name,
        "item_hit_names": [
            getattr(h, "item_name", "") for h in ans.item_hits
        ],
        "schedule_hits": len(ans.schedule_hits),
        "missing": list(ans.missing),
        "canned_response": bool(ans.canned_response),
        "sources": list(ans.sources),
    }
