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

# 「〜はどう捨てますか」「〜の分別」などの質問定型を落として品名だけを残す。
_QUESTION_TAIL = re.compile(
    r"(?:は|の|を|って|とは)?\s*"
    r"(?:どうやって|どのように|どう|なに|何|いつ)?\s*"
    r"(?:捨て(?:方|る|たい|ます|られ)|出し方|出せ|分別|処分(?:方法|の仕方)?|"
    r"ごみ|ゴミ|区分|収集日|回収日|収集|回収)"
    r"[^ぁ-んァ-ヶ一-龥]*.*$"
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

    text = _QUESTION_TAIL.sub("", text)
    text = _LEADING.sub("", text)
    text = _TRAILING.sub("", text).strip()

    # 記号・空白で切り、断片ごとに修飾語を落として候補化する
    # (raw_fragments は修飾語を落とす前の断片。ベクトル検索クエリの選定用)。
    raw_fragments: list[str] = []
    candidates: list[str] = []
    for frag in re.split(r"[\s、。「」()（）]+", text):
        frag = _TRAILING.sub("", _LEADING.sub("", frag))
        if not frag:
            continue
        raw_fragments.append(frag)
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


def _classify_item_match(item_name: str, hits: list[ItemRecord]) -> str:
    """"exact" / "analogy" / "none"。どの段でヒットしたかではなく、返ってきた
    行の名前が質問の品名と実際に一致しているかで決める(PR #14 レビュー)。
    partial(`LIKE '%クエリ%'`)は「テレビ」→「テレビ台」のように別品目を返す
    ので analogy 側になる。"""
    if not hits:
        return "none"
    q = _norm(item_name)
    if any(_norm(getattr(h, "item_name", "")) == q for h in hits):
        return "exact"
    return "analogy"


def _apply_timeline(ans: RoutedAnswer, item_name: str, today: "date | None" = None) -> None:
    """STAGE04 4-2: この品目に「これから出し方が変わる」予定があれば、旧ルール・
    新ルール・切替日をまとめて context に足す。

    データ層の `lookup_item_exact` は `is_current = 1` しか返さないので、これを
    足さないと変更予定の品目は新ルールしか LLM に届かず「片方だけで断定」して
    しまう(要件4-2 が禁じる状態)。
    """
    tl = pending_change(item_name, today or date.today())
    if tl is None:
        return

    ref = today or date.today()
    now_seg = tl.segment_on(ref)
    future = [s for s in tl.segments if s.valid_from and s.valid_from > ref]

    lines = ["## この品目は出し方が変わります(旧・新の両方を必ず時系列で案内する)"]
    if now_seg is not None:
        until = now_seg.valid_until.isoformat() if now_seg.valid_until else "当面"
        lines.append(f"- 現在(〜{until}): {now_seg.disposal_method}")
        if now_seg.disputed and now_seg.hayamihyo_disposal_method:
            lines.append(
                f"  (早見表には「{now_seg.hayamihyo_disposal_method}」と記載。"
                "お知らせの内容を優先する)"
            )
    for seg in future:
        lines.append(f"- {seg.valid_from.isoformat()} から: {seg.disposal_method}")

    ans.context = (ans.context + "\n\n" + "\n".join(lines)).strip()
    ans.sources = _dedupe(ans.sources + [s.source_url for s in tl.segments])
    ans.timeline_change = True


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
    ans.item_from_image = bool(image_item_name)

    if item_name:
        ans.item_hits, ans.item_match_stage = _search_item(
            item_name, item_candidates[1:], vector_query
        )
        ans.item_match = _classify_item_match(item_name, ans.item_hits)

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
    if item_name:
        _apply_timeline(ans, item_name)
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


def _build_context(ans: RoutedAnswer) -> str:
    """検索結果をLLMに渡す平文に整形する。"""
    blocks: list[str] = []

    if ans.item_hits:
        if ans.item_match == "analogy":
            q = ans.detected_item or "この品名"
            header = f"## 早見表に「{q}」の個別記載なし ― 類似する品目の情報"
        else:
            header = "## 分別早見表の検索結果"
        lines = [header]
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

    if ans.found:
        parts.append(_RULES_WITH_CONTEXT)
    else:
        parts.append(_RULES_NO_CONTEXT)

    if ans.timeline_change:
        parts.append(
            "- この品目は将来出し方が変わる。「現在は〜。YYYY年MM月DD日からは〜」の"
            "形で、旧ルールと新ルールの両方を時系列に沿って説明する。"
            "どちらか一方だけを答えて断定しない。\n"
        )

    if ans.item_match == "analogy":
        parts.append(
            "- 『類似する品目の情報』欄の品目は、質問された品名そのものではなく"
            "早見表にある別の品目である。その欄に書かれている実際の品目名を挙げて"
            "「類似する『(その品目名)』に準じると〜」の形で案内し、断定しない。"
            "個別の記載がないため、正確な分別は北九州市の公式ページや窓口で"
            "確認するよう必ず添える。収集日など他の欄の情報はこの限りではない。\n"
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
