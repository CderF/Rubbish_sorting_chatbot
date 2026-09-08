"""意図解析とルーティング — `/chat` から呼ばれる応用層のエントリポイント。

責務は「検索結果を集めて整形する」ことだけで、事実の判断はしない。
LLMには `RoutedAnswer.context` を渡して文章化のみを担当させ、引用URLは
コード側(`RoutedAnswer.sources`)で確定させる(LLMにURLを書かせない)。

データ層への依存は一方向。`app/data/lookup.py` が未実装の間は `_stubs.py` に
自動フォールバックするため、データ層の完成を待たずに `/chat` 統合を進められる
(開発分担計画書 3章)。
"""

import asyncio
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date

from app.rag.session_state import (
    ADD_WORDS,
    CORRECTION_WORDS,
    DialogState,
    TurnInput,
    is_plausible_item_name,
)
from app.rag.session_state import merge as merge_state
from app.rag.types import ItemRecord, ItemResult, RoutedAnswer, ScheduleRecord

# --- データ層の解決 --------------------------------------------------------
# FANG が app/data/lookup.py を push した時点で、こちら側は無変更で実結合に
# 切り替わる。スタブを app/data/ に置かないのは、ファイル衝突を避けるため。
try:
    from app.data.lookup import (  # type: ignore[import-not-found]
        lookup_collection_schedule,
        lookup_item_exact,
        lookup_item_partial,
    )

    _USING_STUBS = False
except ImportError:
    from app.rag._stubs import (
        lookup_collection_schedule,
        lookup_item_exact,
        lookup_item_partial,
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

# 「〜でお願いします」「〜でした」など、品名の後ろに付く丁寧語・依頼形。
# `_QUESTION_TAIL` は「捨て方」「分別」等の質問語を要求するため、
# 「冷蔵庫も追加でお願いします」のような依頼文を1語も削れない。
_POLITE_TAIL = re.compile(
    r"(?:で)?(?:お願い(?:します|いたします)?|おねがい(?:します)?|でした|です|ます|ました|"
    r"してください|ください|したい(?:です)?|しま(?:す|した))\s*[。、!！?？]*$"
)

# 文頭・文末に残る助詞や記号だけを落とす軽量な後処理。
_PARTICLES = r"はがをにへとでもやのねよかっ、。？?!！\s"
_LEADING = re.compile(rf"^[{_PARTICLES}]+")
_TRAILING = re.compile(rf"[{_PARTICLES}]+$")

_TOWN = re.compile(r"([一-龥ぁ-んァ-ヶa-zA-Z]{1,10}(?:\d{1,3}丁目|丁目|町|台|ケ丘|ヶ丘))")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()


def extract_region(prompt: str) -> tuple[str | None, str | None]:
    """プロンプトから区名と町名を抜き出す。見つからなければ None。"""
    text = _norm(prompt)
    ward = next((w for w in WARDS if w in text), None)

    town = None
    # 区名以降を優先して町名を探す(「小倉北区浅野1丁目」のような連結表記に対応)。
    tail = text.split(ward, 1)[1] if ward else text
    for cand in _TOWN.findall(tail) or _TOWN.findall(text):
        if cand not in WARDS and not any(w.startswith(cand) for w in WARDS):
            town = cand
            break
    return ward, town


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


def extract_item_name(prompt: str, strip_waste_words: bool | None = None) -> str | None:
    """質問文から品名らしき部分を取り出す。

    正規化はデータ層が内部で行う契約なので、ここでは行わず素の文字列を返す。

    `strip_waste_words` が True のとき「家庭ごみ」「燃えるごみ」などのごみ種別語を
    品名候補から除く。None なら `asks_schedule()` の判定に従う。
    """
    text = _norm(prompt)
    if not text:
        return None

    if strip_waste_words is None:
        strip_waste_words = asks_schedule(text)

    # 訂正されたターンでは、訂正マーカーより後ろが新しい品名
    # (「冷蔵庫じゃなくて洗濯機でした」→「洗濯機でした」)。前半を残すと
    # 断片の長さ比べで訂正前の品名を拾ってしまう。
    for marker in CORRECTION_WORDS:
        if marker in text:
            text = text.rsplit(marker, 1)[1]

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

    # 「も追加で」「それと」などの接続語は品名ではない。
    for word in ADD_WORDS:
        text = text.replace(word, " ")

    text = _QUESTION_TAIL.sub("", text)
    text = _POLITE_TAIL.sub("", text)
    text = _LEADING.sub("", text)
    text = _TRAILING.sub("", text).strip()

    # 記号・空白で切って最も長い断片を品名候補とする。
    fragments = []
    for frag in re.split(r"[\s、。「」()（）]+", text):
        frag = _TRAILING.sub("", _LEADING.sub("", frag))
        if frag:
            fragments.append(frag)
    if not fragments:
        return None

    candidate = max(fragments, key=len)
    # 助詞・丁寧語の残骸(「浅野1丁目です」→「す」)や指示語(「それ」)を検索キーに
    # すると、LIKE検索が無関係な早見表の行を拾って根拠に混入する(STAGE05で実測)。
    # 指示語の場合は品名を返さないことで、会話状態に既にある品目が保たれる。
    if not is_plausible_item_name(candidate):
        return None
    return candidate


# --- 町名の解決(STAGE05) --------------------------------------------------
# `_TOWN` は「町・台・丁目」等の接尾辞に依存するため、「黒崎」「三ケ森」のように
# 接尾辞を持たない町名を取りこぼす。語尾の活用を自前で推測する代わりに、
# 候補を長い順にDBへ当てて実在するものを選ぶ(DBを町名辞書として使う)。
# `collection_schedule` は (ward, town, waste_type) にインデックスがあり、
# 1回の照会は点クエリなので、数回試しても実質コストは無い。
_TOWN_FRAGMENT = re.compile(r"[一-龥ぁ-んァ-ヶーa-zA-Z0-9]{2,12}")
_MAX_TOWN_PROBES = 24


def _town_candidates(text: str, ward: str | None, exclude: "set[str]" = frozenset()) -> list[str]:
    """町名になりうる文字列を、確からしい順に並べて返す。"""
    text = _norm(text)
    tail = text.split(ward, 1)[1] if ward and ward in text else text

    ordered: list[str] = []

    def push(cand: str) -> None:
        cand = cand.strip()
        if not cand or cand in ordered or cand in WARDS or cand in exclude:
            return
        if any(w.startswith(cand) for w in WARDS):
            return
        ordered.append(cand)

    # 1段目: 接尾辞つき(「浅野1丁目」「本城台」)。最も確からしい。
    for cand in _TOWN.findall(tail) or _TOWN.findall(text):
        push(cand)

    # 2段目: 素の断片とその前方一致(「黒崎です」→「黒崎で」→「黒崎」)。
    bare: list[str] = []
    for frag in _TOWN_FRAGMENT.findall(tail):
        bare.append(frag)
        bare.extend(frag[:n] for n in range(len(frag) - 1, 1, -1))
    for cand in sorted(dict.fromkeys(bare), key=len, reverse=True):
        push(cand)

    return ordered[:_MAX_TOWN_PROBES]


def _resolve_town(
    ward: str, text: str, exclude: "set[str]" = frozenset()
) -> "tuple[str | None, list[ScheduleRecord] | None]":
    """区が分かっている前提で、本文から実在する町名を1つ決める。

    戻り値は (町名, その町の収集日全件)。見つからなければ (None, None)。
    ごみ種別での絞り込みは呼び出し側で行う(0件だったときに種別なしで引き直す
    処理を、DBへの再照会なしで済ませるため)。
    """
    for cand in _town_candidates(text, ward, exclude):
        rows = lookup_collection_schedule(ward, cand, None)
        if rows:
            return cand, rows
    return None, None


def extract_turn(prompt: str, image_item_name: str | None = None) -> TurnInput:
    """1ターンぶんの発話からスロット候補を抜き出す(STAGE05)。

    抽出は従来どおり文字列ヒューリスティクスで、ここでは会話状態を見ない。
    「今の発話から何が読み取れたか」と「それを状態にどう畳み込むか」を分けて
    おくと、畳み込み側(`app/rag/session_state.merge`)がDBもLLMも要らない
    純関数のままテストできる。
    """
    text = _norm(prompt)
    wants_schedule = asks_schedule(text)
    ward, town = extract_region(text)
    item = image_item_name or extract_item_name(text, strip_waste_words=wants_schedule)
    return TurnInput(
        text=text,
        item=item,
        ward=ward,
        town=town,
        waste_type=extract_waste_type(text) if wants_schedule else None,
        asks_schedule=wants_schedule,
        from_image=bool(image_item_name),
    )


# --- 検索 ------------------------------------------------------------------
def _search_item(item_name: str) -> tuple[list[ItemRecord], str | None]:
    """完全一致 → 部分一致 → ベクトル類似度 の順にフォールバックする(早見表パス)。

    ベクトル類似度は前2段がどちらも0件のときだけ発火する(issue #7)。
    `vector_search_items` はChroma未構築・関連ヒットなしのいずれも空リストを返す。

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
    vector = vector_search_items(item_name, top_k=3)
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


def _apply_timeline(result: ItemResult, today: "date | None" = None) -> list[str]:
    """STAGE04 4-2: この品目に「これから出し方が変わる」予定があれば、旧ルール・
    新ルール・切替日を `ItemResult` に足し、引用URLを返す。

    データ層の `lookup_item_exact` は `is_current = 1` しか返さないので、これを
    足さないと変更予定の品目は新ルールしか LLM に届かず「片方だけで断定」して
    しまう(要件4-2 が禁じる状態)。
    """
    tl = pending_change(result.item_name, today or date.today())
    if tl is None:
        return []

    ref = today or date.today()
    now_seg = tl.segment_on(ref)
    future = [s for s in tl.segments if s.valid_from and s.valid_from > ref]

    lines = [f"## 「{result.item_name}」は出し方が変わります(旧・新の両方を必ず時系列で案内する)"]
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

    result.timeline_change = True
    result.timeline_lines = lines
    return [s.source_url for s in tl.segments]


def _filter_by_waste_type(
    schedules: list[ScheduleRecord], waste_type: str | None
) -> list[ScheduleRecord]:
    """ごみ種別が明示されていれば絞り込む。0件になるなら絞り込まない。

    設計提案書 3-1 は「品名と地区を伝えると出し方と収集日まで案内できる」ことを
    求めるので、種別が一致しないことを理由に収集日そのものを落とさない。
    """
    if not waste_type:
        return schedules
    narrowed = [s for s in schedules if _norm(s.waste_type) == _norm(waste_type)]
    return narrowed or schedules


def _blocking_route(
    prompt: str,
    image_item_name: str | None = None,
    state: "DialogState | None" = None,
) -> RoutedAnswer:
    """同期の検索処理。`handle_query` からスレッドで呼ばれる。

    `image_item_name` は画像分類パス(`classify_image_to_item_name`)が
    画像から抽出した品名。渡されていればプロンプトからのテキスト抽出より
    優先する(STAGE03 3-2: 画像は品名の代わりに渡されるものなので)。

    `state` は前ターンまでの会話スロット(STAGE05)。`None` なら1ターン目として
    扱うため、STAGE02/03 と同じ「1リクエスト内で完結する」呼び出しがそのまま
    動く(既存のテスト・呼び出し側は無改修)。

    検索は毎ターン**マージ後のスロット全体に対してやり直す**。会話履歴を事実の
    出どころにしないことで、「検索結果に書かれている内容だけを使う」という
    システムプロンプトの規則を緩めずに 5-1 を満たせる。
    """
    turn = extract_turn(prompt, image_item_name)

    # 地区だけを言い直したターン(「やっぱり八幡西区黒崎です」)は、品名抽出が
    # 残りの断片(「やっぱり」「黒崎です」)を品名として拾ってしまう。町名解決と
    # 同じく、決め打ちで語尾を判別する代わりに早見表に当てて確かめる:
    # 地区を含むターンで、候補がどの検索段にも掛からなければ品名ではない。
    # 地区を伴わないターンでは落とさない — 早見表に無い品物を尋ねられた場合は
    # 「見つかりませんでした」と答える必要がある(設計提案書 3-7、要件4-3)。
    prefetched: "dict[str, tuple[list[ItemRecord], str | None]]" = {}
    if turn.item and not turn.from_image and (turn.ward or turn.town):
        found = _search_item(turn.item)
        # 採用するのは完全一致・部分一致まで。ベクトル段は「意味が近い別の品目」を
        # 返すので、「黒崎です」のような地区の言い直しにも何かしら当たってしまう。
        if found[1] in ("exact", "partial"):
            prefetched[turn.item] = found
        else:
            turn = replace(turn, item=None)

    state = merge_state(state, turn)

    ans = RoutedAnswer()

    # --- 早見表パス: 相談に出ている品目すべてを引き直す -----------------------
    timeline_sources: list[str] = []
    for name in state.items:
        hits, stage = prefetched.get(name) or _search_item(name)
        result = ItemResult(
            item_name=name,
            hits=hits,
            match_stage=stage,
            item_match=_classify_item_match(name, hits),
            from_image=turn.from_image and name == turn.item,
        )
        timeline_sources += _apply_timeline(result)
        ans.item_results.append(result)

    ans.item_hits = [h for r in ans.item_results for h in r.hits]
    ans.item_match_stage = next(
        (r.match_stage for r in reversed(ans.item_results) if r.match_stage), None
    )
    ans.timeline_change = any(r.timeline_change for r in ans.item_results)
    # 要件4-3 の断り書きは、1件でも類推が混じっていれば出す(PR #14)。
    matches = {r.item_match for r in ans.item_results}
    ans.item_match = (
        "analogy" if "analogy" in matches else ("exact" if "exact" in matches else "none")
    )

    # --- 収集日パス --------------------------------------------------------
    town = state.town
    if state.ward:
        schedules = lookup_collection_schedule(state.ward, town, None) if town else None
        if not schedules:
            # 町名が未確定、または前ターンの町名が引けない。今回の発話から引き直す。
            resolved, schedules = _resolve_town(
                state.ward, turn.text, exclude=set(state.items)
            )
            if schedules:
                town = resolved

        if schedules:
            ans.schedule_hits = _filter_by_waste_type(schedules, state.waste_type)
        elif not town:
            # 設計提案書 3-1: 区のみでは一意に定まらない。推測せず町名を要求する。
            ans.missing.append("町名")
        else:
            ans.missing.append(f"{state.ward}{town}の収集日データ")
    elif state.wants_schedule:
        ans.missing.append("区名と町名")

    # 解決できた町名は状態に書き戻す。次ターン以降は再解決なしで引ける。
    state = replace(state, town=town)

    ans.state = state
    ans.detected_item = state.primary_item
    ans.detected_ward, ans.detected_town = state.ward, state.town
    ans.item_from_image = turn.from_image

    ans.found = bool(ans.item_hits or ans.schedule_hits)
    ans.sources = _dedupe(
        [r.source_url for r in ans.item_hits]
        + [r.source_url for r in ans.schedule_hits]
        + timeline_sources
    )
    ans.context = _build_context(ans)
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
    state: "DialogState | None" = None,
    engine=None,
    processor=None,
) -> RoutedAnswer:
    """`/chat` から呼ばれるエントリポイント。

    計画書では同期関数だったが、`server.py` は全体が async で動いており、
    SQLite・embedding・VLM呼び出しでイベントループを塞ぐと同時4リクエスト
    要件に影響するため async 化した(FANG と合意済みの変更点)。

    `state` は同じ `session_id` の前ターンまでの会話スロット(STAGE05)。呼び出し
    側は戻り値の `RoutedAnswer.state` を保存し、次のターンでそれを渡し直す。
    タプルではなく `RoutedAnswer` に載せて返すのは、既存の呼び出し側の
    アンパック形を変えないため。省略時は1ターン目として扱う。

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
        _db_executor, _blocking_route, prompt, image_item_name, state
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


def _matching_item_labels(ans: RoutedAnswer, waste_type: str) -> list[str]:
    """この収集日の行(ごみ種別)に該当する品目名を返す。

    5-1 は「分別・出し方・収集日を組み合わせ」て案内することを求めている。
    早見表の出し方(例「粗大ごみ」)と収集日の種別(例「粗大ごみ」)を突き合わせて
    context 側で明示しないと、その町の全種別を並べたまま「どれがこの品物の日
    なのか」の判断をLLMに任せることになる。

    照合は返ってきた行の `waste_type` 文字列そのものに対して行い、種別語彙を
    こちら側で決め打ちしない(データ層が持つ表記が唯一の正)。
    """
    wt = _norm(waste_type)
    labels = []
    for result in ans.item_results:
        for rec in result.hits:
            method = _norm(_attr(rec, "disposal_method", ""))
            if wt and (wt in method or method in wt):
                labels.append(result.item_name)
                break
    return labels


def _build_context(ans: RoutedAnswer) -> str:
    """検索結果をLLMに渡す平文に整形する。

    品目は1件とは限らない(5-2 の「追加」で1回の回答が複数品目を扱う)。
    品目ごとに見出しを分け、ヒットしなかった品目も「該当なし」として残す
    — 黙って落とすと、LLMが前ターンの記憶から補って断定する余地が生まれる。
    """
    blocks: list[str] = []

    for result in ans.item_results:
        if result.item_match == "analogy":
            # 要件4-3: ヒットはあるが質問された品名そのものではない(PR #14)。
            lines = [f"## 早見表に「{result.item_name}」の個別記載なし ― 類似する品目の情報"]
        else:
            lines = [f"## 分別早見表の検索結果: {result.item_name}"]
        if result.hits:
            for r in result.hits:
                lines.append(f"- 品名: {_attr(r, 'item_name', '')}")
                lines.append(f"  出し方: {_attr(r, 'disposal_method', '')}")
                notes = _attr(r, "notes", "")
                if notes:
                    lines.append(f"  備考: {notes}")
        else:
            lines.append("- 該当する品目は見つかりませんでした(推測で分別を断定しない)")
        blocks.append("\n".join(lines))

        if result.timeline_lines:
            blocks.append("\n".join(result.timeline_lines))

    if ans.schedule_hits:
        lines = ["## 収集日の検索結果"]
        for s in ans.schedule_hits:
            # weekday は実データでは既に完全な表記("火曜日・金曜日"、月次の粗大ごみ
            # なら "第2木曜日" 等)なので、ここで「曜日」を足さない(PR #20)。
            labels = _matching_item_labels(ans, s.waste_type)
            suffix = f" ← {'・'.join(labels)}はこの区分" if labels else ""
            lines.append(f"- {s.ward}{s.town} / {s.waste_type}: {s.weekday}{suffix}")
        blocks.append("\n".join(lines))

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
    "- 検索結果にない分別区分・曜日を推測して書かない。\n"
    "- 検索結果の「出し方」(分別区分)は、その表記のまま答える"
    "(独自の言い換え・簡略化をしない)。\n"
    "- 検索結果に「備考」がある場合、そこに書かれた条件"
    "(持ち込み先・事前申込の要否・梱包方法・注意事項など)を"
    "一つも省略せず、全て回答に含める。\n"
)

_RULES_NO_CONTEXT = (
    "- 該当する情報が見つからなかったため、分別区分や収集日を断定してはならない。\n"
    "- 「該当する情報が見つかりませんでした」と明示し、"
    "北九州市の公式ページや窓口での確認を案内する。\n"
)


# STAGE05: マルチターンで前提が変わったときの組み立て方(5-2)。
_RULE_MULTI_ITEM = (
    "- 相談に出ている品物は複数ある({items})。それぞれについて分けて案内し、"
    "前のターンで案内済みの品物も省略しない。\n"
)

_RULE_CORRECTED = (
    "- 利用者は直前のやりとりで前提を訂正・変更した。訂正前の品物・地区に基づく"
    "案内を繰り返さず、新しい前提で組み直して答える。\n"
)

_RULE_KNOWN_REGION = (
    "- 利用者の地区は「{region}」と既に判明している。同じことを聞き返さない。\n"
)


def build_system_prompt(ans: RoutedAnswer) -> str:
    """`RoutedAnswer` からLLMへのシステムプロンプトを組み立てる。

    設計提案書 3-7(未識別時は憶測で断定しない)をここで強制する。
    STAGE05 では、会話状態(`ans.state`)から「複数品目」「訂正があった」
    「地区は判明済み」を明示的に指示に落とす。検索結果は毎ターン全スロットで
    引き直されているので、「検索結果だけを使う」規則は緩めない。
    """
    parts = [_BASE_RULES]

    if ans.found:
        parts.append(_RULES_WITH_CONTEXT)
    else:
        parts.append(_RULES_NO_CONTEXT)

    state = ans.state
    if state is not None:
        if len(state.items) > 1:
            parts.append(_RULE_MULTI_ITEM.format(items="・".join(state.items)))
        if state.corrected:
            parts.append(_RULE_CORRECTED)
        if state.has_region:
            parts.append(_RULE_KNOWN_REGION.format(region=f"{state.ward}{state.town}"))

    if ans.timeline_change:
        parts.append(
            "- 出し方が将来変わる品目がある。「現在は〜。YYYY年MM月DD日からは〜」の"
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
        # STAGE05: いま会話で保持しているスロット。5-1/5-2 が正しく効いているかを
        # メンターがUI上で確認できるようにする(6-4 の「後から確認できる」にも効く)。
        "slots": ans.state.describe() if ans.state is not None else None,
        "item_results": [
            {
                "item_name": r.item_name,
                "match_stage": r.match_stage,
                "match_stage_label": MATCH_STAGE_LABELS.get(r.match_stage or ""),
                "item_match": r.item_match,
                "hits": len(r.hits),
                "timeline_change": r.timeline_change,
            }
            for r in ans.item_results
        ],
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
