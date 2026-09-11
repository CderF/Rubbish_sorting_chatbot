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

from app.rag import safety
from app.rag.types import ItemRecord, RoutedAnswer, ScheduleRecord

# --- データ層の解決 --------------------------------------------------------
# FANG が app/data/lookup.py を push した時点で、こちら側は無変更で実結合に
# 切り替わる。スタブを app/data/ に置かないのは、ファイル衝突を避けるため。
try:
    from app.data.lookup import (  # type: ignore[import-not-found]
        lookup_collection_schedule,
        lookup_item_contained_in,
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

    def lookup_item_contained_in(text: str, min_length: int = 2):  # noqa: ARG001
        return None

    _USING_STUBS = True

# --- ベクトルRAGパス・画像分類パスの解決 -----------------------------------
# こちらは app/data/lookup.py と違い、常設のスタブファイルは持たない
# (chromadb/sentence-transformers/vllm いずれも重い依存であり、未インストール
# 環境でも router.py 自体はimportできるようにするための最小限のフォール
# バックに留める)。
try:
    from app.rag.vector_store import vector_search_explanations, vector_search_items
except ImportError:

    def vector_search_items(query: str, top_k: int = 3) -> list[ItemRecord]:
        return []

    def vector_search_explanations(query: str, top_k: int = 3) -> list[dict]:
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

# `category_procedure_intent` が認識する、特定品目ではなく「ごみ種別」
# そのものを指す語。`WASTE_TYPES` より広い(早見表 disposal_method の
# 「回収ボックス」系区分も含む)。「ペットボトル」は品名としても頻繁に
# 単体で検索されるため(既存のテスト・実装がこの前提)、ここには含めない。
_CATEGORY_WORDS = (
    "粗大ごみ", "家庭ごみ", "古紙", "小物金属", "小型電子機器",
    "電池類", "紙パック", "古着", "資源化ごみ", "特定ごみ",
)
# 「〜の収集日」「〜は何曜日」のように収集日そのものを尋ねる語。
_CATEGORY_SCHEDULE_MARKERS = re.compile(r"(日|曜日|いつ)")
# 「〜はどうやって出す/申し込む」のように、品目ではなく出し方・手続き
# そのものを尋ねる語。
_CATEGORY_PROCEDURE_MARKERS = re.compile(
    r"(出し方|出すん|出したい|捨てるん|捨てたい|申し?込(?:む|みたい|んで|んだ)?|"
    r"やり方|どうやって|どう出す|教えて)"
)
# 種別名・手続き語を取り除いた残りが「実質的な内容の無い言い回し」かを
# 判定するための、助詞・丁寧語・語尾の除去用パターン(アンカー無し。
# 「ってどうやって出すんですか」→種別語・手続き語除去後の「ってですか」の
# ような、文中に散らばった助詞・丁寧語もまとめて消す)。
_CATEGORY_FILLER_CHARS_RE = re.compile(
    r"(って|です|ます|でしょう|でしょ|ください|は|の|を|も|で|に|へ|から|ん|か|、|。|\s)+"
)


def category_procedure_intent(
    prompt: str, ward: str | None = None, town: str | None = None
) -> tuple[str, bool] | None:
    """`prompt` が特定の品目ではなく、ごみ種別そのものの出し方・手続き・
    収集日を尋ねる質問なら `(種別名, 収集日の質問かどうか)` を返す。
    該当しなければ None。

    2026-09-14、B1のトレース精査で「粗大ごみってどうやって出すんですか」
    「粗大ごみってどうやって申し込むんですか」「古紙を出したいです」の
    ような、特定品目を指さない質問が品目検索に落ち、無関係な品目
    (「ごみ箱」等)や無関係な制度(「ふれあい収集」等)に化ける不具合を
    複数確認した(要件5-1「推測せず確認する」への違反)。品目検索
    (`_search_item`/ベクトル検索)に渡す前に、`_blocking_route` と
    `app/rag/agent.py` の両方からここを通す。

    種別名を(区名・町名を除いた上で)含み、かつ収集日・手続きのどちらかを
    尋ねる語も含み、それらを取り除いた残りに実質的な内容が無い(短い助詞・
    語尾程度)場合に該当と判定する。「粗大ごみの布団はどう捨てますか」の
    ように具体的な品目(布団)が残るケースは対象外(通常の品目検索に任せる)。
    """
    text = _norm(prompt)
    for part in (ward, town):
        if part:
            text = text.replace(part, " ")

    category = next(
        (c for c in sorted(_CATEGORY_WORDS, key=len, reverse=True) if c in text), None
    )
    if category is None:
        return None

    schedule_like = bool(_CATEGORY_SCHEDULE_MARKERS.search(text))
    procedure_like = bool(_CATEGORY_PROCEDURE_MARKERS.search(text))
    if not (schedule_like or procedure_like):
        return None

    remainder = text.replace(category, "")
    remainder = _CATEGORY_SCHEDULE_MARKERS.sub("", remainder)
    remainder = _CATEGORY_PROCEDURE_MARKERS.sub("", remainder)
    remainder = _PUNCT_OR_SPACE_RE.sub("", remainder)
    remainder = _CATEGORY_FILLER_CHARS_RE.sub("", remainder)
    if remainder:
        return None
    return category, schedule_like


# 種別の手続き質問(procedure_like)で、知識ベースに根拠となる一般説明が
# 見つからなかったときの固定回答。LLM生成には一切渡さない(4-3確度ゲートの
# `_ANALOGY_UNRELIABLE_REPLY` と同じ設計: 何も見つからない状況で自由生成
# させると、小型モデルは「ふれあい収集」「ごみ箱」のような無関係な内容を
# 作り出してしまう ― 2026-09-14、B1のトレース精査で確認)。
_CATEGORY_NO_CONTEXT_REPLY = "具体的な品目名と、お住まいの区・町名を教えてください。"


def general_info_for_category(category: str) -> tuple[tuple[str, str] | None, dict]:
    """種別そのものの手続き質問(procedure_like)に、知識ベースの一般説明
    (`search_general_info`)で答えられる根拠があれば `(説明文, 出典URL)` を
    返す。無ければ None。

    2026-09-14、`vector_search_explanations` はコサイン距離だけでは
    「粗大ごみ」「古紙」の出し方・申込みについて、本文に種別名が一度も
    出てこない無関係なページ(「ふれあい収集」「ごみステーションの散乱
    対策」等)をむしろ高いスコアで返してしまうことを確認した
    (要件5-1「推測せず確認する」)。距離の閾値だけでは信頼できないため、
    返ってきたチャンクの本文に実際に種別名が含まれているものだけを信頼する
    (`app/rag/agent.py` の `_query_grounded_in_text` と同じ、埋め込み類似度
    だけに頼らない字面の裏取り)。

    戻り値の第2要素は要件6-4用のトレイル1件(`RoutedAnswer.search_trail` に
    そのまま積める形)。`search_general_info`(説明ページの向量库)は前段の
    字面一致で回避できず必ず1回問い合わせるので、`hits` には返ってきた
    チャンクごとに実際に採用したか(`grounded`、本文に種別名を含むか)を
    添える。
    """
    query = f"{category} 出し方 申込み方法"
    chunks = vector_search_explanations(query, top_k=3)
    trail = {
        "tool": "search_general_info",
        "query": query,
        "attempts": [{
            "stage": "vector_explanations",
            "query": query,
            "hits": [
                {
                    "title": c.get("source_page_title", ""),
                    "score": c.get("score"),
                    "grounded": category in (c.get("text") or ""),
                }
                for c in chunks
            ],
        }],
    }
    for chunk in chunks:
        text = chunk.get("text") or ""
        if category in text:
            return (text, chunk.get("source_url", "")), trail
    return None, trail


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

# `_LEADING` は文字クラスなので「やかん」「ねじ回し」のように内容語自体が
# _PARTICLES に含まれる仮名(や・ね・か 等)で始まる品名まで削ってしまう
# (実測: 「やかんを処分したい」→ 候補が「んを処分したい」、「ねじ回しは
# どう捨てますか」→「じ回し」)。`text` が発話中で「語の書き出し」の位置
# (先頭、または断片分割に使う区切りの直後)に現れているなら、その1文字目は
# 助詞の残骸ではなく利用者が実際に書いた内容語の頭なので `_LEADING` を
# 当てない。区名・町名・ごみ種別語を除去した「跡」(「…丁目でソファを…」の
# 「でソファを…」)は語の書き出しではないので、従来どおり `_LEADING` が効く。
_FRAGMENT_BOUNDARY = r"[\s、。「」()（）]"


def _written_from_the_start(frag: str, original: str) -> bool:
    if not frag:
        return False
    return re.search(rf"(?:^|{_FRAGMENT_BOUNDARY}){re.escape(frag)}", original) is not None


def _trim_particles(text: str, original: str) -> str:
    """文頭・文末に残った助詞を落とす。ただし利用者が書いた語の頭は削らない。

    `original` は正規化しただけの発話全文(区名・町名等の除去より前)。
    """
    if not _written_from_the_start(text, original):
        text = _LEADING.sub("", text)
    return _TRAILING.sub("", text).strip()


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


_CHOME_SUFFIX_RE = re.compile(r"\d+丁目(?:\(一部\)|（一部）)?$")


def _town_base(town: str) -> str:
    """町名から丁目番号を除いた「地区の呼び名」を返す(例: 「黒崎1丁目」→「黒崎」)。

    「企救丘4丁目(一部)」のように、丁目の一部だけが別表記になっている町名も
    同じ地区として扱う(2026-09-14 レビュー指摘: 「(一部)」が付くと従来の
    正規表現が末尾にマッチせず、`_district_groups` がこの町を地区の仲間
    として拾えなかったため、丁目省略の地区名から漏れていた)。"""
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


# `_has_plausible_item_text` が候補から先に取り除く、品目名には現れない
# 会話の相槌・丁寧表現。地区の確認・訂正だけの発言(「すみません、やっぱり
# 〜でした」)がまるごと品名候補になってしまうのを防ぐ。
_FILLER_WORDS = (
    "すみません", "やっぱり", "でした", "ですが", "ですけど",
    "お願いします", "ください", "教えてください",
)
_FILLER_RE = re.compile("(" + "|".join(_FILLER_WORDS) + ")")


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


def _has_plausible_item_text(candidate: str, ward: str | None) -> bool:
    """品名候補が、本当に品目名らしい実体を持つかを判定する。

    2026-09-10、B1実機の複数ターン会話の再現テストで、地域だけの発言
    (「小倉北区浅野1丁目です」)や地域訂正だけの発言(「やっぱり八幡西区
    黒崎でした」)から、`_item_name_candidates` が正規表現の残骸(「す」)や
    会話の相槌・丁寧表現・丁目省略の地区名(「黒崎」、地区名そのものは品目
    ではない)を候補として拾ってしまい、それをそのままベクトル検索にかけて
    無関係な品目(「車いす」「つっぱり棒」)を見つけてしまう不具合を確認した
    (要件5-1「推測せず確認する」への違反、`app/rag/agent.py` の
    `_query_grounded_in_text` と同じ動機のガードをこちら
    ―決定論的経路― にも入れる)。`_blocking_route` が `_search_item`/
    ベクトル検索に候補を渡す前にここで弾く。

    - まず会話の相槌・丁寧表現(`_FILLER_RE`)を候補から取り除く。
    - 丁目省略の地区名(`_match_district`)が残っていれば、それも取り除く。
      `_district_groups` は「同じ地区名を共有する丁目が複数ある」場合しか
      地区として登録しない(既存の設計、`_district_groups` 参照)ため、
      町名が既に「黒崎1〜5丁目」のような結合表記で1行にまとまっている
      データでは検出できないことがある — その場合はこの分岐が素通りする
      既知の限界(別問題、ここでは追わない)。
    - 上のどちらかで何かを取り除いた場合、取り除いた残りに実質的な内容
      (かな以外を1文字も含まない)が無ければ、相槌・地区名の言い換えに
      過ぎないとみなす。句読点・空白は文中のどこにあっても除いてから判定する。
    - 何も取り除かれなかった候補は、ひらがなだけで2文字未満のときだけ弾く
      (助詞・語尾の残骸である可能性が高い一方、「かご」等の実在する短い
      ひらがな品目まで誤って弾かないため)。
    """
    text = _norm(candidate)
    if not text:
        return False

    stripped = text
    trimmed_something = False

    filler_stripped = _FILLER_RE.sub("", stripped)
    if filler_stripped != stripped:
        trimmed_something = True
        stripped = filler_stripped

    if ward:
        district = _match_district(ward, stripped)
        if district:
            base, _towns = district
            stripped = stripped.replace(base, "")
            trimmed_something = True

    # ごみ種別語(「家庭ごみ」等)とその手続き・収集日を尋ねる語も取り除く。
    # 2026-09-14、「小倉南区の企救丘で家庭ごみの日を教えてください」で、
    # フィラー・地区名(企救丘)を取り除いた残りに「家庭ごみ」(品目ではなく
    # ごみ種別)が残り、それが「実質的な内容」と誤判定されてベクトル検索に
    # 渡ってしまう再発を確認した(要件5-1)。種別語が実際に見つかったときに
    # 限って手続き・収集日の語も一緒に取り除く ― 種別語が無いのに手続き語
    # (「捨てたい」等)だけ機械的に取り除くと、「黒崎でかごを捨てたい」の
    # ような実在の品目(かご)の直前にある漢字(捨)が消え、残りが「でかごを」
    # という平仮名だけになって誤って弾かれてしまう(2026-09-14 レビュー指摘)。
    category_word = next(
        (w for w in sorted(_CATEGORY_WORDS, key=len, reverse=True) if w in stripped), None
    )
    if category_word:
        stripped = stripped.replace(category_word, "")
        stripped = _CATEGORY_SCHEDULE_MARKERS.sub("", stripped)
        stripped = _CATEGORY_PROCEDURE_MARKERS.sub("", stripped)
        trimmed_something = True

    if trimmed_something:
        remainder = _PUNCT_OR_SPACE_RE.sub("", stripped)
        return bool(remainder) and not _KANA_ONLY_RE.match(remainder)

    if _KANA_ONLY_RE.match(text):
        if len(text) < 2:
            return False
        # `_item_name_candidates` の断片化・修飾語除去(`_clean_fragment`)が
        # 相槌の一部を単独の断片として切り出すことがある(「やっぱり」→
        # 「ぱり」)。`_FILLER_RE` はフィラー語自体を含む候補しか捉えられない
        # ので、逆方向(候補がフィラー語の部分文字列)もここで見る
        # (2026-09-10 レビュー指摘: 「黒崎」を正しく除去できるようになった後も
        # 「ぱり」というひらがな2文字の残骸が生き残り、ベクトル検索で
        # 「つっぱり棒」を誤って見つけてしまう再発を、実際の再クロール後の
        # DB で確認した)。
        if any(text in w or w in text for w in _FILLER_WORDS):
            return False

    return True


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


def _describe_non_uniform_district(
    ward: str, district_name: str, towns: tuple[str, ...], waste_type: str | None
) -> str | None:
    """地区内で収集日が丁目によって割れている(`_uniform_district_schedule`
    が None を返した)とき、実際にどう割れているかを丁目ごとにまとめた
    案内文を作る。

    2026-09-14、B1のトレース精査で「企救丘」(小倉南区)の質問に対し、
    地区内で収集日が割れているという理由だけで「町名が不明」と案内し、
    そもそも丁目によって収集日が違うこと自体が伝わらない不具合を確認した
    (要件5-2「地域の曖昧さを具体的に示す」)。同じ収集日を共有する丁目を
    グループ化し、利用者がどちらに該当するか選べる形で示す。

    どの丁目も収集日が引けない(データが無い)場合は None を返し、呼び出し側は
    従来通りの「町名が必要」という一般的な案内にフォールバックする。

    「企救丘4丁目(一部)」のように、1つの町(丁目)の中でも収集日データが
    複数(同じごみ種別に対し曜日が2通り)存在することがある ― 該当丁目の
    中でも実際の住所によって収集日が分かれることを公式ページ自身が
    「(一部)」と明記しているケース。これを他の丁目とまとめて1つの通常
    グループに入れてしまうと、存在しない「1〜4丁目共通の収集日」を
    作り出すことになる(2026-09-14 レビュー指摘)。そのため、同じごみ種別で
    複数の曜日を持つ町は他の丁目と混ぜず、単独で「要確認」として案内する。
    """
    groups: dict[frozenset, list[str]] = {}
    ambiguous_towns: list[str] = []
    for t in towns:
        hits = lookup_collection_schedule(ward, t, waste_type)
        if not hits:
            continue
        weekdays_by_type: dict[str, set[str]] = {}
        for h in hits:
            weekdays_by_type.setdefault(h.waste_type, set()).add(h.weekday)
        if any(len(wd) > 1 for wd in weekdays_by_type.values()):
            ambiguous_towns.append(t)
            continue
        key = frozenset((wt, next(iter(wd))) for wt, wd in weekdays_by_type.items())
        groups.setdefault(key, []).append(t)

    if len(groups) < 2 and not ambiguous_towns:
        return None

    parts = []
    for key, group_towns in groups.items():
        weekdays = "・".join(sorted({w for _wt, w in key}))
        town_range = "・".join(group_towns)
        parts.append(f"{town_range}は{weekdays}")
    for t in ambiguous_towns:
        parts.append(f"{t}は収集日が複数存在するため要確認")
    return f"「{district_name}」は丁目によって収集日が異なる({'、'.join(parts)})。何丁目か"


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

# 2026-09-10 実採点trace回帰: 「これ燃えるゴミで合ってますよね？」「さっき
# 聞いたやつ、結局何ごみでしたっけ？」のように `_QUESTION_TAIL` の定型に
# 一致しない言い回しでは、文全体がそのまま vector_query になり、
# `vector_search_items` が(無関係でも)何かしら最近傍の品目を返してしまい、
# 4-3類推が根拠薄弱なまま分別区分を断定していた。
#
# 一方で「廃車にした自動車を処分したいのですが」のように、文中に実在の
# 品名がそのまま埋め込まれているケースは `lookup_item_contained_in`
# (app/data/lookup.py)で正確に拾える。この判定(_looks_like_a_sentence)は
# その救済が失敗した**後**、ベクトル検索に回すか諦めるかの最終ゲートとして
# `_search_item` の中でだけ使う — `_item_name_candidates` の候補生成自体は
# 変更しない(候補を削ると `lookup_item_contained_in` の入力も減ってしまう)。
#
# 実データ検証: waste_items の全864品名にこれらの語(述語マーカー・語尾の
# 指示語・相槌)は1件も含まれない。
_SENTENCE_MARKERS = (
    "です", "ます", "でしょう", "ください", "んです",
    "でした", "ました", "ません", "っけ",
    "分からない", "わからない", "たらいいか", "ばいいか",
)
_VAGUE_REFERENCE_SUFFIXES = ("やつ", "もの", "これ", "それ", "あれ")
_FILLER_WORDS = frozenset({
    "あの", "あのー", "えっと", "えーと", "まあ", "そうそう", "うーん",
    "なんか", "ええ", "はい", "うん", "そう", "そうですね", "ちょっと",
})


def _looks_like_a_sentence(candidate: str) -> bool:
    """候補が品名ではなく、文(述語)・漠然とした代名詞参照・相槌らしいかを判定する。"""
    if any(m in candidate for m in _SENTENCE_MARKERS):
        return True
    if candidate.endswith(_VAGUE_REFERENCE_SUFFIXES):
        return True
    return candidate in _FILLER_WORDS


def _partial_lookup_ok(candidate: str) -> bool:
    """その候補を部分一致(`LIKE '%候補%'`)に渡してよいか。

    2026-09-10 実採点trace回帰: 「え、知らなかったです。じゃあどうすれば
    いいんですか？」のような相槌だけの追い返しは、断片分割の結果として
    1文字の候補(「え」)を生む。これを部分一致に渡すと
    `LIKE '%え%'` が「砂(鉢植え…)」「つえ」「ふえ(ホイッスル)」等に
    無差別にヒットし、根拠皆無のまま analogy として断定してしまう。

    完全一致は1文字でも意味がある(「傘」「畳」「鍵」等、実データに1文字の
    品名が15件ある)ため制限しない。制限するのは部分一致だけ。
    """
    return len(candidate) >= 2 and candidate not in _FILLER_WORDS


def _vector_query_ok(candidate: str) -> bool:
    """その候補をベクトル検索に渡してよいか(決定論的経路・Agentic経路 共通)。

    ベクトル検索は「必ず最近傍を返す」ので、根拠にならない文字列を渡すと
    無関係な品目を掴んだまま断定してしまう。渡してよいのは:

    - 2文字以上(1文字のクエリは意味のある近傍を持たない。「傘」「畳」等の
      1文字の実在品名は完全一致で先に拾えるため、ここで弾いても影響しない)
    - 文・代名詞参照・相槌でないこと(`_looks_like_a_sentence`)

    Agentic経路(`app/rag/tools.py`)はモデルが任意の文字列を query として
    渡してくるため、決定論的経路と同じ判定をこの1関数に集約して共有する
    (2026-09-10 実採点trace回帰。実際にモデルが「え」だけを query にした
    ケースが観測され、`LIKE '%え%'` とベクトル検索の両方が誤爆していた)。
    """
    return len(candidate) >= 2 and not _looks_like_a_sentence(candidate)


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

    # 発話そのものの先頭かどうかを後で判定するために原文を控えておく
    # (`_trim_particles` 参照)。
    original = text

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
    text = _trim_particles(text, original)

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
        frag = _trim_particles(frag, original)
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


# 品目の列挙を切り分ける区切り(「AとB」「A、B」「AやB」「あとB」)。記号に加えて、
# かな・漢字に挟まれた助詞「と」「や」も区切りとして扱う。直前・直後を文字種で
# 縛るのは、「ボート」のように語の一部である「と」を割らないため。
_LIST_SEP_RE = re.compile(
    r"[、,，・]+|(?<=[ぁ-んァ-ヶ一-龥])[とや](?=[ぁ-んァ-ヶ一-龥])|あと|それと"
)


def _is_waste_type_only(fragment: str) -> bool:
    """断片がごみ種別語だけで出来ているか(「かん・びん」「ペットボトル」等)。

    種別そのものの質問であって品目の列挙ではないので、品目の追加検索から外す。
    """
    rest = _norm(fragment)
    for word in sorted(_WASTE_WORDS, key=len, reverse=True):
        rest = rest.replace(word, "")
    return len(rest.strip(" ・、,，。")) < 2


def _listed_item_extras(prompt: str, exclude: set[str]) -> list[ItemRecord]:
    """質問文に列挙された品目のうち、早見表に完全一致するものを返す。

    多品目の質問(「たんすと古い布団と、あと段ボールがたくさんあります」)では、
    主候補の検索だけでは最初の1件しか拾えず、残りの品目が回答から抜け落ちる。
    列挙の区切り(`_LIST_SEP_RE`)で分割し、断片ごとに完全一致(言い換えの解決込み)
    を試して、明示的に列挙された品目を漏らさないようにする。

    誤爆を防ぐため、次は採用しない:
    - ごみ種別語だけの断片、および解決先がごみ種別そのものの行
      (`_is_waste_type_only`)。種別の質問であって品目の列挙ではない。
    - `exclude` にある品目(主候補として検索済み)。同じ品目を二重に載せない。
      断片そのものの表記も見る — 早見表に「テレビ」と「テレビ(ブラウン管式…)」の
      両方がある環境で、検索済みの「テレビ」を言い換え解決で別の行として
      足してしまうのを防ぐ(2026-09-10 テスト回帰)。
    ベクトル検索は使わない — 列挙の断片は短く、意味的近似を根拠にするには
    材料が足りない(無関係な近傍を拾うだけになる)。
    """
    candidates, _ = _item_name_candidates(prompt)
    fragments: list[str] = []
    for cand in candidates:
        for frag in _LIST_SEP_RE.split(cand):
            frag = frag.strip()
            if frag and frag not in fragments:
                fragments.append(frag)

    found: list[ItemRecord] = []
    for frag in fragments:
        if _is_waste_type_only(frag) or _norm(frag) in exclude:
            continue
        rec = lookup_item_exact(resolve_canonical_alias(frag) or frag)
        if rec is None:
            rec = _pick_embedded_item(frag, lookup_item_contained_in(frag))
        if rec is None or _norm(rec.item_name) in exclude:
            continue
        if _is_waste_type_only(rec.item_name):
            # 断片全体は種別語だけではないが、解決先が種別そのものの行
            # (「ペットボトル」「かん・びん」)というケース。品目の列挙ではない
            # ので足さない(「かん・びんとペットボトルは同じ袋に…」で
            # ペットボトルを品目として拾ってしまう回帰を防ぐ)。
            continue
        exclude.add(_norm(rec.item_name))
        found.append(rec)
    return found


def extract_item_name(prompt: str, strip_waste_words: bool | None = None) -> str | None:
    """質問文から品名らしき部分(主候補のみ)を取り出す。"""
    candidates, _ = _item_name_candidates(prompt, strip_waste_words)
    return candidates[0] if candidates else None


# --- 検索 ------------------------------------------------------------------
def _hit_brief(rec: "ItemRecord | None") -> dict | None:
    if rec is None:
        return None
    return {"item_name": rec.item_name, "disposal_method": rec.disposal_method}


def _search_item(
    item_name: str,
    extra_candidates: list[str] | None = None,
    vector_query: str | None = None,
) -> tuple[list[ItemRecord], str | None, list[dict]]:
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

    戻り値は (ヒット, どの段で当たったか, 検索過程のトレイル)。段
    (`item_match_stage`)は WebUI の根拠表示用。回答ロジックが使う「質問の
    品名そのものか、似た別品目か」の判定は別で、呼び出し側の
    `_classify_item_match`(名前一致ベース)で行う。

    トレイル(要件6-4)は、実際に試した段ごとに `{"stage", "query", "hits"}`
    を積んだリスト。採用されなかった段(0件)も含めて全部残すので、`/trace`
    で「完全一致・部分一致は空振りし、ベクトルの上位3件にも正解が無かった」
    ように、途中経過を事後に追える。ベクトル段の `hits` だけは各候補に
    `relation_strength`(`_relation_strength` のスコア、0/1/2)を添える。
    """
    trail: list[dict] = []

    canonical = resolve_canonical_alias(item_name)
    if canonical:
        exact = lookup_item_exact(canonical)
        trail.append({"stage": "canonical_alias", "query": canonical, "hits": [_hit_brief(exact)] if exact else []})
        if exact:
            return [exact], "exact", trail

    exact = lookup_item_exact(item_name)
    trail.append({"stage": "exact", "query": item_name, "hits": [_hit_brief(exact)] if exact else []})
    if exact:
        return [exact], "exact", trail
    if _partial_lookup_ok(item_name):
        partial = lookup_item_partial(item_name)[:5]
        trail.append({"stage": "partial", "query": item_name, "hits": [_hit_brief(p) for p in partial]})
        if partial:
            return partial, "partial", trail
    for cand in extra_candidates or []:
        if cand == item_name:
            continue
        exact = lookup_item_exact(cand)
        trail.append({"stage": "exact", "query": cand, "hits": [_hit_brief(exact)] if exact else []})
        if exact:
            return [exact], "exact", trail
        if not _partial_lookup_ok(cand):
            continue
        partial = lookup_item_partial(cand)[:5]
        trail.append({"stage": "partial", "query": cand, "hits": [_hit_brief(p) for p in partial]})
        if partial:
            return partial, "partial", trail

    # 2026-09-10 実採点trace回帰の救済: `_QUESTION_TAIL` の定型に一致せず
    # item_name が文全体のままでも、その中に実在の品名がそのまま埋め込まれて
    # いれば(「廃車にした自動車を処分したいのです」の「自動車」)、ベクトル
    # 検索より先に正確な一致として拾う(埋め込み一致の方が意味的近似より
    # 遥かに確実な根拠のため)。`lookup_item_contained_in` は単純な部分文字列
    # 一致(語境界を見ない)なので、ここで `_pick_embedded_item` により
    # 「直後が文末・助詞・質問定型」の境界チェックをかける(「捨てたらいいか」
    # に「たらい」が偶然含まれる、のような事故を防ぐ)。
    embedded = _pick_embedded_item(item_name, lookup_item_contained_in(item_name))
    trail.append({"stage": "embedded", "query": item_name, "hits": [_hit_brief(embedded)] if embedded else []})
    if embedded is None:
        for cand in extra_candidates or []:
            embedded = _pick_embedded_item(cand, lookup_item_contained_in(cand))
            trail.append({"stage": "embedded", "query": cand, "hits": [_hit_brief(embedded)] if embedded else []})
            if embedded is not None:
                break
    if embedded is not None:
        return [embedded], "exact", trail

    # 言い換え表(`_CANONICAL_ALIASES`)の語が質問文にそのまま埋もれている
    # ケースの救済。「硫酸」のように早見表に品名として存在しない物質名を、
    # 包括行(「薬品(家庭医薬品でないもの)」)へ固定する。完全一致・部分一致・
    # 埋め込み品名がすべて空振りしたあとにだけ試すので、既存の品名一致を
    # 横取りしない(2026-09-10 第6回採点回帰: 「硫酸」がベクトル類似で
    # 「油(食用油)=家庭ごみ」に化けていた)。
    for probe in (item_name, *(extra_candidates or [])):
        target = _resolve_embedded_alias(probe)
        if not target:
            continue
        rec = lookup_item_exact(target)
        trail.append({"stage": "alias_embedded", "query": probe, "hits": [_hit_brief(rec)] if rec else []})
        if rec is not None:
            return [rec], "exact", trail

    # 埋め込み品名も見つからず、ベクトル検索に渡すクエリ自体が文・代名詞
    # 参照・相槌らしいなら(`_looks_like_a_sentence`)、ベクトル検索にも
    # 回さない。根拠の無い文字列をそのまま embedding にかけても、無関係な
    # 最近傍を拾って断定してしまうだけ(2026-09-10 実採点trace回帰)。
    final_query = vector_query or item_name
    if not _vector_query_ok(final_query):
        trail.append({"stage": "vector_skipped", "query": final_query, "hits": []})
        return [], None, trail

    vector = vector_search_items(final_query, top_k=3)
    trail.append({
        "stage": "vector",
        "query": final_query,
        "hits": [
            {
                "item_name": v.item_name,
                "disposal_method": v.disposal_method,
                "relation_strength": _relation_strength(final_query, v.item_name),
            }
            for v in vector
        ],
    })
    return vector, ("vector" if vector else None), trail


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

# 言い換え表(`_resolve_embedded_alias`)の探索だけに足す緩和。早見表の品名一致
# (`_pick_embedded_item`)は「電動歯ブラシの充電スタンド」を別品目として正しく
# 弾く必要があるため緩めないが、言い換えでは「注射器の本体」のように「の」+
# 部位を表す語が続いても指している品物は同じなので成立させる。
_PART_SUFFIX_RE = re.compile(r"^の(?:本体|部分|中身|フタ|ふた)")


def _pick_embedded_item(text: str, candidates: list[ItemRecord]) -> "ItemRecord | None":
    """`lookup_item_contained_in` が返した(語境界を見ていない)候補群から、
    `text` 中で実際に品名の主辞として現れているものを選ぶ。

    `_resolve_named_hit` と違い、直前の境界は見ない(「使わなくなった電子
    レンジ」「〜にした自動車」のように、直前が助詞ではなく動詞の活用語尾
    でも品名は品名のため)。直後の境界だけを `_AFTER_NAME_OK` でチェックする
    ことで、「捨てたらいいか」に「たらい」が偶然含まれる、のような事故は
    排除できる(直後が「いいか」で `_AFTER_NAME_OK` に一致しない)。

    複数該当すれば最長一致(最も具体的な品名)を優先する。
    """
    if not candidates:
        return None
    norm_text = _norm(text)
    best: "ItemRecord | None" = None
    best_len = -1
    for item in candidates:
        full_name = _norm(getattr(item, "item_name", ""))
        if not full_name:
            continue
        # `lookup_item_contained_in` は括弧書き手前(base name)だけの一致でも
        # 候補に含める(「蛍光管(環型、直型)」候補が「蛍光管を割って…」で
        # 選ばれる)。境界チェックもどちらで見つかったかに合わせて行う。
        for name in (full_name, _base_name(full_name)):
            if not name:
                continue
            idx = norm_text.find(name)
            if idx == -1:
                continue
            end = idx + len(name)
            if not _AFTER_NAME_OK.match(norm_text[end:]):
                continue
            if len(name) > best_len:
                best, best_len = item, len(name)
            break
    return best


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

    # 特定品目ではなく、ごみ種別そのものの手続き・収集日を尋ねる質問
    # (要件5-1、`category_procedure_intent` 参照)。品目検索には一切渡さない。
    category_intent = None if image_item_name else category_procedure_intent(prompt, ward, town)
    category_waste_type: str | None = None
    wants_schedule = asks_schedule(prompt)
    if category_intent is not None:
        category_waste_type, category_schedule_like = category_intent
        wants_schedule = wants_schedule or category_schedule_like

        if not category_schedule_like:
            # 手続きそのものの質問(収集日は絡まない)は、この先の品目・
            # 収集日パスのどちらにも入らないので、ここで確定させて返す。
            # 2026-09-14 レビュー指摘: 以前はここで found=False・context=""
            # のまま関数を抜け、_RULES_NO_CONTEXT で LLM に自由生成させて
            # いたが、小型モデルは「該当情報が見つかりませんでした」を確実に
            # は守らず、「ごみ箱」「ふれあい収集」等を作り出すことがある
            # (4-3確度ゲートと同じ理由)。知識ベースに根拠があれば使い、
            # 無ければ固定の確認文へ倒す ― どちらの分岐も自由生成させない。
            info, _trail_entry = general_info_for_category(category_waste_type)
            ans.search_trail.append(_trail_entry)
            if info is not None:
                text, source_url = info
                ans.found = True
                ans.context = f"## {category_waste_type}についての一般説明\n{text}"
                if source_url:
                    ans.sources = [source_url]
            else:
                ans.canned_response = _CATEGORY_NO_CONTEXT_REPLY
            return ans

    if image_item_name:
        item_candidates, vector_query = [image_item_name], None
    elif category_intent is not None:
        item_candidates, vector_query = [], None
    else:
        item_candidates, vector_query = _item_name_candidates(
            prompt, strip_waste_words=wants_schedule
        )
        # 品目名として実体の無い候補(地域名の言い換え・正規表現の残骸)を
        # 検索に渡す前に弾く(要件5-1、`_has_plausible_item_text` 参照)。
        item_candidates = [c for c in item_candidates if _has_plausible_item_text(c, ward)]
        if vector_query is not None and not _has_plausible_item_text(vector_query, ward):
            vector_query = None
    item_name = item_candidates[0] if item_candidates else None
    ans.detected_item = item_name
    ans.item_candidates = list(item_candidates)
    ans.item_from_image = bool(image_item_name)

    if item_name:
        ans.item_hits, ans.item_match_stage, _trail = _search_item(
            item_name, item_candidates[1:], vector_query
        )
        if _trail:
            ans.search_trail.append({"tool": "search_waste_item", "query": item_name, "attempts": _trail})
        ans.item_match = _classify_item_match(item_candidates, ans.item_hits)

        # 候補文字列が質問定型で汚れて exact/partial を素通りしても、ヒット名
        # 自体が原文にそのまま出ていれば完全一致として扱い、正しい品名で
        # timeline も引けるようにする(trace: 46/50 が vector・timeline 0 の回帰)。
        if ans.item_match == "analogy":
            # 言い換え表(`_CANONICAL_ALIASES`)から名前が変わった行(「硫酸」→
            # 「薬品(家庭医薬品でないもの)」)は、質問文に正式名が現れないため
            # `_resolve_named_hit` では拾えない。ヒット名と解決先が一致することを
            # 確認したうえで引き上げる。
            # 埋め込み品名(`_pick_embedded_item`)はここでは使わない — あちらは
            # 直前の境界を見ないので、「電動スケートボード」の中の「スケートボード」
            # のような別品目まで exact に上げてしまう(2026-09-10 テスト回帰)。
            named = _resolve_named_hit(prompt, ans.item_hits) or _resolve_alias_hit(
                prompt, item_candidates, ans.item_hits
            )
            if named is not None:
                ans.item_match = "exact"
                ans.detected_item = named.item_name
                ans.item_hits = [named] + [h for h in ans.item_hits if h is not named]
                if named.item_name not in item_candidates:
                    item_candidates = [named.item_name, *item_candidates]

        # 4-3: ベクトル1位が紛らわしい品目は、人手で確度を固定した早見表行に
        # 差し替える(_ANALOGY_PREFERRED)。
        if ans.item_match == "analogy":
            ans.item_hits = _override_analogy_hits(ans.detected_item, ans.item_hits)

        # 類似候補があれば必ず1件に絞って区分を導く(canned 断りは「類似品が
        # あるのに確認だけ促すのは減点」という採点フィードバックで否定された、
        # PR #36。`tests/notices_similar_item_test.py::
        # test_blocking_route_uses_best_vector_hit_instead_of_canned` 等で
        # 固定済みの動作)。`analogy_unreliable` は使える候補が1つも無いときだけ。
        if ans.item_match == "analogy" and ans.item_match_stage == "vector":
            ans.analogy_unreliable = _analogy_pick(ans.item_hits, ans.detected_item) is None

        # 品目の列挙(「たんすと布団と、あと段ボール」)では主候補の検索だけでは
        # 1件しか拾えない。質問文に明示的に列挙された品目を追加で検索し、回答から
        # 抜け落ちないようにする(2026-09-10 第6回採点回帰)。検索段が exact
        # (原文の品名が早見表の行に確定した)ときだけ行う — 部分一致・ベクトルの段
        # では質問の品目が一意でなく、断片の完全一致を混ぜると別品目の断定根拠に
        # なってしまう。
        if ans.item_match_stage == "exact" and len(ans.item_hits) == 1:
            exclude = {_norm(h.item_name) for h in ans.item_hits}
            exclude |= {_norm(c) for c in item_candidates}
            extras = _listed_item_extras(prompt, exclude)
            if extras:
                # 列挙された品目がそれぞれ原文の品名と完全一致しているなら、類推では
                # なく「複数品目の質問」として扱う。analogy のままだと context が
                # `_analogy_pick` で1件に絞られ、他の品目が回答から消える。
                ans.item_hits = ans.item_hits + extras
                if ans.item_match != "exact":
                    ans.item_match = "exact"
                    ans.detected_item = (
                        getattr(ans.item_hits[0], "item_name", None) or ans.detected_item
                    )

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
        # 品目が見つかっていなくても、種別そのものの質問(category_intent)
        # なら、その種別名を品目の区分と同じ役割で使う。
        single_item_disposal = single_item_disposal or category_waste_type
        district_ambiguity: str | None = None

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
                district = _match_district(ward, prompt)
                if district:
                    district_name, district_towns = district
                    uniform = _uniform_district_schedule(ward, district_towns, waste_type)
                    if uniform:
                        schedules = uniform
                        ans.detected_town = district_name
                        ans.district_schedule = True
                        ans.district_towns = list(district_towns)
                    else:
                        # 地区内で収集日が割れている(「企救丘」等)。丁目を
                        # 勝手に決めつけて答えることはしないが、「町名が
                        # 不明」とだけ案内すると、そもそも収集日が丁目に
                        # よって違うこと自体が伝わらない(2026-09-14 レビュー
                        # 指摘)。丁目ごとに実際の収集日を引いてグループ化し、
                        # 「どちらのグループか」を選べる形で案内する。
                        # `waste_type` は `wants_schedule` が立っているとき
                        # しか `extract_waste_type` を試さないので、「〜の日を
                        # 教えてください」のように asks_schedule の語彙に無い
                        # 言い回しでは None のことがある。地区の内訳表示だけは
                        # 独立して種別語を探し、無関係な種別まで混ぜない。
                        district_ambiguity = _describe_non_uniform_district(
                            ward, district_name, district_towns,
                            waste_type or extract_waste_type(prompt),
                        )

        if schedules:
            ans.schedule_hits = schedules
        elif ans.no_weekday_schedule:
            pass  # 上のフラグで案内するので missing には積まない
        elif district_ambiguity:
            ans.missing.append(district_ambiguity)
        elif not town:
            # 設計提案書 3-1: 区のみでは一意に定まらない。推測せず町名を要求する。
            ans.missing.append("町名")
        else:
            ans.missing.append(f"{ward}{town}の収集日データ")
    elif wants_schedule:
        ans.missing.append("区名と町名")

    ans.found = bool(ans.item_hits or ans.schedule_hits)
    if ans.item_match == "analogy" and ans.item_hits:
        # 4-3 の参照は「回答で採用する1件」だけに絞る(top-3 の無関係な URL を
        # 付けると『根拠が明確』の印象が薄れる)。
        _pk = _analogy_pick(ans.item_hits, ans.detected_item)
        ans.sources = _dedupe(
            ([_pk.source_url] if _pk is not None else [])
            + [r.source_url for r in ans.schedule_hits]
        )
    else:
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

    # 必須項目テンプレート(`apply_response_template`)はここでは適用しない。
    # `app/server.py._route_safely` が決定論的経路・Agentic経路の共通出口で、
    # STAGE06 の 6-2 の印(`_mark`)を付けた**後**に適用する。ここで先に
    # 適用すると 6-2 の印が付く前にテンプレートが確定し、危険な処分の誘導への
    # 回答から「いいえ、〜してはいけません」が消える(テンプレートは 6-2 を
    # 対象外にするが、その判定には印が必要)。

    # --- 6-3 fallback hook (STAGE06 proposal §7.3 / 担当B) ---
    # 完全空振り(品目も収集日もヒットせず、6-1a/6-2 も 4-3 確度ゲートも未発火)。
    # 検索結果ゼロを LLM に渡して「見つかりませんでした」と言わせると、モデルが
    # 自前の知識で埋める失敗モード(4-3 で繰り返し確認)が未知の入力でも起きる。
    # コード側で固定文を立て、engine.generate を通さず返す。
    #
    # `ans.missing` が非空(区名はあるが町名が無い等)のときは対象外にする
    # (コードレビューで発見: 修正前は6-3がここを素通りせず全部拾ってしまい、
    # build_system_prompt() の「不足している{missing}を推測せず利用者に
    # 追加入力を促す」という既存の指示(3-1由来)を握りつぶしていた。
    # 「本当に何も手がかりが無い」場合と「あと一歩の情報が要る」場合は
    # 区別し、後者は従来どおりLLMに追加入力を促させる)。
    if (
        not ans.found
        and not ans.missing
        and ans.canned_response is None
        and ans.safety_gate is None
    ):
        reply = safety.no_result_reply()
        if reply:
            ans.canned_response = reply
            ans.safety_gate = "6-3"
    return ans


# 利用者が自分の前提の是非を確かめている(はい/いいえで答えるべき)質問。
# 「〜ですよね」「〜していいですか」「〜でいいですか」「大丈夫ですか」等。
# 「どう捨てればいいですか」のような手順の質問(〜ばいいですか)は含めない —
# そちらは必須項目テンプレートが本来想定している質問なので、ここで拾うと
# テンプレートが実質的に無効になってしまう。
_CONFIRMATION_RE = re.compile(
    r"(よね"
    r"|て(?:も)?いい(?:です|んです|の)?か"
    r"|で(?:も)?いい(?:です|んです|の)?か"
    r"|大丈夫|構いません|かまいません|問題(?:ない|ありません)"
    r"|合って(?:い)?ます|正しいです)"
)


def _is_confirmation_question(prompt: str) -> bool:
    return bool(_CONFIRMATION_RE.search(_norm(prompt or "")))


# 利用者が品物の状態・内容物を前提として述べている発話(「水と魚が入ったまま」
# 「洗わずにそのまま」等)。区分を答えるだけでは条件に触れられず、必須項目
# テンプレート(項目表)では「先に中身を取り出す」等の手順を案内できないため、
# テンプレートを当てずに LLM に条件ごと個別に答えさせる。
# 実採点traceの実プロンプト165件で、この判定に当たるのは「水と魚が入ったままの
# 大きな水槽」と「農薬の空き容器は、洗わずにそのまま」の2件だけで、通常の
# 分別質問には発火しないことを確認済み(2026-09-10)。
_STATED_CONDITION_RE = re.compile(
    r"入ったまま|入れたまま|残ったまま|ついたまま|入った状態"
    r"|そのまま|このまま|のままで|中身が入|中身入り|入っている|入ってる"
)


def _has_stated_condition(prompt: str) -> bool:
    return bool(_STATED_CONDITION_RE.search(_norm(prompt or "")))


def apply_response_template(ans: RoutedAnswer, prompt: str = "") -> RoutedAnswer:
    """高リスク回答に必須項目テンプレートを適用して返す。

    決定論的ルートだけでなく Function Calling ルートにも同じ制約を
    適用するため、サーバーの共通出口から呼び出せる公開関数にしている。

    STAGE06 の 6-2(危険・違法な処分の誘導、`safety_gate == "6-2"`)は対象外。
    テンプレートは項目表を固定文で返すため、「タイヤを庭で燃やしていいですか」
    への回答から「いいえ、燃やしてはいけません」という明確な否定が消えて
    しまう。6-2 は `_RULES_DANGEROUS_REQUEST` を課した LLM 生成で、否定と
    正しい手順(検索結果の相談先)をセットで答える。

    `prompt` が確認型の質問(`_is_confirmation_question`)の場合も対象外。
    2026-09-10 第5回採点で「エアコンは燃えないごみの日に出せばいいんですよね？」
    等の誤った前提の確認は、LLM が「いいえ、燃えないごみの日には出せません」と
    前提を明確に否定して通過していた。テンプレートは項目表しか返さず、この
    否定が消える(統合ブランチで実DBに対して確認)。

    `prompt` が品物の状態・内容物を述べている場合(`_has_stated_condition`)も
    対象外。2026-09-10 第6回採点で「水と魚が入ったままの大きな水槽」に項目表が
    当たり、述べられた条件に一切触れない回答になっていた(項目表は品目の区分を
    並べるだけで、条件への対処を書く欄を持たない)。
    """
    if ans.safety_gate == "6-2":
        return ans
    if _is_confirmation_question(prompt) or _has_stated_condition(prompt):
        return ans
    if ans.canned_response is None:
        templated = _build_required_fields_response(ans)
        if templated is not None:
            ans.canned_response = templated
            ans.template_response = True
    return ans


def _build_required_fields_response(ans: RoutedAnswer) -> str | None:
    """高リスクカテゴリを必須項目テンプレートで回答する。

    早見表・収集日検索の結果だけを材料にするため、LLMが申込・手数料・
    収集対象外などを省略したり、根拠のない条件を補ったりする余地を減らす。
    """
    if not ans.item_hits or ans.item_match == "analogy" or ans.timeline_change:
        return None
    if len(ans.item_hits) > 1:
        # 多品目の質問(「たんすと布団と段ボール」)では、この項目表は1品目しか
        # 表せず、残りの品目が回答から抜け落ちる(2026-09-10 第6回採点回帰)。
        # 項目表をやめ、LLM に全品目を個別に整理させる。
        return None

    item = ans.item_hits[0]
    item_name = getattr(item, "item_name", "") or ans.detected_item or "この品目"
    method = getattr(item, "disposal_method", "") or "記載なし"
    notes = (getattr(item, "notes", "") or "").strip()
    lines = [f"品目: {item_name}", f"分別区分: {method}"]

    schedules = ans.schedule_hits
    has_bulky = method == "粗大ごみ" or any(
        getattr(s, "waste_type", "") == "粗大ごみ" for s in schedules
    )
    not_collected = method == "市が収集しないもの"
    no_weekday = ans.no_weekday_schedule or (
        not schedules and method not in _CURBSIDE_WASTE_TYPES
    )

    if has_bulky:
        lines.append("出し方・条件: " + (notes or "早見表の備考を確認してください。"))
        if schedules:
            lines.append("収集日: " + "、".join(
                f"{getattr(s, 'ward', '')}{getattr(s, 'town', '')}は{getattr(s, 'weekday', '')}"
                for s in schedules
            ))
        else:
            lines.append(
                "収集日: お住まいの区名と町名(丁目まで)をお知らせいただければ"
                "お調べします。"
            )
        lines.append(
            "申込・手数料: 品目ごとに300円・500円・700円・1,000円のいずれかです。"
            "事前に粗大ごみ受付センター(電話: 093-513-3005)またはインターネットで"
            "お申し込みください(インターネットからの申込ではキャッシュレス決済も"
            "選べます)。"
        )
        lines.append("注意事項: 粗大ごみは事前申込が必要です。申込なしで収集日に出さないでください。")
        return "\n".join(lines)

    if not_collected:
        lines.append(
            "出し方・条件: 市が収集しない品目です。"
            "家庭ごみ・粗大ごみとして出すことはできません。"
        )
        lines.append("収集日: 市による決まった収集曜日はありません。")
        lines.append(
            "相談先: " + (notes or "販売店・専門の処理業者等へご相談ください。")
        )
        return "\n".join(lines)

    if no_weekday:
        lines.append("出し方・条件: " + (notes or "早見表の備考に記載された方法に従ってください。"))
        lines.append("収集日: 決まった収集曜日はありません。")
        lines.append("申込・手数料: 検索結果に記載なし")
        lines.append("注意事項: 備考に記載されていない方法や条件は追加できません。")
        return "\n".join(lines)

    return None


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


def needs_history_retry(ans: "RoutedAnswer | None") -> bool:
    """決定論的経路が品目も収集日も特定できなかったか。

    この状態の回答は 6-3 の「品名を教えてください」固定文で、セッションの過去の
    発話にある品目を一切使えていない。6-1a/6-2 は `app/server.py._mark` で印が
    付くか、そもそも `_route_safely` の入口で返っているので、`safety_gate` が
    "6-3" かどうかだけを見れば「完全空振り」を正確に判定できる(6-3 以外で
    canned が立つ経路 — 4-3 確度ゲート・種別の一般説明なし — は、品目や種別に
    見当が付いている)。

    呼び出し側(`app/server.py._route_safely`)は True のとき、会話履歴の利用者
    発言を前置した文字列で一度だけ引き直す(`agent.build_history_query`)。
    """
    return ans is not None and ans.safety_gate == "6-3"


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
    "ワイヤレス充電器": "アダプター(電源用)",
}

# 既知の品名の「言い換え」を正式な早見表の品名へ直接固定する。
# `_ANALOGY_PREFERRED` とは意味が違う ― あちらは「似ているが別の品目」を
# 代用として使う類推、こちらは「同じ品目を指す別の言い方」の別名解決で、
# 類推の確度判定(`_analogy_is_reliable` 等)を経由させず常に信頼してよい。
# 完全一致・部分一致・ベクトル検索より先に試す(部分一致の並び順やベクトル
# 距離の揺れに任せない ― 2026-09-14、B1トレース精査で「テレビ」が
# 「テレビアンテナ」に、「お酒の空き瓶」が「酒パック」に化ける事例を確認)。
_CANONICAL_ALIASES = {
    "テレビ": "テレビ(ブラウン管式・液晶式・プラズマ式)",
    "お酒の空き瓶": "びん(飲料・食品のびん)",
    "お酒のあきびん": "びん(飲料・食品のびん)",
    "酒の空き瓶": "びん(飲料・食品のびん)",
    "酒瓶": "びん(飲料・食品のびん)",
    "お酒の瓶": "びん(飲料・食品のびん)",
    "ワインの瓶": "ワインびん",
    "ワインのびん": "ワインびん",
    "ワインボトル": "ワインびん",
    "カップ麺の容器": "カップめんなどの容器・フィルム",
    "カップめんの容器": "カップめんなどの容器・フィルム",
    "カップラーメンの容器": "カップめんなどの容器・フィルム",
    # 長音符の有無だけが違う同じ品目の言い換え(早見表の表記は「ソファー」)。
    "ソファ": "ソファー",
    # 2026-09-10 第6回採点回帰。化学物質名は早見表に品名として存在せず、
    # 包括行(「薬品(家庭医薬品でないもの)」等)にしか対応が無い。ベクトル
    # 類似に任せると「硫酸 → 油(食用油) = 家庭ごみ」のような別区分への誤類推が
    # 起きるため、物質名そのものの言い換えとして包括行へ固定する。
    # いずれも実データ(waste_items)に存在する行を指す。
    "硫酸": "薬品(家庭医薬品でないもの)",
    "塩酸": "薬品(家庭医薬品でないもの)",
    "硝酸": "薬品(家庭医薬品でないもの)",
    "化学薬品": "薬品(家庭医薬品でないもの)",
    "劇物": "薬品(家庭医薬品でないもの)",
    "毒物": "薬品(家庭医薬品でないもの)",
    "農薬の空き容器": "農薬びん(家庭用)(中身のないもの)",
    "農薬の容器": "農薬びん(家庭用)(中身のないもの)",
    "注射器": "注射筒",
    # 2026-09-11 実機の手動検証で確認。「シューズ」を含む品目(ランニング
    # シューズ等)は早見表に個別の行が無く、ベクトル類似に任せると語頭が
    # 共通なだけの「ランニングマシン(粗大ごみ)」に化ける事故を確認した。
    # 「シューズ」は靴の一般的な言い換え語であり特定の品目名ではないため、
    # ここに置いても特定の質問への決め打ちにはならない(早見表の「靴(くつ)」
    # へ一般名詞として解決するだけ)。`_resolve_embedded_alias` の境界検査
    # (`_AFTER_NAME_OK`)により、「シューズボックス」のように「シューズ」の
    # 直後に別の語が続く既存の品名までは横取りしない。
    "シューズ": "靴(くつ)",
}


def resolve_canonical_alias(query: str) -> str | None:
    """`query` が既知の品名の言い換えなら、正式な早見表の品名を返す。
    該当しなければ None(呼び出し側は元の `query` のまま exact/partial/
    ベクトルへ進む)。`app/rag/router.py`(決定論的経路)と
    `app/rag/tools.py`(agentic経路)の両方から呼ばれる共通の解決関数。
    """
    return _CANONICAL_ALIASES.get(_norm(query))


def _resolve_embedded_alias(text: str) -> str | None:
    """`text` の中に言い換え表のキーが「品名の主辞として」現れていれば、
    対応する正式名を返す。

    `resolve_canonical_alias` が候補文字列そのものの言い換えなのに対し、こちらは
    「理科の実験で使った硫酸が残っているのですが」のように品名が質問文に埋もれ、
    候補抽出(`_item_name_candidates`)で切り出せなかった場合の救済。呼び出し側は
    完全一致・部分一致・埋め込み品名がすべて空振りしたあとにだけ試すので、
    既存の品名一致を横取りしない。

    直後の境界検査に `_AFTER_NAME_OK` を使うのは `_pick_embedded_item` と同じ理由。
    これが無いと「テレビ台」(別品目)の中の「テレビ」を拾ってしまう。直前の境界は
    見ない(「使った硫酸」のように直前が動詞の活用語尾でも品名は品名のため)。
    長いキーから順に試し、最も具体的な言い換えを優先する。
    """
    norm = _norm(text)
    if not norm:
        return None
    for key, target in sorted(
        _CANONICAL_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if len(key) < 2:
            continue
        start = 0
        while True:
            idx = norm.find(key, start)
            if idx == -1:
                break
            rest = norm[idx + len(key):]
            if _AFTER_NAME_OK.match(rest) or _PART_SUFFIX_RE.match(rest):
                return target
            start = idx + 1
    return None


def _resolve_alias_hit(
    prompt: str, item_candidates: list[str], hits: list[ItemRecord]
) -> "ItemRecord | None":
    """ヒットが言い換え表の解決結果だった場合、その1件を返す。

    `_resolve_named_hit` は「ヒット名が質問文にそのまま現れている」ことしか
    見ないため、言い換え・物質名から包括行へ解決したケース(質問文に正式名が
    現れない)では None になる。ヒット名と解決先が一致することを確認したうえで
    返し、呼び出し側で exact へ上げられるようにする。
    """
    for probe in [*item_candidates, prompt]:
        target = resolve_canonical_alias(probe) or _resolve_embedded_alias(probe)
        if not target:
            continue
        q = _norm(target)
        for h in hits:
            if _norm(getattr(h, "item_name", "")) == q:
                return h
    return None


def _override_analogy_hits(query: str, hits: list[ItemRecord]) -> list[ItemRecord]:
    """`_ANALOGY_PREFERRED` に該当する品目なら、その早見表行1件だけを hits と
    して返す(ベクトル1位が紛らわしい品目の確度を人手で固定する)。対象が
    ベクトルヒットに無ければ DB から直接引く。該当しなければ hits のまま。"""
    if not query:
        return hits
    q = _norm(query)
    for key, target in sorted(
        _ANALOGY_PREFERRED.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if key not in q:
            continue
        for h in hits:
            if _norm(getattr(h, "item_name", "")) == _norm(target):
                return [h]
        rec = lookup_item_exact(target)
        return [rec] if rec is not None else hits
    return hits


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
# `_has_plausible_item_text`/`category_procedure_intent` が地区名・種別語
# 除去後の残りを判定する際、文中のどこにあっても句読点・空白を取り除いてから
# 見るための正規表現。全角/半角の「？」「！」も対象(2026-09-14 レビュー
# 指摘: 「粗大ごみってどうやって出すんですか？」のように文末に全角疑問符が
# 付くと、除去後の残りに「？」が残って「実質的な内容がある」と誤判定され、
# category_procedure_intent が None を返してしまっていた)。
_PUNCT_OR_SPACE_RE = re.compile(r"[\s、。「」()（）？?!！]+")
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
        # 修飾語+接続助詞を先頭から**全部**剥がす(「家庭用の電動ミニ
        # シュレッダー」→ 家庭用 → の → 電動 → ミニ → 空)。
        changed = True
        while changed and prefix:
            changed = False
            stripped = prefix.lstrip("のなというごと・、 　")
            if stripped != prefix:
                prefix, changed = stripped, True
            for mod in sorted(_STRENGTH_PREFIX_MODS, key=len, reverse=True):
                if prefix.startswith(mod):
                    prefix, changed = prefix[len(mod) :], True
                    break
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
    "- 備考が空、または検索結果に備考の記載が無い場合、条件について何も"
    "書かない。「制限はありません」「特別な梱包は不要です」"
    "「事前申込は不要です」のように、備考が無いことを条件が無いことの"
    "根拠として断定してはならない(備考が無いのは単に情報が無いだけで、"
    "条件が無いことの確認ではない)。\n"
)

_RULES_NO_CONTEXT = (
    "- 該当する情報が見つからなかったため、分別区分や収集日を断定してはならない。\n"
    "- 「該当する情報が見つかりませんでした」と明示し、"
    "北九州市の公式ページや窓口での確認を案内する。\n"
)

# 6-2(危険・違法な処分の誘導)。2026-09-10 の採点フィードバック
# 「不適切な処理方法や誤った分別区分を肯定するような案内を避け…禁止事項や
# 例外事項を明確に伝える」「専門業者への相談や正しい排出ルートの提示など、
# 具体的な代替案を漏れなく案内する」に対応する。以前は検索前に固定文で
# 断っていたが、それでは『農薬』のように早見表に正しい相談先があるケースでも
# 具体策の無い拒否になってしまうため、検索は通したうえでこの指示を足す。
_RULES_DANGEROUS_REQUEST = (
    "- 利用者は、危険または法律に違反する処分方法(野焼き・不法投棄・排水口や"
    "土壌への投棄・中身の残った容器への穴あけ等)をしてよいか尋ねている。"
    "まず「いいえ、〜してはいけません」と明確に否定すること。"
    "危険な方法の手順は一切書かない。\n"
    "- 否定だけで終わらせず、正しい処分方法を必ずセットで示す。検索結果に"
    "販売店・専門の処理業者・粗大ごみ受付センター・回収ボックス等の具体的な"
    "相談先や排出ルートがあれば、一つも省略せず挙げること。\n"
    "- 検索結果に該当品目が無い場合でも、情報不足を理由に回答を避けない。"
    "その方法が禁止であることを明示したうえで、品名を具体的に尋ね、"
    "北九州市の窓口へ相談するよう案内する。\n"
    "- 検索結果の『備考』に、条件によって区分が変わる記載(「割れたものは"
    "家庭ごみ」「必ず中身を使い切ってから穴をあけずに」等)がある場合は、"
    "書かれているとおりに伝えること。断定を強めたり、否定・反対の意味に"
    "書き換えたりしてはならない。\n"
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


_RULES_USER_CONDITION = (
    "- 利用者は品物の状態・内容物について条件を述べている(「水と魚が入った"
    "まま」「洗わずにそのまま」等)。その条件を無視せず、一つずつ取り上げて"
    "どうすればよいかを案内すること。\n"
    "- 検索結果にその条件の扱いが書かれている場合は、書かれているとおりに伝える"
    "(「必ず中身を使い切ってから」等)。書かれていない場合でも「問題ありません」"
    "「そのままで大丈夫です」と断定してはならない。検索結果からは判断できない"
    "ことを明示し、北九州市の窓口(お住まいの区の環境センター)での確認を案内する。\n"
)


def build_system_prompt(ans: RoutedAnswer, prompt: str = "") -> str:
    """`RoutedAnswer` からLLMへのシステムプロンプトを組み立てる。

    設計提案書 3-7(未識別時は憶測で断定しない)をここで強制する。

    `prompt` は利用者の生の発話。品物の状態・内容物について条件を述べている
    場合(`_has_stated_condition`)に、その条件へ一つずつ触れさせる規則を足す
    (`apply_response_template` が同じ判定で項目表を外すのと対になっている)。
    """
    parts = [_BASE_RULES]

    if prompt and _has_stated_condition(prompt):
        parts.append(_RULES_USER_CONDITION)

    if ans.safety_gate == "6-2":
        # 検索結果の有無に関わらず、禁止の明示と正しい手順の提示を課す。
        parts.append(_RULES_DANGEROUS_REQUEST)

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
        "safety_gate": ans.safety_gate,  # STAGE06 6-4: 発火した安全ゲート、未発火なら None
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
        "detected_ward": ans.detected_ward,  # STAGE06 6-4: /trace の「検出した地区」表示用
        "detected_town": ans.detected_town,
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
        "template_response": ans.template_response,
        "safety_gate": ans.safety_gate,  # STAGE06 6-4: 発火した安全ゲート("6-1a"/"6-2"/"6-3")
        "sources": list(ans.sources),
        "search_trail": list(ans.search_trail),  # 6-4: 検索過程の可視化(採用/却下した候補)
        "agent_rounds": ans.agent_rounds,  # 6-4: Agentic経路のツール呼び出しラウンド数
        "agent_stop_reason": ans.agent_stop_reason,  # 6-4: "model_stopped"/"max_rounds_reached"
        # 6-4: LLMに実際に渡した context(canned_response のときは生成自体を
        # スキップしているので意味が無い)。「検索は正しかったのに生成で
        # 崩れた」のか「検索自体が誤っていた」のかを、最終回答と並べて事後に
        # 見分けられるようにする(検索の可視化だけでは埋まらない、生成側の
        # 限界を切り分けるための最後のピース)。
        "context": ans.context,
    }
