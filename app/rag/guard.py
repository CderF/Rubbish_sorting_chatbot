"""PIJ(プロンプトインジェクション)対策 — 入力の正規化と攻撃判定。

課題の「PIJ要件」が挙げる9種類の攻撃のうち、入力側で止められるものをここで
止める(間接注入はデータ取り込み側、システムプロンプト漏洩の最終確認と
出力形式ハイジャックは出力側で別途扱う)。

設計の前提:

- 8Bの小型モデルにプロンプトで「従うな」と書いても守らせきれないことは、
  4-3 の確度ゲート(#29)や 4-2 の時系列回答(#30)で実証済み。検知した攻撃は
  `RoutedAnswer.canned_response` に載せ、`engine.generate` を通さない。
  モデルが関与しなければ、その入力に対する挙動は完全に決定的になる。
- 「ほぼ同一のプロンプトはブロックされます」という課題ルール上、攻撃側は
  言い換えを重ねてくる。単語の列挙では取りこぼすため、判定は
  「メタ語(指示・プロンプト・役割)×操作語(無視・出力・変更)」の共起で見る。
  単語1つでは発火させない。
- 誤検知は分別回答の精度をそのまま削る。ごみ分別の語彙と衝突する語
  (「ルール」「規則」「設定」「パスワード」単体、URL 中の `.html` など)は
  意図的に判定から外している。`tests/prompt_injection_test.py` が実採点50問
  に対する誤検知ゼロを固定している。
"""

from __future__ import annotations

import re
import secrets
import unicodedata

# 検索結果をシステムプロンプトに埋めるときの境界マーカー。プロセス起動ごとに
# 引き直すのが要点で、固定文字列だと外部データ側に同じ文字列を書いて「囲いを
# 閉じてから指示を書く」ことができてしまう。攻撃者は事前に値を知り得ない。
DATA_FENCE = f"DATA-{secrets.token_hex(4)}"

# 生成に渡すプロンプトの上限。実採点の質問は最長でも50文字程度で、これを
# 超える入力は正当な分別質問ではない。`MAX_MODEL_LEN`(8192)を溢れさせて
# `engine.generate` を例外にする攻撃(=SSEが途中で切れ、採点上は失点)も
# ここで同時に塞ぐ。
MAX_PROMPT_CHARS = 2000

# 画像分類パスが返す品名の上限。早見表の実データで最長は30文字弱。
MAX_ITEM_NAME_CHARS = 40

# 検知時に返す固定回答。攻撃を検知したこと自体は伝えない(伝えると攻撃側に
# 判定境界の手がかりを与える)。案内できる範囲だけを述べる。
REFUSAL = (
    "ごみの分別・収集日に関するご質問にのみお答えできます。"
    "知りたい品物の名前や、お住まいの区・町名を教えてください。"
)

# --- 正規化 ---------------------------------------------------------------
# chat template の特殊トークン。`processor.apply_chat_template` は平文を
# そのまま埋め込むため、これを剥がさないとユーザー入力から偽の system ターンを
# 合成できてしまう。役割偽装の中で唯一、本当に役割境界を越えられる経路。
_SPECIAL_TOKEN_RE = re.compile(r"<\s*\|[^|<>]{0,64}\|\s*>")

# 他モデル系列のテンプレート記法。Qwen では意味を持たないが、モデルが
# 学習データ中でこれらを境界として見ている可能性があるため同様に落とす。
_ROLE_TAG_RE = re.compile(
    r"(?:\[/?INST\]|<</?SYS>>|<\|?(?:endoftext|start_header_id|eot_id)\|?>)",
    re.IGNORECASE,
)


def _strip_invisible(text: str) -> str:
    """ゼロ幅文字・双方向制御文字・制御文字を除く(改行とタブは残す)。

    NFKC はこれらを落とさないので、正規化の前に別途行う必要がある。
    `シ​ステムプロンプト` のようにゼロ幅スペースを挟んで語を割る難読化は、
    これを通さないと以降のパターン照合を素通りする。
    """
    return "".join(
        ch
        for ch in text
        if ch in "\n\t"
        or unicodedata.category(ch) not in ("Cc", "Cf", "Co", "Cs")
    )


def normalize_input(text: object) -> str:
    """ユーザー入力を、判定にもモデルにも安全に渡せる平文へ正規化する。

    順序に意味がある: ゼロ幅除去 → NFKC → 特殊トークン除去。全角の
    `＜｜im_start｜＞` は NFKC を通して初めて `<|im_start|>` になるため、
    特殊トークン除去は NFKC の後でなければ素通りする。
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = _strip_invisible(text)
    text = unicodedata.normalize("NFKC", text)
    text = _SPECIAL_TOKEN_RE.sub(" ", text)
    text = _ROLE_TAG_RE.sub(" ", text)
    return text[:MAX_PROMPT_CHARS]


def sanitize_item_name(text: object) -> str:
    """画像分類パス(VLM出力)や検索結果の品名を、context に埋める前に均す。

    画像に書き込んだ文字は VLM がそのまま読み上げるため、品名として
    `RoutedAnswer.detected_item` に入り、`_build_context` の f-string 経由で
    システムプロンプトに載る。改行を潰して1行に畳み、長さを切る。
    """
    cleaned = normalize_input(text)
    cleaned = " ".join(cleaned.split())
    return cleaned[:MAX_ITEM_NAME_CHARS]


# 取り込むテキストの上限。配布お知らせの実データは本文が最長109文字、
# タイトル36文字。5倍以上の余裕を取りつつ、注入されたペイロードの長さは
# 抑える。正当なデータが切られることは実質ない。
MAX_BODY_CHARS = 600
MAX_TITLE_CHARS = 200

# 行頭のマークダウン見出し・引用記号。`_build_context` は `## 見出し` で
# セクションを区切るので、取り込むテキストにこれが残っていると偽のセクション
# ("## 分別早見表の検索結果" 等)を作られる。
_LINE_MARKUP_RE = re.compile(r"^[ \t]*[#>*\-=+]+[ \t]*", re.MULTILINE)

# 行頭の疑似ロール宣言。同上の理由で落とす。
_LINE_ROLE_RE = re.compile(
    r"^[ \t]*(?:system|assistant|user|developer)[ \t]*[:：]",
    re.MULTILINE | re.IGNORECASE,
)


def sanitize_text(text: object, max_chars: int = MAX_BODY_CHARS) -> str:
    """外部データ由来のテキストを、`context` に載せられる形に均す。

    PIJ「間接注入」対策。配布お知らせの本文やクロールしたページの備考は、
    `build_system_prompt` を通してシステムプロンプトの一部になる — ユーザー
    入力より強い位置に入るので、入力側より丁寧に潰す必要がある。

    改行を空白に畳むのは、注入対策であると同時に構造上の修正でもある。
    `_build_context` は `- 品名: …` / `  出し方: …` のインデントで項目を
    並べるが、実データの本文には改行が含まれており(配布15件中の 4-1 全件)、
    そのまま埋めると2行目がインデントの外に出て、あたかも別の項目や別の
    セクションであるかのように見える。
    """
    cleaned = normalize_input(text)
    cleaned = _LINE_MARKUP_RE.sub("", cleaned)
    cleaned = _LINE_ROLE_RE.sub("", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_chars]


# --- 攻撃判定 -------------------------------------------------------------
def _cooccur(a: str, b: str, window: int = 40) -> re.Pattern[str]:
    """語 a と語 b が `window` 文字以内に共起する(順不同)パターン。

    単語1つでの発火を避けるための基本形。窓を切ってあるのは、長い正当な
    文の中でたまたま両方が出現しただけのケースを拾わないため。
    """
    return re.compile(
        rf"(?:{a}.{{0,{window}}}{b})|(?:{b}.{{0,{window}}}{a})",
        re.IGNORECASE | re.DOTALL,
    )


# モデル自身への指示を指す語。「ルール」「規則」「設定」は
# 「ごみ出しのルール」等と衝突するため入れない。
_META = (
    r"(?:システム\s*プロンプト|システム\s*メッセージ|初期(?:の)?(?:指示|プロンプト)"
    r"|上記(?:の)?(?:指示|命令|文章|プロンプト)|これまでの(?:指示|命令)"
    r"|前(?:の|述の)(?:指示|命令)|あなた(?:へ|に)の(?:指示|命令)"
    r"|system\s*prompt|initial\s*instructions?|your\s*instructions?"
    r"|系统提示(?:词)?|提示词)"
)

# 「指示」「命令」単体。無効化系の動詞とだけ組ませる(開示系と組ませると
# 「市の指示を教えて」のような正当な質問を拾ってしまう)。
_ORDER = r"(?:指示|命令|instructions?|prompt|指令)"

_NEGATE = (
    r"(?:無視|忘れ|破棄|解除|上書き|書き換え|従わ(?:ない|ず|なくて)"
    r"|ignore|disregard|forget|override|bypass|忽略|无视)"
)

_REVEAL = (
    r"(?:出力|表示|開示|全文|そのまま|復唱|教えて|見せて|answer with"
    r"|reveal|show me|print|repeat|display|输出|显示|告诉我)"
)

# 「あなたは今から◯◯」の◯◯にあたる役割語。これが無ければ発火させない
# (「あなたは今から私のごみ分別を手伝って」のような正当文を弾かないため)。
_ROLE_NOUN = (
    r"(?:ai|アシスタント|モデル|ボット|専門家|エンジニア|翻訳(?:家|者)?"
    r"|モード|ロール|キャラクター|人格|管理者|開発者|assistant|model)"
)

# 各パターンは PIJ要件のスライドが挙げる攻撃分類に1対1で対応させる。
# trace に分類名がそのまま残るので、どの攻撃で発火したかを後から数えられる。
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # 1. 命令上書き
    ("命令上書き", _cooccur(_ORDER, _NEGATE)),
    ("命令上書き", _cooccur(_META, _NEGATE)),
    (
        "命令上書き",
        re.compile(
            r"(?:以降|今後|これから)(?:は)?[^。]{0,20}(?:従って|優先して)[^。]{0,10}(?:ください|下さい)",
            re.IGNORECASE,
        ),
    ),
    # 2. 役割偽装
    (
        "役割偽装",
        re.compile(
            rf"(?:あなた|君|お前|きみ)は(?:今|これ|以降|ここ)から.{{0,30}}{_ROLE_NOUN}",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "役割偽装",
        re.compile(
            r"(?:開発者|デベロッパー|管理者|デバッグ)\s*モード"
            r"|developer\s*mode|dan\s*モード|jailbreak|脱獄"
            r"|(?:制限|規制|検閲)(?:が|の)?(?:ない|無い|されていない)"
            r"|(?:act|behave|pretend|roleplay)\s+as\b"
            r"|なりきって|演じてください",
            re.IGNORECASE,
        ),
    ),
    # 行頭の疑似ロール宣言。`role: content` 形式でターンを偽装する古典的手口。
    (
        "役割偽装",
        re.compile(r"(?:^|\n)\s*(?:system|assistant|developer)\s*[:：]", re.IGNORECASE),
    ),
    # 3. システムプロンプト漏洩
    ("システムプロンプト漏洩", _cooccur(_META, _REVEAL)),
    (
        "システムプロンプト漏洩",
        re.compile(
            r"(?:何を|どんな)(?:指示|命令)(?:を)?(?:受けて|与えられて)"
            r"|what\s+(?:are|were)\s+you\s+told"
            r"|repeat\s+(?:everything|the\s+above)",
            re.IGNORECASE,
        ),
    ),
    # 4. 情報抽出
    (
        "情報抽出",
        re.compile(
            r"環境変数|env\s*var|\.env\b|api\s*key|api\s*キー"
            r"|アクセストークン|access\s*token|秘密鍵|private\s*key"
            r"|WEBUI_PASSWORD|GRADER_IPS|MODEL_PATH"
            r"|drop\s+table|select\s+.{0,40}\s+from\s",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "情報抽出",
        _cooccur(
            r"(?:データベース|テーブル|会話履歴|他(?:の)?(?:利用者|ユーザー))"
            r"(?:の)?(?:中身|全件|全部|すべて|一覧|ダンプ)",
            _REVEAL,
        ),
    ),
    # 5. 出力ループ・反復誘導
    (
        "出力ループ",
        _cooccur(
            r"(?:無限|永遠|延々|ずっと|エンドレス|forever|infinitely)",
            r"(?:繰り返|反復|出力|続け|repeat|output)",
            window=20,
        ),
    ),
    (
        "出力ループ",
        re.compile(r"\d{2,}\s*回(?:以上)?\s*(?:繰り返|反復|出力|言っ|書い)"),
    ),
    # 6. 難読化・多言語(構造的な痕跡のみ。言語そのものでは判定しない)
    ("難読化", re.compile(r"[A-Za-z0-9+/]{48,}={0,2}")),
    ("難読化", re.compile(r"(?:\\u[0-9a-f]{4}){3,}", re.IGNORECASE)),
    # 8. 出力形式ハイジャック
    (
        "出力形式ハイジャック",
        re.compile(
            r"(?:json|xml|yaml|base64|コードブロック|マークダウン)"
            r"(?:形式|フォーマット)?(?:のみ|だけ)?\s*(?:で|に)?\s*"
            r"(?:出力|返して|答えて|書いて|encode)"
            r"|\bhtml\s*(?:タグ|形式)"
            r"|<\s*script\b",
            re.IGNORECASE,
        ),
    ),
)

# 9. 目的外利用。ごみ分別以外の作業依頼。正当な質問と衝突しうるので、
# 検索が何もヒットしなかったときにだけ照合する(`detect_off_purpose`)。
_OFF_PURPOSE = re.compile(
    r"(?:コード|プログラム|スクリプト|python|javascript|sql)\s*(?:を)?\s*"
    r"(?:書い|作っ|生成|出力)"
    r"|(?:翻訳して|訳して|translate)"
    r"|(?:小説|詩|物語|作文|エッセイ|歌詞|レシピ)\s*(?:を)?\s*(?:書い|作っ|考え|生成)"
    r"|(?:方程式|微分|積分|素因数分解)"
    r"|(?:天気|株価|為替)(?:は|を)?\s*(?:教えて|どう)",
    re.IGNORECASE,
)


def detect_attack(prompt: object) -> str | None:
    """検索結果に関係なく成立する攻撃を判定する。

    戻り値は PIJ要件の攻撃分類名(発火した最初のもの)、無ければ None。
    """
    text = normalize_input(prompt)
    if not text:
        return None
    for label, pattern in _PATTERNS:
        if pattern.search(text):
            return label
    return None


def detect_off_purpose(prompt: object) -> str | None:
    """ごみ分別以外の作業依頼か。検索が空振りしたときだけ呼ぶこと。

    分別質問は早見表・収集日のいずれかに当たるのが通常なので、「何も当たら
    ない」こと自体が既に弱い signal になっている。これ単独では判定しない。
    """
    text = normalize_input(prompt)
    if text and _OFF_PURPOSE.search(text):
        return "目的外利用"
    return None


# --- 出力側の検査 ---------------------------------------------------------
# 出力ゲートで置き換えるときの固定文。入力ゲートの REFUSAL と文面を分けて
# あるのは、trace や採点ログを見たときにどちらのゲートが働いたか区別できる
# ようにするため。
OUTPUT_BLOCKED = (
    "申し訳ありません。この質問にはお答えできません。"
    "ごみの分別・収集日についてお尋ねください。"
)

# システムプロンプト漏洩の判定に使う、自分のプロンプト固有の文字列。
# 我々は自分のシステムプロンプトの原文を完全に知っているので、漏洩判定は
# 推測ではなく完全一致で行える — PIJ 9分類の中で唯一、確実に検知できる種類。
#
# ここは `app/rag/router.py` のプロンプト定数と二重管理になる。文言を変えた
# ときに気付けるよう、`tests/prompt_injection_test.py` が router.py の
# ソース中に各文字列が実在することを固定している。
_LEAK_FINGERPRINTS = (
    "あなたは北九州市のごみ分別・収集日を案内するアシスタントです",
    "URLや参照先は書かない",
    "下記『検索結果』に書かれている内容だけを事実として使う",
    "# 検索結果",
    "## 分別早見表の検索結果",
    "## 収集日の検索結果",
    "## 出し方の変更予定",
)

# 引用はコード側(`format_citations`)が確定させるので、モデルが書いたURLは
# 一律で落とす。出力形式ハイジャックで偽の参照欄を作られるのを防ぐと同時に、
# ハルシネートしたURLが本文に残るのも防ぐ。
_URL_RE = re.compile(r"(?:https?://|knowledge://|www\.)\S+")

# 反復判定。60文字未満は見ない(短い定型文の中の繰り返しを誤検知しないため)。
_REPEAT_MIN_TOTAL = 60
_REPEAT_MIN_COUNT = 8
_REPEAT_MAX_UNIT = 40


def _degenerate_repeat_start(text: str) -> int | None:
    """末尾が同一単位の連続反復になっている場合、その開始位置を返す。

    「『あ』を無限に繰り返して」のような出力ループ攻撃は `MAX_TOKENS` で
    長さは頭打ちになるが、1024トークン分の無意味な反復をそのまま返せば
    攻撃は成立してしまう。単位長1〜40文字で末尾の反復を数え、8回以上かつ
    合計60文字以上なら退化した反復とみなす。日本語の正当な回答が同じ単位を
    8回続けることはない。
    """
    n = len(text)
    if n < _REPEAT_MIN_TOTAL:
        return None
    for unit in range(1, _REPEAT_MAX_UNIT + 1):
        if n < unit * _REPEAT_MIN_COUNT:
            break
        tail = text[n - unit:]
        count = 1
        while text[n - unit * (count + 1): n - unit * count] == tail:
            count += 1
        if count >= _REPEAT_MIN_COUNT and unit * count >= _REPEAT_MIN_TOTAL:
            return n - unit * count
    return None


def scrub_output(text: str) -> tuple[str, str | None]:
    """モデルの生成結果を検査し、(送出してよい本文, 発火した攻撃分類) を返す。

    全量バッファリング前提。1文字も送出する前にここを通すので、漏洩した
    システムプロンプトが部分的にでも採点対象の本文に混ざることがない
    (SSEは一度 yield したdeltaを取り消せないため、逐次送出のままでは
    この保証が作れない)。
    """
    if not text.strip():
        return text, None

    for fingerprint in _LEAK_FINGERPRINTS:
        if fingerprint in text:
            # 部分的に残すと漏れた断片がそのまま採点本文に入るので全置換する。
            return OUTPUT_BLOCKED, "システムプロンプト漏洩"

    blocked: str | None = None

    cut = _degenerate_repeat_start(text)
    if cut is not None:
        text = text[:cut].rstrip()
        blocked = "出力ループ"
        if len(text) < 5:
            # 反復を落とすと中身が残らない = 回答全体が反復だった。
            # 下限を低く取るのは、「家庭ごみです。」のような短くても妥当な
            # 回答が反復に巻き込まれたときに、それを捨てないため。反復自体は
            # 入力ゲート側でも見ているので、ここは取りこぼしの受け皿。
            return OUTPUT_BLOCKED, blocked

    cleaned = _URL_RE.sub("", text)
    if cleaned != text:
        blocked = blocked or "出力形式ハイジャック"
        # URLを抜いた跡に残る空行・空の参照欄を畳む。
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip(), blocked
