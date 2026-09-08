"""Parser for distributed お知らせ knowledge files (STAGE 04 / 4-1 and 4-2).

Pure text processing: no I/O beyond reading the file, no DB, no network. The
ingest layer turns the `Notice` objects produced here into `WasteItem` rows.

A file looks like this (both distributed files share the shape)::

    # 北九州市 環境局からのお知らせ

    ## 目次

    1. [スプレー缶](#お知らせ-01-スプレー缶)
    ...

    ## お知らせ 01: スプレー缶

    **【市からのお知らせ】スプレー缶の出し方が変わります**

    2026年10月から、スプレー缶の出し方を変更します。...
    現行（早見表）の「燃えないごみ」としての扱いは2026年9月まで有効です。

    北九州市 環境局 循環社会推進部業務課

Two kinds of notice occur, and they are told apart by *content* rather than by
filename, so that a future mixed file still routes correctly:

* **change notice** (要件 4-2) — carries a 現行（早見表）の…扱い clause naming the
  treatment it replaces. It must be linked to the existing 早見表 row so both the
  old and the new rule stay answerable.
* **addition notice** (要件 4-1) — a品目 absent from the 早見表. Nothing to
  supersede; it is inserted standalone.
"""

from __future__ import annotations

import calendar
import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# "## お知らせ 01: スプレー缶"
_HEADING_RE = re.compile(r"^お知らせ\s*(\d+)\s*[:：]\s*(.+?)\s*$")

# "**【市からのお知らせ】スプレー缶の出し方が変わります**"
_TITLE_RE = re.compile(r"^\*\*(.+?)\*\*\s*$")

# trailing publisher line
_ISSUER_RE = re.compile(r"^北九州市\s+環境局.*$")

# "2026年10月から、" -> the date the new rule starts applying
_EFFECTIVE_RE = re.compile(r"(\d{4})年(\d{1,2})月から")

# The 現行（早見表）clause appears in three shapes across the 15 notices:
#     現行（早見表）の「燃えないごみ」としての扱いは2026年9月まで有効です。
#     現行（早見表）の家庭ごみとしての扱いは2026年9月まで有効です。
#     現行（早見表）の「そのまま家庭ごみに出してよい」扱いは2026年9月まで有効です。
#     現行（早見表）の「電池類。…出せる」という扱いは2026年9月まで有効です。
# so the treatment is either bracketed or bare, and としての / という are optional.
_PRIOR_RE = re.compile(
    r"現行（早見表）の"
    r"(?:「(?P<quoted>[^」]+)」|(?P<bare>[^「」]+?))"
    r"(?:としての|という)?扱いは"
    r"(?P<year>\d{4})年(?P<month>\d{1,2})月まで有効[^。]*。?"
)

_TRAILING_KAISHU_RE = re.compile(r"回収$")


@dataclass(frozen=True)
class Notice:
    """One `## お知らせ NN: …` section."""

    number: int
    heading: str
    item_name: str
    title: str
    body: str
    issuer: str
    effective_from: date | None
    prior_disposal_method: str | None
    prior_valid_until: date | None
    anchor: str
    content_hash: str
    source_file: str
    warnings: tuple[str, ...] = field(default=())

    @property
    def is_change_notice(self) -> bool:
        """True when this supersedes an existing 早見表 entry (要件 4-2)."""
        return self.prior_disposal_method is not None


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _split_sections(text: str) -> list[str]:
    """Split on level-2 headings, dropping the document preamble."""
    parts = re.split(r"^## ", text, flags=re.MULTILINE)
    return parts[1:]  # parts[0] is the "# …" title block


def _strip_kaishu(heading: str) -> str:
    """`ホバーボード回収` -> `ホバーボード`.

    The 4-1 file suffixes every heading with 回収 while 4-2 does not. Only strip
    when something is left over, so a品目 literally named 回収 survives.
    """
    stripped = _TRAILING_KAISHU_RE.sub("", heading).strip()
    return stripped or heading


def _parse_section(section: str, source_file: str) -> Notice | None:
    lines = section.splitlines()
    if not lines:
        return None

    heading_match = _HEADING_RE.match(lines[0].strip())
    if heading_match is None:
        return None  # 目次 and any other non-notice section

    number = int(heading_match.group(1))
    heading = heading_match.group(2)

    title = ""
    issuer = ""
    body_lines: list[str] = []
    for raw in lines[1:]:
        line = raw.strip()
        if not line:
            continue
        if not title and (m := _TITLE_RE.match(line)):
            title = m.group(1).strip()
            continue
        if _ISSUER_RE.match(line):
            issuer = line
            continue
        body_lines.append(line)

    body = "\n".join(body_lines)

    prior_method: str | None = None
    prior_valid_until: date | None = None
    if prior := _PRIOR_RE.search(body):
        prior_method = (prior.group("quoted") or prior.group("bare") or "").strip()
        prior_valid_until = _month_end(
            int(prior.group("year")), int(prior.group("month"))
        )
        # The 現行 clause is context, not an instruction — keep it out of the
        # disposal method so the answer layer quotes only the new rule there.
        body = (body[: prior.start()] + body[prior.end() :]).strip()

    effective_from: date | None = None
    if eff := _EFFECTIVE_RE.search(body) or _EFFECTIVE_RE.search(section):
        effective_from = date(int(eff.group(1)), int(eff.group(2)), 1)

    warnings: list[str] = []
    if prior_valid_until and effective_from:
        from datetime import timedelta

        if prior_valid_until + timedelta(days=1) != effective_from:
            warnings.append(
                f"旧扱いの終了日({prior_valid_until})と新扱いの開始日({effective_from})が連続していません"
            )
    if prior_method and effective_from is None:
        warnings.append("変更通知だが適用開始日を抽出できませんでした")

    return Notice(
        number=number,
        heading=heading,
        item_name=_strip_kaishu(heading),
        title=title,
        body=body,
        issuer=issuer,
        effective_from=effective_from,
        prior_disposal_method=prior_method or None,
        prior_valid_until=prior_valid_until,
        anchor=f"お知らせ-{number:02d}-{heading}".replace(" ", "-"),
        content_hash=hashlib.sha256(section.strip().encode("utf-8")).hexdigest(),
        source_file=source_file,
        warnings=tuple(warnings),
    )


def parse_notice_text(text: str, source_file: str) -> list[Notice]:
    notices = [_parse_section(s, source_file) for s in _split_sections(text)]
    return [n for n in notices if n is not None]


def parse_notice_file(path: Path) -> list[Notice]:
    path = Path(path)
    return parse_notice_text(path.read_text(encoding="utf-8"), path.name)
