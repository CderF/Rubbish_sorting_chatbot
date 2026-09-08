"""Crawls the city's 早見表 (waste-item) and 収集日 (collection-schedule) pages
and parses them into WasteItem / CollectionSchedule records.

Network access lives only in `fetch_waste_items` / `fetch_collection_schedules`.
The `parse_*` functions are pure (HTML string in, records out) so tests can
feed them fixture HTML without touching the network (see documents/data_schema.md
and the STAGE02/03 データ層 spec, GitHub issue #4).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from app.data.schema import CollectionSchedule, WasteItem

BASE_URL = "https://www.city.kitakyushu.lg.jp"
WASTE_ITEMS_INDEX = f"{BASE_URL}/kurashi/menu01_0393.html"
COLLECTION_SCHEDULE_INDEX = f"{BASE_URL}/kurashi/menu01_0394.html"

_USER_AGENT = "Mozilla/5.0 (compatible; KitakyushuWasteBot/1.0)"
_WARD_NAME_RE = re.compile(r"資源とごみの収集日[（(](.+?)[）)]")
_WASTE_ITEM_PAGE_RE = re.compile(r"^分別早見表[（(]")
_SCHEDULE_PAGE_RE = re.compile(r"^資源とごみの収集日[（(]")


def _get(session: requests.Session, url: str) -> str:
    resp = session.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


def _discover_pages(index_html: str, link_text_re: re.Pattern[str]) -> list[str]:
    """Sub-page URLs whose link text matches `link_text_re`, in document order,
    de-duplicated. Matching on link text (not just URL shape) matters: the index
    pages also link to unrelated pages (site search, call center...) that happen
    to match the same /contents/<id>.html URL shape. The pattern must require the
    parenthesised category/ward suffix (e.g. r"^分別早見表[（(]") — a bare-prefix
    match would also catch the page's own breadcrumb self-link, which repeats the
    section title with no suffix.
    """
    soup = BeautifulSoup(index_html, "html.parser")
    seen: dict[str, None] = {}
    for a in soup.select("a[href]"):
        if link_text_re.match(a.get_text(strip=True)):
            href = a["href"]
            url = href if href.startswith("http") else f"{BASE_URL}{href}"
            seen.setdefault(url, None)
    return list(seen)


def _is_waste_item_table(table: BeautifulSoup) -> bool:
    if table.get("summary") == "分別早見表":
        return True
    caption = table.find("caption")
    if caption and caption.get_text(strip=True) == "分別早見表":
        return True
    # Some pages tag neither summary nor caption (e.g. 924_10064.html, な行) —
    # fall back to the one signal every variant actually has: the header row.
    first_tr = table.find("tr")
    if first_tr:
        header_text = [c.get_text(strip=True) for c in first_tr.find_all(["th", "td"])]
        if header_text[:3] == ["品名", "出し方", "備考"]:
            return True
    return False


def parse_waste_item_page(html: str, source_url: str) -> list[WasteItem]:
    """Parse one 早見表 sub-page (e.g. 分別早見表（あ行）) into WasteItem records.

    Each page splits its 五十音 sub-groups (あ/い/う/え/お for あ行, etc.) into
    *separate* <table> elements rather than one table with repeated headers — a
    page typically has 5 such tables, identified by a shared summary="分別早見表"
    attribute or, on some pages, a <caption>分別早見表</caption> instead. All
    matching tables on the page are parsed and concatenated.
    """
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    source_page_title = h1.get_text(strip=True) if h1 else None

    now = datetime.now(timezone.utc)
    records: list[WasteItem] = []
    for table in soup.find_all("table"):
        if not _is_waste_item_table(table):
            continue
        for tr in table.find_all("tr"):
            if tr.find_all("th"):
                continue
            cells = tr.find_all("td")
            if len(cells) < 3:
                continue
            item_name_raw = cells[0].get_text(strip=True)
            if not item_name_raw:
                continue
            records.append(
                WasteItem(
                    item_name=item_name_raw,
                    item_name_raw=item_name_raw,
                    disposal_method=cells[1].get_text(strip=True),
                    notes=cells[2].get_text(" ", strip=True),
                    source_url=source_url,
                    source_page_title=source_page_title,
                    crawled_at=now,
                )
            )
    return records


_TOWN_RANGE_RE = re.compile(r"^(.*?)(\d+)[~〜～](\d+)丁目$")


def _expand_town_range(town: str) -> list[str]:
    """収集日ページは複数丁目を1行にまとめて掲載する(例: "浅野1～3丁目")。
    `lookup_collection_schedule` は `town` の完全一致なので、このまま格納すると
    利用者が自然に言う「浅野1丁目」が永久にヒットしない(PR #16、藤井の再現
    テスト参照)。ここで丁目単位に展開してから個別レコードとして格納する
    ("正規化はデータ層の責務"という既存の契約(lookup.pyのdocstring)に合わせ、
    展開もデータ層=クロール時点で行う)。

    範囲でない町名(例: "八王寺町")はそのまま1件のリストで返す。
    """
    m = _TOWN_RANGE_RE.match(town)
    if not m:
        return [town]
    prefix, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    if start > end:
        return [town]
    return [f"{prefix}{n}丁目" for n in range(start, end + 1)]


def parse_schedule_page(html: str, source_url: str) -> list[CollectionSchedule]:
    """Parse one 収集日 ward page into CollectionSchedule records: one row per
    (town, waste type) pair, pivoted out of the page's per-town table row.

    Ward pages are not structurally consistent: some wrap a single header row in
    <thead>/<tbody> (e.g. 小倉北区), others have no thead/tbody at all and instead
    repeat a <tr> of <th> cells once per 五十音 group directly among the data rows
    (e.g. 門司区). So this walks every <tr> in document order and classifies each
    one as header (has <th> cells) or data (has <td> cells) rather than relying on
    thead/tbody.
    """
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    title_text = h1.get_text(strip=True) if h1 else ""
    ward_match = _WARD_NAME_RE.search(title_text)
    ward = ward_match.group(1) if ward_match else title_text

    table = soup.find("table")
    if table is None:
        return []

    # columns: [gojuon index letter, 町名, <waste type 1>, <waste type 2>, ...]
    waste_types: list[str] | None = None
    now = datetime.now(timezone.utc)
    records: list[CollectionSchedule] = []

    for tr in table.find_all("tr"):
        header_cells = tr.find_all("th")
        if header_cells:
            if waste_types is None:
                waste_types = [
                    re.sub(r"の収集日.*$", "", th.get_text(" ", strip=True)).strip()
                    for th in header_cells[2:]
                ]
            continue
        if not waste_types:
            continue

        cells = tr.find_all("td")
        if len(cells) < 2 + len(waste_types):
            continue
        town_raw = cells[1].get_text(strip=True)
        if not town_raw:
            continue
        for town in _expand_town_range(town_raw):
            for i, waste_type in enumerate(waste_types):
                weekday = cells[2 + i].get_text(strip=True)
                if not weekday:
                    continue
                records.append(
                    CollectionSchedule(
                        ward=ward,
                        town=town,
                        waste_type=waste_type,
                        weekday=weekday,
                        source_url=source_url,
                        crawled_at=now,
                    )
                )
    return records


def fetch_waste_items(session: requests.Session | None = None) -> list[WasteItem]:
    session = session or requests.Session()
    index_html = _get(session, WASTE_ITEMS_INDEX)
    records: list[WasteItem] = []
    for url in _discover_pages(index_html, _WASTE_ITEM_PAGE_RE):
        records.extend(parse_waste_item_page(_get(session, url), url))
    return records


def fetch_collection_schedules(
    session: requests.Session | None = None,
) -> list[CollectionSchedule]:
    session = session or requests.Session()
    index_html = _get(session, COLLECTION_SCHEDULE_INDEX)
    records: list[CollectionSchedule] = []
    for url in _discover_pages(index_html, _SCHEDULE_PAGE_RE):
        records.extend(parse_schedule_page(_get(session, url), url))
    return records


def crawl_all() -> tuple[list[WasteItem], list[CollectionSchedule]]:
    """Convenience entrypoint: one session shared across both crawls."""
    session = requests.Session()
    return fetch_waste_items(session), fetch_collection_schedules(session)


if __name__ == "__main__":
    waste_items, schedules = crawl_all()
    print(f"waste_items: {len(waste_items)} rows")
    print(f"collection_schedule: {len(schedules)} rows")
