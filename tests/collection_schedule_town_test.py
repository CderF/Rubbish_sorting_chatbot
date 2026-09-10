"""収集日パスの町名照合に関する回帰テスト(PR #16, 藤井の再現テストを元に修正確認用に調整)。

北九州市の収集日ページは複数の丁目を1行にまとめて掲載している(例:
"浅野1～3丁目")。修正前は `crawler.parse_schedule_page` がその表記のまま
`collection_schedule.town` に格納しており、`lookup_collection_schedule` は
`WHERE town = ?` の完全一致なので、利用者が自然に言う「浅野1丁目」は
永久にヒットしなかった。

実測(2026-09-08 時点、`crawler.fetch_collection_schedules()` の結果):
    収集日レコード 2472件 / ユニーク町名 777件
    うち範囲表記   340件 (43.8%)  <- 修正前はこの町の住民が収集日を引けなかった

修正: 取り込み時点(`crawler._expand_town_range`)で丁目単位に展開してから
格納するようにした(正規化はデータ層の責務という既存の契約に合わせ、
`lookup_collection_schedule` 側は完全一致のまま変更していない)。

このテストの `setup()` は、藤井の元の再現テストが直接DBへ投入していた
生の範囲表記(修正前のcrawlerが格納していた形)を、`_expand_town_range` に
通してから投入するよう変更した — 修正後は crawler が範囲表記のままDBに
入れることはないため、そのままでは修正の効果を再現できないため。

    python3 tests/collection_schedule_town_test.py

ネットワークには接続しない(実ページから採取した固定データを使う)。
"""

import datetime
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(tempfile.mkdtemp(), "town_range_test.db")
# `os.environ["DB_PATH"]` は意図的にここでは設定しない(下の
# `if __name__ == "__main__":` 内でのみ設定する)。このファイルには
# pytest 用の `test_*` 関数が無く単体スクリプトとして実行する前提だが、
# ファイル名が `*_test.py` に一致するため `pytest tests/` はこのモジュール
# を import してしまう。モジュール直下で環境変数を書き換えると、
# `app.rag.router._district_groups`/`_town_candidates` 等の
# `functools.lru_cache` がこの空の一時DBを掴んだまま、同じプロセスで動く
# 他のテストファイル全部に波及して汚染する(2026-09-14 レビュー指摘で
# 実害を確認・特定)。実行方法(モジュール先頭のdocstring参照)は
# `python3 tests/collection_schedule_town_test.py` なので、
# `if __name__ == "__main__":` の中でだけ設定すれば実害なく両立できる。

from app.data import db  # noqa: E402
from app.data.schema import CollectionSchedule  # noqa: E402

CRAWLED_AT = datetime.datetime(2026, 9, 8, 1, 0, 0)
URL = "https://www.city.kitakyushu.lg.jp/contents/924_10102.html"

# 実ページをクロールして得られた町名表記をそのまま使う。
FIXTURES = [
    ("小倉北区", "浅野1~3丁目", "家庭ごみ", "火曜日・金曜日"),
    ("小倉北区", "浅野1~3丁目", "プラスチック", "月曜日"),
    ("小倉北区", "青葉1~2丁目", "家庭ごみ", "月曜日・木曜日"),
    ("八幡東区", "中央1~3丁目", "家庭ごみ", "火曜日・金曜日"),
    ("門司区", "大里本町1~3丁目", "家庭ごみ", "火曜日・金曜日"),
    # 範囲表記でない町名。こちらは現状でも正しく引ける(対照群)。
    ("八幡東区", "八王寺町", "家庭ごみ", "火曜日・金曜日・日曜日"),
]

_results = []


def check(label, ok, detail=""):
    _results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"\n         {detail}" if detail and not ok else ""))


def setup():
    # 修正は取り込み時点(crawler._expand_town_range)で丁目単位に展開する方式
    # なので、DBには最初から展開済みの行しか入らない。ここでも同じ展開関数を
    # 通してから格納し、実際の crawl 結果を模す(生の範囲表記をDBへ直接投入
    # すると、この修正方式では再現できないケースになってしまうため)。
    from app.data.crawler import _expand_town_range

    conn = db.connect(DB_PATH)
    db.rebuild_db(conn, [], [
        CollectionSchedule(ward=w, town=town, waste_type=k, weekday=d,
                           source_url=URL, crawled_at=CRAWLED_AT)
        for w, t, k, d in FIXTURES
        for town in _expand_town_range(t)
    ])
    conn.close()


def main():
    setup()
    from app.data import lookup

    # 対照群: 範囲表記でない町名は現状でも引ける
    hits = lookup.lookup_collection_schedule("八幡東区", "八王寺町")
    check("対照群: 範囲でない町名『八王寺町』は引ける", bool(hits))

    # 本題: 利用者が言う丁目単位の町名で引けること
    cases = [
        ("小倉北区", "浅野1丁目", "浅野1~3丁目"),
        ("小倉北区", "浅野2丁目", "浅野1~3丁目"),
        ("小倉北区", "浅野3丁目", "浅野1~3丁目"),
        ("八幡東区", "中央2丁目", "中央1~3丁目"),
        ("門司区", "大里本町1丁目", "大里本町1~3丁目"),
        ("小倉北区", "青葉2丁目", "青葉1~2丁目"),
    ]
    for ward, town, stored in cases:
        hits = lookup.lookup_collection_schedule(ward, town)
        check(f"{ward}{town} で引ける(DB上の表記は {stored!r})",
              bool(hits), f"lookup_collection_schedule({ward!r}, {town!r}) -> {hits}")

    # 範囲外の丁目は引けてはいけない(過剰な展開の検出)
    hits = lookup.lookup_collection_schedule("小倉北区", "浅野4丁目")
    check("範囲外の『浅野4丁目』は引けない(1~3丁目なので)", hits is None,
          f"想定 None、実際 {hits}")

    # ごみ種別での絞り込みも丁目単位で効くこと
    hits = lookup.lookup_collection_schedule("小倉北区", "浅野2丁目", "プラスチック")
    check("丁目単位+ごみ種別の絞り込みが効く", bool(hits),
          f"lookup_collection_schedule('小倉北区', '浅野2丁目', 'プラスチック') -> {hits}")

    passed = sum(_results)
    print(f"\n=== {passed}/{len(_results)} checks passed ===")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    os.environ["DB_PATH"] = DB_PATH
    sys.exit(main())
