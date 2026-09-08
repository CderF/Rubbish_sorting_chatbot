"""収集日パスの町名照合に関する再現テスト(現状は失敗する)。

北九州市の収集日ページは複数の丁目を1行にまとめて掲載しており、
`crawler.parse_schedule_page` はその表記のまま `collection_schedule.town` に
格納する(例: "浅野1~3丁目")。一方 `lookup_collection_schedule` は
`WHERE town = ?` の完全一致なので、利用者が自然に言う「浅野1丁目」は
永久にヒットしない。

実測(2026-09-08 時点、`crawler.fetch_collection_schedules()` の結果):
    収集日レコード 2472件 / ユニーク町名 777件
    うち範囲表記   340件 (43.8%)  <- この町の住民は収集日を引けない
    展開すると     1526件

本番(B1)での確認:
    「八幡東区八王寺町の…」   -> 火曜日・金曜日・日曜日 と正しく回答(範囲でない)
    「八幡東区中央1丁目の…」  -> 該当なし              (実データは "中央1~3丁目")

データ層・応用層のどちらも各自の契約どおりに動いており、範囲表記を展開する
責務がどちらにも割り当てられていないために落ちている。本テストは期待する
振る舞いを実行可能な形で固定するもので、修正が入るまで失敗する。

    python3 tests/collection_schedule_town_test.py

ネットワークには接続しない(実ページから採取した固定データを使う)。
"""

import datetime
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(tempfile.mkdtemp(), "town_range_test.db")
os.environ["DB_PATH"] = DB_PATH

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
    conn = db.connect(DB_PATH)
    db.rebuild_db(conn, [], [
        CollectionSchedule(ward=w, town=t, waste_type=k, weekday=d,
                           source_url=URL, crawled_at=CRAWLED_AT)
        for w, t, k, d in FIXTURES
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
    sys.exit(main())
