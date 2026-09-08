"""Thin pytest wrapper around tests/notices_eval.py so CI fails on a regression."""

from tests.notices_eval import run


def test_notices_data_side_eval_all_pass():
    report = run()
    assert report.all_ok, "\n" + report.table()
