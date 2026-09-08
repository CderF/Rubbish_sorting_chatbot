"""Thin pytest wrapper around tests/stage04_eval.py so CI fails on a regression."""

from tests.stage04_eval import run


def test_stage04_data_side_eval_all_pass():
    report = run()
    assert report.all_ok, "\n" + report.table()
