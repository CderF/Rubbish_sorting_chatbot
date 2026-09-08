"""STAGE 04 data-side evaluation: 『質問 → 期待DB状態』.

Given the two distributed お知らせ files plus a seeded 早見表, ingest + relink,
then assert the DB ends up in the state each requirement needs:

* 4-1 — every added品目 is a single current row, cited to the notice
* 4-2 (前身あり) — a two-row version chain: old superseded, new dated 2026-10-01,
  and `version_window` yields the right validity spans
* 4-2 (前身なし) — a single dated row that still carries the prior treatment and
  its end date in `notes`, and shows up as unlinked
* disputed — when the seeded 早見表 disagrees with the notice's stated 現行, the
  link is still made and the discrepancy is reported
* no-match — an unknown品目 returns [] (the 4-3 fallback trigger), no exception

The answer-text evaluation (本田さん) is built on top of this, not here.

Run standalone for a grader-style table::

    python -m tests.stage04_eval

or as a test::

    pytest tests/stage04_eval_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from app.data.schema import WasteItem
from app.stage04 import store
from app.stage04.ingest import IngestResult, ingest_notice_file, relink

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = Path(__file__).resolve().parent / "fixtures" / "stage04_eval_cases.json"
SEED_URL = "fixture://test-only"
SEED_CRAWLED_AT = datetime(2026, 9, 1, tzinfo=timezone.utc)
INGEST_AT = datetime(2026, 9, 8, 3, 0, 0, tzinfo=timezone.utc)


@dataclass
class Check:
    case: str
    ok: bool
    detail: str = ""


@dataclass
class EvalReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, case: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(case, ok, detail))

    def expect(self, case: str, cond: bool, detail: str = "") -> None:
        self.add(case, bool(cond), "" if cond else detail)

    @property
    def passed(self) -> int:
        return sum(c.ok for c in self.checks)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def all_ok(self) -> bool:
        return not self.failed

    def table(self) -> str:
        width = max((len(c.case) for c in self.checks), default=10)
        lines = [f"{'CASE':<{width}}  RESULT  DETAIL", f"{'-' * width}  ------  ------"]
        for c in self.checks:
            mark = "PASS" if c.ok else "FAIL"
            lines.append(f"{c.case:<{width}}  {mark:<6}  {c.detail}")
        lines.append("")
        lines.append(f"{self.passed}/{len(self.checks)} passed")
        return "\n".join(lines)


def load_cases() -> dict:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def build_eval_db(conn, cases: dict) -> "tuple[list[IngestResult], object]":
    """Seed the 早見表 rows the cases ask for, then ingest + relink."""
    store.init_schema(conn)
    for change in cases["changes"]:
        haya = change["hayamihyo"]
        if haya is None:
            continue
        store.insert_waste_item(
            conn,
            WasteItem(
                item_name=change["item"],
                item_name_raw=change["item"],
                disposal_method=haya["method"],
                notes="",
                source_url=SEED_URL,
                source_page_title="ごみの分別早見表（テストシード）",
                crawled_at=SEED_CRAWLED_AT,
                is_current=True,
            ),
        )
    conn.commit()

    results = [
        ingest_notice_file(REPO_ROOT / f, conn, now=INGEST_AT)
        for f in cases["knowledge_files"]
    ]
    relink_result = relink(conn)
    return results, relink_result


def evaluate(conn, cases: dict, ingest_results, relink_result) -> EvalReport:
    r = EvalReport()
    change_start = date.fromisoformat(cases["change_start"])
    prior_until = date.fromisoformat(cases["prior_valid_until"])
    ingest_unlinked = {n for res in ingest_results for n in res.unlinked}
    ingest_disputed = {name for res in ingest_results for name, _, _ in res.disputed}

    # ---- 4-1 additions ----
    for c in cases["additions"]:
        name = c["item"]
        vs = store.lookup_item_history(conn, name)
        if len(vs) != 1:
            r.expect(f"4-1 {name}", False, f"expected 1 version, got {len(vs)}")
            continue
        v = vs[0]
        r.expect(
            f"4-1 {name}",
            v.is_current
            and v.effective_from is None
            and v.superseded_by is None
            and v.source_url == f"knowledge://お知らせ-4-1.md#{c['anchor']}"
            and all(s in v.disposal_method for s in c["method_contains"]),
            f"is_current={v.is_current} eff={v.effective_from} url={v.source_url} "
            f"missing={[s for s in c['method_contains'] if s not in v.disposal_method]}",
        )

    # ---- 4-2 changes ----
    for c in cases["changes"]:
        name = c["item"]
        vs = store.lookup_item_history(conn, name)
        ks = {
            row["anchor"]: row for row in store.list_knowledge_sources(conn)
        }.get(c["anchor"])

        if c["hayamihyo"] is not None:
            # version chain: [old, new]
            ok2 = len(vs) == 2
            if not ok2:
                r.expect(f"4-2 {name} (chain)", False, f"expected 2 versions, got {len(vs)}")
                continue
            old, new = vs
            f0, u0 = store.version_window(vs, 0)
            f1, u1 = store.version_window(vs, 1)
            r.expect(
                f"4-2 {name} (chain)",
                old.is_current is False
                and old.superseded_by == new.id
                and old.effective_from is None
                and old.disposal_method == c["hayamihyo"]["method"]
                and new.is_current is True
                and new.effective_from == change_start
                and all(s in new.disposal_method for s in c["new_method_contains"])
                and (f0, u0) == (None, prior_until)
                and (f1, u1) == (change_start, None),
                f"old(cur={old.is_current},sup={old.superseded_by},method={old.disposal_method!r}) "
                f"new(eff={new.effective_from}) window0={(f0,u0)} window1={(f1,u1)} "
                f"missing={[s for s in c['new_method_contains'] if s not in new.disposal_method]}",
            )
            # disputed iff seed method disagrees with the notice's stated 現行
            should_dispute = c["hayamihyo"]["method"] != c["notice_prior"]
            r.expect(
                f"4-2 {name} (disputed={should_dispute})",
                (name in ingest_disputed) == should_dispute,
                f"in disputed list: {name in ingest_disputed}",
            )
        else:
            # no predecessor: single dated row, prior state kept in notes
            ok1 = len(vs) == 1
            if not ok1:
                r.expect(f"4-2 {name} (no-pred)", False, f"expected 1 version, got {len(vs)}")
                continue
            v = vs[0]
            r.expect(
                f"4-2 {name} (no-pred)",
                v.is_current is True
                and v.effective_from == change_start
                and c["notice_prior"] in v.notes
                and prior_until.isoformat() in v.notes
                and name in ingest_unlinked
                and name in relink_result.still_unlinked,
                f"eff={v.effective_from} notes={v.notes!r} "
                f"unlinked={name in ingest_unlinked} still={name in relink_result.still_unlinked}",
            )

        r.expect(
            f"4-2 {name} (manifest)",
            ks is not None and ks["prior_disposal_method"] == c["notice_prior"],
            f"knowledge_sources.prior_disposal_method={ks and ks['prior_disposal_method']!r}",
        )

    # ---- item_timeline (the answer layer's entry point) ----
    ref = date.fromisoformat(cases["reference_date"])
    for c in cases["additions"]:
        tl = store.item_timeline(conn, c["item"])
        r.expect(
            f"timeline 4-1 {c['item']}",
            tl.found and len(tl.segments) == 1
            and tl.segments[0].valid_from is None and tl.segments[0].valid_until is None
            and tl.changes_after(ref) == (),
            f"found={tl.found} segs={len(tl.segments)}",
        )
    for c in cases["changes"]:
        tl = store.item_timeline(conn, c["item"])
        seg_now = tl.segment_on(ref)
        r.expect(
            f"timeline 4-2 {c['item']}",
            tl.found
            and len(tl.segments) == 2
            and tl.changes_after(ref) == (change_start,)
            and seg_now is not None
            and c["notice_prior"] in seg_now.disposal_method
            and tl.segment_on(change_start).from_notice
            and (tl.segments[0].synthesized == (c["hayamihyo"] is None)),
            f"segs={len(tl.segments)} now={seg_now and seg_now.disposal_method[:20]!r} "
            f"changes_after={tl.changes_after(ref)} synth0={tl.segments and tl.segments[0].synthesized}",
        )
    for c in cases["no_match"]:
        tl = store.item_timeline(conn, c["item"])
        r.expect(f"timeline no-match {c['item']}", tl.found is False and tl.segments == (), "")

    # ---- no match (4-3 trigger) ----
    for c in cases["no_match"]:
        name = c["item"]
        try:
            vs = store.lookup_item_history(conn, name)
            r.expect(f"no-match {name}", vs == [], f"expected [], got {len(vs)} rows")
        except Exception as e:  # noqa: BLE001 - the point is that it must not raise
            r.expect(f"no-match {name}", False, f"raised {type(e).__name__}: {e}")

    return r


def run() -> EvalReport:
    cases = load_cases()
    with tempfile.TemporaryDirectory() as td:
        conn = store.connect(Path(td) / "eval.db")
        try:
            ingest_results, relink_result = build_eval_db(conn, cases)
            return evaluate(conn, cases, ingest_results, relink_result)
        finally:
            conn.close()


def main() -> int:
    report = run()
    print(report.table())
    return 0 if report.all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------
# answer-text evaluation (本田さん) — plug an answer function in here
# --------------------------------------------------------------------------
def evaluate_answers(conn, cases: dict, answer_fn) -> EvalReport:
    """`answer_fn(conn, item_name, today) -> str` produces the user-facing
    answer; this checks it against the `answer_must_*` keys in the cases file.

    Not wired into CI yet — call it from a test once the answer function exists.
    """
    r = EvalReport()
    today = date.fromisoformat(cases["reference_date"])

    def check(case_id: str, text: str, c: dict) -> None:
        for s in c.get("answer_must_contain", []):
            r.expect(f"{case_id} contains {s!r}", s in text, f"answer={text!r}")
        any_ok = c.get("answer_must_contain_any")
        if any_ok:
            r.expect(f"{case_id} any of {any_ok}", any(s in text for s in any_ok), f"answer={text!r}")
        only = c.get("answer_must_not_be_only")
        if only:
            stripped = text
            for s in only:
                stripped = stripped.replace(s, "")
            r.expect(
                f"{case_id} not only {only}",
                stripped.strip() != "" and len(stripped) < len(text),
                "answer restates only the prior rule",
            )

    for c in cases["additions"]:
        check(f"ans 4-1 {c['item']}", answer_fn(conn, c["item"], today), c)
    for c in cases["changes"]:
        check(f"ans 4-2 {c['item']}", answer_fn(conn, c["item"], today), c)
    for c in cases["no_match"]:
        check(f"ans no-match {c['item']}", answer_fn(conn, c["item"], today), c)
    return r
