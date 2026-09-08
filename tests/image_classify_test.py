"""Offline seam test for app/rag/image_classify.py (issue #7 Testing Decisions).

`classify_image_to_item_name` itself needs the shared vLLM engine (GPU-only,
see tests/spec_test.py for the live-server check on the L4 box). This tests
only `parse_classification_output`, the pure function that turns the VLM's
raw text into an item name or None — no GPU, no network.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.image_classify import parse_classification_output as parse

results = []


def check(name, condition, detail=""):
    verdict = "PASS" if condition else "FAIL"
    print(f"[{verdict}] {name}" + (f" — {detail}" if detail and not condition else ""))
    results.append(condition)


check("plain item name passes through", parse("テレビ") == "テレビ")
check("trailing 句点 is stripped", parse("テレビ。") == "テレビ")
check("trailing ！ is stripped", parse("ペットボトル！") == "ペットボトル")
check("only the first line is used", parse("携帯電話\nこれはスマートフォンです") == "携帯電話")
check("leading/trailing whitespace is stripped", parse("  テレビ  ") == "テレビ")

check("sentinel「不明」→ None", parse("不明") is None)
check("sentinel embedded in a longer reply → None", parse("画像からは不明です") is None)
check("「わかりません」→ None", parse("わかりません") is None)
check("「判別できません」→ None", parse("判別できません") is None)

check("empty string → None", parse("") is None)
check("whitespace-only → None", parse("   ") is None)
check("None input → None", parse(None) is None)
check("punctuation-only line → None", parse("。") is None)

print()
print(f"=== {sum(1 for r in results if r)}/{len(results)} checks passed ===")
