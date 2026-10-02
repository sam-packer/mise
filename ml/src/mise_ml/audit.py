"""Find words that the labeler puts into too many profiles of one category.

A word in the items prompt, or a habit of the labeler, can show up in most profiles of a category
("quiet" in art, "tense" in books). Such a word makes the works of that category look alike to the
models. leaks() finds the prompt words that a prompt puts into profiles, against a baseline prompt;
the leaks job uses it. The audit reports the catalog shares of the words that the leaks job
blocked. This check only warns; it never stops a run.

Run it alone with: uv run python -m mise_ml.audit
"""

import logging
import math
import re
from collections import Counter

from mise_ml.config import CATEGORIES, DATA_AUDIT, LEAK_BLOCK, ML_ROOT
from mise_ml.data import load_catalog
from mise_ml.log import get, num
from mise_ml.util import iter_jsonl, write_json

log = get(__name__)
# Flag a word when more than this share of a category's works use it.
VIBE_LIMIT = 0.25
TEXT_LIMIT = 0.40
# The report keeps every word above this share.
REPORT_SHARE = 0.10
TOP = 6
# The check skips function words, and words about the form of a work rather than its mood.
FUNCTION_WORDS = """
    a about after against all also an and any are as at be because been before being between both
    but by can could did do does down during each even every for from had has have he her here him
    his how i if in into is it its just like may me more most much my no nor not now of off on once
    one only or other our out over own s same she so some such than that the their them then there
    these they this those through to too under until up very was we were what when where which
    while who whom why will with would yet you your
"""
FORM_WORDS = """
    art film song poem book painting image work story lyrics singer speaker narrator protagonist
    readers characters figures scene moment moments feel feels feeling sense mood emotion tone
    atmosphere create creates creating captures shows comes carries expresses describes balances
    focuses remains rather despite without
"""
SKIP = frozenset(FUNCTION_WORDS.split() + FORM_WORDS.split())
# A word of a category's items prompt leaks when its share of the category's vibes and
# descriptions under that prompt is at least LEAK_GAIN above, and LEAK_RATIO times, its share
# under the neutral baseline prompt. The baseline share counts as at least one work.
LEAK_GAIN = 0.05
LEAK_RATIO = 2.0
# One-sided Fisher exact test: the rise must be unlikely by chance in the sample.
LEAK_P = 0.05


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower())) - SKIP


def shares(texts: list[str]) -> dict[str, float]:
    counts = Counter(w for text in texts for w in words(text))
    return {w: n / len(texts) for w, n in counts.most_common()}


def leaks(
    prompts: dict[str, str], texts: dict[str, list[str]], control: dict[str, list[str]]
) -> dict[str, dict[str, dict[str, float]]]:
    """Per category, the content words of its prompt that the prompt puts into its texts.

    `texts` were written with `prompts`; `control` with the baseline prompt, for the same works.
    Each leaked word maps to its share under the prompt and under the baseline.
    """
    found = {}
    for category, items in texts.items():
        table = shares(items) if items else {}
        base = shares(control[category]) if control[category] else {}
        n, m = len(items), len(control[category])
        floor = 1 / max(m, 1)
        found[category] = {
            w: {"prompt": table[w], "baseline": base.get(w, 0.0)}
            for w in sorted(words(prompts[category]))
            # The tolerance keeps exact ties, such as 3/40 - 1/40, from float rounding.
            if table.get(w, 0.0) - base.get(w, 0.0) >= LEAK_GAIN - 1e-9
            and table.get(w, 0.0) >= LEAK_RATIO * max(base.get(w, 0.0), floor) - 1e-9
            # A few works in a small sample happen by chance: 3 of 40 against 0 of 40 is not a leak.
            and fisher_greater(round(table.get(w, 0.0) * n), n, round(base.get(w, 0.0) * m), m)
            < LEAK_P
        }
    return found


def fisher_greater(a: int, n: int, b: int, m: int) -> float:
    """One-sided Fisher exact p-value that a of n texts use a word more often than b of m."""
    total = a + b
    if not total:
        return 1.0

    def p(k: int) -> float:
        return math.comb(n, k) * math.comb(m, total - k) / math.comb(n + m, total)

    return sum(p(k) for k in range(a, min(n, total) + 1))


def run() -> None:
    catalog = load_catalog()
    by_field: dict[str, dict[str, dict[str, float]]] = {"vibe": {}, "text": {}}
    works = {}
    for category in CATEGORIES:
        items = [it for it in catalog.items if it["category"] == category]
        if not items:
            continue
        works[category] = len(items)
        by_field["vibe"][category] = shares([it["vibe"] for it in items])
        by_field["text"][category] = shares([f"{it['vibe']} {it['description']}" for it in items])
    # The catalog has no baseline control, so report the catalog shares of the words that the
    # leaks job found against its control.
    blocked = list(iter_jsonl(LEAK_BLOCK)) if LEAK_BLOCK.is_file() else []
    prompt_leaks = [
        {
            "category": row["category"],
            "word": word,
            "sample": row["shares"][word],
            "catalog": {c: round(by_field["text"][c].get(word, 0.0), 4) for c in works},
        }
        for row in blocked
        for word in row["words"]
    ]
    flags = []
    for field, limit in (("vibe", VIBE_LIMIT), ("text", TEXT_LIMIT)):
        for category, table in by_field[field].items():
            for word, share in table.items():
                if share <= limit:
                    break
                flags.append(
                    {
                        "category": category,
                        "field": field,
                        "word": word,
                        "share": share,
                        # The same word in the other categories: a general habit or a local one.
                        "other": {
                            other: by_field[field][other].get(word, 0.0)
                            for other in works
                            if other != category
                        },
                    }
                )
    write_json(
        DATA_AUDIT,
        {
            "limits": {
                "vibe": VIBE_LIMIT,
                "text": TEXT_LIMIT,
                "report": REPORT_SHARE,
                "leak_gain": LEAK_GAIN,
                "leak_ratio": LEAK_RATIO,
            },
            "works": works,
            "shares": {
                field: {
                    category: {w: round(s, 4) for w, s in table.items() if s > REPORT_SHARE}
                    for category, table in tables.items()
                }
                for field, tables in by_field.items()
            },
            "flags": flags,
            "prompt_leaks": prompt_leaks,
        },
    )
    log.info("word shares per category (vibe = vibe only; text = vibe and description)")
    for category, n in works.items():
        for field in ("vibe", "text"):
            top = list(by_field[field][category].items())[:TOP]
            log.info(
                "%-5s %-4s (%s works): %s",
                category,
                field,
                num(n),
                ", ".join(f"{w} {s:.2f}" for w, s in top),
            )
    for flag in flags:
        limit = VIBE_LIMIT if flag["field"] == "vibe" else TEXT_LIMIT
        log.warning(
            "%s %s: %r in %.0f%% of works (limit %.0f%%); other categories: %s",
            flag["category"],
            flag["field"],
            flag["word"],
            flag["share"] * 100,
            limit * 100,
            ", ".join(f"{c} {s:.0%}" for c, s in flag["other"].items()),
        )
    for leak in prompt_leaks:
        log.warning(
            "%s: blocked prompt word %r (leaks sample %.0f%%, baseline %.0f%%); catalog: %s",
            leak["category"],
            leak["word"],
            leak["sample"]["prompt"] * 100,
            leak["sample"]["baseline"] * 100,
            ", ".join(f"{c} {s:.0%}" for c, s in leak["catalog"].items()),
        )
    log.info(
        "%d flagged words, %d blocked prompt words; report -> %s",
        len(flags),
        len(prompt_leaks),
        DATA_AUDIT.relative_to(ML_ROOT).as_posix(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    run()
