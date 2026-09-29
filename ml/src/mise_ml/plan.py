"""Command plans. No GPU preflight, model loads, logging files, or output writes."""

import json

from mise_ml import config as c
from mise_ml.cache import CachePlan, inspect_job, unused_job
from mise_ml.cache import prune as prune_caches
from mise_ml.provenance import ContentInput, stale_reasons
from mise_ml.steps import JUDGE_POOL, STEPS, command_steps, judge_pool_state
from mise_ml.util import iter_jsonl, sha256_file


def judge_plan() -> tuple[CachePlan, list[str], bool]:
    from mise_ml.data import load_catalog, load_eval_texts
    from mise_ml.evaluate import SEP, judge_job

    catalog = load_catalog()
    texts = set(load_eval_texts())
    saved = json.loads(JUDGE_POOL.read_text(encoding="utf-8")) if JUDGE_POOL.is_file() else None
    historical = saved["keys"] if saved else [r["key"] for r in iter_jsonl(c.JUDGMENTS)]
    keys = [k for k in historical if k.split(SEP)[0] in texts and k.split(SEP)[2] in catalog.index]
    current = saved is not None and saved["state"] == judge_pool_state()
    result = inspect_job(judge_job(keys))
    notes = []
    if not current:
        result.membership_verified = False
        known_texts = {k.split(SEP)[0] for k in keys}
        new = sorted(texts - known_texts)
        if new:
            minimum = sum(
                min(
                    c.ProfileConfig().judge_pool_per_system,
                    sum(it["category"] == cat for it in catalog.items),
                )
                for cat in c.CATEGORIES
            )
            notes.append(
                f"new keys: at least {minimum * len(new)} for {len(new)} unranked feelings; "
                f"examples: {json.dumps(new[:5], ensure_ascii=True)}"
            )
        notes.append(
            "pool needs model ranking: no verified pool for these inputs; "
            "exact keys and request count are unknown until ranking runs"
        )
        notes.append(
            "cache counts above cover recorded pool keys only; unused judge records/bytes "
            "are unknown and are retained"
        )
        result.unused.clear()
        result.unused_bytes = 0
        result.unused_examples.clear()
    return result, notes, current


def cache_plans() -> list[CachePlan]:
    plans = []
    for step in [*command_steps("label"), STEPS["judge"]]:
        try:
            require_inputs(step)
            plans.append(judge_plan()[0] if step.name == "judge" else inspect_job(step.call()))
        except (OSError, ValueError, KeyError, SystemExit) as exc:
            pattern = "labels-*.jsonl" if step.name == "labels" else f"{step.name}.jsonl"
            paths = list((c.DATA / "llm").glob(pattern)) or [c.DATA / "llm" / f"{step.name}.jsonl"]
            for path in paths:
                plans.append(
                    CachePlan(
                        path.stem,
                        path,
                        sha256_file(path) if path.is_file() else None,
                        membership_verified=False,
                        notes=[f"not used: {exc}"],
                    )
                )
    return plans


def require_inputs(step) -> None:
    missing = [
        entry.path if isinstance(entry, ContentInput) else entry
        for entry in step.inputs()
        if not (entry.path if isinstance(entry, ContentInput) else entry).is_file()
    ]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(str(p) for p in missing))


def show(command: str, *, prune: bool = False) -> None:
    print(f"{command} plan (no models; {'prune only' if prune else 'no writes'})")
    print("step           | status     | reasons")
    dirty: set[str] = set()
    caches: list[CachePlan] = []
    steps = command_steps(command)
    if command == "label":
        steps = [*steps, STEPS["judge"]]
    for step in steps:
        reasons: list[str] = []
        details: list[str] = []
        try:
            if step.name == "publish":
                from mise_ml.publish import plan

                print(
                    "  publish: online comparison needs read-only R2 HEAD/LIST requests", flush=True
                )
                details = plan()
                reasons = (
                    []
                    if details[0] == "up to date"
                    else ["online files or local bundle URL need update"]
                )
            elif step.llm:
                require_inputs(step)
                if step.name == "judge":
                    result, notes, verified = judge_plan()
                    if not verified:
                        reasons.append("retrieval pool must be ranked")
                else:
                    result, notes = inspect_job(step.call()), []
                upstream = [name for name in step.needs if name in dirty]
                if upstream:
                    result.unused.clear()
                    result.unused_bytes = 0
                    result.unused_examples.clear()
                    notes.append(
                        "unused records/bytes unknown until upstream outputs are current; "
                        "retain this cache"
                    )
                caches.append(result)
                details = result.lines() + notes
                if result.requests:
                    reasons.append("LLM requests pending")
                if result.output_changed:
                    reasons.append("materialize current records")
            else:
                reasons = stale_reasons(step.name, step.state(), step.outputs())
            upstream = [name for name in step.needs if name in dirty]
            reasons.extend(
                f"upstream {name} will run; recheck after its outputs change" for name in upstream
            )
            if step.name == "install" and c.EVAL_REPORT.is_file():
                report = json.loads(c.EVAL_REPORT.read_text(encoding="utf-8"))
                if not report["ship"]["ok"]:
                    reasons.append("install requires a passing ship gate; current report fails")
            if upstream and step.llm:
                details.append(
                    "request count above uses current files; upstream outputs can change it"
                )
        except (OSError, ValueError, KeyError, SystemExit) as exc:
            reasons.append(f"cannot inspect current inputs: {exc}")
            if prune:
                # Never delete records when the current key set could not be established.
                raise RuntimeError("prune stopped: incomplete cache inspection") from exc
        status = "will run" if reasons else "up to date"
        if reasons:
            dirty.add(step.name)
        print(
            f"{step.name:14} | {status:10} | "
            + (reasons[0] if reasons else "current inputs and outputs match")
        )
        for reason in reasons[1:]:
            print(f"  {reason}")
        for detail in details:
            print(f"  {detail}")
    if command == "label":
        known = {p.path for p in caches}
        for path in sorted((c.DATA / "llm").glob("*.jsonl")):
            if path in known:
                continue
            old = unused_job(path)
            if old.unused:
                caches.append(old)
                print(f"{old.name:14} | up to date | inactive job; no requests")
                for line in old.lines():
                    print(f"  {line}")
        if prune:
            prune_caches(caches)
            print(
                f"pruned {sum(len(p.unused) for p in caches)} unused request records; "
                f"{sum(p.unused_bytes for p in caches)} bytes"
            )
