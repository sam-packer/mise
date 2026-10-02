"""Inspect command dependencies and caches without loading models.

Plans report pending work without writes. An explicit prune request removes unused cache records.
"""

import json

from mise_ml import config as c
from mise_ml.cache import CachePlan, inspect_job, unused_job
from mise_ml.cache import prune as prune_caches
from mise_ml.provenance import ContentInput, stale_reasons
from mise_ml.steps import command_steps
from mise_ml.util import sha256_file


def cache_plans() -> list[CachePlan]:
    plans = []
    for step in command_steps("label"):
        try:
            require_inputs(step)
            plans.append(inspect_job(step.call()))
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


def show(command: str, *, prune: bool = False, tune: bool = False) -> None:
    print(f"{command} plan (no models; {'prune only' if prune else 'no writes'})")
    print("step           | status     | reasons")
    dirty: set[str] = set()
    caches: list[CachePlan] = []
    for step in command_steps(command):
        reasons: list[str] = []
        details: list[str] = []
        upstream = [name for name in step.needs if name in dirty]
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
                result, notes = inspect_job(step.call()), []
                if upstream:
                    result.membership_verified = False
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
            if tune and step.name == "train-teacher":
                reasons.insert(0, "--tune runs a new search")
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
            try:
                old = unused_job(path)
            except (OSError, ValueError, KeyError) as exc:
                print(f"{path.stem:14} | will run   | cannot inspect inactive cache: {exc}")
                if prune:
                    raise RuntimeError("prune stopped: incomplete cache inspection") from exc
                continue
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
