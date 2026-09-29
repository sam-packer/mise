"""Read-only LLM cache inspection and explicit removal of unused request records."""

import json
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mise_ml.cache_io import cache_lock
from mise_ml.llm import JobSpec, cached_records, field_digests
from mise_ml.util import atomic_write, iter_jsonl, sha256_file


@dataclass
class CachePlan:
    name: str
    path: Path
    digest: str | None
    keys: int = 0
    used: int = 0
    requests: int = 0
    reasons: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    unused: set[int] = field(default_factory=set)
    unused_bytes: int = 0
    unused_examples: list[str] = field(default_factory=list)
    output_changed: bool = False
    notes: list[str] = field(default_factory=list)
    membership_verified: bool = True

    def lines(self) -> list[str]:
        lines = [
            f"{self.keys} keys; {self.used} records used; "
            f"{self.requests} requests would run (before retries)"
        ]
        for reason, keys in sorted(self.reasons.items()):
            lines.append(
                f"{reason}: {len(keys)}; examples: {json.dumps(keys[:5], ensure_ascii=True)}"
            )
        lines.append(
            f"records no longer used: {len(self.unused)}; {self.unused_bytes} bytes; "
            f"examples: {json.dumps(self.unused_examples[:5], ensure_ascii=True)}"
        )
        if self.output_changed:
            lines.append("materialized output differs from current accepted cache records")
        return lines + self.notes


def entries(path: Path, notes: list[str]):
    """Match replay's interrupted-append rule, retaining the incomplete bytes."""
    lines = path.read_bytes().splitlines(keepends=True) if path.is_file() else []
    last = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), -1)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            if index != last:
                raise
            notes.append(f"incomplete final cache line: {len(line)} bytes; ignored and retained")
            continue
        yield index, line, entry


def inspect_job(spec: JobSpec) -> CachePlan:
    job = spec.job
    keys = sorted(set(spec.keys))
    prints = {k: spec.fingerprint(k) for k in keys}
    fields = {k: field_digests(spec.fields(k)) for k in keys}
    probe = spec.build(keys[:1])[0].request if keys else None
    sig = job.signature(probe) if probe else ""
    parts = job.signature_parts(probe) if probe else {}
    result = CachePlan(
        job.name, job.cache, sha256_file(job.cache) if job.cache.is_file() else None, len(keys)
    )
    latest: dict[str, dict[str, Any]] = {}
    records: dict[str, dict] = {}
    owners: dict[str, int] = {}
    sizes, examples = {}, {}
    for index, line, entry in entries(job.cache, result.notes):
        sizes[index] = len(line)
        examples[index] = entry["keys"]
        for key in entry["keys"]:
            if key in prints:
                latest[key] = entry
        if entry["sig"] != sig or entry.get("data") is None:
            continue
        for record in cached_records(entry["keys"], entry["data"], spec.parse):
            key = record["key"]
            if key in prints and prints[key] == entry["prints"].get(key):
                records[key], owners[key] = record, index
    pending = [k for k in keys if k not in records]
    for key in pending:
        previous = latest.get(key)
        if previous is None:
            result.reasons["new keys"].append(key)
            continue
        explained = False
        if previous["sig"] != sig:
            old = previous.get("signature")
            if old is None:
                result.reasons[
                    "signature changed: model/revision/prompt/schema/token limit "
                    "(parts not recorded)"
                ].append(key)
            else:
                for part in sorted(parts.keys() | old.keys()):
                    if parts.get(part) != old.get(part):
                        result.reasons[f"signature changed: {part}"].append(key)
            explained = True
        if prints[key] != previous["prints"].get(key):
            old = previous.get("fields", {}).get(key)
            if old is None:
                result.reasons["prompt changed (fields not recorded)"].append(key)
            else:
                names = [
                    name
                    for name in sorted(old.keys() | fields[key].keys())
                    if old.get(name) != fields[key].get(name)
                ]
                for name in names or ["prompt rendering"]:
                    result.reasons[f"fingerprint changed: {name}"].append(key)
            explained = True
        if not explained:
            result.reasons["cached answer rejected by current parser"].append(key)
    result.used = len(records)
    result.requests = len(spec.build(pending))
    result.unused = sizes.keys() - set(owners.values())
    result.unused_bytes = sum(sizes[i] for i in result.unused)
    result.unused_examples = list(
        dict.fromkeys(k for i in sorted(result.unused) for k in examples[i])
    )[:5]
    rows = [records[k] for k in sorted(records)]
    expected = spec.project(rows) if spec.project else rows
    result.output_changed = not job.output.is_file() or list(iter_jsonl(job.output)) != expected
    return result


def unused_job(path: Path) -> CachePlan:
    """An old job namespace (for example labels for a removed vocabulary)."""
    result = CachePlan(path.stem, path, sha256_file(path))
    for i, line, entry in entries(path, result.notes):
        if not {"sig", "prints", "keys"} <= entry.keys():
            # Derived files such as distill-seeds.jsonl are not request caches.
            return CachePlan(path.stem, path, result.digest)
        result.unused.add(i)
        result.unused_bytes += len(line)
        result.unused_examples.extend(entry["keys"][: max(0, 5 - len(result.unused_examples))])
    return result


def prune(plans: list[CachePlan]) -> None:
    # Check the whole report before changing any file. Preserve retained bytes exactly.
    with ExitStack() as locks:
        for plan in sorted(plans, key=lambda p: str(p.path)):
            if plan.unused:
                locks.enter_context(cache_lock(plan.path))
        for plan in plans:
            current = sha256_file(plan.path) if plan.path.exists() else None
            if current != plan.digest:
                raise RuntimeError(
                    f"cache changed after inspection: {plan.path}; run --prune again"
                )
        for plan in plans:
            if not plan.unused:
                continue
            with plan.path.open("rb") as stream:
                kept = b"".join(line for i, line in enumerate(stream) if i not in plan.unused)
            atomic_write(plan.path, kept)
