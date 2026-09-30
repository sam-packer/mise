"""Track step inputs and outputs in stamps, and record each command's provenance in run.json."""

import dataclasses
import importlib.metadata
import json
import platform
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mise_ml import config
from mise_ml.log import get
from mise_ml.util import sha256_file, write_json

log = get(__name__)
_reads: set[Path] | None = None
_read_lock = threading.Lock()


def observe_read(path: Path) -> None:
    """Record HTTP cache reads across the worker threads of the active step."""
    with _read_lock:
        if _reads is not None:
            _reads.add(path)


@contextmanager
def capture_reads():
    """Collect HTTP cache paths read by all worker threads during one pipeline step."""
    global _reads
    with _read_lock:
        if _reads is not None:
            raise RuntimeError("nested pipeline steps cannot share a read receipt")
        _reads = set()
    try:
        yield _reads
    finally:
        with _read_lock:
            _reads = None


PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "xgrammar",
    "onnx",
    "onnxruntime",
    "onnxscript",
    "numpy",
    "pandas",
    "pillow",
)


@dataclasses.dataclass(frozen=True)
class ContentInput:
    """A file dependency hashed from the content a step actually consumes."""

    path: Path
    digest: str
    label: str = ""


StampInput = Path | ContentInput


def file_hashes(paths: list[StampInput]) -> dict[str, str | None]:
    """Hash repository paths, or use ContentInput labels and digests for selected content."""
    return {
        str(p.relative_to(config.REPO_ROOT)).replace("\\", "/")
        + (f"#{entry.label}" if isinstance(entry, ContentInput) and entry.label else ""): (
            entry.digest
            if isinstance(entry, ContentInput)
            else sha256_file(p)
            if p.exists()
            else None
        )
        for entry in paths
        for p in [entry.path if isinstance(entry, ContentInput) else entry]
    }


def snapshot(inputs: list[StampInput], cfg: Any, sources: list[Path]) -> dict[str, Any]:
    return json.loads(
        json.dumps(
            {
                "version": 2,
                "inputs": file_hashes(inputs),
                "config": dataclasses.asdict(cfg),
                "sources": file_hashes(sources),
            },
            sort_keys=True,
        )
    )


def stamp_path(step: str) -> Path:
    return config.MODELS / f"{step}.stamp"


def read_stamp(step: str) -> dict[str, Any] | None:
    """Return a supported stamp, or None when it is missing, invalid, or from another format."""
    path = stamp_path(step)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return None
    return value if isinstance(value, dict) and value.get("version") == 2 else None


def changed(old: dict, new: dict, prefix: str) -> list[str]:
    reasons = []
    for key in sorted(old.keys() | new.keys()):
        a, b = old.get(key), new.get(key)
        name = (
            f"{prefix}.{key}"
            if prefix == "config" or prefix.startswith("config.")
            else f"{prefix} {key}"
        )
        if isinstance(a, dict) and isinstance(b, dict):
            reasons.extend(changed(a, b, name))
        elif a != b:
            reasons.append(
                f"{name} changed" + (f": {a!r} -> {b!r}" if prefix.startswith("config") else "")
            )
    return reasons


def stale_reasons(step: str, state: dict[str, Any], outputs: list[Path]) -> list[str]:
    reasons = [
        f"missing output {p.relative_to(config.REPO_ROOT).as_posix()}"
        for p in outputs
        if not p.is_file()
    ]
    old = read_stamp(step)
    if old is None:
        reasons.append(
            "old or invalid stamp (not verified)" if stamp_path(step).exists() else "no stamp"
        )
    else:
        for section in ("inputs", "config", "sources", "observed", "outputs"):
            reasons.extend(
                changed(
                    old.get(section) or {},
                    state.get(section) or {},
                    section.rstrip("s") if section != "config" else section,
                )
            )
    reasons.extend(
        f"missing input {key}" for key, digest in state["inputs"].items() if digest is None
    )
    if state.get("observed") is None and step in ("fetch", "curate", "resolve"):
        reasons.append("HTTP input receipt not recorded; verify cached API inputs on next run")
    for key, digest in state.get("observed", {}).items() if state.get("observed") else ():
        if digest is None:
            reasons.append(f"missing cached API input {key}")
    return reasons


def is_current(step: str, state: dict[str, Any], outputs: list[Path]) -> bool:
    return not stale_reasons(step, state, outputs)


def write_stamp(step: str, state: dict[str, Any]) -> None:
    path = stamp_path(step)
    write_json(path, state)


def configs() -> dict[str, Any]:
    return {
        cls.__name__: dataclasses.asdict(cls())
        for cls in (
            config.CurateConfig,
            config.ResolveConfig,
            config.ProfileConfig,
            config.TeacherConfig,
            config.StudentConfig,
            config.ExportConfig,
        )
    }


def gpu() -> dict[str, Any]:
    import torch

    return {
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_run_json(used: dict[str, Any] | None = None) -> None:
    from mise_ml.fetch import load_sources
    from mise_ml.plan import cache_plans
    from mise_ml.steps import STEPS

    sources = []
    for src in load_sources():
        present = src.dest.exists()
        sources.append(
            {
                "name": src.name,
                "url": src.url,
                "path": src.path,
                "present": present,
                "sha256": sha256_file(src.dest) if present and not src.sha256 else src.sha256,
            }
        )
    teacher, student, labeler = (
        config.TeacherConfig(),
        config.StudentConfig(),
        config.ProfileConfig(),
    )
    bundle_files = sorted(p for p in config.BUNDLE.glob("*") if p.is_file())
    bundle_files += sorted((config.BUNDLE / "model").glob("*"))
    report = json.loads(config.EVAL_REPORT.read_text()) if config.EVAL_REPORT.exists() else None
    write_json(
        config.RUN_JSON,
        {
            "project": {"name": "mise-ml", "version": importlib.metadata.version("mise-ml")},
            "seed": config.SEED,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
            "gpu": gpu(),
            "models": {
                "labeler": {"id": labeler.model, "revision": labeler.revision},
                "teacher": {"id": teacher.backbone, "revision": teacher.revision},
                "student": {"id": student.backbone, "revision": student.revision},
            },
            "config": configs(),
            "sources": sources,
            "inputs": file_hashes([config.SOURCES, config.EVAL_FEELINGS, config.VOCAB_PATH]),
            "intermediate": file_hashes(sorted(config.CURATED.glob("*.jsonl"))),
            "outputs": file_hashes(bundle_files),
            "ship": report["ship"] if report else None,
            "steps": {name: (used or {}).get(name) or read_stamp(name) for name in STEPS},
            "llm": {
                p.name: {
                    "cache": str(p.path.relative_to(config.REPO_ROOT)),
                    "sha256": p.digest,
                    "records_used": p.used,
                    "membership_verified": p.membership_verified,
                    "notes": p.notes,
                }
                for p in cache_plans()
            },
        },
    )
    log.info(f"provenance -> {config.RUN_JSON.relative_to(config.ML_ROOT).as_posix()}")
