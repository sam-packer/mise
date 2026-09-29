"""Stamps that let commands skip finished steps, and the run.json provenance record."""

import dataclasses
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from mise_ml import config
from mise_ml.log import get
from mise_ml.util import sha256_file, write_json

log = get(__name__)

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


StampInput = Path | ContentInput


def file_hashes(paths: list[StampInput]) -> dict[str, str | None]:
    return {
        str(p.relative_to(config.REPO_ROOT)).replace("\\", "/"): (
            entry.digest
            if isinstance(entry, ContentInput)
            else sha256_file(p)
            if p.exists()
            else None
        )
        for entry in paths
        for p in [entry.path if isinstance(entry, ContentInput) else entry]
    }


def stamp_digest(inputs: list[StampInput], cfg: Any) -> str:
    blob = json.dumps(
        {"inputs": file_hashes(inputs), "config": dataclasses.asdict(cfg)}, sort_keys=True
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def stamp_path(step: str) -> Path:
    return config.MODELS / f"{step}.stamp"


def is_current(step: str, inputs: list[StampInput], cfg: Any, outputs: list[Path]) -> bool:
    path = stamp_path(step)
    return (
        all(p.exists() for p in outputs)
        and path.exists()
        and path.read_text(encoding="utf-8") == stamp_digest(inputs, cfg)
    )


def migrate_stamp(step: str, inputs: list[StampInput], cfg: Any, outputs: list[Path]) -> bool:
    """Convert a legacy file stamp only while every original input still matches."""
    if is_current(step, inputs, cfg, outputs):
        return True
    original = [i.path if isinstance(i, ContentInput) else i for i in inputs]
    if is_current(step, original, cfg, outputs):
        write_stamp(step, inputs, cfg)
        return True
    return False


def write_stamp(step: str, inputs: list[StampInput], cfg: Any) -> None:
    path = stamp_path(step)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stamp_digest(inputs, cfg), encoding="utf-8")


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


def write_run_json() -> None:
    from mise_ml.fetch import load_sources

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
        },
    )
    log.info(f"provenance -> {config.RUN_JSON.relative_to(config.ML_ROOT).as_posix()}")
