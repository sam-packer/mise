"""Stamps that let `all` skip finished steps, and the run.json provenance record."""

import dataclasses
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from mise_ml import config
from mise_ml.util import sha256_file, write_json

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


def file_hashes(paths: list[Path]) -> dict[str, str | None]:
    return {
        str(p.relative_to(config.REPO_ROOT)).replace("\\", "/"): sha256_file(p)
        if p.exists()
        else None
        for p in sorted(paths)
    }


def stamp_digest(inputs: list[Path], cfg: Any) -> str:
    blob = json.dumps(
        {"inputs": file_hashes(inputs), "config": dataclasses.asdict(cfg)}, sort_keys=True
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def stamp_path(step: str) -> Path:
    return config.MODELS / f"{step}.stamp"


def is_current(step: str, inputs: list[Path], cfg: Any, outputs: list[Path]) -> bool:
    path = stamp_path(step)
    return (
        all(p.exists() for p in outputs)
        and path.exists()
        and path.read_text(encoding="utf-8") == stamp_digest(inputs, cfg)
    )


def write_stamp(step: str, inputs: list[Path], cfg: Any) -> None:
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
    print(f"provenance -> {config.RUN_JSON}")
