import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mise_ml import log as logs
from mise_ml.config import ML_ROOT

COMMANDS = {
    "download": "Fetch sources, curate the catalog, and resolve media and links.",
    "label": "Write profiles and labels with the local LLM.",
    "train": "Train, export, evaluate, and install only when the ship gate passes.",
    "publish": "Upload the installed public bundle and private catalog to R2.",
}

log = logs.get("cli")


def preflight(command: str) -> None:
    from mise_ml import keys
    from mise_ml.config import EVAL_FEELINGS, VOCAB_PATH

    problems = []
    if command == "download":
        problems.extend(keys.missing(list(keys.KEYS)))
    if command == "publish":
        problems.extend(f"missing {key}; set it in ml/.env" for key in keys.missing_r2())
    if command in ("label", "train"):
        import torch

        if not torch.cuda.is_available():
            problems.append("no CUDA GPU is visible to PyTorch")
        for path in (VOCAB_PATH, EVAL_FEELINGS):
            if not path.is_file():
                problems.append(f"missing {path}")
    for problem in problems:
        log.error("cannot start %s: %s", command, problem)
    if problems:
        raise SystemExit(1)


class Run:
    """Run a command with step headers and a summary."""

    def __init__(self, total: int) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.total = total

    def step(
        self,
        name: str,
        fn: Callable[[], Any],
        stamp: tuple[list[Path], Any, list[Path]] | None = None,
    ) -> Any:
        from mise_ml import provenance as pv

        n = len(self.rows) + 1
        if stamp and pv.is_current(name, *stamp):
            log.info("step %d/%d %s: up to date, skip", n, self.total, name)
            self.rows.append((name, "skipped", "-"))
            return None
        log.info("step %d/%d %s", n, self.total, name)
        start = time.perf_counter()
        try:
            result = fn()
        except BaseException:
            self.rows.append((name, "failed", logs.elapsed(start)))
            raise
        if stamp:
            pv.write_stamp(name, stamp[0], stamp[1])
        self.rows.append((name, "done", logs.elapsed(start)))
        return result

    def summary(self) -> None:
        log.info("summary:")
        for name, status, took in self.rows:
            log.info("  %-14s %-8s %s", name, status, took)


def run_train(run: Run) -> None:
    from mise_ml import evaluate, export, install, student, teacher
    from mise_ml.config import (
        BUNDLE,
        CATALOG_BUNDLE,
        EVAL_FEELINGS,
        EVAL_REPORT,
        MODELS,
        PAT,
        PAT_SENTENCES,
        PROFILES,
        RESOLVED,
        VOCAB_PATH,
        ExportConfig,
        StudentConfig,
        TeacherConfig,
    )
    from mise_ml.data import load_catalog
    from mise_ml.vocab import labels_path, load_vocab

    labels = labels_path(load_vocab())
    source = Path(__file__).parent
    common = [source / f"{name}.py" for name in ("data", "heads", "training", "inference")]
    curated = [RESOLVED, PROFILES, labels, PAT, PAT_SENTENCES, EVAL_FEELINGS, VOCAB_PATH]
    teacher_in = [*curated, *common, source / "teacher.py", source / "features.py"]
    teacher_out = [MODELS / "teacher.pt", teacher.OUTPUTS]
    run.step("train-teacher", teacher.run, (teacher_in, TeacherConfig(), teacher_out))
    student_out = [
        student.STUDENT_DIR / "meta.json",
        student.STUDENT_DIR / "heads.pt",
        student.STUDENT_DIR / "projection.pt",
        student.STUDENT_DIR / "encoder" / "model.safetensors",
        student.STUDENT_DIR / "encoder" / "config.json",
        student.STUDENT_DIR / "encoder" / "tokenizer.json",
        student.STUDENT_DIR / "encoder" / "tokenizer_config.json",
    ]
    student_in = [teacher.OUTPUTS, RESOLVED, PROFILES, VOCAB_PATH, *common, source / "student.py"]
    run.step("train-student", student.run, (student_in, StudentConfig(), student_out))
    encoder_files = sorted(p for p in (student.STUDENT_DIR / "encoder").rglob("*") if p.is_file())
    export_in = [*student_out, *encoder_files, *curated, *common, source / "export.py"]
    images = [
        BUNDLE / "img" / item["image"]["src"].rsplit("/", 1)[-1]
        for item in load_catalog().items
        if item.get("image")
    ]
    run.step(
        "export",
        export.run,
        (
            export_in,
            ExportConfig(),
            [
                BUNDLE / "manifest.json",
                BUNDLE / "vocab.json",
                BUNDLE / "model" / "model.onnx",
                BUNDLE / "model" / "tokenizer.json",
                BUNDLE / "model" / "tokenizer_config.json",
                BUNDLE / "model" / "config.json",
                CATALOG_BUNDLE / "catalog.json",
                CATALOG_BUNDLE / "items.json",
                CATALOG_BUNDLE / "vectors.bin",
                *images,
            ],
        ),
    )
    run.step("eval-judge", evaluate.judge)
    report = run.step("eval", evaluate.run)
    if not report["ship"]["ok"]:
        run.rows.append(("install", "not run", "-"))
        log.error(
            "ship gate failed: %s; report: %s", "; ".join(report["ship"]["reasons"]), EVAL_REPORT
        )
        raise SystemExit(1)
    run.step("install", install.run)


def command(name: str) -> None:
    argparse.ArgumentParser(prog=name, description=COMMANDS[name]).parse_args()
    load_dotenv(ML_ROOT / ".env", override=False)
    # The student encodes about 1,030-1,120 items per step. Without rounding, the allocator
    # caches a block for each size, reserves more than the GPU has, and a kernel launch fails.
    # Torch reads this before its first CUDA allocation, and no torch import happens above.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "roundup_power2_divisions:4")
    run = Run({"download": 3, "label": 1, "train": 6, "publish": 1}[name])
    with logs.session(name):
        try:
            preflight(name)
            if name == "download":
                from mise_ml import curate, fetch, resolve
                from mise_ml.config import CATALOG, CATALOG_META, PAT, SOURCES, CurateConfig

                run.step("fetch", fetch.run)
                raw = [src.dest for src in fetch.load_sources() if src.dest.exists()]
                run.step(
                    "curate",
                    curate.run,
                    ([*raw, SOURCES], CurateConfig(), [CATALOG, CATALOG_META, PAT]),
                )
                run.step("resolve", resolve.run)
            elif name == "label":
                from mise_ml import profile

                run.step("profile", profile.run)
            elif name == "train":
                run_train(run)
            else:
                from mise_ml import publish as delivery

                run.step("publish", delivery.run)
        finally:
            run.summary()


def download() -> None:
    command("download")


def label() -> None:
    command("label")


def train() -> None:
    command("train")


def publish() -> None:
    command("publish")
