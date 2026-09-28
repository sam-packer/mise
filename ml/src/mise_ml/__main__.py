import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mise_ml import log as logs
from mise_ml.config import CATEGORIES, ML_ROOT

STEPS = {
    "fetch": "download the raw sources and verify their checksums",
    "curate": "select films, books, songs, art, poems (cached API calls); write catalog.jsonl",
    "resolve": "resolve media and links, download images (cached, resumable)",
    "profile": "local LLM pass: item profiles, moods, PAT sentences, labels (resumable)",
    "train-teacher": "cache Qwen3-Embedding-8B features and train the teacher heads",
    "train-student": "distill the teacher into MiniLM",
    "export": "export ONNX, quantize, and write out/bundle",
    "eval": "report recall@10, palette delta E, choice accuracy, latency; write out/run.json",
    "install": "copy out/bundle into ../static/bundle for the web app",
    "all": "run every step above in order, skip finished work, install if the student ships",
}

log = logs.get("all")


def preflight() -> None:
    """Name everything that is missing before hours of work start."""
    import torch

    from mise_ml import keys
    from mise_ml.config import EVAL_FEELINGS, VOCAB_PATH

    problems = keys.missing(list(keys.KEYS))
    if not torch.cuda.is_available():
        problems.append("no CUDA GPU is visible to PyTorch")
    if not VOCAB_PATH.exists():
        problems.append(f"missing {VOCAB_PATH} (the vocab source of truth)")
    if not EVAL_FEELINGS.exists():
        problems.append(f"missing {EVAL_FEELINGS}; eval needs your human-written feelings")
    for problem in problems:
        log.error("cannot start: %s", problem)
    if problems:
        raise SystemExit(1)


class Run:
    """Runs the steps of `all`, with a header line each and a summary at the end."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.total = sum(1 for name in STEPS if name != "all") + 1  # eval --judge is its own

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
            self.summary()
            raise
        if stamp:
            pv.write_stamp(name, stamp[0], stamp[1])
        self.rows.append((name, "done", logs.elapsed(start)))
        return result

    def summary(self) -> None:
        log.info("summary:")
        for name, status, took in self.rows:
            log.info("  %-14s %-8s %s", name, status, took)


def run_all() -> None:
    from mise_ml import curate, evaluate, export, fetch, install, profile, resolve, student, teacher
    from mise_ml.config import (
        BUNDLE,
        CATALOG,
        CATALOG_META,
        EVAL_FEELINGS,
        EVAL_REPORT,
        MODELS,
        PAT,
        PAT_SENTENCES,
        PROFILES,
        RESOLVED,
        SOURCES,
        VOCAB_PATH,
        CurateConfig,
        ExportConfig,
        StudentConfig,
        TeacherConfig,
    )
    from mise_ml.vocab import labels_path, load_vocab

    preflight()
    run = Run()
    run.step("fetch", fetch.run)
    raw = [src.dest for src in fetch.load_sources() if src.dest.exists()]
    run.step("curate", curate.run, ([*raw, SOURCES], CurateConfig(), [CATALOG, CATALOG_META, PAT]))
    run.step("resolve", resolve.run)
    run.step("profile", profile.run)
    labels = labels_path(load_vocab())
    curated = [RESOLVED, PROFILES, labels, PAT, PAT_SENTENCES, EVAL_FEELINGS, VOCAB_PATH]
    teacher_out = [MODELS / "teacher.pt", teacher.OUTPUTS]
    run.step("train-teacher", teacher.run, (curated, TeacherConfig(), teacher_out))
    student_out = [student.STUDENT_DIR / "meta.json", student.STUDENT_DIR / "heads.pt"]
    student_in = [teacher.OUTPUTS, RESOLVED, PROFILES, VOCAB_PATH]
    run.step("train-student", student.run, (student_in, StudentConfig(), student_out))
    export_in = [*student_out, student.STUDENT_DIR / "encoder" / "model.safetensors", *curated]
    run.step("export", export.run, (export_in, ExportConfig(), [BUNDLE / "manifest.json"]))
    run.step("eval-judge", evaluate.judge)
    report = run.step("eval", evaluate.run)
    if not report["ship"]["ok"]:
        run.rows.append(("install", "not run", "-"))
        run.summary()
        log.warning(
            "the student is %.1f points below the teacher on %s (limit 5), so `all` does not "
            "install it; report: %s",
            report["ship"]["gap"] * 100,
            report["ship"]["metric"],
            EVAL_REPORT,
        )
        log.warning("to use the student anyway, run: uv run mise-ml install")
        raise SystemExit(1)
    run.step("install", install.run)
    run.summary()


def print_steps() -> None:
    print("usage: uv run mise-ml <step>   (no step needs an argument)\n")
    print("Steps, in run order:")
    for name, text in STEPS.items():
        print(f"  {name:<14} {text}")
    print("\nFrom zero: uv sync, then uv run mise-ml all")


def main() -> None:
    # API keys live in ml/.env (see ml/.env.example). A variable set in the shell wins.
    load_dotenv(ML_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(prog="mise-ml", description="mise ML pipeline")
    sub = parser.add_subparsers(dest="step", metavar="step")
    for name, text in STEPS.items():
        p = sub.add_parser(name, help=text, description=text)
        if name == "resolve":
            p.add_argument("categories", nargs="*", help=f"any of {', '.join(CATEGORIES)}")
        if name == "eval":
            p.add_argument(
                "--judge", action="store_true", help="local LLM relevance judgments first"
            )
    args = parser.parse_args()
    if args.step is None:
        print_steps()
        return
    if args.step == "resolve":
        unknown = set(args.categories) - set(CATEGORIES)
        if unknown:
            parser.error(f"unknown categories: {', '.join(sorted(unknown))}")

    with logs.session(args.step):
        match args.step:
            case "all":
                run_all()
            case "resolve":
                from mise_ml import resolve

                resolve.run(args.categories or None)
            case "eval":
                from mise_ml import evaluate

                if args.judge:
                    evaluate.judge()
                evaluate.run()
            case _:
                import importlib

                module = {"train-teacher": "teacher", "train-student": "student"}.get(
                    args.step, args.step
                )
                importlib.import_module(f"mise_ml.{module}").run()


if __name__ == "__main__":
    main()
