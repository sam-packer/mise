"""Run command plans or execute their steps with shared logging and provenance."""

import argparse
import os
import time
from typing import Any

from dotenv import load_dotenv

from mise_ml import log as logs
from mise_ml.config import ML_ROOT

COMMANDS = {
    "download": "Fetch sources, curate the catalog, and resolve media and links.",
    "label": "Write profiles and labels with the local LLM.",
    "refine": "Grade the profiles with GPT-6.1 Sol, drop failed works, and refill them.",
    "train": "Train, export, evaluate, and install only when the ship gate passes.",
    "publish": "Upload the installed public bundle to R2.",
}

log = logs.get("cli")


def preflight(command: str) -> None:
    from mise_ml import keys
    from mise_ml.config import DISTILL, EVAL_FEELINGS, VOCAB_PATH

    problems = []
    if command == "train" and not DISTILL.is_file():
        problems.append(f"missing {DISTILL}; run uv run label")
    if command == "download":
        problems.extend(keys.missing(["tmdb", "hardcover", "listenbrainz", "lastfm"]))
    if command == "publish":
        problems.extend(f"missing {key}; set it in ml/.env" for key in keys.missing_r2())
    if command == "refine":
        # Refills download new works, and the grader is a cloud model.
        problems.extend(keys.missing(["tmdb", "hardcover", "listenbrainz", "lastfm", "openai"]))
    if command in ("label", "train", "refine"):
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
    """Execute the registry and record the exact stamps used by this command."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.used: dict[str, Any] = {}
        self.llm = None

    def step(self, step: Any) -> Any:
        from mise_ml import provenance as pv
        from mise_ml.cache import inspect_job
        from mise_ml.config import ProfileConfig
        from mise_ml.profile import lazy_llm

        state = step.state()
        if step.llm:
            spec = step.call()
            plan = inspect_job(spec)
            current = not plan.requests and not plan.output_changed
        else:
            current = step.name != "publish" and pv.is_current(step.name, state, step.outputs())
        if current:
            self.used[step.name] = state if step.llm else pv.read_stamp(step.name)
            self.rows.append((step.name, "up to date", "-"))
            log.info("%s: up to date", step.name)
            return None
        log.info("%s: will run", step.name)
        start = time.perf_counter()
        if step.llm:
            if self.llm is None:
                self.llm = lazy_llm(ProfileConfig())
            spec.run(self.llm)
            result = None
        else:
            with pv.capture_reads() as reads:
                result = step.call()
            if step.observed_http:
                state["observed"] = pv.file_hashes(sorted(reads))
        state["outputs"] = pv.file_hashes(step.outputs())
        pv.write_stamp(step.name, state)
        self.used[step.name] = state
        self.rows.append((step.name, "done", logs.elapsed(start)))
        return result

    def summary(self) -> None:
        for name, status, took in self.rows:
            log.info("%-14s %-12s %s", name, status, took)


def command(name: str) -> None:
    import json
    import sys

    parser = argparse.ArgumentParser(prog=name, description=COMMANDS[name])
    if name != "refine":
        parser.add_argument(
            "--plan",
            action="store_true",
            help="Show what would run and why, without writes or model loads.",
        )
    if name == "train":
        parser.add_argument(
            "--tune",
            action="store_true",
            help="Search new teacher settings and replace teacher_params.json.",
        )
    if name == "label":
        parser.add_argument(
            "--prune",
            action="store_true",
            help="Report and remove only unused LLM cache records; do not generate.",
        )
    args = parser.parse_args()
    load_dotenv(ML_ROOT / ".env", override=False)
    if getattr(args, "plan", False) or getattr(args, "prune", False):
        sys.dont_write_bytecode = True
        from mise_ml.plan import show

        show(
            name,
            prune=getattr(args, "prune", False) and not args.plan,
            tune=getattr(args, "tune", False),
        )
        return
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "roundup_power2_divisions:4")
    from mise_ml.config import EVAL_REPORT
    from mise_ml.provenance import write_run_json
    from mise_ml.steps import command_steps

    run = Run()
    with logs.session(name):
        try:
            preflight(name)
            if name in ("label", "refine"):
                from mise_ml.config import SEED
                from mise_ml.util import make_deterministic

                make_deterministic(SEED, warn_only=True)
            if getattr(args, "tune", False):
                from mise_ml.config import TEACHER_PARAMS

                # Without the file, train-teacher searches and writes a new one.
                TEACHER_PARAMS.unlink(missing_ok=True)
                log.info("removed teacher_params.json; train-teacher searches new settings")
            if name == "refine":
                from mise_ml.refine import loop
                from mise_ml.steps import STEPS

                loop(lambda step_name: run.step(STEPS[step_name]))
            for step in command_steps(name):
                if step.name == "install":
                    report = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
                    if not report["ship"]["ok"]:
                        log.error("ship gate failed: %s", "; ".join(report["ship"]["reasons"]))
                        raise SystemExit(1)
                run.step(step)
            write_run_json(run.used)
        finally:
            run.summary()


def download() -> None:
    command("download")


def label() -> None:
    command("label")


def refine() -> None:
    command("refine")


def train() -> None:
    command("train")


def publish() -> None:
    command("publish")
