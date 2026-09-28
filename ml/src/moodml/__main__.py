import argparse

from moodml.config import CATEGORIES

STEPS = {
    "fetch": "download the raw sources and verify their checksums",
    "curate": "apply the curation rules; write data/curated/catalog.jsonl",
    "resolve": "resolve media and links, download images (cached, resumable)",
    "profile": "local LLM pass: item profiles, moods, PAT sentences, labels (resumable)",
    "train-teacher": "cache Qwen3 features and train the teacher heads",
    "train-student": "distill the teacher into MiniLM",
    "export": "export ONNX, quantize, and write out/bundle",
    "eval": "report recall@10, palette delta E, choice accuracy, latency; write out/run.json",
    "all": "run every step in order and skip work that is already done",
}


def preflight() -> None:
    """Name everything that is missing before hours of work start."""
    import torch

    from moodml.config import EVAL_FEELINGS, VOCAB_PATH

    problems = []
    if not torch.cuda.is_available():
        problems.append("no CUDA GPU is visible to PyTorch")
    if not VOCAB_PATH.exists():
        problems.append(f"missing {VOCAB_PATH} (the vocab source of truth)")

    if not EVAL_FEELINGS.exists():
        problems.append(f"missing {EVAL_FEELINGS}; eval needs your human-written feelings")
    if problems:
        raise SystemExit("Cannot start:\n- " + "\n- ".join(problems))


def run_all() -> None:
    from moodml import curate, evaluate, export, fetch, profile, resolve, student, teacher
    from moodml import provenance as pv
    from moodml.config import (
        BUNDLE,
        CATALOG,
        EVAL_FEELINGS,
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
    from moodml.vocab import labels_path, load_vocab

    preflight()
    fetch.run()
    raw = [src.dest for src in fetch.load_sources() if src.dest.exists()]

    def step(name, fn, inputs, cfg, outputs) -> None:
        if pv.is_current(name, inputs, cfg, outputs):
            print(f"== {name}: up to date, skip")
            return
        print(f"== {name}")
        fn()
        pv.write_stamp(name, inputs, cfg)

    step("curate", curate.run, [*raw, SOURCES], CurateConfig(), [CATALOG, PAT])
    print("== resolve")
    resolve.run()
    print("== profile")
    profile.run()
    labels = labels_path(load_vocab())
    curated = [RESOLVED, PROFILES, labels, PAT, PAT_SENTENCES, EVAL_FEELINGS, VOCAB_PATH]
    teacher_out = [MODELS / "teacher.pt", teacher.OUTPUTS]
    step("train-teacher", teacher.run, curated, TeacherConfig(), teacher_out)
    student_out = [student.STUDENT_DIR / "meta.json", student.STUDENT_DIR / "heads.pt"]
    step(
        "train-student",
        student.run,
        [teacher.OUTPUTS, RESOLVED, PROFILES, VOCAB_PATH],
        StudentConfig(),
        student_out,
    )
    step(
        "export",
        export.run,
        [*student_out, student.STUDENT_DIR / "encoder" / "model.safetensors", *curated],
        ExportConfig(),
        [BUNDLE / "manifest.json"],
    )
    print("== eval --judge")
    evaluate.judge()
    print("== eval")
    evaluate.run()


def main() -> None:
    parser = argparse.ArgumentParser(prog="moodml", description="moodboard ML pipeline")
    sub = parser.add_subparsers(dest="step", required=True, metavar="step")
    for name, text in STEPS.items():
        p = sub.add_parser(name, help=text, description=text)
        if name == "resolve":
            p.add_argument("categories", nargs="*", help=f"any of {', '.join(CATEGORIES)}")
        if name == "eval":
            p.add_argument(
                "--judge", action="store_true", help="local LLM relevance judgments first"
            )
    args = parser.parse_args()

    match args.step:
        case "all":
            run_all()
        case "resolve":
            from moodml import resolve

            unknown = set(args.categories) - set(CATEGORIES)
            if unknown:
                parser.error(f"unknown categories: {', '.join(sorted(unknown))}")
            resolve.run(args.categories or None)
        case "eval":
            from moodml import evaluate

            if args.judge:
                evaluate.judge()
            evaluate.run()
        case _:
            import importlib

            module = {"train-teacher": "teacher", "train-student": "student"}.get(
                args.step, args.step
            )
            importlib.import_module(f"moodml.{module}").run()


if __name__ == "__main__":
    main()
