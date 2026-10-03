"""The command graph and the inputs consumed by each step.

Factories run at the step boundary, after upstream outputs have been written.
They read data and metadata only; they never load a model or contact a service.
"""

import dataclasses
import hashlib
import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mise_ml import config as c
from mise_ml.provenance import ContentInput, StampInput, file_hashes, read_stamp, snapshot
from mise_ml.util import iter_jsonl

SOURCE = Path(__file__).resolve().parent
STUDENT = c.MODELS / "student"
TARGET = c.REPO_ROOT / "static" / "bundle"


@dataclass(frozen=True)
class Settings:
    values: dict[str, Any] = field(default_factory=dict)


def settings(*classes: type, **values: Any) -> Settings:
    return Settings({**{cls.__name__: dataclasses.asdict(cls()) for cls in classes}, **values})


def publish_settings() -> Settings:
    from mise_ml import keys

    # Credentials permit access; only these public settings select the upload content.
    return settings(
        account_id=keys.env("R2_ACCOUNT_ID"),
        bucket=keys.env("R2_BUCKET") or keys.R2_BUCKET_DEFAULT,
        public_url=keys.env("R2_PUBLIC_URL").rstrip("/"),
    )


def curate_settings() -> Settings:
    from mise_ml import keys

    return settings(
        c.CurateConfig, c.ResolveConfig, smithsonian_enabled=bool(keys.env("SMITHSONIAN_API_KEY"))
    )


def content(path: Path, value: Any, label: str = "") -> ContentInput:
    return ContentInput(
        path, hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), label
    )


def files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def constants(*names: str) -> ContentInput:
    return content(
        SOURCE / "config.py", {name: getattr(c, name) for name in names}, ",".join(names)
    )


def catalog_inputs(queries: bool = True) -> list[StampInput]:
    from mise_ml.data import load_catalog

    if not c.RESOLVED.is_file() or not c.PROFILES.is_file():
        return [c.RESOLVED, c.PROFILES]
    catalog = load_catalog()
    return [
        content(
            c.RESOLVED,
            [[it[k] for k in ("id", "category", "title", "creator")] for it in catalog.items],
            "profiled id, category, title, creator",
        ),
        content(
            c.PROFILES,
            [
                [it[k] for k in ("id", "vibe", "description", *(("queries",) if queries else ()))]
                for it in catalog.items
            ],
            "profiled identity, vibe, description" + (", queries" if queries else ""),
        ),
    ]


def teacher_item_inputs() -> list[StampInput]:
    """The text that the teacher embeds for each work: source facts and profile."""
    from mise_ml.data import load_catalog
    from mise_ml.teacher import item_texts

    paths = [c.CATALOG, c.RESOLVED, c.PROFILES]
    if not all(p.is_file() for p in paths):
        return paths
    return [content(c.CATALOG, item_texts(load_catalog()), "teacher item texts")]


def labels() -> Path:
    from mise_ml.vocab import labels_path, load_vocab

    return labels_path(load_vocab())


def query_inputs() -> list[StampInput]:
    return [
        *catalog_inputs(),
        selected(labels(), "key", "palette", "light", "typeface"),
        selected(c.PAT, "phrase", "rgb"),
        selected(c.PAT_SENTENCES, "key", "text"),
        selected(c.EVAL_FEELINGS, "text"),
        c.VOCAB_PATH,
    ]


def selected(path: Path, *fields: str) -> StampInput:
    if not path.is_file():
        return path
    return content(path, [[r.get(f) for f in fields] for r in iter_jsonl(path)], ",".join(fields))


def public_items() -> ContentInput:
    from mise_ml.data import load_catalog
    from mise_ml.export import bundle_item

    return content(c.RESOLVED, [bundle_item(it) for it in load_catalog().items], "public items")


def item_inputs() -> list[StampInput]:
    from mise_ml.profile import item_fields, item_image, item_records

    rows = item_records()
    return [
        content(c.RESOLVED, {k: item_fields(r) for k, r in rows.items()}, "prompt fields"),
        *[path for r in rows.values() if (path := item_image(r)) is not None],
    ]


def leak_inputs() -> list[StampInput]:
    # The block lists depend on the prompts and decoding settings, not on catalog changes.
    from mise_ml.llm import Job
    from mise_ml.profile import item_records, leak_prints

    job = Job("leaks", c.LEAK_BLOCK, c.ProfileConfig())
    return [content(SOURCE / "profile.py", leak_prints(item_records(), job), "leak prompts")]


def theme_inputs() -> list[StampInput]:
    if not c.FACTS.is_file():
        return [c.FACTS]
    return [
        content(
            c.FACTS,
            {
                r["id"]: hashlib.sha256(r["lyrics"].encode()).hexdigest()
                for r in iter_jsonl(c.FACTS)
                if r.get("lyrics")
            },
            "lyrics",
        )
    ]


def mood_inputs() -> list[StampInput]:
    from mise_ml.profile import moods_job

    spec = moods_job()
    return [content(SOURCE / "profile.py", {k: spec.fields(k) for k in spec.keys}, "mood seeds")]


def distill_inputs() -> list[StampInput]:
    from mise_ml.profile import distill_job

    spec = distill_job()
    return [
        selected(c.EVAL_FEELINGS, "text"),
        content(SOURCE / "profile.py", {k: spec.fields(k) for k in spec.keys}, "distill seeds"),
    ]


def student_files() -> list[Path]:
    required = [
        STUDENT / name
        for name in (
            "meta.json",
            "heads.pt",
            "projection.pt",
            "items.npy",
            "encoder/model.safetensors",
            "encoder/config.json",
            "encoder/tokenizer.json",
            "encoder/tokenizer_config.json",
        )
    ]
    return sorted(set(required + files(STUDENT / "encoder")))


def images(root: Path = c.IMG) -> list[Path]:
    return sorted(
        {
            root / r["image"]["src"].rsplit("/", 1)[-1]
            for r in iter_jsonl(c.RESOLVED)
            if r.get("image")
        }
    )


def bundle_files(root: Path = c.BUNDLE) -> list[Path]:
    required = [
        root / name
        for name in (
            "manifest.json",
            "items.json",
            "vocab.json",
            "vectors.bin",
            "penalty.bin",
            "search-index.json",
            "model/model.onnx",
            "model/config.json",
            "model/tokenizer.json",
            "model/tokenizer_config.json",
        )
    ]
    return sorted(set(required + files(root)))


def gate_files(root: Path) -> list[Path]:
    """The bundle files that eval reads to run the graph and rank the catalog."""
    names = [
        "manifest.json",
        "vectors.bin",
        "model/model.onnx",
        "model/config.json",
        "model/tokenizer.json",
        "model/tokenizer_config.json",
    ]
    # An older installed bundle has no penalty; its manifest says so, and eval ranks without one.
    if (root / "penalty.bin").is_file():
        names.append("penalty.bin")
    return [root / name for name in names]


def source_outputs() -> list[Path]:
    from mise_ml.fetch import load_sources

    return [
        c.RAW / s.skip_if if s.skip_if and (c.RAW / s.skip_if).is_file() else s.dest
        for s in load_sources()
    ]


def raw_curate() -> list[Path]:
    return [
        c.MET_CSV,
        c.RAW / "poetrydb" / "poems.jsonl",
        *[
            c.RAW / "text2colors" / f"{split}_{part}.pkl"
            for split in ("train", "test")
            for part in ("names", "palettes_rgb")
        ],
    ]


@dataclass(frozen=True)
class Step:
    name: str
    command: str
    inputs: Callable[[], list[StampInput]]
    config: Callable[[], Any]
    modules: tuple[str, ...]
    outputs: Callable[[], list[Path]]
    action: str
    needs: tuple[str, ...] = ()
    llm: bool = False
    observed_http: bool = False

    def state(self) -> dict[str, Any]:
        state = snapshot(self.inputs(), self.config(), [SOURCE / f"{m}.py" for m in self.modules])
        state["outputs"] = file_hashes(self.outputs())
        if self.observed_http:
            old = read_stamp(self.name) or {}
            paths = old.get("observed")
            state["observed"] = (
                file_hashes([c.REPO_ROOT / p for p in paths]) if paths is not None else None
            )
        return state

    def call(self) -> Any:
        module, function = self.action.split(":")
        return getattr(importlib.import_module(f"mise_ml.{module}"), function)()


COMMON = ("data", "color", "vocab", "util", "heads", "training", "inference", "rated")
PROFILE = ("profile", "llm", "audit", "data", "color", "vocab", "util")

# Every command, including the individually resumable LLM jobs, uses this graph.
STEPS = {
    step.name: step
    for step in (
        Step(
            "fetch",
            "download",
            lambda: [c.SOURCES, constants("SEED")],
            lambda: settings(c.CurateConfig, c.ResolveConfig),
            ("fetch", "http", "util"),
            source_outputs,
            "fetch:run",
            observed_http=True,
        ),
        Step(
            "curate",
            "download",
            lambda: [*raw_curate(), constants("SEED", "CATALOG_VERSION")],
            curate_settings,
            ("curate", "http", "util", "threads"),
            lambda: [c.CATALOG, c.CATALOG_META, c.PAT],
            "curate:run",
            ("fetch",),
            observed_http=True,
        ),
        Step(
            "resolve",
            "download",
            lambda: [c.CATALOG, c.CATALOG_META, constants("CATEGORIES", "RESOLVE_VERSION")],
            lambda: settings(c.ResolveConfig, c.CurateConfig),
            ("resolve", "curate", "http", "color", "util", "threads"),
            lambda: [c.RESOLVED, c.RESOLVE_DROPPED, c.RESOLVE_META, *images()],
            "resolve:run",
            ("curate",),
            observed_http=True,
        ),
        Step(
            "facts",
            "download",
            lambda: [
                content(
                    c.RESOLVED,
                    [
                        [
                            r["id"],
                            r["title"],
                            r.get("creator"),
                            r.get("source", {}).get("artist_mbid"),
                            r.get("source", {}).get("deezer"),
                        ]
                        for r in iter_jsonl(c.RESOLVED)
                        if r["category"] == "song"
                    ],
                    "song identity",
                )
            ],
            lambda: settings(c.ResolveConfig),
            ("facts", "curate", "http", "util", "threads"),
            lambda: [c.FACTS],
            "facts:run",
            ("resolve",),
            observed_http=True,
        ),
        Step(
            "themes",
            "label",
            theme_inputs,
            c.ProfileConfig,
            PROFILE,
            lambda: [c.THEMES],
            "profile:themes_job",
            ("facts",),
            True,
        ),
        # Label a sample first and block the prompt words that leak into it.
        Step(
            "leaks",
            "label",
            leak_inputs,
            c.ProfileConfig,
            PROFILE,
            lambda: [c.LEAK_BLOCK],
            "profile:leaks_job",
            ("themes",),
            True,
        ),
        Step(
            "items",
            "label",
            lambda: [*item_inputs(), c.LEAK_BLOCK],
            c.ProfileConfig,
            PROFILE,
            lambda: [c.PROFILES],
            "profile:items_job",
            ("themes", "leaks"),
            True,
        ),
        Step(
            "moods",
            "label",
            mood_inputs,
            c.ProfileConfig,
            PROFILE,
            lambda: [c.MOODS],
            "profile:moods_job",
            llm=True,
        ),
        Step(
            "pat",
            "label",
            lambda: [content(c.PAT, sorted({r["phrase"] for r in iter_jsonl(c.PAT)}), "phrases")],
            c.ProfileConfig,
            PROFILE,
            lambda: [c.PAT_SENTENCES],
            "profile:pat_job",
            llm=True,
        ),
        Step(
            "labels",
            "label",
            lambda: [
                selected(c.EVAL_FEELINGS, "text"),
                selected(c.MOODS, "feelings"),
                selected(c.PROFILES, "queries"),
                c.VOCAB_PATH,
            ],
            c.ProfileConfig,
            PROFILE,
            lambda: [labels()],
            "profile:labels_job",
            ("items", "moods"),
            True,
        ),
        Step(
            "distill",
            "label",
            distill_inputs,
            c.ProfileConfig,
            PROFILE,
            lambda: [c.DISTILL],
            "profile:distill_job",
            llm=True,
        ),
        # A cheap check of the profiles before training. It warns and never fails.
        Step(
            "audit",
            "train",
            lambda: [*catalog_inputs(False), constants("CATEGORIES"), c.LEAK_BLOCK],
            Settings,
            ("audit", "data", "util"),
            lambda: [c.DATA_AUDIT],
            "audit:run",
        ),
        Step(
            "train-teacher",
            "train",
            lambda: [
                *query_inputs(),
                *teacher_item_inputs(),
                selected(c.DISTILL, "text"),
                c.FIT_LABELS,
                c.TUNING_SET,
                constants("SEED", "CATEGORIES"),
            ],
            c.TeacherConfig,
            (*COMMON, "teacher", "features"),
            # The step writes the params file when it searches, so the file is an output.
            # A changed or missing file runs the step again.
            lambda: [c.MODELS / "teacher.pt", c.MODELS / "teacher_outputs.pt", c.TEACHER_PARAMS],
            "teacher:run",
        ),
        Step(
            "train-student",
            "train",
            lambda: [
                c.MODELS / "teacher_outputs.pt",
                *catalog_inputs(False),
                c.VOCAB_PATH,
                c.TUNING_SET,
                # Vocabulary pruning keeps the tokens of the eval feelings.
                selected(c.EVAL_FEELINGS, "text"),
                constants("SEED", "CATEGORIES"),
            ],
            c.StudentConfig,
            (*COMMON, "student", "wordpiece"),
            student_files,
            "student:run",
            ("train-teacher",),
        ),
        Step(
            "export",
            "train",
            lambda: [
                *student_files(),
                # The exposure penalty counts wins over the training texts in this file.
                c.MODELS / "teacher_outputs.pt",
                *query_inputs(),
                *images(),
                c.REPO_ROOT / "scripts" / "build-name-data.ts",
                c.REPO_ROOT / "scripts" / "build-samples.ts",
                c.REPO_ROOT / "src" / "lib" / "examples.ts",
                c.REPO_ROOT / "src" / "lib" / "code.ts",
                *[
                    c.REPO_ROOT / "src" / "lib" / "mood" / name
                    for name in (
                        "name-data.ts",
                        "types.ts",
                        "anchor-search.ts",
                        "search.ts",
                        "engine.ts",
                        "fp16.ts",
                        "samples.ts",
                    )
                ],
                public_items(),
                c.TUNING_SET,
                constants("SEED", "CATEGORIES"),
            ],
            lambda: settings(c.ExportConfig, c.TeacherConfig),
            # Room calibration drops the texts near an eval feeling with the teacher's rule.
            (*COMMON, "student", "export", "teacher", "features"),
            lambda: [*bundle_files(), *images(c.BUNDLE / "img")],
            "export:run",
            ("train-student",),
        ),
        Step(
            "eval",
            "train",
            lambda: [
                *query_inputs(),
                # The teacher's item vectors come from these texts.
                *teacher_item_inputs(),
                c.MODELS / "teacher.pt",
                *student_files(),
                *gate_files(c.BUNDLE),
                # The ship gate compares the new student with the installed bundle.
                *gate_files(TARGET),
                TARGET / "items.json",
                selected(c.EVAL_FEELINGS, "text", "set"),
                c.TUNING_SET,
                c.GOLD_SET,
                constants("SEED", "CATEGORIES"),
            ],
            # The export settings set the exposure penalty that eval subtracts.
            lambda: settings(c.TeacherConfig, c.ExportConfig),
            (*COMMON, "evaluate", "export", "student", "teacher", "features"),
            lambda: [c.EVAL_REPORT, c.EVAL_FAILURES],
            "evaluate:run",
            ("export", "train-teacher"),
        ),
        Step(
            "install",
            "train",
            lambda: [*bundle_files(), c.EVAL_REPORT],
            Settings,
            ("install", "delivery", "util"),
            lambda: [TARGET / p.relative_to(c.BUNDLE) for p in bundle_files()],
            "install:run",
            ("eval",),
        ),
        Step(
            "publish",
            "publish",
            lambda: bundle_files(TARGET),
            publish_settings,
            ("publish", "delivery", "keys", "threads"),
            lambda: [c.REPO_ROOT / "src" / "lib" / "bundle.ts"],
            "publish:run",
        ),
    )
}


def command_steps(command: str) -> list[Step]:
    return [step for step in STEPS.values() if step.command == command]
