import gc
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from mise_ml.config import (
    BUNDLE,
    CATEGORIES,
    EVAL_REPORT,
    JUDGMENTS,
    ML_ROOT,
    SEED,
    ProfileConfig,
    StudentConfig,
    TeacherConfig,
)
from mise_ml.data import Catalog, load_catalog, load_eval_texts, load_queries, recall_at_k
from mise_ml.export import onnx_run, onnx_session
from mise_ml.features import shared_encoder
from mise_ml.llm import Job, Record, Request, Unit, sha
from mise_ml.log import elapsed, get, num, progress
from mise_ml.profile import (
    FEELING_RULES,
    array_of,
    chunks,
    lazy_llm,
    numbered,
    obj,
    parse_numbered,
)
from mise_ml.provenance import write_run_json
from mise_ml.student import Student, encode_texts, load_student
from mise_ml.teacher import item_store, load_teacher, query_store
from mise_ml.util import iter_jsonl, make_deterministic, write_json
from mise_ml.vocab import load_vocab

log = get(__name__)
SHIP_MARGIN = 0.05
SEP = "\x1f"

JUDGE_SYSTEM = f"""You judge picks for mise, a mood app. A user typed a feeling, and the app \
shows works that should feel like it.

{FEELING_RULES}

For each numbered work, answer fit = true when a thoughtful person would say the work \
feels like the moment, and false otherwise. Judge the mood, not the literal topic."""


@dataclass
class Outputs:
    query: np.ndarray
    items: np.ndarray
    palette: np.ndarray
    choices: list[np.ndarray]


def teacher_outputs(texts: list[str], catalog: Catalog) -> Outputs:
    model, ckpt = load_teacher()
    if ckpt["item_ids"] != [it["id"] for it in catalog.items]:
        raise SystemExit("catalog changed since train-teacher; rerun train-teacher")
    cfg = TeacherConfig(**ckpt["cfg"])
    fq = torch.tensor(query_store(cfg).get(texts), dtype=torch.float32, device="cuda")
    fi = torch.tensor(item_store(cfg).get(catalog.texts), dtype=torch.float32, device="cuda")
    with torch.no_grad():
        palette, *choices = model.heads(fq)
        return Outputs(
            query=model.embed_queries(fq).cpu().numpy(),
            items=model.embed_items(fi).cpu().numpy(),
            palette=palette.cpu().numpy(),
            choices=[c.cpu().numpy() for c in choices],
        )


def student_outputs(texts: list[str], catalog: Catalog) -> tuple[Outputs, list[float]]:
    """The shipped path: the int8 ONNX graph and vectors.bin from the bundle."""
    model_path = BUNDLE / "model" / "model.onnx"
    if not model_path.exists():
        raise SystemExit(f"no bundle at {BUNDLE}; run export first")
    _, tokenizer, _ = load_student()
    items = np.fromfile(BUNDLE / "vectors.bin", dtype="<f4").reshape(len(catalog.items), -1)
    session = onnx_session(model_path, threads=1)
    max_tokens = StudentConfig().max_length
    rows, latency = [], []
    onnx_run(session, tokenizer, [texts[0]], max_tokens)
    for text in progress(texts, desc="student (1 thread)", unit="text"):
        start = time.perf_counter()
        out = onnx_run(session, tokenizer, [text], max_tokens)
        latency.append((time.perf_counter() - start) * 1000)
        rows.append(out)
    return (
        Outputs(
            query=np.concatenate([r["embedding"] for r in rows]),
            items=items,
            palette=np.concatenate([r["palette"] for r in rows]),
            choices=[np.concatenate([r[k] for r in rows]) for k in ("light", "typeface", "scent")],
        ),
        latency,
    )


def baseline_outputs(texts: list[str], catalog: Catalog) -> Outputs:
    """Off-the-shelf MiniLM, used only to widen the judging pool."""
    cfg = StudentConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, revision=cfg.revision)
    model = Student(
        AutoModel.from_pretrained(cfg.backbone, revision=cfg.revision), (1, 1, 1), 8
    ).cuda()
    return Outputs(
        query=encode_texts(model, tokenizer, texts, cfg.max_length),
        items=encode_texts(model, tokenizer, catalog.texts, cfg.item_max_length),
        palette=np.zeros((len(texts), 5, 3)),
        choices=[],
    )


def top_by_category(out: Outputs, catalog: Catalog, k: int) -> list[dict[str, list[int]]]:
    sims = out.query @ out.items.T
    result = []
    for row in sims:
        per_cat = {}
        for c, name in enumerate(CATEGORIES):
            idx = np.flatnonzero(catalog.categories == c)
            per_cat[name] = idx[np.argsort(-row[idx])[:k]].tolist()
        result.append(per_cat)
    return result


def judge() -> None:
    """Local LLM relevance judgments for pooled items. Resumable like profile."""
    make_deterministic(SEED, warn_only=True)
    cfg = ProfileConfig()
    texts = load_eval_texts()
    if not texts:
        raise SystemExit("no eval feelings; write eval_feelings.jsonl first")
    catalog = load_catalog()
    k = cfg.judge_pool_per_system
    log.info(
        f"judge: {num(len(texts))} eval feelings; pool = top {k} per category from the "
        "teacher, the student, and the untrained MiniLM"
    )
    systems = []
    for name, fn in progress(
        (
            ("teacher", teacher_outputs),
            ("student", lambda t, c: student_outputs(t, c)[0]),
            ("baseline", baseline_outputs),
        ),
        desc="judge pool",
        unit="system",
    ):
        log.debug(f"judge pool: ranking with the {name}")
        systems.append(top_by_category(fn(texts, catalog), catalog, k))
    shared_encoder.cache_clear()  # free the 8B teacher before the 9B labeler loads
    gc.collect()
    torch.cuda.empty_cache()
    keys = [
        SEP.join((text, cat, catalog.items[i]["id"]))
        for t, text in enumerate(texts)
        for cat in CATEGORIES
        for i in dict.fromkeys(i for s in systems for i in s[t][cat])
    ]
    schema = obj({"fits": array_of({"n": {"type": "integer"}, "fit": {"type": "boolean"}})})

    def describe(key: str) -> str:
        return catalog.texts[catalog.index[key.split(SEP)[2]]]

    def build(pending: list[str]) -> list[Unit]:
        groups: dict[str, list[str]] = defaultdict(list)
        for key in pending:
            groups[key.split(SEP)[0]].append(key)
        units = []
        for text, group in sorted(groups.items()):
            for chunk in chunks(group, 20):
                prompt = f"Feeling: {text}\n\nWorks:\n{numbered([describe(k) for k in chunk])}"
                units.append(Unit(chunk, Request(JUDGE_SYSTEM, prompt, schema, 20 * len(chunk))))
        return units

    def parse(unit_keys: list[str], data: dict[str, Any]) -> list[Record]:
        return parse_numbered(unit_keys, data["fits"], lambda row: {"fit": bool(row["fit"])})

    def fingerprint(key: str) -> str:
        return sha(key + describe(key))

    log.info(f"judge: {num(len(keys))} (feeling, item) pairs in the pool")
    Job("judge", JUDGMENTS, cfg).run(keys, build, parse, fingerprint, lazy_llm(cfg))
    fits = sum(1 for r in iter_jsonl(JUDGMENTS) if r["fit"])
    log.info(f"judge: {num(fits)} pairs judged a fit")


def judged_recall(out: Outputs, texts: list[str], catalog: Catalog) -> float:
    fits: dict[tuple[str, str], set[int]] = defaultdict(set)
    for r in iter_jsonl(JUDGMENTS):
        text, cat, item_id = r["key"].split(SEP)
        if r["fit"] and item_id in catalog.index:
            fits[(text, cat)].add(catalog.index[item_id])
    if not fits:
        return float("nan")
    tops = top_by_category(out, catalog, 10)
    scores = []
    for t, text in enumerate(texts):
        for cat in CATEGORIES:
            rel = fits.get((text, cat))
            if rel:
                scores.append(len(rel & set(tops[t][cat])) / min(len(rel), 10))
    return float(np.mean(scores)) if scores else float("nan")


def label_metrics(out: Outputs, rows: np.ndarray, qs: Any, offset: int) -> dict[str, float]:
    """Palette delta E (OKLab) and choice accuracy against the LLM labels."""
    m = qs.has_palette[rows]
    metrics = {}
    if m.any():
        pred = out.palette[offset : offset + len(rows)][m]
        metrics["palette_dE"] = float(np.linalg.norm(pred - qs.palette[rows][m], axis=-1).mean())
    for name, labels, pred in zip(
        ("light", "typeface", "scent"), (qs.light, qs.typeface, qs.scent), out.choices, strict=True
    ):
        y = labels[rows]
        has = y >= 0
        if has.any():
            p = pred[offset : offset + len(rows)].argmax(-1)
            metrics[f"{name}_acc"] = float((p[has] == y[has]).mean())
    return metrics


def run() -> dict[str, Any]:
    start = time.perf_counter()
    make_deterministic(SEED)
    vocab = load_vocab()
    catalog = load_catalog()
    qs = load_queries(catalog, vocab, TeacherConfig())
    eval_rows = qs.where("eval")
    held_rows = qs.where("heldout")
    texts = [qs.texts[i] for i in eval_rows] + [qs.texts[i] for i in held_rows]
    log.info(
        f"reading the teacher, the bundle, and {num(len(eval_rows))} eval feelings plus "
        f"{num(len(held_rows))} held-out item feelings"
    )
    if not len(eval_rows):
        log.warning("no eval feelings found; the report covers the held-out feelings only")
    if not JUDGMENTS.exists():
        log.info("no judgments yet; run `uv run mise-ml eval --judge` for the judged recall@10")

    teacher = teacher_outputs(texts, catalog)
    student, latency = student_outputs(texts, catalog)
    n_eval = len(eval_rows)
    eval_texts = texts[:n_eval]
    held_pos = qs.pos[held_rows]

    report: dict[str, Any] = {"counts": {"eval": n_eval, "heldout": len(held_rows)}}
    for name, out in (("teacher", teacher), ("student", student)):
        report[name] = {
            "recall@10_heldout": recall_at_k(
                out.query[n_eval:], held_pos, out.items, catalog.categories
            ),
            "recall@10_judged": judged_recall(
                Outputs(out.query[:n_eval], out.items, out.palette, out.choices),
                eval_texts,
                catalog,
            ),
            "eval": label_metrics(out, eval_rows, qs, 0),
            "heldout": label_metrics(out, held_rows, qs, n_eval),
        }
    # The floor: off-the-shelf MiniLM, what the stub bundle runs. It has no trained heads,
    # so only its retrieval numbers mean anything.
    baseline = baseline_outputs(texts, catalog)
    report["baseline"] = {
        "recall@10_heldout": recall_at_k(
            baseline.query[n_eval:], held_pos, baseline.items, catalog.categories
        ),
        "recall@10_judged": judged_recall(
            Outputs(baseline.query[:n_eval], baseline.items, baseline.palette, []),
            eval_texts,
            catalog,
        ),
        "eval": {},
        "heldout": {},
    }
    report["student"]["latency_ms"] = {
        "median": statistics.median(latency),
        "p95": float(np.percentile(latency, 95)),
        "threads": 1,
    }
    key = "recall@10_judged"
    if np.isnan(report["teacher"][key]) or np.isnan(report["student"][key]):
        key = "recall@10_heldout"
    gap = report["teacher"][key] - report["student"][key]
    report["ship"] = {"metric": key, "gap": gap, "ok": bool(gap <= SHIP_MARGIN)}
    write_json(EVAL_REPORT, report)

    def cell(value: float | None, width: int, digits: int = 3) -> str:
        if value is None or np.isnan(value):
            return f"{'—':>{width}}"
        return f"{value:>{width}.{digits}f}"

    log.info(
        f"{'':8} {'recall@10':>10} {'judged':>8} {'palette dE':>11} "
        f"{'light':>6} {'type':>6} {'scent':>6}"
    )
    for name in ("teacher", "student", "baseline"):
        r = report[name]
        e = r["eval"] or r["heldout"]
        log.info(
            f"{name:8} {cell(r['recall@10_heldout'], 10)} {cell(r['recall@10_judged'], 8)} "
            f"{cell(e.get('palette_dE'), 11, 4)} {cell(e.get('light_acc'), 6)} "
            f"{cell(e.get('typeface_acc'), 6)} {cell(e.get('scent_acc'), 6)}"
        )
    log.info("baseline = off-the-shelf MiniLM (the stub); it has no palette or choice heads")
    lat = report["student"]["latency_ms"]
    log.info(f"student latency: median {lat['median']:.1f} ms, p95 {lat['p95']:.1f} ms (1 thread)")
    verdict = "ship" if report["ship"]["ok"] else "do not ship"
    line = f"{verdict}: teacher - student on {key} = {gap * 100:.1f} points (limit 5)"
    (log.info if report["ship"]["ok"] else log.warning)(line)
    write_run_json()
    log.info(f"done in {elapsed(start)}: report -> {EVAL_REPORT.relative_to(ML_ROOT).as_posix()}")
    return report
