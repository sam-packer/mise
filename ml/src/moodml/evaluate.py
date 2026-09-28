import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from moodml.config import (
    BUNDLE,
    CATEGORIES,
    EVAL_REPORT,
    JUDGMENTS,
    SEED,
    ProfileConfig,
    StudentConfig,
    TeacherConfig,
)
from moodml.data import Catalog, load_catalog, load_eval_texts, load_queries, recall_at_k
from moodml.export import onnx_run, onnx_session
from moodml.features import shared_encoder
from moodml.llm import Job, Record, Request, Unit, sha
from moodml.profile import (
    FEELING_RULES,
    array_of,
    chunks,
    lazy_llm,
    numbered,
    obj,
    parse_numbered,
)
from moodml.provenance import write_run_json
from moodml.student import Student, encode_texts, load_student
from moodml.teacher import item_store, load_teacher, query_store
from moodml.util import iter_jsonl, make_deterministic, write_json
from moodml.vocab import load_vocab

SHIP_MARGIN = 0.05
SEP = "\x1f"

JUDGE_SYSTEM = f"""You judge picks for a moodboard app. A user typed a feeling, and the app \
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
    for text in texts:
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
    systems = [
        top_by_category(teacher_outputs(texts, catalog), catalog, k),
        top_by_category(student_outputs(texts, catalog)[0], catalog, k),
        top_by_category(baseline_outputs(texts, catalog), catalog, k),
    ]
    shared_encoder.cache_clear()
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

    Job("judge", JUDGMENTS, cfg).run(keys, build, parse, fingerprint, lazy_llm(cfg))


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


def run() -> None:
    make_deterministic(SEED)
    vocab = load_vocab()
    catalog = load_catalog()
    qs = load_queries(catalog, vocab, TeacherConfig())
    eval_rows = qs.where("eval")
    held_rows = qs.where("heldout")
    texts = [qs.texts[i] for i in eval_rows] + [qs.texts[i] for i in held_rows]
    if not len(eval_rows):
        print("no eval feelings found; report covers the held-out paraphrases only")

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
    for name in ("teacher", "student"):
        r = report[name]
        print(
            f"{name}: recall@10 heldout {r['recall@10_heldout']:.3f}, "
            f"judged {r['recall@10_judged']:.3f}, eval {r['eval']}, heldout {r['heldout']}"
        )
    lat = report["student"]["latency_ms"]
    print(f"student latency: median {lat['median']:.1f} ms, p95 {lat['p95']:.1f} ms (1 thread)")
    verdict = "SHIP" if report["ship"]["ok"] else "DO NOT SHIP"
    print(f"{verdict}: teacher - student {key} = {gap * 100:.1f} points (limit 5) -> {EVAL_REPORT}")
    write_run_json()
