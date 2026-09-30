"""Compare trained models with the baseline and record the gate for bundle installation."""

import gc
import json
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
    CATALOG,
    CATEGORIES,
    EVAL_REPORT,
    JUDGMENTS,
    ML_ROOT,
    SEED,
    ProfileConfig,
    StudentConfig,
    TeacherConfig,
)
from mise_ml.data import (
    Catalog,
    load_catalog,
    load_eval_sets,
    load_eval_texts,
    load_queries,
    recall_at_k,
    recall_scores,
)
from mise_ml.export import onnx_run, onnx_session
from mise_ml.features import shared_encoder
from mise_ml.llm import Job, JobSpec, Record, Request, Unit, sha
from mise_ml.log import elapsed, get, num, progress
from mise_ml.profile import (
    chunks,
    item_prompt,
    lazy_llm,
    numbered,
    numbered_array,
    obj,
    parse_numbered,
)
from mise_ml.student import Student, encode_texts, load_student
from mise_ml.teacher import item_store, load_teacher, query_store
from mise_ml.util import iter_jsonl, make_deterministic, write_json
from mise_ml.vocab import load_vocab

log = get(__name__)
SHIP_MARGIN = 0.05
SEP = "\x1f"
JUDGED_K = 10

JUDGE_SYSTEM = """You judge picks for mise, a mood app. A user typed a feeling, and the app \
shows works that should feel like it.

Read the feeling as a person would. It can be short, casual, misspelled, figurative, \
sarcastic, or mixed. Interpret idioms and slang by their meaning. For sarcasm, judge the \
underlying feeling, not the literal claim.

For each numbered work, default to fit = false. Answer true only when source evidence \
supports the same dominant emotion as the user's feeling. Use the source facts and tags to \
check the generated profile. The profile is an interpretation, not independent evidence. Mark a \
pick false when it matches only a surface word, object, setting, or title but not the \
feeling. Require the work's central feeling to match the user's specific emotional \
experience; a broad shared mood such as sadness or unease is not enough. When source \
facts and the profile disagree, trust the source facts. Do not invent events or feelings \
to justify a match. Answer true only with direct evidence that the work shares the \
dominant emotion. Answer false when the source describes a different or opposing \
emotion, or when the match needs an unsupported interpretation. For example, 'over the \
moon' means delight; a lonely lunar adventure is not a fit just because it involves the \
moon. In 'beautiful morning but i cannot face getting up', exhaustion dominates; a \
cheerful morning scene is not a fit. Judge the emotional core and tone of the work, \
not the literal topic."""


def judge_schema(count: int) -> dict[str, Any]:
    return obj({"fits": numbered_array({"fit": {"type": "boolean"}}, count)})


@dataclass
class Outputs:
    query: np.ndarray
    items: np.ndarray
    palette: np.ndarray
    choices: list[np.ndarray]


def teacher_outputs(texts: list[str], catalog: Catalog) -> Outputs:
    model, ckpt = load_teacher()
    if ckpt["item_ids"] != [it["id"] for it in catalog.items]:
        raise SystemExit("catalog changed since train-teacher; rerun uv run train")
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
    """Read the public int8 ONNX graph and public catalog vectors."""
    model_path = BUNDLE / "model" / "model.onnx"
    if not model_path.exists():
        raise SystemExit(f"no bundle at {BUNDLE}; run uv run train first")
    _, tokenizer, meta = load_student()
    items = (
        np.fromfile(BUNDLE / "vectors.bin", dtype="<f2")
        .astype(np.float32)
        .reshape(len(catalog.items), -1)
    )
    session = onnx_session(model_path, threads=1)
    max_tokens = meta["cfg"]["max_length"]
    rows, latency = [], []
    onnx_run(session, tokenizer, [texts[0]], max_tokens)
    for text in progress(texts, desc="student (1 thread)", unit="text"):
        start = time.perf_counter()
        out = onnx_run(session, tokenizer, [text], max_tokens)
        latency.append((time.perf_counter() - start) * 1000)
        rows.append(out)
    heads = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))["heads"]
    choices = []
    for name in ("light", "typeface", "scent"):
        logits = np.concatenate([r[name] for r in rows])
        correction = heads.get("corrections", {}).get(name)
        if correction:
            logits = logits - correction["tau"] * np.log(np.asarray(correction["prior"]))
        choices.append(logits)
    return (
        Outputs(
            query=np.concatenate([r["embedding"] for r in rows]),
            items=items,
            palette=np.concatenate([r["palette"] for r in rows]),
            choices=choices,
        ),
        latency,
    )


def baseline_outputs(texts: list[str], catalog: Catalog) -> Outputs:
    """Untrained student backbone for the retrieval floor and judging pool."""
    from mise_ml.student import STUDENT_DIR

    saved = json.loads((STUDENT_DIR / "meta.json").read_text(encoding="utf-8"))["cfg"]
    cfg = StudentConfig(**saved)
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, revision=cfg.revision)
    model = Student(
        AutoModel.from_pretrained(cfg.backbone, revision=cfg.revision, attn_implementation="sdpa"),
        (1, 1, 1),
        8,
    ).cuda()
    model.projection = torch.nn.Identity()
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
    from mise_ml.steps import JUDGE_POOL, judge_pool_state

    state = judge_pool_state()
    saved = json.loads(JUDGE_POOL.read_text(encoding="utf-8")) if JUDGE_POOL.is_file() else None
    if saved and saved["state"] == state:
        keys = saved["keys"]
    else:
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
        write_json(JUDGE_POOL, {"state": state, "keys": keys})
    judge_job(keys).run(lazy_llm(cfg))
    fits = sum(1 for r in iter_jsonl(JUDGMENTS) if r["fit"])
    log.info(f"judge: {num(fits)} pairs judged a fit")


def judge_job(keys: list[str]) -> JobSpec:
    cfg = ProfileConfig()
    catalog = load_catalog()
    sources = {row["id"]: row for row in iter_jsonl(CATALOG)}

    def describe(key: str) -> str:
        index = catalog.index[key.split(SEP)[2]]
        resolved = catalog.items[index]
        source = sources.get(resolved["id"], {})
        facts = {
            **source,
            **resolved,
            "signal": {**source.get("signal", {}), **resolved.get("signal", {})},
        }
        return f"Source facts:\n{item_prompt(facts)}\nGenerated profile: {catalog.texts[index]}"

    def prompt_for(text: str, group: list[str]) -> str:
        return (
            f"Feeling: {text}\n\nWorks:\n{numbered([describe(k) for k in group])}"
            f"\n\nUser's feeling: {text}\n"
            "For each work, compare its central emotion with this feeling. Default to false. "
            "Answer true only when source evidence supports the same dominant emotion."
        )

    def build(pending: list[str]) -> list[Unit]:
        groups: dict[str, list[str]] = defaultdict(list)
        for key in pending:
            groups[key.split(SEP)[0]].append(key)
        units = []
        for text, group in sorted(groups.items()):
            for chunk in chunks(group, 20):
                prompt = prompt_for(text, chunk)
                units.append(
                    Unit(
                        chunk,
                        Request(JUDGE_SYSTEM, prompt, judge_schema(len(chunk)), 20 * len(chunk)),
                    )
                )
        return units

    def parse(unit_keys: list[str], data: dict[str, Any]) -> list[Record]:
        return parse_numbered(unit_keys, data["fits"], lambda row: {"fit": bool(row["fit"])})

    def fingerprint(key: str) -> str:
        return sha(key + prompt_for(key.split(SEP)[0], [key]))

    return JobSpec(
        Job("judge", JUDGMENTS, cfg),
        keys,
        build,
        parse,
        fingerprint,
        lambda k: {
            "feeling": k.split(SEP)[0],
            "category": k.split(SEP)[1],
            "item_id": k.split(SEP)[2],
            "text": describe(k),
        },
    )


def judged_fits(catalog: Catalog) -> dict[tuple[str, str], set[int]]:
    fits: dict[tuple[str, str], set[int]] = defaultdict(set)
    for r in iter_jsonl(JUDGMENTS):
        text, cat, item_id = r["key"].split(SEP)
        if r["fit"] and item_id in catalog.index:
            fits[(text, cat)].add(catalog.index[item_id])
    return fits


def judged_scores(
    out: Outputs,
    texts: list[str],
    catalog: Catalog,
    fits: dict[tuple[str, str], set[int]],
) -> np.ndarray:
    tops = top_by_category(out, catalog, JUDGED_K)
    scores = []
    for t, text in enumerate(texts):
        per_feeling = []
        for cat in CATEGORIES:
            rel = fits.get((text, cat))
            if rel:
                per_feeling.append(len(rel & set(tops[t][cat])) / min(len(rel), 10))
        scores.append(float(np.mean(per_feeling)) if per_feeling else float("nan"))
    return np.asarray(scores)


def paired_gap(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Resample paired feelings with a fixed seed for a percentile interval."""
    if not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"gap": None, "ci95": None, "feelings": len(a)}
    differences = a - b
    rng = np.random.default_rng(SEED)
    means = np.empty(10000)
    for i in range(len(means)):
        means[i] = differences[rng.integers(0, len(a), len(a))].mean()
    return {
        "gap": float(differences.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "feelings": len(a),
    }


def ship_gate(
    held: dict[str, np.ndarray],
    judged: dict[str, np.ndarray],
    n_judged: int,
    judgments_complete: bool,
) -> dict[str, Any]:
    gaps = {
        "teacher_student_judged": paired_gap(judged["teacher"], judged["student"]),
        "teacher_student_heldout": paired_gap(held["teacher"], held["student"]),
        "student_baseline_heldout": paired_gap(held["student"], held["baseline"]),
    }
    reasons = []
    if n_judged < 100:
        reasons.append(
            f"judged set is too small: {n_judged} feelings with fits; at least 100 required"
        )
    if not judgments_complete:
        reasons.append("judgments are missing for the current retrieval pool; run uv run train")
    for name, result in gaps.items():
        gap, interval = result["gap"], result["ci95"]
        if gap is None:
            reasons.append(f"{name}: no complete paired recall scores")
            log.warning("%s: confidence interval unavailable", name)
            continue
        log.info(
            "%s: %.2f points, paired bootstrap 95%% CI [%.2f, %.2f]",
            name,
            gap * 100,
            interval[0] * 100,
            interval[1] * 100,
        )
        if name == "student_baseline_heldout":
            if gap < 0.10:
                reasons.append(f"student is only {gap * 100:.2f} points above baseline; need 10")
        elif gap > SHIP_MARGIN:
            reasons.append(f"{name}: student is {gap * 100:.2f} points behind; limit 5")
    for reason in reasons:
        log.warning(reason)
    return {
        "ok": not reasons,
        "reasons": reasons,
        "gaps": gaps,
        "bootstrap": {"seed": SEED, "resamples": 10000, "unit": "feeling"},
    }


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
        log.info("no judgments yet; run `uv run train` for the judged recall@10")

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
            "eval": label_metrics(out, eval_rows, qs, 0),
            "heldout": label_metrics(out, held_rows, qs, n_eval),
        }
    # Compare retrieval with the same backbone before fine-tuning.
    shared_encoder.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = baseline_outputs(texts, catalog)
    report["baseline"] = {
        "recall@10_heldout": recall_at_k(
            baseline.query[n_eval:], held_pos, baseline.items, catalog.categories
        ),
        "eval": {},
        "heldout": {},
    }
    report["student"]["latency_ms"] = {
        "median": statistics.median(latency),
        "p95": float(np.percentile(latency, 95)),
        "threads": 1,
    }
    report["student"]["room_usage"] = {}
    for name, logits in zip(("light", "typeface", "scent"), student.choices, strict=True):
        counts = np.bincount(logits[:n_eval].argmax(-1), minlength=logits.shape[1])
        report["student"]["room_usage"][name] = {
            "distinct": int(np.count_nonzero(counts)),
            "top_share": float(counts.max() / n_eval) if n_eval else None,
        }
    systems = {"teacher": teacher, "student": student, "baseline": baseline}
    held_scores = {
        name: recall_scores(out.query[n_eval:], held_pos, out.items, catalog.categories)
        for name, out in systems.items()
    }
    fits = judged_fits(catalog)
    judged_mask = np.asarray(
        [any(fits.get((text, cat)) for cat in CATEGORIES) for text in eval_texts], dtype=bool
    )
    n_judged = int(judged_mask.sum())
    dropped = n_eval - n_judged
    log.info("judged recall: dropped %d feelings with no fits; %d remain", dropped, n_judged)
    report["counts"].update(judged=n_judged, judged_dropped_no_fits=dropped)
    judged = {
        name: judged_scores(
            Outputs(out.query[:n_eval], out.items, out.palette, out.choices),
            eval_texts,
            catalog,
            fits,
        )[judged_mask]
        for name, out in systems.items()
    }
    for name, scores in judged.items():
        report[name]["recall@10_judged"] = float(scores.mean()) if len(scores) else float("nan")
    eval_sets = load_eval_sets()
    set_names = np.asarray([eval_sets[text] for text in eval_texts])
    for name, scores in judged.items():
        report[name]["judged_by_set"] = {}
        for set_name in sorted(set(set_names)):
            mask = set_names[judged_mask] == set_name
            selected = scores[mask]
            report[name]["judged_by_set"][str(set_name)] = {
                "feelings": int((set_names == set_name).sum()),
                "judged": int(mask.sum()),
                "recall@10": float(selected.mean()) if len(selected) else None,
            }
    known = {r["key"] for r in iter_jsonl(JUDGMENTS)}
    # Check only the depth used by the metric.
    # The deeper judge pool covers near-tied items that floating-point noise can reorder.
    complete = all(
        SEP.join((text, cat, catalog.items[i]["id"])) in known
        for out in systems.values()
        for text, tops in zip(
            eval_texts,
            top_by_category(
                Outputs(out.query[:n_eval], out.items, out.palette, out.choices),
                catalog,
                JUDGED_K,
            ),
            strict=True,
        )
        for cat, indices in tops.items()
        for i in indices
    )
    report["ship"] = ship_gate(held_scores, judged, n_judged, complete)
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
    log.info("baseline = untrained student backbone; it has no palette or choice heads")
    for set_name in sorted(set(set_names)):
        results = [report[name]["judged_by_set"][set_name] for name in systems]
        log.info(
            "judged set %s (%d/%d): teacher %s, student %s, baseline %s",
            set_name,
            results[0]["judged"],
            results[0]["feelings"],
            *(cell(result["recall@10"], 5) for result in results),
        )
    for name, usage in report["student"]["room_usage"].items():
        log.info(
            "student %s: %d distinct; top share %s",
            name,
            usage["distinct"],
            cell(usage["top_share"], 5),
        )
    lat = report["student"]["latency_ms"]
    log.info(f"student latency: median {lat['median']:.1f} ms, p95 {lat['p95']:.1f} ms (1 thread)")
    verdict = "ship" if report["ship"]["ok"] else "do not ship"
    (log.info if report["ship"]["ok"] else log.warning)(verdict)
    log.info(f"done in {elapsed(start)}: report -> {EVAL_REPORT.relative_to(ML_ROOT).as_posix()}")
    return report
