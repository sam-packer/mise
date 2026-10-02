"""Compare trained models with the baseline and record the gate for bundle installation."""

import gc
import json
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from mise_ml.config import (
    BUNDLE,
    CATALOG,
    EVAL_FAILURES,
    EVAL_REPORT,
    GOLD_SET,
    ML_ROOT,
    SEED,
    TUNING_SET,
    StudentConfig,
    TeacherConfig,
)
from mise_ml.data import (
    Catalog,
    fidelity_at_k,
    load_catalog,
    load_eval_sets,
    load_queries,
    recall_at_k,
)
from mise_ml.export import onnx_run, onnx_session
from mise_ml.features import shared_encoder
from mise_ml.install import TARGET
from mise_ml.log import elapsed, get, num, progress
from mise_ml.profile import item_prompt
from mise_ml.rated import FIT, RatedSet, brief, load_rated, rated_scores
from mise_ml.student import Student, encode_texts
from mise_ml.teacher import item_store, load_teacher, query_store
from mise_ml.util import atomic_write, iter_jsonl, make_deterministic, write_json
from mise_ml.vocab import load_vocab

log = get(__name__)
# The student's tuning objective must beat the untrained backbone by this much.
SHIP_MARGIN = 0.10
# The student may fall at most this far below the installed bundle on the shared tuning pairs.
INSTALLED_MARGIN = 0.01
FACTS_CHARS = 700
# The failures report lists this many pairs of each kind. A non-fit has a rating at most NON_FIT.
WORST = 25
NON_FIT = 0.5
# World fidelity samples this many works. The app shows WORLD_SAME neighbors in a work's category
# and WORLD_OTHER in each other category (src/lib/mood/search.ts).
WORLD_SAMPLE = 1000
WORLD_SAME = 2
WORLD_OTHER = 2


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


def bundle_outputs(root: Path, texts: list[str], desc: str) -> tuple[Outputs, list[float]]:
    """Read a public bundle: its int8 ONNX graph, tokenizer, and catalog vectors."""
    model_path = root / "model" / "model.onnx"
    if not model_path.exists():
        raise SystemExit(f"no bundle at {root}; run uv run train first")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(root / "model")
    items = (
        np.fromfile(root / "vectors.bin", dtype="<f2")
        .astype(np.float32)
        .reshape(-1, manifest["encoder"]["dims"])
    )
    session = onnx_session(model_path, threads=1)
    max_tokens = manifest["encoder"]["maxTokens"]
    rows, latency = [], []
    onnx_run(session, tokenizer, [texts[0]], max_tokens)
    for text in progress(texts, desc=f"{desc} (1 thread)", unit="text"):
        start = time.perf_counter()
        out = onnx_run(session, tokenizer, [text], max_tokens)
        latency.append((time.perf_counter() - start) * 1000)
        rows.append(out)
    heads = manifest["heads"]
    choices = []
    for name in ("light", "typeface"):
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


def student_outputs(texts: list[str], catalog: Catalog) -> tuple[Outputs, list[float]]:
    """Read the new bundle; its vectors follow the current catalog order."""
    out, latency = bundle_outputs(BUNDLE, texts, "student")
    if len(out.items) != len(catalog.items):
        raise SystemExit("catalog changed since export; rerun uv run train")
    return out, latency


def baseline_outputs(texts: list[str], catalog: Catalog) -> Outputs:
    """Untrained student backbone for the retrieval floor and the ship gate."""
    from mise_ml.student import STUDENT_DIR

    saved = json.loads((STUDENT_DIR / "meta.json").read_text(encoding="utf-8"))["cfg"]
    cfg = StudentConfig(**saved)
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, revision=cfg.revision)
    model = Student(
        AutoModel.from_pretrained(cfg.backbone, revision=cfg.revision, attn_implementation="sdpa"),
        (1, 1),
        8,
    ).cuda()
    model.projection = torch.nn.Identity()
    return Outputs(
        query=encode_texts(model, tokenizer, texts, cfg.max_length),
        items=encode_texts(model, tokenizer, catalog.texts, cfg.item_max_length),
        palette=np.zeros((len(texts), 5, 3)),
        choices=[],
    )


def work_facts(item_ids: list[str]) -> dict[str, str]:
    """Source facts and the generated profile of each work, as raters read them."""
    catalog = load_catalog()
    sources = {row["id"]: row for row in iter_jsonl(CATALOG)}
    result = {}
    for item_id in item_ids:
        index = catalog.index[item_id]
        resolved = catalog.items[index]
        source = sources.get(item_id, {})
        facts = {
            **source,
            **resolved,
            "signal": {**source.get("signal", {}), **resolved.get("signal", {})},
        }
        # Cap the facts: full overviews and poems make a batch of works too long to rate.
        text = item_prompt(facts)
        if len(text) > FACTS_CHARS:
            text = text[:FACTS_CHARS].rsplit(" ", 1)[0] + " ..."
        result[item_id] = f"Source facts:\n{text}\nGenerated profile: {catalog.texts[index]}"
    return result


def installed_scores(
    tuning: RatedSet, student: Outputs, rows: list[int], catalog: Catalog
) -> dict[str, Any] | None:
    """Score the installed bundle and the new student on the tuning pairs whose works are in
    both catalogs. rows holds the student's query row for each tuning feeling."""
    if not (TARGET / "manifest.json").is_file():
        return None
    ids = [it["id"] for it in json.loads((TARGET / "items.json").read_text(encoding="utf-8"))]
    index = {item_id: i for i, item_id in enumerate(ids)}
    keep = np.asarray([catalog.items[i]["id"] in index for i in tuning.item], dtype=bool)
    shared = replace(
        tuning, feeling=tuning.feeling[keep], item=tuning.item[keep], rating=tuning.rating[keep]
    )
    installed, _ = bundle_outputs(TARGET, tuning.feelings, "installed")
    return {
        "pairs": int(keep.sum()),
        "student": rated_scores(shared, student.query[rows], student.items),
        "installed": rated_scores(
            replace(
                shared,
                item=np.asarray([index[catalog.items[i]["id"]] for i in shared.item], dtype=int),
            ),
            installed.query,
            installed.items,
        ),
    }


def ship_gate(
    tuning: dict[str, dict[str, float]], installed: dict[str, Any] | None
) -> dict[str, Any]:
    """Gate the bundle on the tuning set. The gold set is never part of the gate."""
    student, baseline = tuning["student"]["objective"], tuning["baseline"]["objective"]
    lead = student - baseline
    checks: dict[str, Any] = {
        "student_tuning": student,
        "baseline_tuning": baseline,
        "student_over_baseline": lead,
    }
    reasons = []
    log.info("student over baseline on the tuning set: %.4f; need %.2f", lead, SHIP_MARGIN)
    # A NaN fails every check.
    if not lead >= SHIP_MARGIN:
        reasons.append(
            f"student tuning objective is only {lead:.4f} above the baseline; need {SHIP_MARGIN}"
        )
    if installed is None:
        checks["installed"] = f"skipped: no installed bundle at {TARGET}"
        log.info("no installed bundle at %s; skip the installed comparison", TARGET)
    else:
        new, old = installed["student"]["objective"], installed["installed"]["objective"]
        checks["installed"] = {
            "pairs": installed["pairs"],
            "student": new,
            "installed": old,
            "student_over_installed": new - old,
        }
        log.info(
            "installed bundle on %d shared tuning pairs: student %.4f, installed %.4f",
            installed["pairs"],
            new,
            old,
        )
        if not new - old >= -INSTALLED_MARGIN:
            reasons.append(
                f"student is {old - new:.4f} below the installed bundle on "
                f"{installed['pairs']} shared tuning pairs; limit {INSTALLED_MARGIN}"
            )
    for reason in reasons:
        log.warning(reason)
    return {"ok": not reasons, "reasons": reasons, "checks": checks}


def label_metrics(out: Outputs, rows: np.ndarray, qs: Any, offset: int) -> dict[str, float]:
    """Palette delta E (OKLab) and choice accuracy against the LLM labels."""
    m = qs.has_palette[rows]
    metrics = {}
    if m.any():
        pred = out.palette[offset : offset + len(rows)][m]
        metrics["palette_dE"] = float(np.linalg.norm(pred - qs.palette[rows][m], axis=-1).mean())
    for name, labels, pred in zip(
        ("light", "typeface"), (qs.light, qs.typeface), out.choices, strict=True
    ):
        y = labels[rows]
        has = y >= 0
        if has.any():
            p = pred[offset : offset + len(rows)].argmax(-1)
            metrics[f"{name}_acc"] = float((p[has] == y[has]).mean())
    return metrics


def failures(rs: RatedSet, sims: np.ndarray, catalog: Catalog) -> str:
    """The student's worst tuning pairs as Markdown: fits that it ranks low in their feeling and
    non-fits that it ranks high. sims holds the student's similarity of each pair."""
    rank = np.empty(len(sims), dtype=int)
    size = np.empty(len(sims), dtype=int)
    for f in np.unique(rs.feeling):
        m = np.flatnonzero(rs.feeling == f)
        rank[m[np.argsort(-sims[m], kind="stable")]] = np.arange(1, len(m) + 1)
        size[m] = len(m)
    # The place of a pair among its feeling's pairs: 0 at the top, 1 at the bottom.
    place = (rank - 1) / np.maximum(size - 1, 1)
    fits = np.flatnonzero(rs.rating >= FIT)
    non_fits = np.flatnonzero(rs.rating <= NON_FIT)
    sections = (
        (
            "Missed fits",
            f"rating {FIT:g} or more, lowest in their feeling",
            fits[np.lexsort((sims[fits], -place[fits]))],
        ),
        (
            "False fits",
            f"rating {NON_FIT:g} or less, highest in their feeling",
            non_fits[np.lexsort((-sims[non_fits], place[non_fits]))],
        ),
    )
    lines = ["# The student's worst tuning pairs", ""]
    for title, rule, pairs in sections:
        lines += [f"## {title} ({rule})", ""]
        for n, i in enumerate(pairs[:WORST], 1):
            it = catalog.items[rs.item[i]]
            lines += [
                f"{n}. **{rs.feelings[rs.feeling[i]]}**: {it['category']}, "
                f"{it['title']} by {it['creator']}",
                f"   rating {rs.rating[i]:.2f}; similarity {sims[i]:.3f}; "
                f"rank {rank[i]} of {size[i]}",
                f"   - vibe: {it['vibe']}",
                f"   - description: {it['description']}",
                "",
            ]
    return "\n".join(lines)


def world_neighbors(rows: np.ndarray, items: np.ndarray, catalog: Catalog) -> list[set[int]]:
    """The app's world rule (world() in src/lib/mood/search.ts): the nearest works by item-to-item
    similarity, WORLD_SAME in the work's category and WORLD_OTHER in each other category. Skip the
    work, its creator, and its title, and take one work per creator in a category."""
    creators = np.asarray([str(it["creator"]) for it in catalog.items])
    titles = np.asarray([str(it["title"]) for it in catalog.items])
    # The app sums in double precision, and a stable sort keeps catalog order for equal scores.
    items = items.astype(np.float64)
    orders = np.argsort(-(items[rows] @ items.T), axis=1, kind="stable")
    result = []
    for row, order in zip(rows, orders, strict=True):
        skip = (creators == creators[row]) | (titles == titles[row])
        need = {
            int(cat): WORLD_SAME if cat == catalog.categories[row] else WORLD_OTHER
            for cat in np.unique(catalog.categories)
        }
        seen: dict[int, set[str]] = {cat: set() for cat in need}
        found: set[int] = set()
        for i in order:
            cat = int(catalog.categories[i])
            if skip[i] or len(seen[cat]) >= need[cat] or creators[i] in seen[cat]:
                continue
            seen[cat].add(creators[i])
            found.add(int(i))
            if len(found) == sum(need.values()):
                break
        result.append(found)
    return result


def world_fidelity(
    rows: np.ndarray, student_items: np.ndarray, teacher_items: np.ndarray, catalog: Catalog
) -> float:
    """The share of the teacher's world neighbors of each sampled work that the student's world
    neighbors also contain."""
    teacher = world_neighbors(rows, teacher_items, catalog)
    student = world_neighbors(rows, student_items, catalog)
    shared = sum(len(t & s) for t, s in zip(teacher, student, strict=True))
    return shared / sum(len(t) for t in teacher)


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
    for name, logits in zip(("light", "typeface"), student.choices, strict=True):
        counts = np.bincount(logits[:n_eval].argmax(-1), minlength=logits.shape[1])
        report["student"]["room_usage"][name] = {
            "distinct": int(np.count_nonzero(counts)),
            "top_share": float(counts.max() / n_eval) if n_eval else None,
        }
    systems = {"teacher": teacher, "student": student, "baseline": baseline}
    # Rated pairs: the tuning set picked the settings and gates the bundle; the gold set is the
    # independent check.
    row = {text: i for i, text in enumerate(eval_texts)}
    rated: dict[str, tuple[RatedSet, list[int]]] = {}
    for set_name, path in (("tuning", TUNING_SET), ("gold", GOLD_SET)):
        rs = load_rated(path, catalog)
        missing = [f for f in rs.feelings if f not in row]
        if missing:
            raise SystemExit(f"{path.name}: {len(missing)} feelings are not eval feelings")
        idx = [row[f] for f in rs.feelings]
        rated[set_name] = rs, idx
        report["counts"][set_name] = len(rs.rating)
        for name, out in systems.items():
            report[name][set_name] = rated_scores(rs, out.query[idx], out.items)
    # The share of the teacher's top 10 per category that the student also finds.
    fidelity = fidelity_at_k(
        student.query[:n_eval],
        student.items,
        teacher.query[:n_eval],
        teacher.items,
        catalog.categories,
    )
    eval_sets = load_eval_sets()
    set_names = np.asarray([eval_sets[text] for text in eval_texts])
    report["student"]["fidelity@10"] = float(fidelity.mean()) if n_eval else None
    report["student"]["fidelity_by_set"] = {
        str(set_name): float(fidelity[set_names == set_name].mean())
        for set_name in sorted(set(set_names))
    }
    world = np.random.default_rng(SEED).choice(
        len(catalog.items), min(WORLD_SAMPLE, len(catalog.items)), replace=False
    )
    report["student"]["world_fidelity"] = world_fidelity(
        np.sort(world), student.items, teacher.items, catalog
    )
    tuning, tuning_rows = rated["tuning"]
    atomic_write(
        EVAL_FAILURES,
        failures(
            tuning,
            (student.query[tuning_rows][tuning.feeling] * student.items[tuning.item]).sum(1),
            catalog,
        ).encode("utf-8"),
    )
    installed = installed_scores(tuning, student, tuning_rows, catalog)
    report["ship"] = ship_gate(
        {name: report[name]["tuning"] for name in ("student", "baseline")}, installed
    )
    write_json(EVAL_REPORT, report)

    def cell(value: float | None, width: int, digits: int = 3) -> str:
        if value is None or np.isnan(value):
            return f"{'—':>{width}}"
        return f"{value:>{width}.{digits}f}"

    log.info(f"{'':8} {'recall@10':>10} {'palette dE':>11} {'light':>6} {'type':>6}")
    for name in ("teacher", "student", "baseline"):
        r = report[name]
        e = r["eval"] or r["heldout"]
        log.info(
            f"{name:8} {cell(r['recall@10_heldout'], 10)} "
            f"{cell(e.get('palette_dE'), 11, 4)} {cell(e.get('light_acc'), 6)} "
            f"{cell(e.get('typeface_acc'), 6)}"
        )
    log.info("baseline = untrained student backbone; it has no palette or choice heads")
    for set_name in ("tuning", "gold"):
        log.info(
            "%s set (%d pairs): %s",
            set_name,
            report["counts"][set_name],
            "; ".join(f"{name} {brief(report[name][set_name])}" for name in systems),
        )
    for set_name, value in report["student"]["fidelity_by_set"].items():
        log.info("eval set %s: student fidelity@10 %s", set_name, cell(value, 5))
    log.info(
        "student fidelity@10 (teacher's top 10 per category): %s",
        cell(report["student"]["fidelity@10"], 5),
    )
    log.info(
        "student world fidelity (teacher's world neighbors of %d works): %s",
        len(world),
        cell(report["student"]["world_fidelity"], 5),
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
