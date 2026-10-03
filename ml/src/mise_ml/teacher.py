"""Train heads over frozen Qwen features and write targets for student distillation."""

import dataclasses
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mise_ml.config import (
    CATALOG,
    FIT_LABELS,
    ML_ROOT,
    MODELS,
    SEED,
    TEACHER_PARAMS,
    TUNING_SET,
    TeacherConfig,
)
from mise_ml.data import Catalog, QuerySet, load_catalog, load_queries, recall_at_k
from mise_ml.features import ITEM_TEMPLATE, QUERY_TEMPLATE, FeatureStore, shared_encoder
from mise_ml.heads import ChoiceHeads, Mlp, palette_loss
from mise_ml.log import elapsed, get, num, progress
from mise_ml.profile import item_prompt
from mise_ml.rated import brief as rated_brief
from mise_ml.rated import load_rated, rated_scores
from mise_ml.training import cosine_schedule
from mise_ml.util import iter_jsonl, make_deterministic, write_json
from mise_ml.vocab import load_vocab

log = get(__name__)
CHECKPOINT = MODELS / "teacher.pt"
OUTPUTS = MODELS / "teacher_outputs.pt"
# The teacher embeds each work's source facts before its profile; facts and profile together
# score higher on the tuning set than either alone. The cap keeps a long overview or poem from
# pushing the profile past max_length.
FACTS_CHARS = 700
# On an RTX 5090 with the fit loss, an epoch takes about 1.6 s at batch 512. The last search of
# 50 trials took 24 min, about 28 s a trial.
TRIAL_SECONDS = 28
TRIALS = 50


class Teacher(nn.Module):
    def __init__(self, d_in: int, cfg: TeacherConfig, sizes: tuple[int, int]) -> None:
        super().__init__()
        self.query = Mlp(d_in, cfg.hidden, cfg.dims, cfg.dropout)
        self.item = Mlp(d_in, cfg.hidden, cfg.dims, cfg.dropout)
        self.heads = ChoiceHeads(d_in, 512, sizes, cfg.dropout)
        self.raw_weight = cfg.raw_weight

    def embed(self, head: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Join the head and raw vectors, so a dot product mixes the two cosines by raw_weight."""
        e = F.normalize(head(x), dim=-1)
        if not self.raw_weight:
            return e
        r = self.raw_weight
        return torch.cat([(1 - r) ** 0.5 * e, r**0.5 * F.normalize(x, dim=-1)], -1)

    def embed_queries(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.query, x)

    def embed_items(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.item, x)


def source_facts(catalog: Catalog, item_ids: list[str], chars: int) -> dict[str, str]:
    """The source facts of each work as the raters read them, cut to about `chars` characters."""
    sources = {row["id"]: row for row in iter_jsonl(CATALOG)}
    result = {}
    for item_id in item_ids:
        resolved = catalog.items[catalog.index[item_id]]
        source = sources.get(item_id, {})
        facts = {
            **source,
            **resolved,
            "signal": {**source.get("signal", {}), **resolved.get("signal", {})},
        }
        text = item_prompt(facts)
        if len(text) > chars:
            text = text[:chars].rsplit(" ", 1)[0] + " ..."
        result[item_id] = text
    return result


def item_texts(catalog: Catalog) -> list[str]:
    """What the teacher embeds for each work: its source facts, then its profile."""
    facts = source_facts(catalog, [it["id"] for it in catalog.items], FACTS_CHARS)
    return [
        f"{facts[it['id']]}\n{text}" for it, text in zip(catalog.items, catalog.texts, strict=True)
    ]


def query_store(cfg: TeacherConfig) -> FeatureStore:
    return FeatureStore("queries", QUERY_TEMPLATE, cfg)


def item_store(cfg: TeacherConfig) -> FeatureStore:
    return FeatureStore("items", ITEM_TEMPLATE, cfg)


def load_teacher() -> tuple[Teacher, dict]:
    ckpt = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
    cfg = TeacherConfig(**ckpt["cfg"])
    model = Teacher(ckpt["d_in"], cfg, tuple(ckpt["sizes"])).cuda().eval()
    model.load_state_dict(ckpt["state"])
    return model, ckpt


@torch.no_grad()
def in_chunks(fn, x: torch.Tensor, size: int = 8192) -> torch.Tensor:
    return torch.cat([fn(x[i : i + size]) for i in range(0, len(x), size)])


def drop_near_eval(qs: QuerySet, fq: torch.Tensor, threshold: float) -> None:
    """Mark as "dropped" the train, val, and distill texts whose Qwen query features (`fq`, one
    row per qs text) have a cosine of at least `threshold` with any eval feeling. load_queries
    removes only exact copies, so rewordings such as "i am" for "i'm" otherwise train on the rated
    feelings."""
    evals = fq[torch.as_tensor(qs.where("eval"), device=fq.device)]
    rows = qs.where("train", "val", "distill")
    near = np.zeros(len(qs.texts), dtype=bool)
    if len(evals) and len(rows):
        # The cached features are L2-normalized, so a dot product is the cosine.
        best = in_chunks(
            lambda x: (x @ evals.T).max(1).values, fq[torch.as_tensor(rows, device=fq.device)]
        )
        near[rows] = (best >= threshold).cpu().numpy()
    counts = ", ".join(
        f"{s} {num(int((near & (qs.split == s)).sum()))}" for s in ("train", "val", "distill")
    )
    log.info(
        f"dropped {num(int(near.sum()))} training texts with a query cosine >= {threshold} to "
        f"one of {num(len(evals))} eval feelings ({counts})"
    )
    qs.split = np.where(near, "dropped", qs.split)


def load_fit_groups(
    path: Path, catalog: Catalog, qs: QuerySet
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fit ratings as groups of one feeling and the rated works of one category: the qs row
    of each group, its catalog rows (-1 pads a short group), and their ratings.

    Only feelings in the train split count, so a feeling that drop_near_eval dropped, or a val or
    eval feeling, never trains. A group whose works all share one rating has no order to learn.
    """
    row = {text: i for i, text in enumerate(qs.texts)}
    train = set(qs.where("train").tolist())
    groups: dict[tuple[int, int], list[tuple[int, float]]] = {}
    outside, missing = set(), 0
    for r in iter_jsonl(path):
        q = row.get(r["feeling"], -1)
        if q not in train:
            outside.add(r["feeling"])
            continue
        if r["item_id"] not in catalog.index:
            missing += 1
            continue
        item = catalog.index[r["item_id"]]
        groups.setdefault((q, int(catalog.categories[item])), []).append((item, r["rating"]))
    kept = [(k, g) for k, g in sorted(groups.items()) if len({rating for _, rating in g}) > 1]
    width = max((len(g) for _, g in kept), default=1)
    items = np.full((len(kept), width), -1, dtype=np.int64)
    ratings = np.zeros((len(kept), width), dtype=np.float32)
    for i, (_, g) in enumerate(kept):
        items[i, : len(g)] = [item for item, _ in g]
        ratings[i, : len(g)] = [rating for _, rating in g]
    log.info(
        f"fit ratings: {num(len(kept))} groups over {num(len({q for (q, _), _ in kept}))} train "
        f"feelings; skipped {num(len(groups) - len(kept))} groups with one rating, "
        f"{num(len(outside))} feelings outside the train split, and {num(missing)} pairs with a "
        "work not in the catalog"
    )
    return np.array([q for (q, _), _ in kept], dtype=np.int64), items, ratings


class Trainer:
    """Load the queries and cached features once, then train heads for one config or many."""

    def __init__(self, cfg: TeacherConfig) -> None:
        self.vocab = load_vocab()
        self.catalog = load_catalog()
        self.qs = load_queries(self.catalog, self.vocab, cfg, include_distill=True)
        splits = ", ".join(
            f"{s} {num(len(self.qs.where(s)))}"
            for s in ("train", "val", "heldout", "eval", "distill")
        )
        log.info(
            f"reading data/curated: {num(len(self.catalog.items))} items, "
            f"{num(len(self.qs.texts))} queries ({splits}); backbone {cfg.backbone}"
        )
        dev = "cuda"
        self.fi = torch.tensor(
            item_store(cfg).get(item_texts(self.catalog)), dtype=torch.float32, device=dev
        )
        self.fq = torch.tensor(query_store(cfg).get(self.qs.texts), dtype=torch.float32, device=dev)
        drop_near_eval(self.qs, self.fq, cfg.near_eval_cosine)
        self.tuning = load_rated(TUNING_SET, self.catalog)
        self.tuning_fq = torch.tensor(
            query_store(cfg).get(self.tuning.feelings), dtype=torch.float32, device=dev
        )
        shared_encoder.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()

        def t(a: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(a, device=dev)

        self.pos = t(self.qs.pos)
        self.palette = t(self.qs.palette)
        self.has_palette = t(self.qs.has_palette)
        self.choices = [t(self.qs.light), t(self.qs.typeface)]
        fit_query, fit_items, fit_rating = load_fit_groups(FIT_LABELS, self.catalog, self.qs)
        self.fit_query, self.fit_items, self.fit_rating, self.fit_pad = (
            t(fit_query),
            t(np.maximum(fit_items, 0)),
            t(fit_rating),
            t(fit_items < 0),
        )
        self.build(cfg)

    def build(self, cfg: TeacherConfig) -> None:
        """Start fresh heads for cfg. The fixed seed gives equal configs equal runs."""
        self.cfg = cfg
        torch.manual_seed(SEED)
        self.model = Teacher(self.fi.shape[1], cfg, self.vocab.sizes()).cuda()

    def losses(self, b: torch.Tensor, full_softmax: bool = False) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        x = self.fq[b]
        out: dict[str, torch.Tensor] = {}
        with_pos = b[self.pos[b] >= 0]
        if len(with_pos):
            q = self.model.embed_queries(self.fq[with_pos])
            p = self.pos[with_pos]
            cand = torch.arange(len(self.fi), device=b.device) if full_softmax else p.unique()
            logits = q @ self.model.embed_items(self.fi[cand]).T / cfg.temperature
            out["retrieval"] = F.cross_entropy(logits, torch.searchsorted(cand, p))
        palette, *logits_by_head = self.model.heads(x)
        m = self.has_palette[b]
        if m.any():
            out["palette"] = cfg.palette_weight * palette_loss(
                palette[m], self.palette[b][m], cfg.lightness_weight
            )
        for name, logits, labels in zip(
            ("light", "typeface"), logits_by_head, self.choices, strict=True
        ):
            y = labels[b]
            m = y >= 0
            if m.any():
                # Validation uses no smoothing, so val losses of two configs compare.
                smoothing = cfg.label_smoothing if self.model.training else 0.0
                out[name] = cfg.choice_weight * F.cross_entropy(
                    logits[m], y[m], label_smoothing=smoothing
                )
        return out

    def fit_loss(self, groups: torch.Tensor) -> torch.Tensor:
        """Teach the teacher to order each rated top 10 by the judge's fit ratings.

        ListNet: a softmax cross-entropy from the teacher's scores to targets proportional to
        exp(rating / fit_tau). One listwise term per group uses the whole graded order of the 10
        works, which is what picks the best 2 on a wall. The softmax ignores a shift of all
        ratings in a group, so a lenient judge does not matter. The scores keep the retrieval
        temperature.
        """
        cfg = self.cfg
        pad = self.fit_pad[groups]
        q = self.model.embed_queries(self.fq[self.fit_query[groups]])
        # Embed each rated work once; inverse maps the groups back to those rows.
        unique, inverse = self.fit_items[groups].unique(return_inverse=True)
        items = self.model.embed_items(self.fi[unique])[inverse]
        logits = (q[:, None] * items).sum(-1) / cfg.temperature
        target = (self.fit_rating[groups] / cfg.fit_tau).masked_fill(pad, -torch.inf).softmax(-1)
        log_p = logits.masked_fill(pad, -torch.inf).log_softmax(-1)
        return cfg.fit_weight * -(target * log_p.masked_fill(pad, 0)).sum(-1).mean()

    @torch.no_grad()
    def validate(self, rows: torch.Tensor) -> dict[str, float]:
        self.model.eval()
        parts: dict[str, list[float]] = {}
        for start in range(0, len(rows), self.cfg.batch_size):
            for k, v in self.losses(rows[start : start + self.cfg.batch_size], True).items():
                parts.setdefault(k, []).append(v.item())
        metrics = {k: float(np.mean(v)) for k, v in parts.items()}
        metrics["loss"] = sum(metrics.values())
        items = in_chunks(self.model.embed_items, self.fi).cpu().numpy()
        tuning_q = in_chunks(self.model.embed_queries, self.tuning_fq).cpu().numpy()
        metrics.update(
            {f"tuning_{k}": v for k, v in rated_scores(self.tuning, tuning_q, items).items()}
        )
        r = rows[self.pos[rows] >= 0]
        if len(r):
            q = in_chunks(self.model.embed_queries, self.fq[r]).cpu().numpy()
            metrics["recall@10"] = recall_at_k(
                q, self.pos[r].cpu().numpy(), items, self.catalog.categories
            )
        self.model.train()
        return metrics

    def train(self, save: bool = True) -> dict[str, float]:
        """Return the val metrics of the epoch with the best tuning-set objective.

        With save, write that epoch to the checkpoint. Without it (a search trial), log at DEBUG.
        """
        cfg = self.cfg
        level = logging.INFO if save else logging.DEBUG
        train = torch.as_tensor(self.qs.where("train"), device="cuda")
        val = torch.as_tensor(self.qs.where("val"), device="cuda")
        train_pos = train[self.pos[train] >= 0]
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        steps = cfg.epochs * -(-len(train) // cfg.batch_size)
        sched = cosine_schedule(opt, steps, cfg.warmup_ratio)
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        best, best_epoch, best_metrics = -1.0, -1, {}
        log.log(
            level,
            f"training heads: {cfg.epochs} epochs, batch {cfg.batch_size}, lr {cfg.lr:.3g}, "
            f"{num(len(train))} train rows ({num(len(train_pos))} with an item)",
        )
        start_all = time.perf_counter()
        for epoch in range(cfg.epochs):
            start = time.perf_counter()
            perm = train[torch.randperm(len(train), device="cuda", generator=gen)]
            bar = progress(
                range(0, len(perm), cfg.batch_size), desc=f"epoch {epoch + 1}/{cfg.epochs}"
            )
            total = 0.0
            for i, step in enumerate(bar, 1):
                loss = sum(self.losses(perm[step : step + cfg.batch_size]).values())
                if cfg.fit_weight and len(self.fit_query):
                    groups = torch.randperm(len(self.fit_query), device="cuda", generator=gen)
                    loss = loss + self.fit_loss(groups[: cfg.fit_groups])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item()
                if i % 10 == 0:
                    bar.set_postfix(
                        loss=f"{total / i:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}", refresh=False
                    )
            metrics = self.validate(val)
            improved = metrics["tuning_objective"] > best
            if improved:
                best, best_epoch, best_metrics = metrics["tuning_objective"], epoch + 1, metrics
                if save:
                    self.save()
            log.log(
                level,
                f"epoch {epoch + 1}/{cfg.epochs} ({elapsed(start)}): train loss "
                f"{total / max(i, 1):.4f}; val loss {metrics['loss']:.4f}, "
                f"recall@10 {metrics.get('recall@10', float('nan')):.3f}; tuning "
                f"{rated_brief(tuned(metrics))}; best {best:.4f} (epoch {best_epoch})"
                f"{', saved' if improved and save else ''}",
            )
            log.debug("epoch %d val parts: %s", epoch + 1, metrics)
        if save:
            log.info(
                f"training done in {elapsed(start_all)}: best tuning objective {best:.4f} at epoch "
                f"{best_epoch} -> {CHECKPOINT.relative_to(ML_ROOT).as_posix()}"
            )
        return best_metrics

    def save(self) -> None:
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": self.model.state_dict(),
                "cfg": dataclasses.asdict(self.cfg),
                "d_in": self.fi.shape[1],
                "sizes": self.vocab.sizes(),
                "vocab": self.vocab.digest,
                "item_ids": [it["id"] for it in self.catalog.items],
            },
            CHECKPOINT,
        )

    @torch.no_grad()
    def write_outputs(self) -> None:
        """Teacher targets for distillation: every query outside the held-out and eval sets."""
        self.model.load_state_dict(torch.load(CHECKPOINT, weights_only=False)["state"])
        self.model.eval()
        rows = torch.as_tensor(self.qs.where("train", "val", "distill"), device="cuda")
        x = self.fq[rows]
        palette, light, face = (
            torch.cat(parts)
            for parts in zip(
                *(self.model.heads(x[i : i + 8192]) for i in range(0, len(x), 8192)), strict=True
            )
        )
        torch.save(
            {
                "texts": [self.qs.texts[i] for i in rows.tolist()],
                "split": self.qs.split[rows.cpu().numpy()].tolist(),
                "pos": self.pos[rows].cpu(),
                "query_emb": in_chunks(self.model.embed_queries, x).half().cpu(),
                "item_emb": in_chunks(self.model.embed_items, self.fi).half().cpu(),
                "item_ids": [it["id"] for it in self.catalog.items],
                "palette": palette.cpu(),
                "light": light.cpu(),
                "typeface": face.cpu(),
                "temperature": self.cfg.temperature,
                "vocab": self.vocab.digest,
            },
            OUTPUTS,
        )
        log.info(
            f"teacher outputs for {num(len(rows))} queries and {num(len(self.fi))} items -> "
            f"{OUTPUTS.relative_to(ML_ROOT).as_posix()}"
        )


def tuned(metrics: dict[str, float]) -> dict[str, float]:
    """The tuning-set scores inside a metrics dict, without the prefix."""
    return {k.removeprefix("tuning_"): v for k, v in metrics.items() if k.startswith("tuning_")}


def brief(params: dict[str, Any]) -> str:
    return ", ".join(
        f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in params.items()
    )


def tune(trainer: Trainer, trials: int = TRIALS) -> dict[str, Any]:
    """Search the head settings on the tuning set and write the best to TEACHER_PARAMS.

    Trial 1 is the default config, so the last line compares the best trial with it.
    Trials do not write a checkpoint. No pruner: trials have different epoch counts and
    LR schedules, so the objective at one epoch does not compare across trials.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = TeacherConfig()
    best = -1.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 0.1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "epochs": trial.suggest_int("epochs", 3, 40),
            "hidden": trial.suggest_categorical("hidden", [512, 1024, 2048]),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "temperature": trial.suggest_float("temperature", 0.02, 0.1),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.15),
            "raw_weight": trial.suggest_float("raw_weight", 0.0, 1.0),
        }
        start = time.perf_counter()
        trainer.build(dataclasses.replace(base, **params))
        metrics = trainer.train(save=False)
        trial.set_user_attr("metrics", metrics)
        best = max(best, metrics["tuning_objective"])
        log.info(
            f"trial {trial.number + 1}/{trials} ({elapsed(start)}): {brief(params)}; tuning "
            f"{rated_brief(tuned(metrics))}; val recall@10 {metrics['recall@10']:.4f}; "
            f"best {best:.4f}"
        )
        return metrics["tuning_objective"]

    start = time.perf_counter()
    log.info(
        f"searching head settings: {trials} trials on the tuning set; a trial takes about "
        f"{TRIAL_SECONDS} s, so the search takes about {trials * TRIAL_SECONDS // 60} min"
    )
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(
        {
            k: getattr(base, k)
            for k in (
                "lr",
                "weight_decay",
                "dropout",
                "epochs",
                "hidden",
                "batch_size",
                "temperature",
                "label_smoothing",
                "warmup_ratio",
                "raw_weight",
            )
        }
    )
    study.optimize(objective, n_trials=trials)
    default, top = study.trials[0], study.best_trial
    log.info(
        f"search done in {elapsed(start)}: trial {top.number + 1} tuning objective "
        f"{top.value:.4f} vs default {default.value:.4f}; {brief(top.params)}"
    )
    log.info(
        "val losses, best vs default: "
        + ", ".join(
            f"{k} {top.user_attrs['metrics'].get(k, float('nan')):.4f} vs "
            f"{default.user_attrs['metrics'].get(k, float('nan')):.4f}"
            for k in ("palette", "light", "typeface")
        )
    )
    write_json(TEACHER_PARAMS, top.params)
    log.info(f"teacher settings -> {TEACHER_PARAMS.relative_to(ML_ROOT).as_posix()}")
    return top.params


def run() -> None:
    start = time.perf_counter()
    make_deterministic(SEED)
    trainer = Trainer(TeacherConfig())
    if TEACHER_PARAMS.is_file():
        params = json.loads(TEACHER_PARAMS.read_text(encoding="utf-8"))
    else:
        params = tune(trainer)
    trainer.build(dataclasses.replace(TeacherConfig(), **params))
    trainer.train()
    trainer.write_outputs()
    log.info(f"done in {elapsed(start)}")
