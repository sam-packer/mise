import dataclasses
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mise_ml.config import ML_ROOT, MODELS, SEED, TeacherConfig
from mise_ml.data import load_catalog, load_queries, recall_at_k
from mise_ml.features import ITEM_TEMPLATE, QUERY_TEMPLATE, FeatureStore, shared_encoder
from mise_ml.heads import ChoiceHeads, Mlp, palette_loss
from mise_ml.log import elapsed, get, num, progress
from mise_ml.training import cosine_schedule, teacher_config
from mise_ml.util import make_deterministic
from mise_ml.vocab import load_vocab

log = get(__name__)
CHECKPOINT = MODELS / "teacher.pt"
OUTPUTS = MODELS / "teacher_outputs.pt"


class Teacher(nn.Module):
    def __init__(self, d_in: int, cfg: TeacherConfig, sizes: tuple[int, int, int]) -> None:
        super().__init__()
        self.query = Mlp(d_in, cfg.hidden, cfg.dims, cfg.dropout)
        self.item = Mlp(d_in, cfg.hidden, cfg.dims, cfg.dropout)
        self.heads = ChoiceHeads(d_in, 512, sizes, cfg.dropout)

    def embed_queries(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query(x), dim=-1)

    def embed_items(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item(x), dim=-1)


def query_store(cfg: TeacherConfig) -> FeatureStore:
    return FeatureStore("queries", QUERY_TEMPLATE, cfg)


def item_store(cfg: TeacherConfig) -> FeatureStore:
    return FeatureStore("items", ITEM_TEMPLATE, cfg)


def load_teacher() -> tuple[Teacher, dict]:
    ckpt = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
    cfg = teacher_config(ckpt["cfg"])
    model = Teacher(ckpt["d_in"], cfg, tuple(ckpt["sizes"])).cuda().eval()
    model.load_state_dict(ckpt["state"])
    return model, ckpt


@torch.no_grad()
def in_chunks(fn, x: torch.Tensor, size: int = 8192) -> torch.Tensor:
    return torch.cat([fn(x[i : i + size]) for i in range(0, len(x), size)])


class Trainer:
    def __init__(self, cfg: TeacherConfig) -> None:
        self.cfg = cfg
        self.vocab = load_vocab()
        self.catalog = load_catalog()
        self.qs = load_queries(self.catalog, self.vocab, cfg)
        splits = ", ".join(
            f"{s} {num(len(self.qs.where(s)))}" for s in ("train", "val", "heldout", "eval")
        )
        log.info(
            f"reading data/curated: {num(len(self.catalog.items))} items, "
            f"{num(len(self.qs.texts))} queries ({splits}); backbone {cfg.backbone}"
        )
        dev = "cuda"
        self.fi = torch.tensor(
            item_store(cfg).get(self.catalog.texts), dtype=torch.float32, device=dev
        )
        self.fq = torch.tensor(query_store(cfg).get(self.qs.texts), dtype=torch.float32, device=dev)
        shared_encoder.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()

        def t(a: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(a, device=dev)

        self.pos = t(self.qs.pos)
        self.palette = t(self.qs.palette)
        self.has_palette = t(self.qs.has_palette)
        self.choices = [t(self.qs.light), t(self.qs.typeface), t(self.qs.scent)]
        self.item_cat = t(self.catalog.categories)
        self.hard = torch.full((len(self.qs.texts), cfg.hard_negatives), -1, device=dev)
        self.model = Teacher(self.fi.shape[1], cfg, self.vocab.sizes()).to(dev)

    @torch.no_grad()
    def mine(self, rows: torch.Tensor) -> None:
        """Hard negatives: random picks from the top items of the positive's category."""
        self.model.eval()
        items = in_chunks(self.model.embed_items, self.fi)
        for start in range(0, len(rows), 4096):
            r = rows[start : start + 4096]
            sims = in_chunks(self.model.embed_queries, self.fq[r]) @ items.T
            p = self.pos[r]
            sims[self.item_cat[None, :] != self.item_cat[p][:, None]] = -torch.inf
            sims[torch.arange(len(r), device=sims.device), p] = -torch.inf
            top = sims.topk(self.cfg.hard_negative_pool, dim=1).indices
            pick = torch.randint(
                0, top.shape[1], (len(r), self.cfg.hard_negatives), device=top.device
            )
            self.hard[r] = top.gather(1, pick)
        self.model.train()

    def losses(self, b: torch.Tensor, full_softmax: bool = False) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        x = self.fq[b]
        out: dict[str, torch.Tensor] = {}
        with_pos = b[self.pos[b] >= 0]
        if len(with_pos):
            q = self.model.embed_queries(self.fq[with_pos])
            p = self.pos[with_pos]
            if full_softmax:
                cand = torch.arange(len(self.fi), device=b.device)
            else:
                negs = self.hard[with_pos]
                cand = torch.cat([p, negs[negs >= 0]]).unique()
            logits = q @ self.model.embed_items(self.fi[cand]).T / cfg.temperature
            out["retrieval"] = F.cross_entropy(logits, torch.searchsorted(cand, p))
        palette, *logits_by_head = self.model.heads(x)
        m = self.has_palette[b]
        if m.any():
            out["palette"] = cfg.palette_weight * palette_loss(
                palette[m], self.palette[b][m], cfg.lightness_weight
            )
        for name, logits, labels in zip(
            ("light", "typeface", "scent"), logits_by_head, self.choices, strict=True
        ):
            y = labels[b]
            m = y >= 0
            if m.any():
                out[name] = cfg.choice_weight * F.cross_entropy(
                    logits[m], y[m], label_smoothing=cfg.label_smoothing
                )
        return out

    @torch.no_grad()
    def validate(self, rows: torch.Tensor) -> dict[str, float]:
        self.model.eval()
        parts: dict[str, list[float]] = {}
        for start in range(0, len(rows), self.cfg.batch_size):
            for k, v in self.losses(rows[start : start + self.cfg.batch_size], True).items():
                parts.setdefault(k, []).append(v.item())
        metrics = {k: float(np.mean(v)) for k, v in parts.items()}
        metrics["loss"] = sum(metrics.values())
        r = rows[self.pos[rows] >= 0]
        if len(r):
            items = in_chunks(self.model.embed_items, self.fi).cpu().numpy()
            q = in_chunks(self.model.embed_queries, self.fq[r]).cpu().numpy()
            metrics["recall@10"] = recall_at_k(
                q, self.pos[r].cpu().numpy(), items, self.catalog.categories
            )
        self.model.train()
        return metrics

    def train(self) -> None:
        cfg = self.cfg
        train = torch.as_tensor(self.qs.where("train"), device="cuda")
        val = torch.as_tensor(self.qs.where("val"), device="cuda")
        train_pos = train[self.pos[train] >= 0]
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        steps = cfg.epochs * -(-len(train) // cfg.batch_size)
        sched = cosine_schedule(opt, steps, cfg.warmup_ratio)
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        best, best_epoch = -1.0, -1
        log.info(
            f"training heads: {cfg.epochs} epochs, batch {cfg.batch_size}, lr {cfg.lr}, "
            f"{num(len(train))} train rows ({num(len(train_pos))} with an item), "
            f"hard negatives from epoch {cfg.hard_negative_warmup}"
        )
        start_all = time.perf_counter()
        for epoch in range(cfg.epochs):
            start = time.perf_counter()
            if epoch >= cfg.hard_negative_warmup:
                self.mine(train_pos)
            perm = train[torch.randperm(len(train), device="cuda", generator=gen)]
            bar = progress(
                range(0, len(perm), cfg.batch_size), desc=f"epoch {epoch + 1}/{cfg.epochs}"
            )
            total = 0.0
            for i, step in enumerate(bar, 1):
                loss = sum(self.losses(perm[step : step + cfg.batch_size]).values())
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
            improved = metrics["recall@10"] > best
            if improved:
                best, best_epoch = metrics["recall@10"], epoch + 1
                self.save()
            log.info(
                f"epoch {epoch + 1}/{cfg.epochs} ({elapsed(start)}): train loss "
                f"{total / max(i, 1):.4f}; val loss {metrics['loss']:.4f}, "
                f"recall@10 {metrics.get('recall@10', float('nan')):.3f}; "
                f"best {best:.4f} (epoch {best_epoch}){', saved' if improved else ''}"
            )
            log.debug("epoch %d val parts: %s", epoch + 1, metrics)
            if epoch + 1 - best_epoch >= cfg.patience:
                log.info("early stop: no recall improvement for %d epochs", cfg.patience)
                break
        log.info(
            f"training done in {elapsed(start_all)}: best val recall@10 {best:.4f} at epoch "
            f"{best_epoch} -> {CHECKPOINT.relative_to(ML_ROOT).as_posix()}"
        )

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
        rows = torch.as_tensor(self.qs.where("train", "val"), device="cuda")
        x = self.fq[rows]
        palette, light, face, scent = (
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
                "scent": scent.cpu(),
                "temperature": self.cfg.temperature,
                "vocab": self.vocab.digest,
            },
            OUTPUTS,
        )
        log.info(
            f"teacher outputs for {num(len(rows))} queries and {num(len(self.fi))} items -> "
            f"{OUTPUTS.relative_to(ML_ROOT).as_posix()}"
        )


def run() -> None:
    start = time.perf_counter()
    make_deterministic(SEED)
    trainer = Trainer(TeacherConfig())
    trainer.train()
    trainer.write_outputs()
    log.info(f"done in {elapsed(start)}")
