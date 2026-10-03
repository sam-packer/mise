"""Distill the teacher's predictions into a feeling encoder and heads for the browser.

The catalog side is fixed: the teacher's item vectors in a space of the student's dims.
"""

import dataclasses
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from mise_ml.config import ML_ROOT, MODELS, SEED, TUNING_SET, StudentConfig
from mise_ml.data import fidelity_at_k, load_catalog, load_eval_texts, recall_at_k, typo_feeling
from mise_ml.heads import ChoiceHeads, kl_logits, palette_loss
from mise_ml.inference import encode_batches
from mise_ml.log import elapsed, get, num, progress
from mise_ml.rated import brief, load_rated, rated_scores
from mise_ml.teacher import OUTPUTS as TEACHER_OUTPUTS
from mise_ml.training import cosine_schedule
from mise_ml.util import hash_fraction, make_deterministic
from mise_ml.vocab import load_vocab
from mise_ml.wordpiece import prune

log = get(__name__)
STUDENT_DIR = MODELS / "student"
OUTPUT_NAMES = ("embedding", "palette", "light", "typeface")


class Student(nn.Module):
    """Text encoder with masked mean pooling and retrieval, palette, and choice heads."""

    def __init__(
        self,
        encoder: PreTrainedModel,
        sizes: tuple[int, int],
        head_hidden: int,
        dims: int = 384,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.uses_token_types = "token_type_ids" in inspect.signature(encoder.forward).parameters
        self.projection = (
            nn.Identity()
            if encoder.config.hidden_size == dims
            else nn.Linear(encoder.config.hidden_size, dims, bias=False)
        )
        self.dims = dims
        self.heads = ChoiceHeads(encoder.config.hidden_size, head_hidden, sizes, dropout=0.1)

    def pool(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.uses_token_types:
            inputs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**inputs).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def embed(self, **batch: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.pool(**batch)), dim=-1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        pooled = self.pool(input_ids, attention_mask, token_type_ids)
        palette, light, typeface = self.heads(pooled)
        return F.normalize(self.projection(pooled), dim=-1), palette, light, typeface


def item_space(item_emb: torch.Tensor, dims: int) -> torch.Tensor:
    """The (teacher dims x dims) map onto the top right singular vectors of the teacher's item
    vectors: the subspace that best keeps the teacher's item inner products. On the tuning set the
    teacher scored 0.701 in it and 0.704 in full; raw Qwen vectors cut to their first 384 dims
    scored 0.596."""
    _, _, v = torch.linalg.svd(item_emb.double().cpu(), full_matrices=False)
    return v[:dims].T.float()


def save_student(
    model: Student, tokenizer: PreTrainedTokenizerBase, meta: dict, path: Path = STUDENT_DIR
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path / "encoder")
    torch.save(model.heads.state_dict(), path / "heads.pt")
    torch.save(model.projection.state_dict(), path / "projection.pt")
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_student(path: Path = STUDENT_DIR) -> tuple[Student, PreTrainedTokenizerBase, dict]:
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    encoder = AutoModel.from_pretrained(path / "encoder", attn_implementation="sdpa")
    model = Student(encoder, tuple(meta["sizes"]), meta["head_hidden"], meta["cfg"]["dims"])
    if not isinstance(model.projection, nn.Identity):
        model.projection.load_state_dict(torch.load(path / "projection.pt", weights_only=True))
    model.heads.load_state_dict(torch.load(path / "heads.pt", weights_only=True))
    return model.eval(), AutoTokenizer.from_pretrained(path / "encoder"), meta


class Pretokenized:
    """Keep tokenized texts on the GPU, padded to max_length.

    Return each batch with padding only up to its longest row.
    Right padding lets rows() trim trailing columns without removing tokens.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, texts: list[str], max_length: int, device: str
    ) -> None:
        assert tokenizer.padding_side == "right"
        batch = tokenizer(
            texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )
        self.tensors = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "token_type_ids": batch.get("token_type_ids", torch.zeros_like(batch["input_ids"])).to(
                device
            ),
        }
        self.lengths = self.tensors["attention_mask"].sum(1)

    def rows(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        width = int(self.lengths[idx].max())
        return {k: v[idx, :width] for k, v in self.tensors.items()}


@torch.no_grad()
def encode_texts(
    model: Student,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
    device: str = "cuda",
) -> np.ndarray:
    model.eval()

    def encode(batch):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.setdefault("token_type_ids", torch.zeros_like(batch["input_ids"]))
        with torch.autocast(
            torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            return model.embed(**batch).float().cpu().numpy()

    return encode_batches(tokenizer, texts, max_length, encode, device)


def item_candidates(
    teacher_sims: torch.Tensor,
    positives: torch.Tensor,
    by_category: list[torch.Tensor],
    cfg: StudentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Items for one step: sorted indices, and a (queries x candidates) mask of the items that
    the teacher picked for each query.

    Take the teacher's top items in each category for each query, plus the positives and
    random items.
    """
    top = torch.cat(
        [
            idx[teacher_sims[:, idx].topk(min(cfg.teacher_topk, len(idx)), dim=1).indices]
            for idx in by_category
        ],
        dim=1,
    )
    rand = torch.randint(
        0, sum(map(len, by_category)), (cfg.random_items,), device=positives.device
    )
    cand = torch.cat([top.flatten(), positives[positives >= 0], rand]).unique()
    picked = torch.zeros(len(positives), len(cand), dtype=torch.bool, device=cand.device)
    picked.scatter_(1, torch.searchsorted(cand, top), True)
    return cand, picked


def category_kl(
    student: torch.Tensor, teacher: torch.Tensor, categories: torch.Tensor, n_categories: int
) -> torch.Tensor:
    """Mean over categories of the KL between the rankings inside each category."""
    losses = [
        kl_logits(student[:, categories == c], teacher[:, categories == c], 1.0)
        for c in range(n_categories)
        if (categories == c).any()
    ]
    return torch.stack(losses).mean()


def noisy_texts(texts: list[str], rows: np.ndarray, share: float, epoch: int) -> list[str]:
    """Copy of texts where a `share` of `rows` gets typing noise that changes each epoch."""
    noisy = list(texts)
    for i in rows:
        if hash_fraction(f"typo:{epoch}:{texts[i]}") < share:
            noisy[i] = typo_feeling(texts[i], str(epoch))
    return noisy


def run(cfg: StudentConfig | None = None, out: Path = STUDENT_DIR) -> None:
    cfg = cfg or StudentConfig()
    make_deterministic(SEED)
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    vocab = load_vocab()
    catalog = load_catalog()
    tuning = load_rated(TUNING_SET, catalog)
    t = torch.load(TEACHER_OUTPUTS, weights_only=False)
    if t["vocab"] != vocab.digest:
        raise SystemExit("vocab changed since train-teacher; rerun uv run train")
    if t["item_ids"] != [it["id"] for it in catalog.items]:
        raise SystemExit("catalog changed since train-teacher; rerun uv run train")

    dev = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, revision=cfg.revision)
    encoder = AutoModel.from_pretrained(
        cfg.backbone, revision=cfg.revision, attn_implementation="sdpa"
    )
    texts: list[str] = t["texts"]
    if cfg.prune_vocab:
        full = len(tokenizer)
        corpus = [*texts, *load_eval_texts(), *tuning.feelings, *catalog.texts]
        tokenizer = prune(tokenizer, encoder, corpus, cfg.prune_keep_below)
        log.info(f"vocabulary pruned from {num(full)} to {num(len(tokenizer))} tokens")
    model = Student(encoder, vocab.sizes(), cfg.head_hidden, cfg.dims).to(dev)

    split = np.array(t["split"])
    train = np.flatnonzero(np.isin(split, ("train", "distill")))
    val = np.flatnonzero(split == "val")
    pos = t["pos"].to(dev)
    # The student encodes feelings only. The catalog vectors are the teacher's item vectors in a
    # fixed space of the student's dims; they ship as vectors.bin and get no gradient.
    space = item_space(t["item_emb"], cfg.dims).to(dev)
    # Round to fp16 as vectors.bin does, so training, selection, and eval score the shipped values.
    items = F.normalize(t["item_emb"].float().to(dev) @ space, dim=-1).half().float()
    tq = F.normalize(t["query_emb"].float().to(dev) @ space, dim=-1)
    t_palette = t["palette"].float().to(dev)
    t_choices = [t[k].float().to(dev) for k in ("light", "typeface")]
    n_items = len(catalog.items)
    item_categories = torch.as_tensor(catalog.categories, device=dev)
    n_categories = int(catalog.categories.max()) + 1
    by_category = [torch.where(item_categories == c)[0] for c in range(n_categories)]

    groups = [
        {"params": model.encoder.parameters(), "lr": cfg.encoder_lr},
        {"params": model.heads.parameters(), "lr": cfg.head_lr},
        {"params": model.projection.parameters(), "lr": cfg.head_lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, fused=True)
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    sched = cosine_schedule(opt, total, cfg.warmup_ratio)
    rng = np.random.default_rng(SEED)
    best, best_epoch = -1.0, -1
    tau = cfg.temperature
    log.info(
        f"reading teacher outputs: {num(len(texts))} queries "
        f"({num(int((split == 'train').sum()))} train, "
        f"{num(int((split == 'distill').sum()))} distill, "
        f"{num(len(val))} val), {num(n_items)} items; backbone {cfg.backbone}"
    )
    log.info(
        f"training: {cfg.epochs} epochs x {num(steps_per_epoch)} steps, batch {cfg.batch_size}, "
        f"encoder lr {cfg.encoder_lr}, head lr {cfg.head_lr}; items per step: per query and "
        f"category the teacher's top {cfg.teacher_topk}, plus positives and "
        f"{cfg.random_items} random; KL weight {cfg.kl_weight}; regression weight "
        f"{cfg.regression_weight}; typing noise on {cfg.typo_share:.0%} of train rows"
    )
    log.info(
        f"fixed item space: the teacher's {t['item_emb'].shape[1]}-dim item vectors on their "
        f"top {cfg.dims} singular vectors"
    )
    item_emb = items.cpu().numpy()
    log.info(
        "one epoch = one pass over train + distill rows; the best epoch has the highest "
        "tuning-set objective (mean of within-feeling AUC and Spearman on rated pairs)"
    )
    start_all = time.perf_counter()

    for epoch in range(cfg.epochs):
        start = time.perf_counter()
        query_tokens = Pretokenized(
            tokenizer, noisy_texts(texts, train, cfg.typo_share, epoch), cfg.max_length, dev
        )
        model.train()
        order = rng.permutation(train)
        bar = progress(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{cfg.epochs}", unit="step")
        running = 0.0
        for step in bar:
            b = torch.as_tensor(
                order[step * cfg.batch_size : (step + 1) * cfg.batch_size], device=dev
            )
            p = pos[b]
            cand, picked = item_candidates(tq[b] @ items.T, p, by_category, cfg)
            queries = query_tokens.rows(b)
            with torch.autocast(dev, dtype=torch.bfloat16):
                q_emb, palette, light, face = model(**queries)
            q_emb = q_emb.float()
            s = q_emb @ items[cand].T / tau
            teacher_sims = tq[b] @ items[cand].T / tau
            loss_kl = cfg.kl_weight * category_kl(
                s, teacher_sims, item_categories[cand], n_categories
            )
            has = p >= 0
            target = torch.searchsorted(cand, p[has])
            # The teacher's own picks for a query often fit it too, so they are not negatives
            # for that query's positive.
            false_negatives = picked[has]
            false_negatives[torch.arange(len(target), device=dev), target] = False
            loss_nce = (
                cfg.infonce_weight
                * F.cross_entropy(s[has].masked_fill(false_negatives, float("-inf")), target)
                if has.any()
                else s.new_zeros(())
            )
            loss_pal = cfg.palette_weight * palette_loss(
                palette.float(), t_palette[b], cfg.lightness_weight
            )
            loss_choice = cfg.choice_weight * sum(
                kl_logits(out.float(), tc[b], cfg.choice_temperature)
                for out, tc in zip((light, face), t_choices, strict=True)
            )
            # The teacher's own feeling vector in the item space is a direct target.
            loss_reg = cfg.regression_weight * (1 - (q_emb * tq[b]).sum(-1)).mean()
            loss = loss_kl + loss_nce + loss_reg + loss_pal + loss_choice
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            if step % 20 == 0:
                bar.set_postfix(
                    loss=f"{running / (step + 1):.3f}",
                    kl=f"{loss_kl.item():.3f}",
                    nce=f"{loss_nce.item():.3f}",
                    reg=f"{loss_reg.item():.3f}",
                    lr=f"{sched.get_last_lr()[0]:.1e}",
                    refresh=False,
                )

        q = encode_texts(model, tokenizer, [texts[i] for i in val], cfg.max_length)
        teacher_q = t["query_emb"].float().numpy()
        teacher_items = t["item_emb"].float().numpy()
        fidelity = float(
            fidelity_at_k(q, item_emb, teacher_q[val], teacher_items, catalog.categories).mean()
        )
        has_pos = t["pos"].numpy()[val] >= 0
        val_pos = t["pos"].numpy()[val][has_pos]
        recall = recall_at_k(q[has_pos], val_pos, item_emb, catalog.categories)
        teacher_recall = recall_at_k(
            teacher_q[val][has_pos], val_pos, teacher_items, catalog.categories
        )
        tuning_q = encode_texts(model, tokenizer, tuning.feelings, cfg.max_length)
        scores = rated_scores(tuning, tuning_q, item_emb)
        improved = scores["objective"] > best
        if improved:
            best, best_epoch = scores["objective"], epoch + 1
            save_student(
                model,
                tokenizer,
                {
                    "backbone": cfg.backbone,
                    "sizes": list(vocab.sizes()),
                    "head_hidden": cfg.head_hidden,
                    "vocab": vocab.digest,
                    "item_ids": [it["id"] for it in catalog.items],
                    "cfg": dataclasses.asdict(cfg),
                    "val_recall@10": recall,
                    "val_fidelity@10": fidelity,
                    "tuning": scores,
                },
                out,
            )
            np.save(out / "items.npy", item_emb.astype(np.float16))
        log.info(
            f"epoch {epoch + 1}/{cfg.epochs} ({elapsed(start)}): train loss "
            f"{running / steps_per_epoch:.4f}; tuning {brief(scores)}; val fidelity@10 "
            f"{fidelity:.3f}; val recall@10 {recall:.3f} (teacher {teacher_recall:.3f}); "
            f"best tuning {best:.4f} (epoch {best_epoch})" + (", saved" if improved else "")
        )
        if epoch + 1 - best_epoch >= cfg.patience:
            log.info("early stop: no tuning-set improvement for %d epochs", cfg.patience)
            break
    log.info(
        f"done in {elapsed(start_all)}: best tuning objective {best:.4f} at epoch {best_epoch} -> "
        f"{out.relative_to(ML_ROOT).as_posix()}"
    )
