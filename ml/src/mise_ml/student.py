"""Distill the teacher's predictions into a text encoder and heads for the browser."""

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

from mise_ml.config import ML_ROOT, MODELS, SEED, StudentConfig
from mise_ml.data import load_catalog, recall_at_k
from mise_ml.heads import ChoiceHeads, kl_logits, palette_loss
from mise_ml.inference import encode_batches
from mise_ml.log import elapsed, get, num, progress
from mise_ml.teacher import OUTPUTS as TEACHER_OUTPUTS
from mise_ml.training import cosine_schedule
from mise_ml.util import make_deterministic
from mise_ml.vocab import load_vocab

log = get(__name__)
STUDENT_DIR = MODELS / "student"
OUTPUT_NAMES = ("embedding", "palette", "light", "typeface", "scent")


class Student(nn.Module):
    """Text encoder with masked mean pooling and retrieval, palette, and choice heads."""

    def __init__(
        self,
        encoder: PreTrainedModel,
        sizes: tuple[int, int, int],
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
        palette, light, typeface, scent = self.heads(pooled)
        return F.normalize(self.projection(pooled), dim=-1), palette, light, typeface, scent


def save_student(model: Student, tokenizer: PreTrainedTokenizerBase, meta: dict) -> None:
    STUDENT_DIR.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(STUDENT_DIR / "encoder")
    tokenizer.save_pretrained(STUDENT_DIR / "encoder")
    torch.save(model.heads.state_dict(), STUDENT_DIR / "heads.pt")
    torch.save(model.projection.state_dict(), STUDENT_DIR / "projection.pt")
    (STUDENT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


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
    top: torch.Tensor, positives: torch.Tensor, n_items: int, cfg: StudentConfig
) -> torch.Tensor:
    """Keep teacher neighbors and positives within the memory budget for item activations."""
    required = torch.cat([top.flatten(), positives[positives >= 0]]).unique()
    rand = torch.randint(0, n_items, (cfg.random_items,), device=top.device)
    cand = torch.cat([required, rand]).unique()
    # Trim only random extras if unusually distinct neighbors exceed the step budget.
    upper = min(n_items, 1120)
    if len(required) > upper:
        raise ValueError("teacher neighbors and positives exceed the 1120-item step budget")
    if len(cand) > upper:
        extras = rand[~torch.isin(rand, required)].unique()
        extras = extras[torch.randperm(len(extras), device=top.device)]
        cand = torch.cat([required, extras[: upper - len(required)]]).sort().values
    return cand


def embed_items(model: "Student", tokens: Pretokenized, cand: torch.Tensor, chunks: int = 4):
    """Item embeddings in `cand` order, encoded in length-sorted chunks so short texts are not
    padded to the longest one in the step."""
    order = tokens.lengths[cand].argsort()
    parts = [model.embed(**tokens.rows(cand[idx])) for idx in order.chunk(chunks)]
    back = torch.empty_like(order)
    back[order] = torch.arange(len(order), device=order.device)
    return torch.cat(parts)[back]


def run() -> None:
    cfg = StudentConfig()
    make_deterministic(SEED)
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    vocab = load_vocab()
    catalog = load_catalog()
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
    model = Student(encoder, vocab.sizes(), cfg.head_hidden, cfg.dims).to(dev)

    split = np.array(t["split"])
    train = np.flatnonzero(np.isin(split, ("train", "distill")))
    val = np.flatnonzero(split == "val")
    texts: list[str] = t["texts"]
    pos = t["pos"].to(dev)
    tq = t["query_emb"].float().to(dev)
    ti = t["item_emb"].float().to(dev)
    t_palette = t["palette"].float().to(dev)
    t_choices = [t[k].float().to(dev) for k in ("light", "typeface", "scent")]
    n_items = len(catalog.items)

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
    query_tokens = Pretokenized(tokenizer, texts, cfg.max_length, dev)
    item_tokens = Pretokenized(tokenizer, catalog.texts, cfg.item_max_length, dev)
    log.info(
        f"reading teacher outputs: {num(len(texts))} queries "
        f"({num(int((split == 'train').sum()))} train, "
        f"{num(int((split == 'distill').sum()))} distill, "
        f"{num(len(val))} val), {num(n_items)} items; backbone {cfg.backbone}"
    )
    log.info(
        f"training: {cfg.epochs} epochs x {num(steps_per_epoch)} steps, batch {cfg.batch_size}, "
        f"encoder lr {cfg.encoder_lr}, head lr {cfg.head_lr}; items per step: teacher top "
        f"{cfg.teacher_topk} per query + {cfg.random_items} random + positives, "
        "at most 1120 unique items"
    )
    log.info("one epoch = one pass over train + distill rows; validation uses val rows only")
    start_all = time.perf_counter()

    for epoch in range(cfg.epochs):
        start = time.perf_counter()
        model.train()
        order = rng.permutation(train)
        bar = progress(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{cfg.epochs}", unit="step")
        running = 0.0
        for step in bar:
            b = torch.as_tensor(
                order[step * cfg.batch_size : (step + 1) * cfg.batch_size], device=dev
            )
            top = (tq[b] @ ti.T).topk(cfg.teacher_topk, dim=1).indices
            p = pos[b]
            cand = item_candidates(top, p, n_items, cfg)
            queries = query_tokens.rows(b)
            with torch.autocast(dev, dtype=torch.bfloat16):
                q_emb, palette, light, face, scent = model(**queries)
                i_emb = embed_items(model, item_tokens, cand)
            s = q_emb.float() @ i_emb.float().T / tau
            teacher_sims = tq[b] @ ti[cand].T / tau
            loss_kl = cfg.kl_weight * kl_logits(s, teacher_sims, 1.0)
            has = p >= 0
            loss_nce = (
                cfg.infonce_weight * F.cross_entropy(s[has], torch.searchsorted(cand, p[has]))
                if has.any()
                else s.new_zeros(())
            )
            loss_pal = cfg.palette_weight * palette_loss(
                palette.float(), t_palette[b], cfg.lightness_weight
            )
            loss_choice = cfg.choice_weight * sum(
                kl_logits(out.float(), tc[b], cfg.choice_temperature)
                for out, tc in zip((light, face, scent), t_choices, strict=True)
            )
            loss = loss_kl + loss_nce + loss_pal + loss_choice
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
                    lr=f"{sched.get_last_lr()[0]:.1e}",
                    refresh=False,
                )

        item_emb = encode_texts(model, tokenizer, catalog.texts, cfg.item_max_length)
        val_pos = val[t["pos"].numpy()[val] >= 0]
        q = encode_texts(model, tokenizer, [texts[i] for i in val_pos], cfg.max_length)
        recall = recall_at_k(q, t["pos"].numpy()[val_pos], item_emb, catalog.categories)
        teacher_recall = recall_at_k(
            t["query_emb"].float().numpy()[val_pos],
            t["pos"].numpy()[val_pos],
            t["item_emb"].float().numpy(),
            catalog.categories,
        )
        improved = recall > best
        if improved:
            best, best_epoch = recall, epoch + 1
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
                },
            )
        log.info(
            f"epoch {epoch + 1}/{cfg.epochs} ({elapsed(start)}): train loss "
            f"{running / steps_per_epoch:.4f}; val recall@10 {recall:.3f} (teacher "
            f"{teacher_recall:.3f}); best {best:.3f} (epoch {best_epoch})"
            + (", saved" if improved else "")
        )
        if epoch + 1 - best_epoch >= cfg.patience:
            log.info("early stop: no recall improvement for %d epochs", cfg.patience)
            break
    log.info(
        f"done in {elapsed(start_all)}: best val recall@10 {best:.3f} at epoch {best_epoch} -> "
        f"{STUDENT_DIR.relative_to(ML_ROOT).as_posix()}"
    )
