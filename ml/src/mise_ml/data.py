"""Join catalog items and labels into deterministic query splits for training and evaluation."""

import hashlib
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np

from mise_ml.color import hex_palette_to_oklab, rgb255_palette_to_oklab
from mise_ml.config import (
    CATEGORIES,
    DISTILL,
    EVAL_FEELINGS,
    PAT,
    PAT_SENTENCES,
    PROFILES,
    RESOLVED,
    SEED,
    TeacherConfig,
)
from mise_ml.log import get as get_logger
from mise_ml.util import hash_fraction, iter_jsonl
from mise_ml.vocab import Vocab, labels_path

log = get_logger(__name__)

# Mirror src/lib/code.ts. JavaScript counts UTF-16 code units.
MAX_FEELING = 500


def normalize_feeling(text: str) -> str:
    """Normalize case and spacing, then remove punctuation at the edges for duplicate checks."""
    text = " ".join(text.lower().split())
    start, end = 0, len(text)
    while start < end and (text[start].isspace() or unicodedata.category(text[start])[0] == "P"):
        start += 1
    while end > start and (
        text[end - 1].isspace() or unicodedata.category(text[end - 1])[0] == "P"
    ):
        end -= 1
    return text[start:end]


def distill_rejection(text: str, evals: set[str], seen: set[str]) -> str | None:
    normalized = normalize_feeling(text)
    if not normalized:
        return "empty"
    if len(text.encode("utf-16-le")) // 2 > MAX_FEELING:
        return "too_long"
    if normalized in evals:
        return "eval"
    if normalized in seen:
        return "duplicate"
    return None


def load_distill_texts(exclude: list[str]) -> list[str]:
    if not DISTILL.is_file():
        raise SystemExit(f"missing {DISTILL}; run uv run label")
    evals = {normalize_feeling(t) for t in load_eval_texts()}
    seen = {normalize_feeling(t) for t in exclude}
    texts = []
    for row in iter_jsonl(DISTILL):
        text = row["text"].strip()
        if distill_rejection(text, evals, seen) is None:
            texts.append(text)
            seen.add(normalize_feeling(text))
    return texts


# Casual spellings that people type. Each pattern matches whole words in any case.
# Phrases come before the contractions that they contain.
CASUAL = (
    (re.compile(r"\bi don['’]t know\b", re.I), "idk"),
    (re.compile(r"\bto be honest\b", re.I), "tbh"),
    (re.compile(r"\bright now\b", re.I), "rn"),
    (re.compile(r"\bwant to\b", re.I), "wanna"),
    (re.compile(r"\bgoing to\b", re.I), "gonna"),
    (re.compile(r"\bgot to\b", re.I), "gotta"),
    (re.compile(r"\btrying to\b", re.I), "tryna"),
    (re.compile(r"\bkind of\b", re.I), "kinda"),
    (re.compile(r"\bbecause\b", re.I), "bc"),
    (
        re.compile(r"\b(i|you|we|they|he|she|it|that|what|there)['’](m|re|ve|ll|d|s)\b", re.I),
        r"\1\2",
    ),
    (
        re.compile(r"\b(can|don|won|isn|didn|doesn|wasn|aren|couldn|wouldn|shouldn)['’]t\b", re.I),
        r"\1t",
    ),
    (re.compile(r"\byou\b", re.I), "u"),
    (re.compile(r"\bvery\b", re.I), "v"),
    (re.compile(r"\b(\w{3,})ing\b", re.I), r"\1in"),
)


def typo_feeling(text: str, salt: str = "") -> str:
    """Type the feeling as a hurried person does: lower case, casual spellings, 1-2 letter slips.

    The result depends only on the text and the salt, so each salt gives one fixed variant.
    """
    seed = int.from_bytes(hashlib.sha256(f"{SEED}:typo:{salt}:{text}".encode()).digest()[:8])
    rng = random.Random(seed)
    if rng.random() < 0.5:
        text = text.lower()
    # Apply each casual spelling that matches, each with an even chance.
    for pattern, repl in CASUAL:
        if pattern.search(text) and rng.random() < 0.5:
            text = pattern.sub(repl, text)
    for _ in range(rng.choice((1, 1, 2))):
        text = letter_slip(text, rng)
    return text


def letter_slip(text: str, rng: random.Random) -> str:
    """Swap, drop, double, or repeat a letter without changing spaces or punctuation."""
    letters = [i for i, c in enumerate(text) if c.isascii() and c.isalpha()]
    if not letters:
        return text
    swaps = [i for i in letters if i + 1 in letters and text[i] != text[i + 1]]
    operation = rng.choice(["drop", "double", "repeat", *(["swap"] if swaps else [])])
    i = rng.choice(swaps if operation == "swap" else letters)
    if operation == "swap":
        return text[:i] + text[i + 1] + text[i] + text[i + 2 :]
    if operation == "drop":
        return text[:i] + text[i + 1 :]
    if operation == "repeat":
        # A stretched vowel, as in "sooo" or "huuurts".
        vowels = [j for j in letters if text[j] in "aeiouy"]
        if vowels:
            i = rng.choice(vowels)
            return text[:i] + text[i] * rng.randint(2, 4) + text[i:]
    return text[:i] + text[i] + text[i:]


def load_eval_sets() -> dict[str, str]:
    return {
        r["text"].strip(): r["set"] for r in iter_jsonl(EVAL_FEELINGS) if r.get("text", "").strip()
    }


def load_eval_texts() -> list[str]:
    return list(load_eval_sets())


def item_text(item: dict[str, Any]) -> str:
    """Same shape as the stub bundle's item text, so both encoders see one format."""
    return (
        f"{item['vibe']}. {item['description']} "
        f"{item['category']}: {item['title']} by {item['creator']}."
    )


@dataclass
class Catalog:
    items: list[dict[str, Any]]
    texts: list[str]
    categories: np.ndarray
    index: dict[str, int]


def load_catalog() -> Catalog:
    """Join resolved items with profiles and sort by category and ID to align all model arrays."""
    profiles = {r["key"]: r for r in iter_jsonl(PROFILES)}
    items = []
    for r in iter_jsonl(RESOLVED):
        p = profiles.get(r["id"])
        if p is None:
            continue
        items.append(
            {**r, "vibe": p["vibe"], "description": p["description"], "queries": p["queries"]}
        )
    items.sort(key=lambda it: (CATEGORIES.index(it["category"]), it["id"]))
    if not items:
        raise SystemExit("no profiled items; run uv run download and uv run label first")
    return Catalog(
        items=items,
        texts=[item_text(it) for it in items],
        categories=np.array([CATEGORIES.index(it["category"]) for it in items], dtype=np.int64),
        index={it["id"]: i for i, it in enumerate(items)},
    )


@dataclass
class Query:
    text: str
    pos: int = -1
    palette: np.ndarray | None = None
    light: int = -1
    typeface: int = -1
    split: str = "train"


@dataclass
class QuerySet:
    texts: list[str]
    pos: np.ndarray
    palette: np.ndarray
    has_palette: np.ndarray
    light: np.ndarray
    typeface: np.ndarray
    split: np.ndarray

    def where(self, *splits: str) -> np.ndarray:
        return np.flatnonzero(np.isin(self.split, splits))


def load_queries(
    catalog: Catalog, vocab: Vocab, cfg: TeacherConfig, *, include_distill: bool = False
) -> QuerySet:
    """Combine labels and queries with stable splits that keep evaluation text out of training."""
    evals = set(load_eval_texts())
    queries: dict[str, Query] = {}

    def get(text: str) -> Query:
        return queries.setdefault(text, Query(text))

    matches: dict[str, set[str]] = {}
    for item in catalog.items:
        for q in item["queries"]:
            matches.setdefault(q, set()).add(item["id"])
    for text, ids in matches.items():
        q = get(text)
        if len(ids) == 1:
            q.pos = catalog.index[next(iter(ids))]

    light_ix = {x: i for i, x in enumerate(vocab.lights)}
    face_ix = {x: i for i, x in enumerate(vocab.typeface_ids)}
    for r in iter_jsonl(labels_path(vocab)):
        lab = hex_palette_to_oklab(r["palette"])
        if lab is None:
            continue
        q = get(r["key"])
        q.palette = lab
        q.light = light_ix.get(r["light"], -1)
        q.typeface = face_ix.get(r["typeface"], -1)

    pat = {r["phrase"]: r["rgb"] for r in iter_jsonl(PAT)}
    for r in iter_jsonl(PAT_SENTENCES):
        if r["key"] in pat:
            q = get(r["text"])
            if q.palette is None:
                q.palette = rgb255_palette_to_oklab(np.array(pat[r["key"]]))

    for text in evals:
        get(text)

    for q in queries.values():
        if q.text in evals:
            q.split = "eval"
        elif q.pos >= 0 and hash_fraction("heldout:" + q.text) < cfg.heldout_paraphrase_fraction:
            q.split = "heldout"
        elif hash_fraction("val:" + q.text) < cfg.val_fraction:
            q.split = "val"

    rows = sorted(
        (
            q
            for q in queries.values()
            if q.split == "eval"
            or q.pos >= 0
            or q.palette is not None
            or any(label >= 0 for label in (q.light, q.typeface))
        ),
        key=lambda q: q.text,
    )
    ambiguous = sum(len(ids) > 1 for ids in matches.values())
    if ambiguous:
        log.warning(
            "%d feelings match multiple works; omitted their retrieval labels; "
            "removed %d rows without another training label",
            ambiguous,
            len(queries) - len(rows),
        )
    if include_distill:
        # Assign splits before adding distillation queries to prevent leakage into training.
        # Exclude punctuation variants of those queries too.
        rows.extend(Query(text, split="distill") for text in load_distill_texts(list(queries)))
    zero = np.zeros((5, 3))
    return QuerySet(
        texts=[q.text for q in rows],
        pos=np.array([q.pos for q in rows], dtype=np.int64),
        palette=np.stack([q.palette if q.palette is not None else zero for q in rows]).astype(
            np.float32
        ),
        has_palette=np.array([q.palette is not None for q in rows]),
        light=np.array([q.light for q in rows], dtype=np.int64),
        typeface=np.array([q.typeface for q in rows], dtype=np.int64),
        split=np.array([q.split for q in rows]),
    )


def recall_scores(
    query_emb: np.ndarray,
    pos: np.ndarray,
    item_emb: np.ndarray,
    item_categories: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """One hit value per query, ranked within the positive item's category."""
    if len(pos) == 0:
        return np.empty(0, dtype=np.float64)
    sims = query_emb @ item_emb.T
    same = item_categories[None, :] == item_categories[pos][:, None]
    sims = np.where(same, sims, -np.inf)
    pos_sim = sims[np.arange(len(pos)), pos]
    rank = (sims > pos_sim[:, None]).sum(axis=1)
    return (rank < k).astype(np.float64)


def fidelity_at_k(
    query_emb: np.ndarray,
    item_emb: np.ndarray,
    teacher_query: np.ndarray,
    teacher_items: np.ndarray,
    item_categories: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Per query: the share of the teacher's top k in each category that a model also ranks in
    its top k, averaged over the categories."""
    sims = query_emb @ item_emb.T
    teacher_sims = teacher_query @ teacher_items.T
    shares = []
    for c in np.unique(item_categories):
        idx = np.flatnonzero(item_categories == c)
        mine = np.argpartition(-sims[:, idx], k, axis=1)[:, :k]
        theirs = np.argpartition(-teacher_sims[:, idx], k, axis=1)[:, :k]
        shares.append([len(np.intersect1d(a, b)) / k for a, b in zip(mine, theirs, strict=True)])
    return np.asarray(shares).mean(axis=0)


def recall_at_k(
    query_emb: np.ndarray,
    pos: np.ndarray,
    item_emb: np.ndarray,
    item_categories: np.ndarray,
    k: int = 10,
) -> float:
    scores = recall_scores(query_emb, pos, item_emb, item_categories, k)
    return float(scores.mean()) if len(scores) else float("nan")
