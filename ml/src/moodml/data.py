from dataclasses import dataclass
from typing import Any

import numpy as np

from moodml.color import hex_palette_to_oklab, rgb255_palette_to_oklab
from moodml.config import (
    CATEGORIES,
    EVAL_FEELINGS,
    PAT,
    PAT_SENTENCES,
    PROFILES,
    RESOLVED,
    TeacherConfig,
)
from moodml.util import hash_fraction, iter_jsonl
from moodml.vocab import Vocab, labels_path

SPLITS = ("train", "val", "heldout", "eval")


def load_eval_texts() -> list[str]:
    return [r["text"].strip() for r in iter_jsonl(EVAL_FEELINGS) if r.get("text", "").strip()]


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
        raise SystemExit("no profiled items; run resolve and profile first")
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
    scent: int = -1
    split: str = "train"


@dataclass
class QuerySet:
    texts: list[str]
    pos: np.ndarray
    palette: np.ndarray
    has_palette: np.ndarray
    light: np.ndarray
    typeface: np.ndarray
    scent: np.ndarray
    split: np.ndarray

    def where(self, *splits: str) -> np.ndarray:
        return np.flatnonzero(np.isin(self.split, splits))


def load_queries(catalog: Catalog, vocab: Vocab, cfg: TeacherConfig) -> QuerySet:
    evals = set(load_eval_texts())
    queries: dict[str, Query] = {}

    def get(text: str) -> Query:
        return queries.setdefault(text, Query(text))

    for i, item in enumerate(catalog.items):
        for q in item["queries"]:
            get(q).pos = i

    light_ix = {x: i for i, x in enumerate(vocab.lights)}
    face_ix = {x: i for i, x in enumerate(vocab.typeface_ids)}
    scent_ix = {x: i for i, x in enumerate(vocab.scent_ids)}
    for r in iter_jsonl(labels_path(vocab)):
        lab = hex_palette_to_oklab(r["palette"])
        if lab is None:
            continue
        q = get(r["key"])
        q.palette = lab
        q.light = light_ix.get(r["light"], -1)
        q.typeface = face_ix.get(r["typeface"], -1)
        q.scent = scent_ix.get(r["scent"], -1)

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

    rows = sorted(queries.values(), key=lambda q: q.text)
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
        scent=np.array([q.scent for q in rows], dtype=np.int64),
        split=np.array([q.split for q in rows]),
    )


def recall_at_k(
    query_emb: np.ndarray,
    pos: np.ndarray,
    item_emb: np.ndarray,
    item_categories: np.ndarray,
    k: int = 10,
) -> float:
    """Share of queries whose positive item ranks in the top k of its own category."""
    if len(pos) == 0:
        return float("nan")
    sims = query_emb @ item_emb.T
    same = item_categories[None, :] == item_categories[pos][:, None]
    sims = np.where(same, sims, -np.inf)
    pos_sim = sims[np.arange(len(pos)), pos]
    rank = (sims > pos_sim[:, None]).sum(axis=1)
    return float((rank < k).mean())
