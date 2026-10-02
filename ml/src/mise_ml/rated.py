"""Score a model on rated pairs: a feeling, a work, and a 0-3 rating from careful raters.

The tuning set picks every setting, epoch and recipe. The gold set is the final check only, so
no step selects by it. A pair fits when its rating (the mean of the raters) is 2 or higher.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mise_ml.data import Catalog
from mise_ml.util import iter_jsonl

FIT = 2.0


@dataclass
class RatedSet:
    feelings: list[str]
    feeling: np.ndarray  # per pair: the row in feelings
    item: np.ndarray  # per pair: the catalog row
    rating: np.ndarray


def load_rated(path: Path, catalog: Catalog) -> RatedSet:
    """Read the pairs whose work is in the catalog."""
    rows = [r for r in iter_jsonl(path) if r["item_id"] in catalog.index]
    if not rows:
        raise SystemExit(f"no rated pairs in {path.name} match the catalog")
    feelings = sorted({r["feeling"] for r in rows})
    row = {f: i for i, f in enumerate(feelings)}
    return RatedSet(
        feelings,
        np.array([row[r["feeling"]] for r in rows]),
        np.array([catalog.index[r["item_id"]] for r in rows]),
        np.array([r["rating"] for r in rows], dtype=np.float64),
    )


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks from 0, where tied values share the mean of their ranks."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(inverse, ranks) / counts)[inverse]


def within_auc(rs: RatedSet, sims: np.ndarray) -> float:
    """The share of (fit, non-fit) pairs with the same feeling that the model orders correctly.
    Ties count half. A score offset per feeling does not change it."""
    fit = rs.rating >= FIT
    right = total = 0.0
    for f in np.unique(rs.feeling):
        m = rs.feeling == f
        pos, neg = sims[m & fit], sims[m & ~fit]
        right += (pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()
        total += len(pos) * len(neg)
    return right / total if total else float("nan")


def rated_scores(rs: RatedSet, query_emb: np.ndarray, item_emb: np.ndarray) -> dict[str, float]:
    """Within-feeling AUC, Spearman over all pairs, and their mean as the objective.

    query_emb holds one row per rs.feelings; item_emb holds one row per catalog item.
    """
    sims = (query_emb[rs.feeling] * item_emb[rs.item]).sum(1)
    if not np.isfinite(sims).all():
        # A diverged model never wins a selection.
        return {"auc": float("nan"), "spearman": float("nan"), "objective": float("-inf")}
    auc = within_auc(rs, sims)
    spearman = float(np.corrcoef(average_ranks(rs.rating), average_ranks(sims))[0, 1])
    objective = (auc + spearman) / 2
    return {
        "auc": auc,
        "spearman": spearman,
        "objective": objective if np.isfinite(objective) else float("-inf"),
    }


def brief(scores: dict[str, float]) -> str:
    return f"{scores['objective']:.4f} (AUC {scores['auc']:.3f}, Spearman {scores['spearman']:.3f})"
