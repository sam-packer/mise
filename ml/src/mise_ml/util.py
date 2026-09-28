import hashlib
import json
import os
import random
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def dumps_line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"


def sort_jsonl(path: Path, key: str) -> None:
    """Rewrite a JSONL file sorted by one field, so git diffs stay small and stable."""
    if path.exists():
        rows = sorted(iter_jsonl(path), key=lambda r: (str(r[key]), dumps_line(r)))
        write_jsonl(path, rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps_line(row))
            n += 1
    return n


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps_line(row))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def slugify(text: str, max_len: int = 60) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text[:max_len].rstrip("-") or "untitled"


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def hash_fraction(text: str) -> float:
    """A deterministic value in [0, 1) for splits that survive reruns."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000


def make_deterministic(seed: int, warn_only: bool = False) -> None:
    """Fixed seeds and deterministic kernels. Call before the first CUDA operation."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def word_count(text: str) -> int:
    return len(text.split())
