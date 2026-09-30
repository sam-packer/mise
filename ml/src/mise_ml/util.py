"""Share safe file writes, stable hashes, and deterministic setup across pipeline steps."""

import hashlib
import json
import logging
import os
import random
import re
import tempfile
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger("mise_ml.util")


def atomic_write(path: Path, data: bytes) -> None:
    """Replace a file atomically so an interrupted write cannot leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


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


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read JSONL rows, or yield nothing if the file is missing.

    An interrupted append can leave a partial final line.
    Skip that line with a warning so a rerun can retry it.
    Raise an error for an invalid line anywhere else.
    """
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    lines = [line for line in lines if line]
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            if i < len(lines) - 1:
                raise
            log.warning(f"{path.name}: the last line is cut short (a killed run?); skip it")
            return
        yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    lines = [dumps_line(row) for row in rows]
    atomic_write(path, "".join(lines).encode("utf-8"))
    return len(lines)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                # A killed append left half a line. Cut it, so the next row starts clean.
                f.seek(0)
                data = f.read()
                f.truncate(data.rfind(b"\n") + 1)
                log.warning(f"{path.name}: removed a last line cut short by a killed run")
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps_line(row))


def write_json(path: Path, data: Any) -> None:
    atomic_write(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


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
