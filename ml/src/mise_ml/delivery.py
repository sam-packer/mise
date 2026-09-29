"""Read and hash the files in a public bundle."""

import hashlib
import json
from pathlib import Path


def public_files(root: Path) -> dict[str, Path]:
    """Every file in the bundle folder; the folder holds only public files."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    files = {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*")) if p.is_file()}
    named = [manifest["encoder"]["model"], *(f["path"] for f in manifest["files"].values())]
    for name in named:
        if name not in files:
            raise SystemExit(f"missing bundle file: {name}")
    return files


def content_hash(files: dict[str, Path | bytes]) -> str:
    digest = hashlib.sha256()
    for name, body in sorted(files.items()):
        data = body.read_bytes() if isinstance(body, Path) else body
        digest.update(f"{name}\0{hashlib.sha256(data).hexdigest()}\n".encode())
    return digest.hexdigest()[:8]
