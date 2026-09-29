"""Read and hash the files in a public bundle."""

import hashlib
import json
from pathlib import Path


def public_files(root: Path) -> dict[str, Path]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    names = {"manifest.json", *manifest["assets"], *(f["path"] for f in manifest["files"].values())}
    files = {name: root / name for name in sorted(names)}
    for name, path in files.items():
        if not path.is_file():
            raise SystemExit(f"missing bundle file: {name}")
    return files


def content_hash(files: dict[str, Path | bytes]) -> str:
    digest = hashlib.sha256()
    for name, body in sorted(files.items()):
        data = body.read_bytes() if isinstance(body, Path) else body
        digest.update(f"{name}\0{hashlib.sha256(data).hexdigest()}\n".encode())
    return digest.hexdigest()[:8]
