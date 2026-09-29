"""Prepare public bundle files and private catalog files for delivery."""

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from mise_ml.config import REPO_ROOT

CATALOG_TS = REPO_ROOT / "src" / "lib" / "server" / "catalog.ts"
PRIVATE_FILES = {"items.json", "vectors.bin", "anchors.json", "anchors.bin", "catalog.json"}


def check_binding(bucket: str) -> None:
    """Require the configured bucket to match the Worker binding."""
    text = (REPO_ROOT / "wrangler.jsonc").read_text(encoding="utf-8")
    text = re.sub(
        r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/',
        lambda match: match[0] if match[0].startswith('"') else "",
        text,
    )
    text = re.sub(r",\s*([}\]])", r"\1", text)
    bindings = json.loads(text).get("r2_buckets", [])
    binding = next((entry for entry in bindings if entry["binding"] == "CATALOG"), None)
    if not binding or binding.get("bucket_name") != bucket:
        raise SystemExit(
            f"set the CATALOG bucket_name in wrangler.jsonc to {bucket} before install or publish"
        )


def public_files(root: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if path.name in PRIVATE_FILES:
            raise SystemExit(f"private file in public bundle: {path}; build the bundle again")
        if name not in ("manifest.json", "vocab.json") and name.split("/")[0] not in (
            "model",
            "img",
        ):
            raise SystemExit(f"unexpected public bundle file: {path}")
        files[name] = path
    return files


def catalog_files(root: Path, manifest: dict) -> dict[str, Path | bytes]:
    path = root / "catalog.json"
    if not path.exists():
        raise SystemExit(f"no catalog at {root}; build and install the bundle and catalog first")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if (
        meta.get("version") != manifest["version"]
        or meta["dims"] != manifest["encoder"]["dims"]
        or meta["counts"] != manifest["counts"]
        or meta["heads"] != manifest["heads"]
    ):
        raise SystemExit("the public bundle and private catalog do not match; install them again")
    names = {"catalog.json", "items.json", "vectors.bin"}
    if meta["heads"]["kind"] == "anchors":
        names.update(("anchors.json", "anchors.bin"))
    files: dict[str, Path | bytes] = {}
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            raise SystemExit(f"missing catalog file: {path}")
        files[name] = path
    return files


def content_hash(files: dict[str, Path | bytes]) -> str:
    digest = hashlib.sha256()
    for name, body in sorted(files.items()):
        data = body.read_bytes() if isinstance(body, Path) else body
        digest.update(f"{name}\0{hashlib.sha256(data).hexdigest()}\n".encode())
    return digest.hexdigest()[:8]


def write_catalog_ts(prefix: str) -> bool:
    content = (
        "// Run `uv run publish` to set the published catalog prefix.\n"
        f"export const CATALOG_PREFIX = '{prefix}';\n"
    )
    if CATALOG_TS.exists() and CATALOG_TS.read_text(encoding="utf-8") == content:
        return False
    CATALOG_TS.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_TS.write_text(content, encoding="utf-8")
    return True


def install_local_catalog(files: dict[str, Path | bytes], prefix: str) -> None:
    """Put the exact catalog bytes in local R2 and set the development override."""
    bun = shutil.which("bun")
    if not bun:
        raise SystemExit("install Bun before installing the local catalog")
    with TemporaryDirectory(prefix="mise-catalog-") as directory:
        for name, body in files.items():
            target = Path(directory) / name
            if isinstance(body, Path):
                shutil.copyfile(body, target)
            else:
                target.write_bytes(body)
        subprocess.run(
            [
                bun,
                "--no-env-file",
                str(REPO_ROOT / "scripts" / "local-catalog.ts"),
                directory,
                prefix,
            ],
            cwd=REPO_ROOT,
            check=True,
        )
