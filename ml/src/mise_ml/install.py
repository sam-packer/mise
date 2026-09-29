"""Install the public bundle and put the private catalog in local R2."""

import json
import shutil
from collections import Counter
from datetime import UTC, datetime

from mise_ml import keys
from mise_ml.config import BUNDLE, CATALOG_BUNDLE, INSTALLED_CATALOG, REPO_ROOT
from mise_ml.delivery import (
    catalog_files,
    check_binding,
    content_hash,
    install_local_catalog,
    public_files,
)
from mise_ml.log import get, num, progress

log = get(__name__)

TARGET = REPO_ROOT / "static" / "bundle"


def run() -> None:
    manifest_path = BUNDLE / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no bundle at {BUNDLE}; run `uv run mise-ml export` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog = catalog_files(CATALOG_BUNDLE, manifest)
    files = list(public_files(BUNDLE).values())
    bucket = keys.catalog_bucket()
    check_binding(bucket)
    prefix = f"catalog/{datetime.now(UTC):%Y-%m-%d}-{content_hash(catalog)}/"
    # Copy next to the target first, then swap, so a failed copy never leaves half a bundle.
    staging = TARGET.with_name("bundle.new")
    previous = TARGET.with_name("bundle.old")
    for path in (staging, previous):
        if path.exists():
            shutil.rmtree(path)
    size = sum(p.stat().st_size for p in files)
    log.info(f"copying {num(len(files))} files ({size / 2**20:.0f} MiB) to {staging}")
    for src in progress(files, desc="install", unit="file"):
        dest = staging / src.relative_to(BUNDLE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    if TARGET.exists():
        TARGET.rename(previous)
    staging.rename(TARGET)
    if previous.exists():
        shutil.rmtree(previous)
    if INSTALLED_CATALOG.exists():
        shutil.rmtree(INSTALLED_CATALOG)
    shutil.copytree(CATALOG_BUNDLE, INSTALLED_CATALOG)
    install_local_catalog(catalog, prefix)

    items = json.loads((INSTALLED_CATALOG / "items.json").read_text(encoding="utf-8"))
    counts = Counter(it["category"] for it in items)
    log.info(
        f"installed bundle {manifest['version']}: {num(len(items))} items "
        f"({', '.join(f'{k} {num(v)}' for k, v in sorted(counts.items()))}) -> {TARGET}"
    )
