"""Copy out/bundle/ into the web app at ../static/bundle/."""

import json
import shutil
from collections import Counter

from mise_ml.config import BUNDLE, REPO_ROOT

TARGET = REPO_ROOT / "static" / "bundle"


def run() -> None:
    manifest_path = BUNDLE / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no bundle at {BUNDLE}; run `uv run mise-ml export` first")
    # Copy next to the target first, then swap, so a failed copy never leaves half a bundle.
    staging = TARGET.with_name("bundle.new")
    previous = TARGET.with_name("bundle.old")
    for path in (staging, previous):
        if path.exists():
            shutil.rmtree(path)
    shutil.copytree(BUNDLE, staging)
    if TARGET.exists():
        TARGET.rename(previous)
    staging.rename(TARGET)
    if previous.exists():
        shutil.rmtree(previous)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = json.loads((TARGET / "items.json").read_text(encoding="utf-8"))
    counts = Counter(it["category"] for it in items)
    print(f"installed bundle {manifest['version']} -> {TARGET}")
    print("items: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
