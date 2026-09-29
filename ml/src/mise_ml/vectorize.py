"""Publish versioned search data through the user's Wrangler login."""

import json
import os
import shutil
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from botocore.exceptions import ClientError

from mise_ml.config import REPO_ROOT
from mise_ml.delivery import content_hash, wrangler_config
from mise_ml.log import get

log = get(__name__)
INDEX = "mise-catalog"
METADATA_INDEXES = {"category": "string", "creator": "number", "title": "number"}
MARKER = "vectorize-v1.json"


class Vectorize:
    def __init__(self, account_id: str):
        bun = shutil.which("bun")
        if not bun:
            raise SystemExit("install Bun before publishing the catalog")
        self.command = [bun, "x", "wrangler", "vectorize"]
        self.env = dict(os.environ, CLOUDFLARE_ACCOUNT_ID=account_id, NO_COLOR="1")

    def run(self, *args: str) -> Any:
        for attempt in range(3):
            result = subprocess.run(
                [*self.command, *args],
                cwd=REPO_ROOT,
                env=self.env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            if (
                not result.returncode
                or args[0] != "get-vectors"
                or "[code: 10000]" not in result.stderr
            ):
                break
            if attempt < 2:
                log.warning("Wrangler rejected a read token; retry with a fresh Wrangler process")
                time.sleep(1)
        if result.returncode:
            raise SystemExit(f"Wrangler {args[0]} failed:\n{result.stderr}\n{result.stdout}")
        # get-vectors and query have no --json flag. Skip their Wrangler banner.
        lines = result.stdout.splitlines()
        start = next((i for i, line in enumerate(lines) if line in ("{", "[")), None)
        if start is None:
            return None
        return json.loads("\n".join(lines[start:]))

    def check(self) -> None:
        binding = next(
            (b for b in wrangler_config().get("vectorize", []) if b["binding"] == "CATALOG_SEARCH"),
            None,
        )
        if not binding or binding.get("index_name") != INDEX:
            raise SystemExit(f"set the CATALOG_SEARCH binding to {INDEX} in wrangler.jsonc")
        live = self.run("get", INDEX, "--json")
        if live["config"] != {"dimensions": 384, "metric": "cosine"}:
            raise SystemExit(
                f"Vectorize configuration differs: {live['config']}; expected 384/cosine"
            )
        indexes = self.run("list-metadata-index", INDEX, "--json")
        actual = {entry["propertyName"]: entry["indexType"].lower() for entry in indexes}
        if any(actual.get(key) != kind for key, kind in METADATA_INDEXES.items()):
            raise SystemExit(
                f"Vectorize metadata indexes differ: {actual}; expected {METADATA_INDEXES}. "
                "Check the live state before running the one-time commands in ml/README.md."
            )

    def get(self, ids: list[str]) -> list[dict]:
        return self.run("get-vectors", INDEX, "--ids", *ids) or []

    def verify(self, batches: list[Path], namespace: str) -> bool:
        """Read every uploaded record. An accepted mutation is not proof of visibility."""
        probes: dict[str, str] = {}
        chunks: list[list[dict]] = []

        def matches(rows: list[dict]) -> bool:
            expected = {row["id"]: row for row in rows}
            actual = self.get(list(expected))
            return len(actual) == len(expected) and all(
                row["id"] in expected
                and row.get("metadata") == expected[row["id"]]["metadata"]
                and len(row.get("values", [])) == 384
                and struct.pack("<384f", *row["values"])
                == struct.pack("<384f", *expected[row["id"]]["values"])
                and row.get("namespace") == namespace
                for row in actual
            )

        for batch in batches:
            rows = [json.loads(line) for line in batch.read_text(encoding="utf-8").splitlines()]
            if not matches([rows[-1]]):
                return False
            chunks.extend(rows[start : start + 20] for start in range(0, len(rows), 20))
            for row in rows:
                probes.setdefault(row["metadata"]["category"], row["id"])
        with ThreadPoolExecutor(max_workers=6) as pool:
            if not all(pool.map(matches, chunks)):
                return False
        for category, vector_id in probes.items():
            result = self.run(
                "query",
                INDEX,
                "--vector-id",
                vector_id,
                "--namespace",
                namespace,
                "--filter",
                json.dumps({"category": category}),
                "--top-k",
                "1",
                "--return-values",
                "--return-metadata",
                "all",
            )
            if not result or not result.get("matches"):
                return False
        return True


def publish_search(
    vectorize: Vectorize,
    s3: Any,
    bucket: str,
    files: dict[str, Path | bytes],
    prefix: str,
) -> bytes:
    """Upload and verify the namespace before recording completion in private R2."""
    expected = {"index": INDEX, "namespace": prefix, "hash": content_hash(files)}
    try:
        obj = s3.get_object(Bucket=bucket, Key=prefix + MARKER)
        marker = json.loads(obj["Body"].read())
    except ClientError as error:
        if error.response["Error"]["Code"] not in ("NoSuchKey", "404"):
            raise
        marker = None
    if marker is not None and marker != expected:
        raise SystemExit(f"Vectorize completion record differs at {bucket}/{prefix}{MARKER}")

    with TemporaryDirectory(prefix="mise-vectorize-") as directory:
        root = Path(directory)
        for name in ("items.json", "vectors.bin", "catalog.json"):
            body = files[name]
            (root / name).write_bytes(body.read_bytes() if isinstance(body, Path) else body)
        output = root / "prepared"
        subprocess.run(
            [
                vectorize.command[0],
                "--no-env-file",
                str(REPO_ROOT / "scripts/vectorize-catalog.ts"),
                str(root),
                prefix,
                str(output),
            ],
            cwd=REPO_ROOT,
            env=vectorize.env,
            check=True,
        )
        upload = json.loads((output / "upload.json").read_text(encoding="utf-8"))
        batches = [output / name for name in upload["batches"]]
        if marker is not None:
            if not vectorize.verify(batches, prefix):
                raise SystemExit("the completed Vectorize namespace differs from the catalog; stop")
            log.info("Vectorize %s is already uploaded and verified, reuse", prefix)
        else:
            for batch in batches:
                result = vectorize.run(
                    "upsert", INDEX, "--file", str(batch), "--batch-size", "5000", "--json"
                )
                log.info("Vectorize accepted %s vectors", result["count"])
            deadline = time.monotonic() + 900
            while not vectorize.verify(batches, prefix):
                if time.monotonic() >= deadline:
                    raise SystemExit(
                        "Vectorize is not fully visible after 15 minutes; rerun publish"
                    )
                log.info("wait for Vectorize records and category filters to become visible")
                time.sleep(15)
            s3.put_object(
                Bucket=bucket,
                Key=prefix + MARKER,
                Body=json.dumps(expected, separators=(",", ":")).encode(),
                ContentType="application/json",
            )
        return (output / "search-index.json").read_bytes()
