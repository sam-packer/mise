"""Upload public assets and the private catalog to separate R2 buckets.

Store images under img/<hash20>.webp and public files under bundles/<date>-<hash8>/.
Store the private catalog under catalog/<date>-<hash8>/ in its own bucket.
Set each item's image URL to the public image URL before hashing the catalog.
"""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from mise_ml import keys
from mise_ml.config import INSTALLED_CATALOG, REPO_ROOT
from mise_ml.delivery import (
    catalog_files,
    check_binding,
    content_hash,
    install_local_catalog,
    public_files,
    write_catalog_ts,
)
from mise_ml.install import TARGET
from mise_ml.log import elapsed, get, num, progress
from mise_ml.threads import run_all

log = get(__name__)

BUNDLE_TS = REPO_ROOT / "src" / "lib" / "bundle.ts"
BUNDLE_TS_COMMENT = "// Run `uv run publish` to set the public bundle URL."
WORKERS = 16
IMMUTABLE = "public, max-age=31536000, immutable"
TYPES = {
    ".json": "application/json",
    ".onnx": "application/octet-stream",
    ".bin": "application/octet-stream",
    ".txt": "text/plain; charset=utf-8",
    ".webp": "image/webp",
}
CORS_RULE = {
    "AllowedOrigins": [
        "https://mise.art",
        "https://www.mise.art",
        "http://localhost:5173",
        "http://localhost:4173",
    ],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 86400,
}


def content_type(key: str) -> str:
    return TYPES.get(Path(key).suffix, "application/octet-stream")


def client(cfg: dict[str, str]) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=f"https://{cfg['account_id']}.r2.cloudflarestorage.com",
        region_name="auto",
        aws_access_key_id=cfg["access_key_id"],
        aws_secret_access_key=cfg["secret_access_key"],
        config=Config(max_pool_connections=WORKERS + 4, retries={"mode": "standard"}),
    )


def require_buckets(s3: Any, cfg: dict[str, str]) -> None:
    """Name each missing or unreachable bucket before any upload starts."""
    problems = []
    for key, role in (("bucket", "public bundle"), ("catalog_bucket", "private catalog")):
        name = cfg[key]
        try:
            s3.head_bucket(Bucket=name)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket", "NotFound"):
                problems.append(
                    f"the {role} bucket {name!r} does not exist; create it with "
                    f"`bunx wrangler r2 bucket create {name}`"
                )
            else:
                problems.append(
                    f"the {role} bucket {name!r} is not reachable ({code}); give the R2 API "
                    "token object read and write access to it"
                )
    for problem in problems:
        log.error(problem)
    if problems:
        raise SystemExit(1)


def existing(s3: Any, bucket: str, prefix: str) -> set[str]:
    keys_ = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys_.update(obj["Key"] for obj in page.get("Contents", []))
    return keys_


def existing_prefix(s3: Any, bucket: str, root: str, hash8: str) -> str | None:
    """Find the prefix for this content hash, from any date."""
    suffix = f"-{hash8}/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=root, Delimiter="/"):
        for common in page.get("CommonPrefixes", []):
            if common["Prefix"].endswith(suffix):
                return common["Prefix"]
    return None


def apply_cors(s3: Any, bucket: str) -> None:
    """Add the CORS rule unless a rule already covers it. Other rules stay as they are."""
    paste = json.dumps([CORS_RULE], indent=2)
    try:
        try:
            rules = s3.get_bucket_cors(Bucket=bucket)["CORSRules"]
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchCORSConfiguration":
                raise
            rules = []
        for rule in rules:
            if set(CORS_RULE["AllowedOrigins"]) <= set(rule.get("AllowedOrigins", [])) and set(
                CORS_RULE["AllowedMethods"]
            ) <= set(rule.get("AllowedMethods", [])):
                log.info("CORS: the bucket already allows the app origins")
                return
        s3.put_bucket_cors(Bucket=bucket, CORSConfiguration={"CORSRules": [*rules, CORS_RULE]})
        log.info(f"CORS: added a rule for {', '.join(CORS_RULE['AllowedOrigins'])}")
    except ClientError as e:
        log.warning(
            f"CORS: cannot read or set the bucket CORS policy ({e.response['Error']['Code']}). "
            f"Paste this into R2 > {bucket} > Settings > CORS policy:\n{paste}"
        )


def write_bundle_ts(url: str) -> bool:
    """Rewrite src/lib/bundle.ts to point at the published bundle. False if it already did."""
    content = f"{BUNDLE_TS_COMMENT}\nexport const BUNDLE_URL = '{url}';\n"
    if BUNDLE_TS.exists() and BUNDLE_TS.read_text(encoding="utf-8") == content:
        return False
    BUNDLE_TS.write_text(content, encoding="utf-8")
    return True


def upload(s3: Any, bucket: str, jobs: list[tuple[str, Path | bytes]], desc: str) -> int:
    """Put each (key, file or bytes) with its Content-Type. Returns the bytes sent."""

    def put(job: tuple[str, Path | bytes]) -> int:
        key, body = job
        data = body.read_bytes() if isinstance(body, Path) else body
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type(key),
            CacheControl=IMMUTABLE,
        )
        log.debug("put %s (%s bytes)", key, num(len(data)))
        return len(data)

    bar = progress(total=len(jobs), desc=desc, unit="file")

    def put_one(job: tuple[str, Path | bytes]) -> int:
        size = put(job)
        bar.update()
        return size

    sent = sum(run_all(put_one, jobs, WORKERS))
    bar.close()
    return sent


def run() -> None:
    start = time.perf_counter()
    cfg = keys.r2()
    check_binding(cfg["catalog_bucket"])
    if cfg["catalog_bucket"] == cfg["bucket"]:
        raise SystemExit("the public bundle and private catalog need separate R2 buckets")
    manifest_path = TARGET / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no bundle at {TARGET}; run `uv run train` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    private = catalog_files(INSTALLED_CATALOG, manifest)
    public = public_files(TARGET)
    items = json.loads((INSTALLED_CATALOG / "items.json").read_text(encoding="utf-8"))

    # Images: key by content, and point items.json at the public URL.
    images: dict[str, Path] = {}
    by_file: dict[Path, str] = {}
    for item in progress(items, desc="hash images", unit="item"):
        if not item["image"]:
            continue
        path = TARGET / "img" / item["image"]["src"].rsplit("/", 1)[-1]
        if path not in by_file:
            if not path.exists():
                raise SystemExit(f"{item['id']}: image {path} is missing; run uv run train again")
            by_file[path] = f"img/{hashlib.sha256(path.read_bytes()).hexdigest()[:20]}.webp"
        key = by_file[path]
        images[key] = path
        item["image"]["src"] = f"{cfg['public_url']}/{key}"

    # Keep catalog data out of the public bucket.
    files: dict[str, Path | bytes] = {
        name: path for name, path in public.items() if not name.startswith("img/")
    }
    private["items.json"] = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode()
    hash8 = content_hash(files)
    catalog_hash = content_hash(private)

    s3 = client(cfg)
    require_buckets(s3, cfg)
    apply_cors(s3, cfg["bucket"])

    reused = existing_prefix(s3, cfg["bucket"], "bundles/", hash8)
    if reused:
        prefix = reused
        log.info(f"bundle {prefix} is already online, reuse")
    else:
        date = datetime.now(UTC).strftime("%Y-%m-%d")
        prefix = f"bundles/{date}-{hash8}/"
    catalog_prefix = existing_prefix(s3, cfg["catalog_bucket"], "catalog/", catalog_hash)
    if catalog_prefix is None:
        catalog_prefix = f"catalog/{datetime.now(UTC):%Y-%m-%d}-{catalog_hash}/"
    log.info(
        f"bundle {manifest['version']} -> {prefix.removeprefix('bundles/').rstrip('/')}: "
        f"{num(len(files))} files, {num(len(images))} images, bucket {cfg['bucket']}"
    )

    online = existing(s3, cfg["bucket"], "img/")
    img_jobs = [(k, p) for k, p in sorted(images.items()) if k not in online]
    log.info(
        f"images: {num(len(images) - len(img_jobs))} already online, {num(len(img_jobs))} to upload"
    )
    img_bytes = upload(s3, cfg["bucket"], img_jobs, "images")

    online = existing(s3, cfg["bucket"], prefix)
    file_jobs = [(prefix + n, b) for n, b in files.items() if prefix + n not in online]
    if not file_jobs and not reused:
        log.info(f"bundle {prefix} is already online with the same files, skip")
    # The manifest goes last, so a prefix with a manifest always has all its files.
    manifest_key = prefix + "manifest.json"
    first = [job for job in file_jobs if job[0] != manifest_key]
    file_bytes = upload(s3, cfg["bucket"], first, "bundle")
    file_bytes += upload(
        s3, cfg["bucket"], [j for j in file_jobs if j[0] == manifest_key], "manifest"
    )
    online = existing(s3, cfg["catalog_bucket"], catalog_prefix)
    catalog_jobs = [
        (catalog_prefix + name, body)
        for name, body in private.items()
        if catalog_prefix + name not in online
    ]
    catalog_key = catalog_prefix + "catalog.json"
    catalog_bytes = upload(
        s3,
        cfg["catalog_bucket"],
        [job for job in catalog_jobs if job[0] != catalog_key],
        "catalog",
    )
    catalog_bytes += upload(
        s3,
        cfg["catalog_bucket"],
        [job for job in catalog_jobs if job[0] == catalog_key],
        "catalog metadata",
    )
    log.info(
        f"uploaded {num(len(catalog_jobs))} private catalog files "
        f"({catalog_bytes / 2**20:.1f} MiB) to {cfg['catalog_bucket']}/{catalog_prefix}"
    )

    log.info(
        f"uploaded {num(len(img_jobs))} images and {num(len(file_jobs))} bundle files "
        f"({(img_bytes + file_bytes) / 2**20:.1f} MiB); skipped "
        f"{num(len(images) - len(img_jobs))} images and {num(len(files) - len(file_jobs))} "
        f"bundle files already online; took {elapsed(start)}"
    )
    install_local_catalog(private, catalog_prefix)
    url = f"{cfg['public_url']}/{prefix}"
    if write_bundle_ts(url):
        log.info("updated src/lib/bundle.ts")
    else:
        log.info(f"src/lib/bundle.ts already points at {url}")
    write_catalog_ts(catalog_prefix)
    log.info("commit src/lib/bundle.ts and src/lib/server/catalog.ts, then run bun run deploy")
