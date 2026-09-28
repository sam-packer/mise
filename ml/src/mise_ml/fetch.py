"""Download the raw sources listed in sources.toml and verify their checksums."""

import time
import tomllib
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

from mise_ml.config import ML_ROOT, RAW, SOURCES, CurateConfig, ResolveConfig
from mise_ml.http import USER_AGENT, CachedClient, FetchError
from mise_ml.log import elapsed, get, progress
from mise_ml.util import sha256_file, write_jsonl

log = get(__name__)

POETRYDB = "https://poetrydb.org"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    path: str
    license: str
    kind: str = "file"
    sha256: str = ""
    bytes: int = 0
    skip_if: str = ""
    note: str = ""

    @property
    def dest(self) -> Path:
        return RAW / self.path


def load_sources() -> list[Source]:
    with SOURCES.open("rb") as f:
        return [Source(**s) for s in tomllib.load(f)["source"]]


def verify(src: Source, path: Path) -> None:
    size, digest = path.stat().st_size, sha256_file(path)
    if (size, digest) != (src.bytes, src.sha256):
        log.error("CHECKSUM MISMATCH for %s (%s)", src.name, path)
        log.error("  expected %s (%d bytes)", src.sha256, src.bytes)
        log.error("  got      %s (%d bytes)", digest, size)
        log.error("the source changed upstream; review the new file, then update sources.toml")
        raise SystemExit(1)
    log.debug("ok %s sha256 %s", src.path, digest)


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    log.info("downloading %s", dest.relative_to(RAW).as_posix())
    try:
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=120, headers={"User-Agent": USER_AGENT}
        ) as resp:
            if resp.status_code != 200:
                log.warning("download failed: %s: HTTP %d", url, resp.status_code)
                return False
            total = int(resp.headers.get("content-length", 0)) or None
            with (
                part.open("wb") as f,
                progress(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
            ):
                for chunk in resp.iter_bytes(1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
    except httpx.HTTPError as e:
        log.warning("download failed: %s: %s", url, e)
        part.unlink(missing_ok=True)
        return False
    part.replace(dest)
    return True


def poems_by_title(client: CachedClient, author: str) -> list[dict]:
    """Fallback for authors whose full list fails (PoetryDB answers 503 for large ones).

    It asks for the titles, then gets each poem in the curation length range on its own.
    """
    cfg = CurateConfig()
    listing = client.get_json(f"{POETRYDB}/author/{quote(author)}/title,linecount") or []
    poems = []
    for entry in listing if isinstance(listing, list) else []:
        if not cfg.poem_min_lines <= int(entry["linecount"]) <= cfg.poem_max_lines:
            continue
        query = f"{quote(author, safe='')}:abs;{quote(entry['title'], safe='')}:abs"
        body = client.get_json(f"{POETRYDB}/author,title/{query}")
        if isinstance(body, list):
            poems.extend(p for p in body if p["author"] == author)
    return poems


def snapshot_poetrydb(dest: Path) -> None:
    """Build the PoetryDB file from the API in a fixed order. PoetryDB has no file to hash."""
    client = CachedClient(ResolveConfig())
    authors = sorted((client.get_json(f"{POETRYDB}/author") or {}).get("authors", []))
    poems: list[dict] = []
    log.info("building the PoetryDB snapshot from the API (%d authors)", len(authors))
    fallbacks = 0
    for author in progress(authors, desc="poetrydb", unit="author"):
        try:
            body = client.get_json(f"{POETRYDB}/author/{quote(author)}")
        except FetchError:
            log.debug("poetrydb: full list failed for %s; fetching poems by title", author)
            fallbacks += 1
            body = poems_by_title(client, author)
        if isinstance(body, list):
            poems.extend(body)
    client.close()
    unique = {(p["author"], p["title"], "\n".join(p["lines"])): p for p in poems}
    rows = [unique[k] for k in sorted(unique)]
    write_jsonl(dest, rows)
    log.info(
        "poetrydb: %d poems by %d authors (%d authors fetched title by title)",
        len(rows),
        len(authors),
        fallbacks,
    )


def fetch(src: Source) -> str:
    """Make one source present and verified. Returns what happened, for the summary."""
    dest = src.dest
    if src.skip_if and (RAW / src.skip_if).exists():
        log.info("skip %s: %s is present", src.name, src.skip_if)
        return "skipped"
    status = "verified"
    if not dest.exists():
        if src.kind == "snapshot":
            snapshot_poetrydb(dest)
        elif not download(src.url, dest):
            return "failed"
        status = "downloaded"
    if src.sha256:
        verify(src, dest)
    elif src.kind == "snapshot":
        status = "snapshot"
    else:
        status = "unchecked"
    if src.kind == "zip" and not (dest.parent / dest.stem).exists():
        log.info("extracting %s", dest.name)
        with zipfile.ZipFile(dest) as z:
            z.extractall(dest.parent)
    return status


def run() -> None:
    start = time.perf_counter()
    sources = load_sources()
    have = sum(1 for s in sources if s.dest.exists())
    log.info(
        f"reading {SOURCES.name}: {len(sources)} sources ({have} present, "
        f"{len(sources) - have} to download)"
    )
    results: dict[str, list[Source]] = defaultdict(list)
    for src in progress(sources, desc="fetch", unit="source"):
        results[fetch(src)].append(src)
    size = sum(s.dest.stat().st_size for s in sources if s.dest.exists())
    checked = len(results["verified"]) + len(results["downloaded"])
    log.info(
        f"done in {elapsed(start)}: {checked} files match their SHA-256 "
        f"({len(results['downloaded'])} new), {len(results['snapshot'])} API snapshot, "
        f"{size / 1e9:.2f} GB of source files in {RAW.relative_to(ML_ROOT).as_posix()}"
    )
    for src in results["failed"]:
        log.warning(f"could not get {src.name}. {' '.join(src.note.split())}")
    if results["failed"]:
        raise SystemExit(1)
