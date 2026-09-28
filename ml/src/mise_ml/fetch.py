"""Download the raw sources listed in sources.toml and verify their checksums."""

import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx
from tqdm import tqdm

from mise_ml.config import RAW, SOURCES, CurateConfig, ResolveConfig
from mise_ml.http import USER_AGENT, CachedClient, FetchError
from mise_ml.util import sha256_file, write_jsonl

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
        raise SystemExit(
            f"\nCHECKSUM MISMATCH for {src.name} ({path})\n"
            f"  expected {src.sha256} ({src.bytes} bytes)\n"
            f"  got      {digest} ({size} bytes)\n"
            "The source changed upstream. Review the new file, then update sources.toml."
        )


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=120, headers={"User-Agent": USER_AGENT}
        ) as resp:
            if resp.status_code != 200:
                print(f"failed {url}: HTTP {resp.status_code}")
                return False
            total = int(resp.headers.get("content-length", 0)) or None
            with (
                part.open("wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
            ):
                for chunk in resp.iter_bytes(1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
    except httpx.HTTPError as e:
        print(f"failed {url}: {e}")
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
    for author in tqdm(authors, desc="poetrydb"):
        try:
            body = client.get_json(f"{POETRYDB}/author/{quote(author)}")
        except FetchError:
            body = poems_by_title(client, author)
        if isinstance(body, list):
            poems.extend(body)
    client.close()
    unique = {(p["author"], p["title"], "\n".join(p["lines"])): p for p in poems}
    rows = [unique[k] for k in sorted(unique)]
    write_jsonl(dest, rows)
    print(f"poetrydb: {len(rows)} poems by {len(authors)} authors")


def fetch(src: Source) -> bool:
    """Make one source present and verified. False when it needs a manual download."""
    dest = src.dest
    if src.skip_if and (RAW / src.skip_if).exists():
        print(f"skip {src.name}: {src.skip_if} is present")
        return True
    if not dest.exists():
        if src.kind == "manual":
            return False
        if src.kind == "snapshot":
            snapshot_poetrydb(dest)
        elif not download(src.url, dest):
            return False
    if src.sha256:
        verify(src, dest)
        print(f"ok {src.path}")
    elif src.kind == "snapshot":
        print(f"ok {src.path} (API snapshot; run.json records its content hash)")
    else:
        print(f"present {src.path} (optional manual source, no checksum)")
    if src.kind == "zip" and not (dest.parent / dest.stem).exists():
        print(f"extracting {dest.name}")
        with zipfile.ZipFile(dest) as z:
            z.extractall(dest.parent)
    return True


def run() -> None:
    missing = [src for src in load_sources() if not fetch(src)]
    if missing:
        print("\nManual downloads:")
        for src in missing:
            print(f"- {src.name}: {src.note.strip()}")
