import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from mise_ml.config import CACHE, ResolveConfig
from mise_ml.util import stable_hash


class FetchError(RuntimeError):
    pass


USER_AGENT = "mise-catalog/0.1 (class project; +https://github.com/sam-packer/mise)"


class RateLimiter:
    def __init__(self, intervals: dict[str, float], default: float) -> None:
        self.intervals = intervals
        self.default = default
        self.next_at: dict[str, float] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.guard = threading.Lock()

    def wait(self, host: str) -> None:
        with self.guard:
            lock = self.locks.setdefault(host, threading.Lock())
        with lock:
            now = time.monotonic()
            at = self.next_at.get(host, 0.0)
            if at > now:
                time.sleep(at - now)
            self.next_at[host] = time.monotonic() + self.intervals.get(host, self.default)


class CachedClient:
    """GET with an on-disk response cache and per-host rate limits.

    A cached response never touches the network, so reruns are fast and free.
    """

    def __init__(self, cfg: ResolveConfig, cache_dir: Path = CACHE / "http") -> None:
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.limiter = RateLimiter(cfg.min_interval, cfg.default_interval)
        self.hits = 0
        self.requests = 0
        self.client = httpx.Client(
            timeout=cfg.timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )

    def _path(self, url: str, suffix: str) -> Path:
        host = urlsplit(url).hostname or "unknown"
        return self.cache_dir / host / f"{stable_hash(url, 24)}{suffix}"

    def _fetch(self, url: str) -> httpx.Response:
        host = urlsplit(url).hostname or ""
        for attempt in range(self.cfg.retries):
            self.limiter.wait(host)
            try:
                resp = self.client.get(url)
            except httpx.HTTPError:
                time.sleep(2**attempt)
                continue
            # 403 too: bot protection (the Met) answers 403 for a while under load.
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                retry_after = resp.headers.get("retry-after", "")
                time.sleep(float(retry_after) if retry_after.isdigit() else 5 * 2**attempt)
                continue
            return resp
        raise FetchError(url)

    def get_json(self, url: str) -> Any | None:
        """The JSON body, or None for a 404. Anything else raises FetchError.

        Only real answers (JSON with 200, or 404) are cached. A block page, a non-JSON 200,
        or any other status raises, so the caller counts an error and a rerun retries it.
        """
        path = self._path(url, ".json")
        if path.exists():
            self.hits += 1
            cached = json.loads(path.read_text(encoding="utf-8"))
            return cached["body"]
        self.requests += 1
        resp = self._fetch(url)
        if resp.status_code == 404:
            body = None
        elif resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError as e:
                raise FetchError(f"{url}: HTTP 200 without JSON") from e
        else:
            raise FetchError(f"{url}: HTTP {resp.status_code}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"url": url, "status": resp.status_code, "body": body}),
            encoding="utf-8",
        )
        return body

    def get_bytes(self, url: str) -> bytes | None:
        """Raw bytes, or None for a 404. Anything else raises FetchError.

        Only 404s are cached; the caller stores what it keeps.
        """
        miss = self._path(url, ".404")
        if miss.exists():
            return None
        resp = self._fetch(url)
        if resp.status_code == 404:
            miss.parent.mkdir(parents=True, exist_ok=True)
            miss.write_text("")
            return None
        if resp.status_code != 200 or not resp.content:
            raise FetchError(f"{url}: HTTP {resp.status_code}, {len(resp.content)} bytes")
        return resp.content

    def close(self) -> None:
        self.client.close()
