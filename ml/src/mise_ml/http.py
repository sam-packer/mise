import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx

from mise_ml.config import CACHE, ResolveConfig
from mise_ml.log import get, num
from mise_ml.util import atomic_write, stable_hash

log = get(__name__)


class FetchError(RuntimeError):
    pass


USER_AGENT = "mise-catalog/0.1 (class project; +https://github.com/sam-packer/mise)"
# Open Library allows 3 requests/s instead of 1 when the User-Agent carries a contact email.
# MusicBrainz and Wikimedia ask for a contact in the User-Agent. The email goes to these only.
IDENTIFIED_AGENT = "mise-catalog/0.1 (me@sampacker.com; +https://github.com/sam-packer/mise)"
IDENTIFIED_HOSTS = {
    "openlibrary.org",
    "covers.openlibrary.org",
    "musicbrainz.org",
    "query.wikidata.org",
    "commons.wikimedia.org",
}
# The Art Institute of Chicago asks for this header on its API.
HOST_HEADERS = {"api.artic.edu": {"AIC-User-Agent": USER_AGENT}}
# After this many failed requests in a row, a host gets no more calls in this run.
BREAKER_LIMIT = 10

# API errors in a 200 body, by host: (codes to retry with backoff, codes that mean "not found").
# Deezer 4: over quota; 800: no data (an unknown ISRC). Last.fm 8, 11, 16: temporary;
# 29: rate limit; 6: no such track.
API_ERRORS: dict[str, tuple[set[Any], set[Any]]] = {
    "api.deezer.com": ({4}, {800}),
    "ws.audioscrobbler.com": ({8, 11, 16, 29}, {6}),
}


def api_error_code(body: Any) -> Any:
    if not (isinstance(body, dict) and "error" in body):
        return None
    error = body["error"]
    return error.get("code") if isinstance(error, dict) else error


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
    """GET and POST with an on-disk response cache and per-host rate limits.

    A cached response never touches the network, so reruns are fast and free. Auth headers
    and `secret` query parameters go to the server but never into the cache key or the
    cache file, so a key never lands on disk and a new key reuses the cached answers.
    """

    def __init__(self, cfg: ResolveConfig, cache_dir: Path = CACHE / "http") -> None:
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.limiter = RateLimiter(cfg.min_interval, cfg.default_interval)
        self.hits = 0
        self.requests = 0
        self.failures: dict[str, int] = {}
        self.guard = threading.Lock()
        self.client = httpx.Client(
            timeout=cfg.timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )

    def _check(self, url: str) -> str:
        """The host of url. A host that the breaker stopped raises FetchError at once."""
        host = urlsplit(url).hostname or ""
        if self.failures.get(host, 0) >= BREAKER_LIMIT:
            raise FetchError(f"{host}: stopped for this run after {BREAKER_LIMIT} failures")
        return host

    def _result(self, host: str, ok: bool) -> None:
        """Count failures in a row for each host. Any success resets the count.

        A blocked host (Cloudflare, an outage) would otherwise cost minutes of backoff for
        every item. Its items count as errors, so the next run tries them again.
        """
        with self.guard:
            if ok:
                self.failures[host] = 0
                return
            self.failures[host] = self.failures.get(host, 0) + 1
            if self.failures[host] == BREAKER_LIMIT:
                log.warning(
                    f"{host}: {num(BREAKER_LIMIT)} failed requests in a row; no more calls "
                    "to it in this run (its items count as errors and the next run retries them)"
                )

    def _path(self, key: str, suffix: str) -> Path:
        host = urlsplit(key).hostname or "unknown"
        return self.cache_dir / host / f"{stable_hash(key, 24)}{suffix}"

    def _fetch(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        secret: dict[str, str] | None = None,
        body: Any = None,
    ) -> httpx.Response:
        host = urlsplit(url).hostname or ""
        agent = IDENTIFIED_AGENT if host in IDENTIFIED_HOSTS else USER_AGENT
        send_headers = {"User-Agent": agent, **HOST_HEADERS.get(host, {}), **(headers or {})}
        if secret:
            url = f"{url}{'&' if '?' in url else '?'}{urlencode(secret)}"
        for attempt in range(self.cfg.retries):
            self.limiter.wait(host)
            try:
                if body is None:
                    resp = self.client.get(url, headers=send_headers)
                else:
                    resp = self.client.post(url, json=body, headers=send_headers)
            except httpx.HTTPError:
                time.sleep(2**attempt)
                continue
            # 403 too: bot protection (the Met) answers 403 for a while under load.
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                wait = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset-in")
                wait = wait if wait and wait.isdigit() else ""
                time.sleep(float(wait) + 1 if wait else 5 * 2**attempt)
                continue
            return resp
        raise FetchError(urlsplit(url)._replace(query="").geturl())

    def get_json(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        secret: dict[str, str] | None = None,
    ) -> Any | None:
        """The JSON body, or None for a 404. Anything else raises FetchError.

        Only real answers (JSON with 200, or 404) are cached. A block page, a non-JSON 200,
        or any other status raises, so the caller counts an error and a rerun retries it.
        Deezer and Last.fm answer 200 with {"error": ...}. API_ERRORS names the codes to
        retry with backoff and the codes that mean "not found" (cached as None). Any other
        error body raises FetchError and is never cached.
        """
        return self._json(url, url, headers, secret, None)

    def post_json(self, url: str, body: Any, headers: dict[str, str] | None = None) -> Any:
        """POST a JSON body; the answer is cached by URL and body.

        A GraphQL answer with "errors" raises FetchError and is never cached.
        """
        key = f"{url}#{json.dumps(body, sort_keys=True)}"
        answer = self._json(url, key, headers, None, body)
        if isinstance(answer, dict) and answer.get("errors"):
            raise FetchError(f"{url}: {answer['errors']}")
        return answer

    def _json(
        self,
        url: str,
        key: str,
        headers: dict[str, str] | None,
        secret: dict[str, str] | None,
        post: Any,
    ) -> Any | None:
        path = self._path(key, ".json")
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                cached = None  # a file cut short by a hard kill: ask again
            if cached is not None:
                self.hits += 1
                return cached["body"]
        host = self._check(url)
        self.requests += 1
        try:
            body, status = self._json_network(url, headers, secret, post)
        except FetchError:
            self._result(host, False)
            raise
        self._result(host, True)
        if post is not None and isinstance(body, dict) and body.get("errors"):
            return body  # post_json raises; an error answer is never cached
        atomic_write(path, json.dumps({"url": key, "status": status, "body": body}).encode())
        return body

    def _json_network(
        self,
        url: str,
        headers: dict[str, str] | None,
        secret: dict[str, str] | None,
        post: Any,
    ) -> tuple[Any, int]:
        retry, not_found = API_ERRORS.get(urlsplit(url).hostname or "", (set(), set()))
        for attempt in range(self.cfg.retries):
            resp = self._fetch(url, headers, secret, post)
            if resp.status_code == 404:
                body = None
            elif resp.status_code == 200:
                try:
                    body = resp.json()
                except ValueError as e:
                    raise FetchError(f"{url}: HTTP 200 without JSON") from e
            else:
                raise FetchError(f"{url}: HTTP {resp.status_code}")
            code = api_error_code(body)
            if code is None:
                break
            if code in not_found:
                body = None
                break
            if code not in retry:
                raise FetchError(f"{url}: API error {body['error']}")
            time.sleep(5 * 2**attempt)
        else:
            raise FetchError(f"{url}: API error {body['error']}")
        return body, resp.status_code

    def get_bytes(self, url: str) -> bytes | None:
        """Raw bytes, or None for a 404. Anything else raises FetchError.

        Only 404s are cached; the caller stores what it keeps.
        """
        miss = self._path(url, ".404")
        if miss.exists():
            return None
        host = self._check(url)
        try:
            resp = self._fetch(url)
            if resp.status_code not in (200, 404) or (resp.status_code == 200 and not resp.content):
                raise FetchError(f"{url}: HTTP {resp.status_code}, {len(resp.content)} bytes")
        except FetchError:
            self._result(host, False)
            raise
        self._result(host, True)
        if resp.status_code == 404:
            atomic_write(miss, b"")
            return None
        return resp.content

    def close(self) -> None:
        self.client.close()
