"""API keys from ml/.env. Only "set" or "missing" is ever logged, never a value."""

import os

from mise_ml.log import get

log = get(__name__)

# Service -> (environment variables, any one of which is enough; what the key is for).
KEYS = {
    "tmdb": (("TMDB_TOKEN", "TMDB_API_KEY"), "films: selection and details"),
    "hardcover": (("HARDCOVER_TOKEN",), "books: selection, tags, and covers"),
    "listenbrainz": (("LISTENBRAINZ_TOKEN",), "songs: the top recordings of each artist"),
    "lastfm": (("LASTFM_API_KEY",), "songs: older artists and listener tags"),
}


def env(name: str) -> str:
    return os.environ.get(name, "").strip()


def missing(services: list[str]) -> list[str]:
    """One line for each service without a key."""
    out = []
    for service in services:
        names, use = KEYS[service]
        if not any(env(n) for n in names):
            out.append(f"missing API key {' or '.join(names)} ({use}); set it in ml/.env")
    return out


def require(services: list[str]) -> None:
    problems = missing(services)
    for problem in problems:
        log.error(problem)
    if problems:
        raise SystemExit(1)


def tmdb() -> tuple[dict[str, str], dict[str, str]]:
    """Headers and query parameters for TMDB. The v4 bearer token wins over the v3 key."""
    token = env("TMDB_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}, {}
    return {}, {"api_key": env("TMDB_API_KEY")}


def hardcover() -> dict[str, str]:
    token = env("HARDCOVER_TOKEN").removeprefix("Bearer ").strip()
    return {"Authorization": f"Bearer {token}"}


def listenbrainz() -> dict[str, str]:
    return {"Authorization": f"Token {env('LISTENBRAINZ_TOKEN')}"}


def lastfm() -> dict[str, str]:
    return {"api_key": env("LASTFM_API_KEY")}
