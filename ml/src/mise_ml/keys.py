"""Read service credentials for catalog downloads and publication.

The CLI loads ml/.env. Log only whether credentials are set, never their values.
"""

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


# Cloudflare R2, for `publish` only. Each variable is required.
R2 = {
    "R2_ACCOUNT_ID": "the Cloudflare account id (the host of the S3 endpoint)",
    "R2_ACCESS_KEY_ID": "the access key id of an R2 API token",
    "R2_SECRET_ACCESS_KEY": "the secret access key of that token",
    "R2_PUBLIC_URL": "the public custom domain of the bucket, such as https://cdn.mise.art",
}
R2_BUCKET_DEFAULT = "mise"


def env(name: str) -> str:
    return os.environ.get(name, "").strip()


def r2() -> dict[str, str]:
    """Return R2 settings, or report every missing required variable and stop."""
    for name in (*R2, "R2_BUCKET"):
        log.debug("%s: %s", name, "set" if env(name) else "missing")
    problems = [
        f"missing {name} ({use}); set it in ml/.env" for name, use in R2.items() if not env(name)
    ]
    for problem in problems:
        log.error(problem)
    if problems:
        raise SystemExit(1)
    return {
        "account_id": env("R2_ACCOUNT_ID"),
        "access_key_id": env("R2_ACCESS_KEY_ID"),
        "secret_access_key": env("R2_SECRET_ACCESS_KEY"),
        "bucket": env("R2_BUCKET") or R2_BUCKET_DEFAULT,
        "public_url": env("R2_PUBLIC_URL").rstrip("/"),
    }


def missing_r2() -> list[str]:
    """Return missing R2 variable names without stopping."""
    return [name for name in R2 if not env(name)]


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
