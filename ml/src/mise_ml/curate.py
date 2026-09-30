"""Select the catalog candidates: films (TMDB), books (Hardcover), songs (ListenBrainz, Last.fm,
and MusicBrainz), art (the Met, Chicago, Cleveland, NASA, and Smithsonian),
and poems (PoetryDB). data/cache/http caches every API answer, so a rerun is fast.

Spread candidates across years within each era and across art sources.
Write extra candidates so resolve can fill each group's quota when media is missing.
"""

import html
import json
import math
import pickle
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any
from urllib.parse import quote, quote_plus, urlencode

import numpy as np
import pandas as pd

from mise_ml import keys
from mise_ml.config import (
    CATALOG,
    CATALOG_META,
    CATALOG_VERSION,
    IMG,
    MET_CSV,
    ML_ROOT,
    PAT,
    RAW,
    RESOLVE_DROPPED,
    RESOLVED,
    SEED,
    CurateConfig,
    Eras,
    ResolveConfig,
)
from mise_ml.http import CachedClient, FetchError
from mise_ml.log import elapsed, get, num, progress
from mise_ml.threads import run_all
from mise_ml.util import slugify, stable_hash, write_json, write_jsonl

log = get(__name__)

Record = dict[str, Any]

TMDB = "https://api.themoviedb.org/3"
HARDCOVER = "https://api.hardcover.app/v1/graphql"
LISTENBRAINZ = "https://api.listenbrainz.org/1"
MUSICBRAINZ = "https://musicbrainz.org/ws/2/recording"
AIC = "https://api.artic.edu/api/v1/artworks/search"
CMA = "https://openaccess-api.clevelandart.org/api/artworks/"

# Exclude alternate song versions from the catalog.
# Resolve accepts these words in a Deezer match only when the catalog title has them too.
VERSION = re.compile(
    r"\b(remix|mix|live|instrumental|karaoke|acoustic|cover|sped up|slowed|a cappella|"
    r"acapella|originally performed|made popular|tribute|demo)\b",
    re.IGNORECASE,
)
FEATURING = re.compile(r"\s*[\(\[]\s*(feat|ft|featuring|with)\b\.?[^\)\]]*[\)\]]", re.IGNORECASE)
# " - Remastered 2009", "(2011 Remaster)", "- Mono": the same song, another master.
REMASTER = re.compile(
    r"\s*(-\s+|[\(\[])[^\(\)\[\]]*\b(remaster(ed)?|mono|stereo|single version|radio edit|"
    r"album version|original mix)\b[^\(\)\[\]]*[\)\]]?\s*$",
    re.IGNORECASE,
)
BOOK_JUNK = re.compile(r"\b(box(ed)? set|boxset|collection|omnibus|books \d+\s*-\s*\d+)\b", re.I)
# TMDB keywords that say nothing about a film's mood.
KEYWORD_JUNK = re.compile(r"stinger|based on|sequel|remake|duringcredits|aftercredits|\(mcu\)")
# AIC terms that name a material or technique, not a subject or mood.
TERM_JUNK = re.compile(r"paint|canvas|panel|oil|tempera|watercolor|century|gouache|ink|chalk")


def base_title(title: str) -> str:
    """The title without a "(feat. ...)" part, which differs between catalogs."""
    return FEATURING.sub("", title).strip()


def clean_song_title(title: str) -> str:
    return REMASTER.sub("", title).strip() or title.strip()


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower().replace("&", "and"))


def artist_name_key(name: str) -> str:
    """Match artist names across service IDs without discarding non-Latin characters."""
    return " ".join(name.casefold().split())


def year_or_none(value: Any) -> int | None:
    try:
        y = int(float(value))
    except (TypeError, ValueError):
        return None
    return y if -5000 < y < 2100 and y != 0 else None


def text_or_none(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def clip(text: Any, limit: int) -> str | None:
    """Plain text without HTML, cut at a word boundary."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()
    if len(text) <= limit:
        return text or None
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + " ..."


def unique(values: Iterable[str], limit: int) -> list[str]:
    out: list[str] = []
    for v in values:
        v = v.strip()
        if v and v.lower() not in {o.lower() for o in out}:
            out.append(v)
    return out[:limit]


def parallel(fn: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    """Return concurrent results in input order; the HTTP client limits each host's request rate."""
    return run_all(fn, items, workers)


# Eras and groups


def era_name(start: int, end: int) -> str:
    return f"{start}-{end}" if start > 0 else f"before-{end + 1}"


def era_groups(category: str, eras: Eras) -> dict[str, int]:
    return {f"{category}:{era_name(s, e)}": quota for s, e, quota in eras}


def targets(cfg: CurateConfig, catalog: list[Record] | None = None) -> dict[str, int]:
    """The items resolve keeps for each group."""
    quota = cfg.art_met_quota
    met = {f"art:met-{c.lower()}": round(cfg.art_met * s) for c, s in quota.items()}
    # Put any rounding remainder in the last class so source fallback keeps the total exact.
    last = next(reversed(met))
    met[last] += cfg.art_met - sum(met.values())
    out = {
        **era_groups("film", cfg.film_eras),
        **era_groups("book", cfg.book_eras),
        **era_groups("song", cfg.song_eras),
        **met,
        "art:aic": cfg.art_aic,
        "art:cma": cfg.art_cma,
        "art:nasa": cfg.art_nasa,
        "art:si": cfg.art_si,
    }
    if catalog is not None:
        # Art selectors record effective quotas after a missing key or source outage.
        defaults = out
        out = {g: n for g, n in out.items() if not g.startswith("art:")}
        for r in catalog:
            if r["category"] == "art":
                out[r["group"]] = r.get("group_target", defaults[r["group"]])
    return out


def wanted(category: str, quota: int, cfg: CurateConfig) -> int:
    return round(quota * cfg.candidate_factor[category])


def spread(rows: list[Record], n: int, cfg: CurateConfig) -> tuple[list[Record], int]:
    """The best n rows of an era, spread over its years.

    Supply rows in rank order. First cap each year at year_cap_factor times its even share.
    Fill free slots with the best remaining rows.
    Return the rows and the count from the capped pass.
    """
    years = {r["year"] for r in rows}
    cap = max(1, math.ceil(cfg.year_cap_factor * n / max(1, len(years))))
    per_year: Counter[int] = Counter()
    first, rest = [], []
    for r in rows:
        if len(first) < n and per_year[r["year"]] < cap:
            per_year[r["year"]] += 1
            first.append(r)
        else:
            rest.append(r)
    return first + rest[: n - len(first)], len(first)


def by_era(category: str, rows: list[Record], eras: Eras, cfg: CurateConfig) -> list[Record]:
    """Keep each era's candidates; rows carry "year" and "rank"."""
    out = []
    for start, end, quota in eras:
        group = f"{category}:{era_name(start, end)}"
        pool = sorted(
            (r for r in rows if start <= r["year"] <= end), key=lambda r: (-r["rank"], r["id"])
        )
        n = wanted(category, quota, cfg)
        if category == "song":
            scenes: dict[str, list[Record]] = defaultdict(list)
            for r in pool:
                scenes[r["signal"]["scene"]].append(r)
            floor = min(wanted("song", cfg.song_scene_floor, cfg), quota // max(1, len(scenes)))
            first = [r for scene in sorted(scenes) for r in scenes[scene][:floor]]
            first.sort(key=lambda r: (-r["rank"], r["id"]))
            ids = {r["source"]["mbid"] for r in first}
            rest, capped = spread(
                [r for r in pool if r["source"]["mbid"] not in ids], n - len(first), cfg
            )
            chosen = first + rest
        else:
            chosen, capped = spread(pool, n, cfg)
        for order, r in enumerate(chosen):
            r["group"] = group
            if category == "song":
                r["selection_order"] = order
        out.extend(chosen)
        line = (
            f"{category} {era_name(start, end)}: {num(len(chosen))} candidates for "
            f"{num(quota)} slots from {num(len(pool))}"
        )
        if category == "song" and chosen:
            line += (
                f"; last candidate {chosen[-1]['rank']:,} listeners, "
                f"minimum including scene floors {min(r['rank'] for r in chosen):,}"
            )
        if len(chosen) < n:
            log.warning(f"{line}; the era is short by {num(n - len(chosen))} candidates")
        elif category == "song":
            log.info(f"{line} ({num(len(first))} scene reserves; the rest spread by year)")
        else:
            log.info(f"{line} ({num(len(chosen) - capped)} over the year cap)")
    return out


# Films: TMDB discover by year, ranked by vote count


def tmdb_get(http: CachedClient, path: str, params: dict[str, Any]) -> Any:
    headers, secret = keys.tmdb()
    return http.get_json(f"{TMDB}{path}?{urlencode(sorted(params.items()))}", headers, secret)


def select_films(http: CachedClient, cfg: CurateConfig) -> list[Record]:
    genres = {g["id"]: g["name"] for g in tmdb_get(http, "/genre/movie/list", {})["genres"]}
    pages: list[tuple[int, int, int]] = []
    for (start, end, quota), votes in zip(cfg.film_eras, cfg.film_min_votes, strict=True):
        n_years = end - start + 1
        cap = math.ceil(cfg.year_cap_factor * wanted("film", quota, cfg) / n_years)
        # 20 films a page; a little more than the cap, for films without a poster.
        per_year = math.ceil(cap * 1.25 / 20)
        pages += [(y, p, votes) for y in range(start, end + 1) for p in range(1, per_year + 1)]

    def page(args: tuple[int, int, int]) -> list[dict]:
        year, number, votes = args
        params = {
            "primary_release_year": year,
            "sort_by": "vote_count.desc",
            "vote_count.gte": votes,
            "include_adult": "false",
            "include_video": "false",
            "page": number,
        }
        body = tmdb_get(http, "/discover/movie", params) or {}
        if number > body.get("total_pages", 0):
            return []
        return [{**m, "_year": year} for m in body.get("results", [])]

    log.info(f"film: {num(len(pages))} TMDB discover pages, {cfg.film_eras[0][0]} to now")
    found: dict[int, Record] = {}
    for rows in progress(parallel(page, pages, 8), desc="film pages", unit="page"):
        for m in rows:
            year = year_or_none((m.get("release_date") or "")[:4])
            if m.get("adult") or not m.get("poster_path") or year != m["_year"]:
                continue
            title = m["title"].strip()
            found.setdefault(
                m["id"],
                {
                    "id": f"film:{slugify(title)}-{year}",
                    "category": "film",
                    "title": title,
                    "creator": "",
                    "year": year,
                    "rank": int(m["vote_count"]),
                    "signal": {
                        "genres": [genres[g] for g in m.get("genre_ids", []) if g in genres],
                        "overview": clip(m.get("overview"), cfg.description_chars),
                    },
                    "source": {"tmdb": m["id"]},
                    "links": {"primary": f"https://www.themoviedb.org/movie/{m['id']}"},
                },
            )
    return by_era("film", list(found.values()), cfg.film_eras, cfg)


# Books: Hardcover GraphQL, ranked by readers

BOOK_FIELDS = (
    "id",
    "title",
    "slug",
    "release_year",
    "users_read_count",
    "compilation",
    "canonical_id",
    "cached_tags",
    "cached_contributors",
    "cached_image",
    "description",
)
BOOK_QUERY = """query Books($start: Int!, $end: Int!, $limit: Int!, $offset: Int!) {
  books(
    where: {release_year: {_gte: $start, _lte: $end}, compilation: {_eq: false},
            canonical_id: {_is_null: true}, users_read_count: {_gt: 0}}
    order_by: [{users_read_count: desc}, {id: asc}]
    limit: $limit
    offset: $offset
  ) { id title slug release_year users_read_count cached_tags cached_contributors
      cached_image description }
}"""


def hardcover(http: CachedClient, query: str, variables: dict[str, Any] | None = None) -> Any:
    body = {"query": query, "variables": variables or {}}
    return http.post_json(HARDCOVER, body, keys.hardcover())["data"]


def check_hardcover_schema(http: CachedClient) -> None:
    """Stop before selection if Hardcover lacks a required field, and report the missing names."""
    data = hardcover(http, '{ __type(name: "books") { fields { name } } }')
    have = {f["name"] for f in (data.get("__type") or {}).get("fields", [])}
    absent = [f for f in BOOK_FIELDS if f not in have]
    if absent:
        raise SystemExit(f"Hardcover's books type has no field {', '.join(absent)}; update curate")


def book_tags(tags: Any, category: str, limit: int) -> list[str]:
    rows = (tags or {}).get(category) or []
    rows = sorted(rows, key=lambda t: -(t.get("count") or 0))
    return unique((t.get("tag", "") for t in rows), limit)


def book_record(b: dict, cfg: CurateConfig) -> Record | None:
    title = (b.get("title") or "").strip()
    authors = [
        c["author"]["name"].strip()
        for c in b.get("cached_contributors") or []
        if (c.get("author") or {}).get("name") and c.get("contribution") in (None, "Author")
    ]
    if not title or not authors or BOOK_JUNK.search(title):
        return None
    image = (b.get("cached_image") or {}).get("url")
    return {
        "id": f"book:{slugify(title)}-{slugify(authors[0], 24)}",
        "category": "book",
        "title": title,
        "creator": authors[0],
        "year": b["release_year"],
        "rank": int(b["users_read_count"]),
        "signal": {
            "moods": [t.lower() for t in book_tags(b.get("cached_tags"), "Mood", cfg.book_tags)],
            "genres": book_tags(b.get("cached_tags"), "Genre", cfg.book_tags),
            "description": clip(b.get("description"), cfg.description_chars),
        },
        "source": {"hardcover": b["id"], "image": image},
        "links": {"primary": f"https://hardcover.app/books/{b['slug']}"},
    }


def select_books(http: CachedClient, cfg: CurateConfig) -> list[Record]:
    check_hardcover_schema(http)
    rows: dict[tuple[str, str], Record] = {}
    bar = progress(total=len(cfg.book_eras), desc="book eras", unit="era")
    for start, end, quota in cfg.book_eras:
        n = wanted("book", quota, cfg)
        era: list[Record] = []
        for page in range(cfg.book_max_pages):
            variables = {"start": start, "end": end, "limit": cfg.book_page}
            variables["offset"] = page * cfg.book_page
            books = hardcover(http, BOOK_QUERY, variables)["books"]
            for b in books:
                r = book_record(b, cfg)
                key = (norm(r["title"]), norm(r["creator"])) if r else None
                if r is not None and key not in rows:
                    rows[key] = r
                    era.append(r)
            ranked = sorted(era, key=lambda r: (-r["rank"], r["id"]))
            if len(books) < cfg.book_page or spread(ranked, n, cfg)[1] >= n:
                break
        log.debug(f"book {era_name(start, end)}: {num(len(era))} books from {page + 1} pages")
        bar.update()
    bar.close()
    return by_era("book", list(rows.values()), cfg.book_eras, cfg)


# Songs: ListenBrainz sitewide stats and artist top recordings, years from MusicBrainz

SITEWIDE_RANGES = (
    "this_week",
    "this_month",
    "this_year",
    "week",
    "month",
    "quarter",
    "half_yearly",
    "year",
    "all_time",
)
LASTFM = "https://ws.audioscrobbler.com/2.0/"
OLD_ERA_TAGS = (
    "50s",
    "60s",
    "70s",
    "oldies",
    "motown",
    "doo wop",
    "rock and roll",
    "classic rock",
    "soul",
    "funk",
    "disco",
    "folk",
    "blues",
    "jazz",
)

SCENE_TAGS = (
    "indie", "indie pop", "alternative r&b", "neo-soul", "bedroom pop", "dream pop",
    "art pop", "shoegaze", "post-punk", "singer-songwriter", "lo-fi", "alternative rock",
    "electronic", "ambient", "trip-hop", "hip-hop", "k-pop", "j-pop", "city pop",
    "afrobeats", "latin", "bossa nova", "reggae", "country", "americana", "classical",
    "soundtrack", "jazz", "indie rock", "synthpop",
)  # fmt: skip


def listenbrainz_pool(http: CachedClient) -> list[dict]:
    """Recordings with a user count: the sitewide top lists, and the top recordings of
    every artist on them."""
    sitewide: dict[str, dict] = {}
    for rng in SITEWIDE_RANGES:
        url = f"{LISTENBRAINZ}/stats/sitewide/recordings?range={rng}&count=1000"
        for x in (http.get_json(url) or {}).get("payload", {}).get("recordings", []):
            if x.get("recording_mbid") and x.get("artist_mbids"):
                sitewide.setdefault(x["recording_mbid"], x)
    found = {x["artist_mbids"][0]: x["artist_name"] for x in sitewide.values()}
    by_recording = len(found)
    for rng in SITEWIDE_RANGES:
        url = f"{LISTENBRAINZ}/stats/sitewide/artists?range={rng}&count=1000"
        for x in (http.get_json(url) or {}).get("payload", {}).get("artists", []):
            if x.get("artist_mbid"):
                found[x["artist_mbid"]] = x.get("artist_name") or found.get(x["artist_mbid"], "")
    by_stats = len(found)
    # Keep the first Last.fm discovery tag, including for artists already in sitewide stats.
    # Round-robin pages give each tag its first 100 artists before fetching the next 100.
    scenes: dict[str, str] = {}
    scene_names: dict[str, str] = {}
    for page in (1, 2):
        for tag in dict.fromkeys((*OLD_ERA_TAGS, *SCENE_TAGS)):
            params = {"method": "tag.gettopartists", "tag": tag, "limit": 100, "format": "json"}
            if page > 1:
                params["page"] = page
            try:
                body = http.get_json(f"{LASTFM}?{urlencode(params)}", secret=keys.lastfm()) or {}
            except FetchError as e:
                log.warning("song: Last.fm artist tag %s page %s failed: %s", tag, page, e)
                continue
            for a in (body.get("topartists") or {}).get("artist", []):
                if a.get("mbid"):
                    found.setdefault(a["mbid"], a.get("name") or "")
                    scenes.setdefault(a["mbid"], tag)
                if a.get("name"):
                    scene_names.setdefault(artist_name_key(a["name"]), tag)
    artists = list(found)
    log.info(
        f"song: {num(len(sitewide))} recordings by {num(by_recording)} artists in the "
        f"ListenBrainz sitewide stats ({len(SITEWIDE_RANGES)} ranges), "
        f"{num(by_stats - by_recording)} more artists from the sitewide artist stats, and "
        f"{num(len(artists) - by_stats)} from Last.fm's decade and scene tags"
    )
    # User counts for the sitewide recordings, in batches.
    ids = sorted(sitewide)
    users: dict[str, int] = {}
    for i in range(0, len(ids), 500):
        body = {"recording_mbids": ids[i : i + 500]}
        for p in http.post_json(f"{LISTENBRAINZ}/popularity/recording", body) or []:
            users[p["recording_mbid"]] = p.get("total_user_count") or 0
    pool = {
        mbid: {
            "mbid": mbid,
            "title": x["track_name"],
            "artist": x["artist_name"],
            "artist_mbid": x["artist_mbids"][0],
            "users": users.get(mbid, 0),
        }
        for mbid, x in sitewide.items()
    }

    def top(artist: str) -> list[dict]:
        url = f"{LISTENBRAINZ}/popularity/top-recordings-for-artist/{artist}"
        return http.get_json(url, keys.listenbrainz()) or []

    bar = progress(total=len(artists), desc="song artists", unit="artist")

    def one(artist: str) -> list[dict]:
        try:
            rows = top(artist)
        except FetchError as e:
            log.warning("song: artist %s failed; keep its sitewide recordings: %s", artist, e)
            rows = []
        bar.update()
        return rows

    for artist, rows in zip(artists, parallel(one, artists, 4), strict=True):
        for x in rows:
            if x.get("recording_mbid") and x.get("recording_name"):
                pool.setdefault(
                    x["recording_mbid"],
                    {
                        "mbid": x["recording_mbid"],
                        "title": x["recording_name"],
                        "artist": x.get("artist_name") or found[artist],
                        "artist_mbid": artist,
                        "users": x.get("total_user_count") or 0,
                    },
                )
    bar.close()
    artist_scenes = {
        artist: scenes.get(artist, scene_names.get(artist_name_key(name), "sitewide"))
        for artist, name in found.items()
    }
    for x in pool.values():
        x["scene"] = artist_scenes[x["artist_mbid"]]
    return list(pool.values())


def per_artist(pool: list[dict], limit: int) -> list[dict]:
    """At most `limit` songs per artist, by user count; one recording per song; no other
    versions (live, remix, karaoke, ...)."""
    out, seen = [], set()
    count: Counter[str] = Counter()
    names: Counter[str] = Counter()
    for x in sorted(pool, key=lambda x: (-x["users"], x["mbid"])):
        x["title"] = clean_song_title(x["title"])
        name = artist_name_key(x["artist"])
        title = base_title(x["title"]).casefold().replace("&", "and")
        key = (x["artist_mbid"], "".join(c for c in title if c.isalnum()))
        if (
            not x["artist"]
            or VERSION.search(x["title"])
            or key in seen
            or count[x["artist_mbid"]] >= limit
            or names[name] >= limit
        ):
            continue
        seen.add(key)
        count[x["artist_mbid"]] += 1
        names[name] += 1
        out.append(x)
    return out


def musicbrainz_dates(http: CachedClient, mbids: list[str]) -> dict[str, dict]:
    """First release year and ISRCs for each recording, 100 recordings per search."""
    ids = sorted(set(mbids))
    out: dict[str, dict] = {}
    for i in progress(range(0, len(ids), 100), desc="musicbrainz", unit="batch"):
        query = "rid:(" + " OR ".join(ids[i : i + 100]) + ")"
        url = f"{MUSICBRAINZ}?{urlencode({'query': query, 'fmt': 'json', 'limit': 100})}"
        try:
            body = http.get_json(url) or {}
        except FetchError as e:
            log.warning("song: MusicBrainz date batch failed: %s", e)
            continue
        for rec in body.get("recordings", []):
            out[rec["id"]] = {
                "year": year_or_none((rec.get("first-release-date") or "")[:4]),
                "isrcs": sorted(rec.get("isrcs") or []),
            }
    return out


def select_songs(http: CachedClient, cfg: CurateConfig) -> list[Record]:
    pool = per_artist(listenbrainz_pool(http), cfg.song_per_artist)
    dates = musicbrainz_dates(http, [x["mbid"] for x in pool])
    rows = []
    for x in pool:
        meta = dates.get(x["mbid"]) or {}
        if meta.get("year") is None:
            continue
        rows.append(
            {
                "id": f"song:{slugify(x['artist'], 24)}-{slugify(x['title'], 40)}",
                "category": "song",
                "title": x["title"],
                "creator": x["artist"],
                "year": meta["year"],
                "rank": int(x["users"]),
                "signal": {"scene": x["scene"]},
                "source": {
                    "mbid": x["mbid"],
                    "artist_mbid": x["artist_mbid"],
                    "isrc": meta["isrcs"][0] if meta["isrcs"] else None,
                },
                "links": {},
            }
        )
    log.info(
        f"song: {num(len(pool))} songs after the per-artist cap of {cfg.song_per_artist}; "
        f"{num(len(rows))} have a release year in MusicBrainz"
    )
    # Non-Latin names and truncated titles can share a slug. Use stable recording IDs for
    # every collision so a wider pool cannot reassign an old song's saved media to another song.
    counts = Counter(r["id"] for r in rows)
    for r in rows:
        if counts[r["id"]] > 1:
            r["id"] += f"-{r['source']['mbid']}"
    return by_era("song", rows, cfg.song_eras, cfg)


# Poems: PoetryDB


def curate_poems(cfg: CurateConfig) -> list[Record]:
    raw = pd.read_json(RAW / "poetrydb" / "poems.jsonl", lines=True)
    records, seen = [], set()
    dropped: Counter[str] = Counter()
    for row in raw.to_dict("records"):
        lines = [str(x).rstrip() for x in row["lines"]]
        n = int(row.get("linecount") or len(lines))
        key = (row["author"].lower(), row["title"].lower())
        if not cfg.poem_min_lines <= n <= cfg.poem_max_lines:
            dropped[f"not {cfg.poem_min_lines}-{cfg.poem_max_lines} lines"] += 1
            continue
        if key in seen:
            dropped["duplicate title"] += 1
            continue
        seen.add(key)
        records.append(poem_record(row, cfg))
    log.info(
        "poem: %s of %s PoetryDB poems kept (%s)",
        num(len(records)),
        num(len(raw)),
        ", ".join(f"{num(n)} {why}" for why, n in dropped.most_common()),
    )
    return records


def poem_record(row: dict[str, Any], cfg: CurateConfig) -> Record:
    excerpt = [str(x).rstrip() for x in row["lines"]][: cfg.poem_excerpt_lines]
    while excerpt and not excerpt[-1].strip():
        excerpt.pop()
    title, author = str(row["title"]).strip(), str(row["author"]).strip()
    return {
        "id": f"poem:{slugify(author, 24)}-{slugify(title, 40)}",
        "category": "poem",
        "title": title,
        "creator": author,
        "year": None,
        "rank": 0,
        "signal": {},
        "source": {"poetrydb": True},
        "text": "\n".join(excerpt),
        "links": {
            "primary": "https://www.poetryfoundation.org/search?query="
            + quote_plus(f"{title} {author}")
        },
    }


# Art: the Met CSV, the Art Institute of Chicago, the Cleveland Museum of Art

MET_COLUMNS = [
    "Object ID",
    "Is Highlight",
    "Is Public Domain",
    "Department",
    "Title",
    "Culture",
    "Artist Display Name",
    "Object Date",
    "Object Begin Date",
    "Medium",
    "Classification",
    "Object Wikidata URL",
    "Tags",
]


def load_met() -> pd.DataFrame:
    met = pd.read_csv(MET_CSV, usecols=MET_COLUMNS, dtype=str, low_memory=False)
    met = met[(met["Is Public Domain"] == "True") & met["Title"].notna()]
    met["cls"] = met["Classification"].fillna("").str.split("|").str[0].str.strip()
    return met


def met_record(row: dict[str, Any], group: str, rank: int) -> Record:
    title = str(row["Title"]).strip()
    artist = text_or_none(row["Artist Display Name"]) or ""
    oid = int(row["Object ID"])
    wikidata = text_or_none(row["Object Wikidata URL"])
    tags = text_or_none(row["Tags"])
    return {
        "id": f"art:{slugify(title, 40)}-{oid}",
        "category": "art",
        "title": title,
        "creator": artist.split("|")[0].strip() or "Unknown artist",
        "year": year_or_none(row["Object Begin Date"]),
        "rank": rank,
        "group": group,
        "signal": {
            "classification": row["cls"],
            "medium": text_or_none(row["Medium"]),
            "date": text_or_none(row["Object Date"]),
            "culture": text_or_none(row["Culture"]),
            "department": row["Department"],
            "subjects": tags.split("|") if tags else [],
        },
        "source": {"met": oid, "wikidata": wikidata.rsplit("/", 1)[-1] if wikidata else None},
        "links": {"primary": f"https://www.metmuseum.org/art/collection/search/{oid}"},
    }


def select_met(cfg: CurateConfig) -> list[Record]:
    """Rank objects by highlights, Wikidata images, and subject tags before a seeded shuffle."""
    met = load_met()
    rng = random.Random(SEED)
    records = []
    for cls in cfg.art_met_quota:
        rows = met[met["cls"] == cls].to_dict("records")
        rng.shuffle(rows)
        rows.sort(
            key=lambda r: (
                r["Is Highlight"] != "True",
                not isinstance(r["Object Wikidata URL"], str),
                not isinstance(r["Tags"], str),
            )
        )
        group = f"art:met-{cls.lower()}"
        n = wanted("art", targets(cfg)[group], cfg)
        records += [met_record(r, group, -rank) for rank, r in enumerate(rows[:n])]
        log.debug(f"art: Met {cls}: {num(len(rows))} public-domain objects, {num(n)} candidates")
    log.info(f"art: {num(len(records))} Met candidates from {num(len(met))} public-domain objects")
    return records


AIC_FIELDS = [
    "id",
    "title",
    "artist_title",
    "date_display",
    "date_start",
    "image_id",
    "medium_display",
    "artwork_type_title",
    "subject_titles",
    "term_titles",
    "style_titles",
    "classification_titles",
]


def select_aic(http: CachedClient, cfg: CurateConfig) -> list[Record]:
    """Public-domain paintings with an image, the museum's boosted works first."""
    n = wanted("art", cfg.art_aic, cfg)
    query = {
        "bool": {
            "filter": [
                {"term": {"is_public_domain": True}},
                {"exists": {"field": "image_id"}},
                {"term": {"artwork_type_id": 1}},  # Painting
            ]
        }
    }
    rows: list[dict] = []
    for page in range(1, math.ceil(n / 100) + 1):
        params = {
            "query": query,
            "sort": [{"is_boosted": "desc"}, {"id": "asc"}],
            "fields": AIC_FIELDS,
            "limit": 100,
            "page": page,
        }
        body = http.get_json(f"{AIC}?{urlencode({'params': json.dumps(params)})}") or {}
        rows += body.get("data", [])
        if page >= body.get("pagination", {}).get("total_pages", 0):
            break
    records = []
    for rank, a in enumerate(rows[:n]):
        title = (a.get("title") or "Untitled").strip()
        terms = [t for t in a.get("term_titles") or [] if not TERM_JUNK.search(t)]
        records.append(
            {
                "id": f"art:{slugify(title, 40)}-aic{a['id']}",
                "category": "art",
                "title": title,
                "creator": (a.get("artist_title") or "").strip() or "Unknown artist",
                "year": year_or_none(a.get("date_start")),
                "rank": -rank,
                "group": "art:aic",
                "signal": {
                    "classification": a.get("artwork_type_title"),
                    "medium": a.get("medium_display"),
                    "date": a.get("date_display"),
                    "subjects": unique(a.get("subject_titles") or [], 15),
                    "styles": unique(a.get("style_titles") or [], 6),
                    "terms": unique(terms, 12),
                },
                "source": {"aic": a["id"]},
                "links": {"primary": f"https://www.artic.edu/artworks/{a['id']}"},
            }
        )
    log.info(f"art: {num(len(records))} Art Institute of Chicago candidates")
    return records


CMA_FIELDS = (
    "id,title,creation_date,creation_date_earliest,creators,images,culture,technique,"
    "description,url,type,department,is_highlight"
)


def select_cma(http: CachedClient, cfg: CurateConfig) -> list[Record]:
    """CC0 paintings with an image: highlights first, then works with a curator's
    description, then a seeded shuffle."""
    rows: list[dict] = []
    while True:
        params = {"cc0": 1, "has_image": 1, "type": "Painting", "limit": 1000, "skip": len(rows)}
        body = http.get_json(f"{CMA}?{urlencode({**params, 'fields': CMA_FIELDS})}") or {}
        rows += body.get("data", [])
        if not body.get("data") or len(rows) >= body.get("info", {}).get("total", 0):
            break
    rows = [r for r in rows if ((r.get("images") or {}).get("web") or {}).get("url")]
    rows.sort(key=lambda r: r["id"])
    random.Random(SEED).shuffle(rows)
    rows.sort(key=lambda r: (not r.get("is_highlight"), not r.get("description")))
    n = wanted("art", cfg.art_cma, cfg)
    records = []
    for rank, a in enumerate(rows[:n]):
        title = (a.get("title") or "Untitled").strip()
        creators = a.get("creators") or []
        creator = (creators[0].get("description") or "").split(" (")[0] if creators else ""
        records.append(
            {
                "id": f"art:{slugify(title, 40)}-cma{a['id']}",
                "category": "art",
                "title": title,
                "creator": creator.strip() or "Unknown artist",
                "year": year_or_none(a.get("creation_date_earliest")),
                "rank": -rank,
                "group": "art:cma",
                "signal": {
                    "classification": a.get("type"),
                    "medium": a.get("technique"),
                    "date": a.get("creation_date"),
                    "culture": ", ".join(a.get("culture") or []) or None,
                    "description": clip(a.get("description"), cfg.description_chars),
                },
                "source": {"cma": a["id"], "image": a["images"]["web"]["url"]},
                "links": {"primary": a.get("url") or f"https://clevelandart.org/art/{a['id']}"},
            }
        )
    log.info(f"art: {num(len(records))} Cleveland Museum of Art candidates of {num(len(rows))}")
    return records


NASA_TOPICS = (
    "aurora", "earth at night", "sunset", "clouds", "storm", "ocean", "desert",
    "glacier", "nebula", "galaxy", "moon surface", "milky way", "lightning", "snow",
    "forest from above", "mountains", "islands", "earth sunrise",
)  # fmt: skip
NASA_JUNK = re.compile(
    r"copyright|\u00a9|all rights reserved|third.party|Getty|Alamy|Reuters|Associated Press|"
    r"\b(portrait|headshot|ceremony|ribbon.cutting|award|press conference|"
    r"test|testing|wind tunnel|calibration|launch pad|"
    r"crates?|offload\w*|unload\w*|assembly|technicians?|clean room|"
    r"spacecraft processing|payload processing|launch vehicle|launch complex|"
    r"rocket|rollout|briefing|hangar|ground support|aircraft|airplane|fuselage|livery|"
    r"diagram|schematic|chart|graph|staff|logo|insignia)\b",
    re.I,
)
SI_TOPICS = (
    "Landscapes",
    "Nature",
    "Gardens",
    "Flowers",
    "Trees",
    "Mountains",
    "Water",
    "Sky",
    "Night",
    "Seascapes",
    "Architecture",
    "Abstraction",
)
SI_PEOPLE = re.compile(r"\b(portraits?|self.portraits?|sitters?|group photograph)\b", re.I)
SI_BIOGRAPHY = re.compile(r",?\s+\(?(?:born|active|died|founded|b\.|d\.|ca\.)(?:\s|$)", re.I)
# Catalog numbers used as titles ("PIA14417", "iss025e012345", "S94-12345").
NASA_TECHNICAL = re.compile(r"\b(PIA\d+|iss\d{3}e\d+|sts\d+-\d+|s\d{2}-\d+|jsc\d+)\b", re.I)


def title_shape(r: Record) -> tuple[str, str]:
    """A title without its numbers, with its creator. Series titles such as "Earth observations
    taken by the STS-59 crew" share one shape, so the catalog keeps one of them."""
    return re.sub(r"\d+", "", norm(r.get("title") or "")), r.get("creator") or ""


def topic_sample(pools: list[list[Record]], n: int, creator_cap: int = 8) -> list[Record]:
    """Seeded round-robin selection; no topic exceeds twice its even share. Keep one record per
    title shape, and at most `creator_cap` records per creator, so one maker cannot fill a
    source."""
    rng = random.Random(SEED)
    cap = max(1, math.ceil(2 * n / max(1, len(pools))))
    for pool in pools:
        pool.sort(key=lambda r: r["id"])
        rng.shuffle(pool)
    out, seen, shapes = [], set(), set()
    creators: Counter[str] = Counter()
    # A skipped record does not use a turn: each pool reads on until it takes one.
    taken, pos = [0] * len(pools), [0] * len(pools)
    while len(out) < n:
        took = False
        for k, pool in enumerate(pools):
            while taken[k] < cap and pos[k] < len(pool):
                r = pool[pos[k]]
                pos[k] += 1
                shape = title_shape(r)
                if r["id"] in seen or shape in shapes or creators[r["creator"]] >= creator_cap:
                    continue
                seen.add(r["id"])
                shapes.add(shape)
                creators[r["creator"]] += 1
                taken[k] += 1
                out.append({**r, "rank": -len(out)})
                took = True
                if len(out) >= n:
                    return out
                break
        if not took:
            return out
    return out


def select_nasa(http: CachedClient, cfg: CurateConfig) -> list[Record]:
    """Nature and space images; exclude explicit third-party rights and work-site shots."""
    n = wanted("art", cfg.art_nasa, cfg)
    if not n:
        return []
    pools = []
    for topic in NASA_TOPICS:
        params = {"q": topic, "media_type": "image", "page_size": 100}
        try:
            body = http.get_json(f"https://images-api.nasa.gov/search?{urlencode(params)}") or {}
        except FetchError as e:
            log.warning("art: NASA topic %s failed: %s", topic, e)
            continue
        rows = []
        for item in body.get("collection", {}).get("items", []):
            data = item.get("data") or []
            if not data:
                continue
            a = data[0]
            oid = a.get("nasa_id")
            text = " ".join(
                [a.get("title", ""), a.get("description", ""), " ".join(a.get("keywords") or [])]
            )
            image = next(
                (
                    x["href"]
                    for x in item.get("links", [])
                    if x.get("render") == "image" and "~medium" in x.get("href", "")
                ),
                None,
            )
            if not oid or not image or NASA_JUNK.search(text):
                continue
            title = clip(a.get("title"), 300) or "Untitled"
            rows.append(
                {
                    "id": f"art:{slugify(title, 40)}-nasa{stable_hash(oid, 12)}",
                    "category": "art",
                    "group": "art:nasa",
                    "title": title,
                    "creator": " / ".join(
                        unique(["NASA", a.get("center") or "", a.get("secondary_creator") or ""], 3)
                    ),
                    "year": year_or_none((a.get("date_created") or "")[:4]),
                    "signal": {
                        "subjects": a.get("keywords") or [],
                        "topic": topic,
                        "description": clip(a.get("description"), cfg.description_chars),
                        "center": a.get("center"),
                    },
                    "source": {"nasa": oid, "image": image},
                    "links": {"primary": f"https://images.nasa.gov/details/{quote(oid, safe='')}"},
                }
            )
        pools.append([r for r in rows if not NASA_TECHNICAL.search(r["title"])])
    # Every NASA image credits a NASA center, so the per-creator cap does not apply.
    records = topic_sample(pools, n, creator_cap=n)
    log.info("art: %s NASA candidates", num(len(records)))
    return records


def select_si(http: CachedClient, cfg: CurateConfig) -> list[Record]:
    """CC0 photographs and paintings, with a topic cap and no metadata-marked portraits."""
    n = wanted("art", cfg.art_si, cfg)
    if not n:
        return []
    key = keys.env("SMITHSONIAN_API_KEY")
    if not key:
        log.warning("art: SMITHSONIAN_API_KEY missing; other museums fill its slots")
        return []
    pools = []
    for topic in SI_TOPICS:
        query = (
            'media_usage:CC0 AND online_media_type:"Images" '
            'AND object_type:("Photographs" OR "Paintings") '
            "AND unit_code:(CHNDM OR SAAM OR NASM OR EEPA OR NMAAHC OR SG) "
            f'AND topic:"{topic}"'
        )
        rows = []
        for start in range(0, 300, 100):
            params = {"q": query, "start": start, "rows": 100}
            try:
                body = (
                    http.get_json(
                        f"https://api.si.edu/openaccess/api/v1.0/search?{urlencode(params)}",
                        secret={"api_key": key},
                    )
                    or {}
                )
            except FetchError as e:
                log.warning("art: Smithsonian topic %s failed: %s", topic, e)
                break
            response = body.get("response") or {}
            for a in response.get("rows") or []:
                content = a.get("content") or {}
                detail = content.get("descriptiveNonRepeating") or {}
                free = content.get("freetext") or {}
                indexed = content.get("indexedStructured") or {}
                media = next(
                    (
                        m
                        for m in (detail.get("online_media") or {}).get("media", [])
                        if m.get("type") == "Images"
                        and m.get("idsId")
                        and (m.get("usage") or {}).get("access") == "CC0"
                    ),
                    None,
                )
                if not media or SI_PEOPLE.search(json.dumps(content)):
                    continue
                oid = detail.get("record_ID") or a["id"]
                # Names carry biography ("William H. Rau, born Philadelphia, PA 1855-died ...").
                # Keep the name itself.
                names = [
                    SI_BIOGRAPHY.split(v["content"], maxsplit=1)[0].strip(" ,;")
                    for v in free.get("name", [])
                    if v.get("label", "").lower() in ("artist", "photographer", "maker")
                ]
                dates = " ".join(v["content"] for v in free.get("date", []))
                years = [int(y) for y in re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", dates)]
                description = " ".join(
                    v["content"] for v in free.get("notes", []) if v.get("label") == "Description"
                )
                credit = "; ".join(v["content"] for v in free.get("creditLine", []))
                rows.append(
                    {
                        "id": f"art:si-{oid}",
                        "category": "art",
                        "group": "art:si",
                        "title": clip(a.get("title"), 300) or "Untitled",
                        # The holding museum is in source.unit and the credit line, not the creator.
                        "creator": ", ".join(n for n in names if n and n.lower() != "unidentified")
                        or "Unknown artist",
                        # The first year in the object's dates: the date it was made.
                        "year": years[0] if years else None,
                        "signal": {
                            "subjects": indexed.get("topic") or [],
                            "topic": topic,
                            "date": "; ".join(v["content"] for v in free.get("date", [])),
                            "description": clip(description, cfg.description_chars),
                        },
                        "source": {
                            "si": oid,
                            "credit": credit,
                            "unit": a.get("unitCode"),
                            "image": "https://ids.si.edu/ids/deliveryService?"
                            + urlencode({"id": media["idsId"], "max": 1200}),
                        },
                        "links": {
                            "primary": detail.get("record_link")
                            or f"https://www.si.edu/object/{oid}"
                        },
                    }
                )
            if start + 100 >= response.get("rowCount", 0):
                break
        pools.append(rows)
    records = topic_sample(pools, n)
    log.info("art: %s Smithsonian candidates", num(len(records)))
    return records


def select_art(http: CachedClient, cfg: CurateConfig) -> list[Record]:
    nasa, si = select_nasa(http, cfg), select_si(http, cfg)
    nasa_quota = min(cfg.art_nasa, int(len(nasa) / cfg.candidate_factor["art"]))
    si_quota = min(cfg.art_si, int(len(si) / cfg.candidate_factor["art"]))
    extra = cfg.art_nasa + cfg.art_si - nasa_quota - si_quota
    met_extra, aic_extra = round(extra * 0.6), round(extra * 0.25)
    effective = replace(
        cfg,
        art_nasa=nasa_quota,
        art_si=si_quota,
        art_met=cfg.art_met + met_extra,
        art_aic=cfg.art_aic + aic_extra,
        art_cma=cfg.art_cma + extra - met_extra - aic_extra,
    )
    if extra:
        log.info("art: move %s unfilled NASA/Smithsonian slots to the other museums", extra)
    records = select_met(effective) + select_aic(http, effective) + select_cma(http, effective)
    records += nasa[: wanted("art", nasa_quota, cfg)] + si[: wanted("art", si_quota, cfg)]
    quotas = targets(effective)
    for r in records:
        r["group_target"] = quotas[r["group"]]
    return records


# PAT palettes


def load_pat() -> list[dict[str, Any]]:
    root = RAW / "text2colors"
    rows = []
    for split in ("train", "test"):
        names_path = root / f"{split}_names.pkl"
        pal_path = root / f"{split}_palettes_rgb.pkl"
        if not names_path.exists() or not pal_path.exists():
            continue
        with names_path.open("rb") as f:
            names = pickle.load(f, encoding="latin1")
        with pal_path.open("rb") as f:
            palettes = pickle.load(f, encoding="latin1")
        for name, pal in zip(names, palettes, strict=False):
            phrase = " ".join(name) if isinstance(name, list | tuple) else str(name)
            arr = np.asarray(pal, dtype=np.float64).reshape(-1, 3)[:5]
            if arr.shape != (5, 3) or not phrase.strip():
                continue
            if arr.max() <= 1.0:
                arr = arr * 255.0
            rows.append({"phrase": phrase.strip(), "rgb": np.rint(arr).astype(int).tolist()})
    return rows


def dedupe_ids(records: list[Record]) -> int:
    seen: Counter[str] = Counter()
    renamed = 0
    for r in records:
        seen[r["id"]] += 1
        if seen[r["id"]] > 1:
            log.debug("duplicate id %s", r["id"])
            r["id"] = f"{r['id']}-{seen[r['id']]}"
            renamed += 1
    return renamed


def catalog_version() -> int | None:
    if not CATALOG_META.exists():
        return None
    return json.loads(CATALOG_META.read_text(encoding="utf-8")).get("version")


def start_fresh_if_old() -> None:
    """Clear catalog results and images on format mismatch; keep raw sources and the HTTP cache."""
    version = catalog_version()
    if version == CATALOG_VERSION:
        return
    files = [p for p in (CATALOG, RESOLVED, RESOLVE_DROPPED) if p.exists()]
    images = sorted(IMG.glob("*.webp")) if IMG.exists() else []
    if not files and not images:
        return
    log.warning(
        f"data/curated holds a catalog in format {version or 1}; this code writes format "
        f"{CATALOG_VERSION} (TMDB, Hardcover, ListenBrainz, three museums). Starting the "
        f"catalog fresh: removing {', '.join(p.name for p in files) or 'no files'} and "
        f"{num(len(images))} images in data/img. data/raw and data/cache/http stay."
    )
    for p in [*files, *images]:
        p.unlink()


def run() -> None:
    start = time.perf_counter()
    cfg = CurateConfig()
    keys.require(["tmdb", "hardcover", "listenbrainz", "lastfm"])
    start_fresh_if_old()
    log.info(
        "selecting: films from TMDB, books from Hardcover, songs from ListenBrainz and "
        "MusicBrainz, art from the Met, Chicago, Cleveland, NASA, and Smithsonian; "
        "poems from PoetryDB (API answers cached in data/cache/http)"
    )
    http = CachedClient(ResolveConfig())
    steps: dict[str, Callable[[], list[Record]]] = {
        "film": lambda: select_films(http, cfg),
        "book": lambda: select_books(http, cfg),
        "song": lambda: select_songs(http, cfg),
        "art": lambda: select_art(http, cfg),
    }
    # Each category talks to its own hosts, so they run in parallel.
    found = run_all(lambda fn: fn(), list(steps.values()), len(steps))
    records = [r for rows in found for r in rows]
    http.close()
    records += curate_poems(cfg)
    renamed = dedupe_ids(records)
    if renamed:
        log.info("%s items shared an id; they got a numeric suffix", num(renamed))
    records.sort(key=lambda r: r["id"])
    write_jsonl(CATALOG, records)
    write_json(CATALOG_META, {"version": CATALOG_VERSION})
    pat = sorted(load_pat(), key=lambda r: (r["phrase"], r["rgb"]))
    write_jsonl(PAT, pat)
    counts = Counter(r["category"] for r in records)
    by_group: dict[str, int] = defaultdict(int)
    for r in records:
        by_group[r.get("group", r["category"])] += 1
    log.debug("candidates by group: " + ", ".join(f"{g} {n}" for g, n in sorted(by_group.items())))
    log.info(
        "done in %s: %s catalog items (%s) -> %s; %s API calls, %s answers from the cache; "
        "%s PAT palettes -> %s",
        elapsed(start),
        num(len(records)),
        ", ".join(f"{c} {num(counts[c])}" for c in ("film", "book", "song", "poem", "art")),
        CATALOG.relative_to(ML_ROOT).as_posix(),
        num(http.requests),
        num(http.hits),
        num(len(pat)),
        PAT.relative_to(ML_ROOT).as_posix(),
    )
