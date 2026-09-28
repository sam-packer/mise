"""Select the catalog: films (TMDB), books (Hardcover), songs (ListenBrainz and MusicBrainz),
art (the Met CSV, the Art Institute of Chicago, the Cleveland Museum of Art), and poems
(PoetryDB). Every API answer is cached in data/cache/http, so a rerun is fast.

Films, books, and songs are chosen by era: each era keeps its quota, spread over its years.
Each era and each art source is a group. curate writes more candidates than a group keeps;
resolve tries them in rank order and keeps the group's target.
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
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import quote_plus, urlencode

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
from mise_ml.http import CachedClient
from mise_ml.log import elapsed, get, num, progress
from mise_ml.util import slugify, write_json, write_jsonl

log = get(__name__)

Record = dict[str, Any]

TMDB = "https://api.themoviedb.org/3"
HARDCOVER = "https://api.hardcover.app/v1/graphql"
LISTENBRAINZ = "https://api.listenbrainz.org/1"
MUSICBRAINZ = "https://musicbrainz.org/ws/2/recording"
AIC = "https://api.artic.edu/api/v1/artworks/search"
CMA = "https://openaccess-api.clevelandart.org/api/artworks/"

# Words that mark another version of a song. A catalog song never has one; resolve rejects
# a Deezer match with one unless the catalog title has it too.
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
    """fn over items with a thread pool, results in item order. The per-host rate limit
    still holds; parallel calls hide the network latency."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


# Eras and groups


def era_name(start: int, end: int) -> str:
    return f"{start}-{end}" if start > 0 else f"before-{end + 1}"


def era_groups(category: str, eras: Eras) -> dict[str, int]:
    return {f"{category}:{era_name(s, e)}": quota for s, e, quota in eras}


def targets(cfg: CurateConfig) -> dict[str, int]:
    """The items resolve keeps for each group."""
    quota = cfg.art_met_quota
    return {
        **era_groups("film", cfg.film_eras),
        **era_groups("book", cfg.book_eras),
        **era_groups("song", cfg.song_eras),
        **{f"art:met-{c.lower()}": round(cfg.art_met * s) for c, s in quota.items()},
        "art:aic": cfg.art_aic,
        "art:cma": cfg.art_cma,
    }


def wanted(category: str, quota: int, cfg: CurateConfig) -> int:
    return round(quota * cfg.candidate_factor[category])


def spread(rows: list[Record], n: int, cfg: CurateConfig) -> tuple[list[Record], int]:
    """The best n rows of an era, spread over its years.

    rows must be in rank order. A year first gets at most year_cap_factor times its even
    share; the free slots then go to the best rows left. Returns the rows and how many came
    from the capped pass.
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
        chosen, capped = spread(pool, n, cfg)
        for r in chosen:
            r["group"] = group
        out.extend(chosen)
        line = (
            f"{category} {era_name(start, end)}: {num(len(chosen))} candidates for "
            f"{num(quota)} slots from {num(len(pool))}"
        )
        if len(chosen) < n:
            log.warning(f"{line}; the era is short by {num(n - len(chosen))} candidates")
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
    """Stop with the field names that Hardcover no longer has, before any selection."""
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


def listenbrainz_pool(http: CachedClient) -> list[dict]:
    """Recordings with a user count: the sitewide top lists, and the top recordings of
    every artist on them."""
    sitewide: dict[str, dict] = {}
    for rng in SITEWIDE_RANGES:
        url = f"{LISTENBRAINZ}/stats/sitewide/recordings?range={rng}&count=1000"
        for x in (http.get_json(url) or {}).get("payload", {}).get("recordings", []):
            if x.get("recording_mbid") and x.get("artist_mbids"):
                sitewide.setdefault(x["recording_mbid"], x)
    found = dict.fromkeys(x["artist_mbids"][0] for x in sitewide.values())
    by_recording = len(found)
    for rng in SITEWIDE_RANGES:
        url = f"{LISTENBRAINZ}/stats/sitewide/artists?range={rng}&count=1000"
        for x in (http.get_json(url) or {}).get("payload", {}).get("artists", []):
            if x.get("artist_mbid"):
                found.setdefault(x["artist_mbid"])
    by_stats = len(found)
    # ListenBrainz listeners play mostly recent music, so the stats alone give too few songs
    # from before 1980. Last.fm's top artists of older decades and genres fill that era.
    for tag in OLD_ERA_TAGS:
        params = {"method": "tag.gettopartists", "tag": tag, "limit": 100, "format": "json"}
        body = http.get_json(f"{LASTFM}?{urlencode(params)}", secret=keys.lastfm()) or {}
        for a in (body.get("topartists") or {}).get("artist", []):
            if a.get("mbid"):
                found.setdefault(a["mbid"])
    artists = list(found)
    log.info(
        f"song: {num(len(sitewide))} recordings by {num(by_recording)} artists in the "
        f"ListenBrainz sitewide stats ({len(SITEWIDE_RANGES)} ranges), "
        f"{num(by_stats - by_recording)} more artists from the sitewide artist stats, and "
        f"{num(len(artists) - by_stats)} from Last.fm's top artists of older decades"
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
        rows = top(artist)
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
                        "artist": x.get("artist_name") or "",
                        "artist_mbid": artist,
                        "users": x.get("total_user_count") or 0,
                    },
                )
    bar.close()
    return list(pool.values())


def per_artist(pool: list[dict], limit: int) -> list[dict]:
    """At most `limit` songs per artist, by user count; one recording per song; no other
    versions (live, remix, karaoke, ...)."""
    out, seen = [], set()
    count: Counter[str] = Counter()
    for x in sorted(pool, key=lambda x: (-x["users"], x["mbid"])):
        x["title"] = clean_song_title(x["title"])
        key = (x["artist_mbid"], norm(base_title(x["title"])))
        if (
            not x["artist"]
            or VERSION.search(x["title"])
            or key in seen
            or count[x["artist_mbid"]] >= limit
        ):
            continue
        seen.add(key)
        count[x["artist_mbid"]] += 1
        out.append(x)
    return out


def musicbrainz_dates(http: CachedClient, mbids: list[str]) -> dict[str, dict]:
    """First release year and ISRCs for each recording, 100 recordings per search."""
    ids = sorted(set(mbids))
    out: dict[str, dict] = {}
    for i in progress(range(0, len(ids), 100), desc="musicbrainz", unit="batch"):
        query = "rid:(" + " OR ".join(ids[i : i + 100]) + ")"
        url = f"{MUSICBRAINZ}?{urlencode({'query': query, 'fmt': 'json', 'limit': 100})}"
        for rec in (http.get_json(url) or {}).get("recordings", []):
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
                "signal": {},
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
    """Public-domain objects of each class: highlights first, then objects with a Wikidata
    item (a Commons image), then objects with subject tags, then a seeded shuffle."""
    met = load_met()
    rng = random.Random(SEED)
    records = []
    for cls, share in cfg.art_met_quota.items():
        rows = met[met["cls"] == cls].to_dict("records")
        rng.shuffle(rows)
        rows.sort(
            key=lambda r: (
                r["Is Highlight"] != "True",
                not isinstance(r["Object Wikidata URL"], str),
                not isinstance(r["Tags"], str),
            )
        )
        n = wanted("art", round(cfg.art_met * share), cfg)
        group = f"art:met-{cls.lower()}"
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
                "source": {"aic": a["id"], "image_id": a["image_id"]},
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


def select_art(http: CachedClient, cfg: CurateConfig) -> list[Record]:
    return select_met(cfg) + select_aic(http, cfg) + select_cma(http, cfg)


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
    """A catalog from an older format (MovieLens, goodbooks-10k, MuSe) cannot mix with this
    one. Remove it, its resolve results, and its images. data/raw and the HTTP cache stay."""
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
        "MusicBrainz, art from the Met CSV, the Art Institute of Chicago, and the Cleveland "
        "Museum of Art; poems from PoetryDB (API answers cached in data/cache/http)"
    )
    http = CachedClient(ResolveConfig())
    steps: dict[str, Callable[[], list[Record]]] = {
        "film": lambda: select_films(http, cfg),
        "book": lambda: select_books(http, cfg),
        "song": lambda: select_songs(http, cfg),
        "art": lambda: select_art(http, cfg),
    }
    # Each category talks to its own hosts, so they run in parallel.
    with ThreadPoolExecutor(max_workers=len(steps)) as pool:
        futures = {c: pool.submit(fn) for c, fn in steps.items()}
        records = [r for c in steps for r in futures[c].result()]
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
