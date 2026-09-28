import ast
import math
import pickle
import random
import re
import time
from collections import Counter
from typing import Any
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

from mise_ml.config import (
    CATALOG,
    ML_ROOT,
    MUSE_KAGGLE,
    MUSE_ZENODO,
    PAT,
    RAW,
    SEED,
    SPOTIFY_TRACKS,
    CurateConfig,
)
from mise_ml.log import elapsed, get, num, progress
from mise_ml.util import slugify, write_jsonl

log = get(__name__)

TITLE_YEAR_RE = re.compile(r"^(.*?)\s*\((\d{4})\)\s*$")
TRAILING_ARTICLE_RE = re.compile(r"^(.*), (The|A|An|Les|La|Le|L'|Il|El|Die|Das|Der)$")
SERIES_RE = re.compile(r"\s*\([^()]*#\s*[\d.]+[^()]*\)\s*$")

SHELF_JUNK = {
    "to",
    "read",
    "reading",
    "reads",
    "currently",
    "own",
    "owned",
    "owns",
    "favorite",
    "favorites",
    "favourite",
    "favourites",
    "fav",
    "favs",
    "default",
    "kindle",
    "ebook",
    "ebooks",
    "e",
    "audio",
    "audiobook",
    "audiobooks",
    "library",
    "borrowed",
    "wishlist",
    "wish",
    "club",
    "reread",
    "re",
    "shelf",
    "series",
    "abandoned",
    "dnf",
    "maybe",
    "buy",
    "bought",
    "have",
    "finished",
    "tbr",
    "hold",
    "english",
    "book",
    "books",
    "novel",
    "novels",
    "i",
    "my",
    "list",
    "want",
    "later",
    "next",
    "nook",
    "calibre",
    "unread",
    "pending",
}


Record = dict[str, Any]


def year_or_none(value: Any) -> int | None:
    try:
        y = int(float(value))
    except (TypeError, ValueError):
        return None
    return y if 0 < y < 2100 else None


def clean_movie_title(raw: str) -> tuple[str, int | None]:
    title, year = raw.strip(), None
    m = TITLE_YEAR_RE.match(title)
    if m:
        title, year = m.group(1), int(m.group(2))
    if title.endswith(")") and " (" in title:
        title = title[: title.index(" (")]
    m = TRAILING_ARTICLE_RE.match(title)
    if m:
        sep = "" if m.group(2).endswith("'") else " "
        title = f"{m.group(2)}{sep}{m.group(1)}"
    return title, year


def curate_films(cfg: CurateConfig) -> list[Record]:
    root = RAW / "movielens" / "ml-latest"
    movies = pd.read_csv(root / "movies.csv")
    links = pd.read_csv(root / "links.csv", dtype={"imdbId": str})
    counts = pd.read_csv(root / "ratings.csv", usecols=["movieId"], dtype="int32")[
        "movieId"
    ].value_counts()
    rated = int((counts >= cfg.film_min_ratings).sum())
    counts = counts[counts >= cfg.film_min_ratings].head(cfg.film_top)
    chosen = set(counts.index.tolist())
    log.info(
        "film: %s of %s movies have at least %s ratings; keep the top %s",
        num(rated),
        num(len(movies)),
        num(cfg.film_min_ratings),
        num(len(chosen)),
    )

    tags = pd.read_csv(root / "genome-tags.csv").set_index("tagId")["tag"].to_dict()
    scores = pd.read_csv(root / "genome-scores.csv", dtype={"movieId": "int32", "tagId": "int32"})
    scores = scores[scores["movieId"].isin(chosen)]
    scores = scores[scores["relevance"] >= cfg.film_genome_min_relevance]
    scores = scores.sort_values(["movieId", "relevance"], ascending=[True, False])
    top_tags = (
        scores.groupby("movieId")
        .head(cfg.film_genome_tags)
        .groupby("movieId")["tagId"]
        .apply(lambda s: [tags[t] for t in s])
        .to_dict()
    )

    movies = movies[movies["movieId"].isin(chosen)].merge(links, on="movieId", how="left")
    records = []
    for row in movies.itertuples(index=False):
        title, year = clean_movie_title(row.title)
        imdb = f"tt{row.imdbId}" if isinstance(row.imdbId, str) else None
        records.append(
            {
                "id": f"film:{slugify(title)}-{year or row.movieId}",
                "category": "film",
                "title": title,
                "creator": "",
                "year": year,
                "rank": int(counts[row.movieId]),
                "signal": {
                    "tags": top_tags.get(row.movieId, []),
                    "genres": [] if row.genres == "(no genres listed)" else row.genres.split("|"),
                },
                "source": {
                    "movielens": int(row.movieId),
                    "imdb": imdb,
                },
                "links": {"primary": f"https://www.imdb.com/title/{imdb}/"} if imdb else {},
            }
        )
    untagged = sum(1 for r in records if not r["signal"]["tags"])
    no_imdb = sum(1 for r in records if not r["source"]["imdb"])
    log.info(
        "film: %s kept; %s without genome tags, %s without an IMDb id",
        num(len(records)),
        num(untagged),
        num(no_imdb),
    )
    return records


def clean_shelves(names: list[str], limit: int) -> list[str]:
    out = []
    for name in names:
        n = name.lower().strip()
        tokens = set(re.split(r"[-_ ]+", n))
        if n and not tokens & SHELF_JUNK and not re.search(r"\d", n) and n not in out:
            out.append(n)
        if len(out) >= limit:
            break
    return out


def normalize_isbn(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().upper()
    if re.fullmatch(r"\d{9}[\dX]|\d{8}[\dX]", v):
        return v.zfill(10)
    return None


def curate_books(cfg: CurateConfig) -> list[Record]:
    root = RAW / "goodbooks"
    books = pd.read_csv(root / "books.csv", dtype={"isbn": str})
    total = len(books)
    books = books.sort_values("ratings_count", ascending=False).head(cfg.book_top)
    tag_names = pd.read_csv(root / "tags.csv").set_index("tag_id")["tag_name"].to_dict()
    book_tags = pd.read_csv(root / "book_tags.csv")
    book_tags = book_tags[book_tags["goodreads_book_id"].isin(books["goodreads_book_id"])]
    book_tags = book_tags.sort_values(["goodreads_book_id", "count"], ascending=[True, False])
    shelves = (
        book_tags.groupby("goodreads_book_id")["tag_id"]
        .apply(lambda s: [str(tag_names.get(t, "")) for t in s])
        .to_dict()
    )
    records = []
    for row in books.itertuples(index=False):
        title = SERIES_RE.sub("", str(row.title)).strip()
        author = str(row.authors).split(",")[0].strip()
        year = year_or_none(row.original_publication_year)
        records.append(
            {
                "id": f"book:{slugify(title)}-{slugify(author, 24)}",
                "category": "book",
                "title": title,
                "creator": author,
                "year": year,
                "rank": int(row.ratings_count),
                "signal": {
                    "shelves": clean_shelves(
                        shelves.get(row.goodreads_book_id, []), cfg.book_shelf_tags
                    )
                },
                "source": {
                    "goodreads": int(row.goodreads_book_id),
                    "isbn": normalize_isbn(row.isbn),
                },
                "links": {},
            }
        )
    no_shelves = sum(1 for r in records if not r["signal"]["shelves"])
    no_isbn = sum(1 for r in records if not r["source"]["isbn"])
    log.info(
        "book: top %s of %s by ratings count; %s without shelf tags, %s without an ISBN",
        num(len(records)),
        num(total),
        num(no_shelves),
        num(no_isbn),
    )
    return records


def parse_list(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.startswith("["):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return [str(x) for x in parsed]


def as_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 3)


def squash(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def with_spotify_popularity(muse: pd.DataFrame) -> pd.DataFrame:
    """Add track and artist popularity from the Spotify tracks dataset.

    A track matches by Spotify ID, else by artist and title. An artist matches by name
    and gets the highest popularity of any track they appear on.
    """
    tracks = pd.read_csv(
        SPOTIFY_TRACKS, usecols=["track_id", "artists", "track_name", "popularity"]
    )
    by_id = tracks.groupby("track_id")["popularity"].max()
    by_name = (
        tracks.assign(
            a=tracks["artists"].astype(str).str.split(";").str[0].map(squash),
            t=tracks["track_name"].map(squash),
        )
        .groupby(["a", "t"])["popularity"]
        .max()
        .to_dict()
    )
    artists = tracks["artists"].astype(str).str.split(";").explode()
    by_artist = (
        pd.Series(tracks.loc[artists.index, "popularity"].to_numpy(), index=artists.map(squash))
        .groupby(level=0)
        .max()
    )
    a, t = muse["artist"].map(squash), muse["track"].map(squash)
    by_name_pop = pd.Series(
        [by_name.get(key, np.nan) for key in zip(a, t, strict=True)], index=muse.index
    )
    return muse.assign(
        artist_key=a,
        track_pop=muse["spotify_id"].map(by_id).fillna(by_name_pop),
        artist_pop=a.map(by_artist),
    )


def select_songs(cfg: CurateConfig) -> tuple[pd.DataFrame, pd.Series]:
    """The top songs and their rank value.

    The Kaggle release ranks by Last.fm listeners, else by emotion tag count. The Zenodo
    release has no popularity column. Only about 2,000 of its tracks match the Spotify
    tracks dataset, so ranking by track popularity alone leaves too few. It ranks by artist
    popularity, then track popularity, with at most song_per_artist tracks per artist.
    """
    path = MUSE_KAGGLE if MUSE_KAGGLE.exists() else MUSE_ZENODO
    muse = pd.read_csv(path)
    total = len(muse)
    muse = muse[muse["spotify_id"].notna() & (muse["spotify_id"].astype(str).str.len() > 0)]
    muse = muse.drop_duplicates("spotify_id")
    log.info(
        "song: %s of %s MuSe tracks (%s) have a Spotify id", num(len(muse)), num(total), path.name
    )
    rank_col = next(
        (c for c in ("listeners", "lastfm_listeners", "number_of_emotion_tags") if c in muse),
        None,
    )
    if rank_col is not None:
        muse = muse.sort_values(rank_col, ascending=False).head(cfg.song_top)
        log.info("song: keep the top %s by %s", num(len(muse)), rank_col)
        return muse, muse[rank_col].fillna(0)
    muse = with_spotify_popularity(muse)
    log.info(
        "song: popularity from the Spotify tracks data: %s tracks match, %s artists match",
        num(int(muse["track_pop"].notna().sum())),
        num(int(muse.loc[muse["artist_pop"].notna(), "artist_key"].nunique())),
    )
    muse = muse.sort_values(["artist_pop", "track_pop"], ascending=False, na_position="last")
    muse = muse.groupby("artist_key", sort=False).head(cfg.song_per_artist).head(cfg.song_top)
    log.info(
        "song: keep the top %s by artist, then track popularity, at most %s per artist",
        num(len(muse)),
        num(cfg.song_per_artist),
    )
    return muse, muse["track_pop"].fillna(muse["artist_pop"]).fillna(0)


def curate_songs(cfg: CurateConfig) -> list[Record]:
    muse, rank = select_songs(cfg)
    records = []
    for (index, row), value in zip(muse.iterrows(), rank, strict=True):
        track, artist = str(row["track"]), str(row["artist"])
        spotify = str(row["spotify_id"])
        genre = row.get("genre")
        records.append(
            {
                "id": f"song:{slugify(artist, 24)}-{slugify(track, 40)}",
                "category": "song",
                "title": track,
                "creator": artist,
                "year": None,
                "rank": int(value),
                "signal": {
                    "valence": as_float(row.get("valence_tags")),
                    "arousal": as_float(row.get("arousal_tags")),
                    "dominance": as_float(row.get("dominance_tags")),
                    "tags": parse_list(row.get("seeds")),
                    "genre": genre if isinstance(genre, str) else None,
                },
                "source": {"spotify": spotify, "muse_row": int(index)},
                "links": {"spotify": f"https://open.spotify.com/track/{spotify}"},
            }
        )
    return records


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
    "Link Resource",
]


def load_met() -> pd.DataFrame:
    met = pd.read_csv(
        RAW / "met" / "MetObjects.csv", usecols=MET_COLUMNS, dtype=str, low_memory=False
    )
    met = met[(met["Is Public Domain"] == "True") & met["Title"].notna()]
    met["cls"] = met["Classification"].fillna("").str.split("|").str[0].str.strip()
    return met


def text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def art_record(row: dict[str, Any], quota: int, rank: int) -> Record:
    title = str(row["Title"]).strip()
    artist = text_or_none(row["Artist Display Name"]) or ""
    oid = int(row["Object ID"])
    return {
        "id": f"art:{slugify(title, 40)}-{oid}",
        "category": "art",
        "title": title,
        "creator": artist.split("|")[0].strip() or "Unknown artist",
        "year": year_or_none(row["Object Begin Date"]),
        "rank": rank,
        "signal": {
            "classification": row["cls"],
            "quota": quota,
            "medium": text_or_none(row["Medium"]),
            "date": text_or_none(row["Object Date"]),
            "culture": text_or_none(row["Culture"]),
            "department": row["Department"],
        },
        "source": {"met": oid},
        "links": {"primary": f"https://www.metmuseum.org/art/collection/search/{oid}"},
    }


def curate_art(cfg: CurateConfig) -> list[Record]:
    met = load_met()
    rng = random.Random(SEED)
    records = []
    for cls, share in cfg.art_quota.items():
        rows = met[met["cls"] == cls].to_dict("records")
        rng.shuffle(rows)
        rows.sort(key=lambda r: r["Is Highlight"] != "True")
        quota = round(cfg.art_target * share)
        picked = rows[: quota * cfg.art_candidate_factor]
        highlights = sum(1 for r in picked if r["Is Highlight"] == "True")
        log.debug(
            "art: %s: %s public-domain objects, %s candidates (%s highlights) for %s slots",
            cls,
            num(len(rows)),
            num(len(picked)),
            num(highlights),
            num(quota),
        )
        for rank, row in enumerate(picked):
            records.append(art_record(row, quota, -rank))
    by_cls = Counter(r["signal"]["classification"] for r in records)
    log.info(
        "art: %s candidates from %s public-domain objects (%s); resolve keeps %s with images",
        num(len(records)),
        num(len(met)),
        ", ".join(f"{k} {num(v)}" for k, v in by_cls.items()),
        num(cfg.art_target),
    )
    return records


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


def run() -> None:
    start = time.perf_counter()
    cfg = CurateConfig()
    log.info("reading data/raw: MovieLens, goodbooks-10k, MuSe, PoetryDB, Met, Text2Colors PAT")
    records: list[Record] = []
    steps = (
        ("film", curate_films),
        ("book", curate_books),
        ("song", curate_songs),
        ("poem", curate_poems),
        ("art", curate_art),
    )
    for _, fn in progress(steps, desc="curate", unit="category"):
        records.extend(fn(cfg))
    renamed = dedupe_ids(records)
    if renamed:
        log.info("%s items shared an id; they got a numeric suffix", num(renamed))
    records.sort(key=lambda r: r["id"])
    write_jsonl(CATALOG, records)
    pat = sorted(load_pat(), key=lambda r: (r["phrase"], r["rgb"]))
    write_jsonl(PAT, pat)
    counts = Counter(r["category"] for r in records)
    log.info(
        "done in %s: %s catalog items (%s) -> %s; %s PAT palettes -> %s",
        elapsed(start),
        num(len(records)),
        ", ".join(f"{c} {num(counts[c])}" for c in ("film", "book", "song", "poem", "art")),
        CATALOG.relative_to(ML_ROOT).as_posix(),
        num(len(pat)),
        PAT.relative_to(ML_ROOT).as_posix(),
    )
