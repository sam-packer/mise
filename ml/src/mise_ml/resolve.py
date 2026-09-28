import io
import re
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import quote, quote_plus, urlencode

import numpy as np
from PIL import Image

from mise_ml import keys
from mise_ml.color import image_tone
from mise_ml.config import (
    CATALOG,
    CATALOG_VERSION,
    CATEGORIES,
    IMG,
    ML_ROOT,
    RESOLVE_DROPPED,
    RESOLVED,
    CurateConfig,
    ResolveConfig,
)
from mise_ml.curate import (
    KEYWORD_JUNK,
    LASTFM,
    VERSION,
    base_title,
    catalog_version,
    norm,
    targets,
    tmdb_get,
    unique,
)
from mise_ml.http import CachedClient, FetchError
from mise_ml.log import elapsed, get, num, progress
from mise_ml.util import append_jsonl, iter_jsonl, sort_jsonl, write_jsonl

log = get(__name__)

Record = dict[str, Any]
Result = tuple[Record | None, str]

TMDB_IMAGES = "https://image.tmdb.org/t/p/w780"
DEEZER = "https://api.deezer.com"
WIKIDATA = "https://query.wikidata.org/sparql"
P18_QUERY = """SELECT ?item ?image WHERE {{
  VALUES ?item {{ {ids} }}
  ?item wdt:P18 ?image.
}}"""
OPENLIBRARY = "https://openlibrary.org"
MET = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
AIC_IIIF = "https://www.artic.edu/iiif/2"

# Last.fm tags whose top-track lists give songs their mood tags in bulk: one call returns up
# to 1,000 tracks. Moods first, then scenes, genres, and decades.
LASTFM_TAGS = (
    # moods
    "chill", "mellow", "sad", "happy", "melancholy", "melancholic", "romantic", "love",
    "energetic", "upbeat", "party", "relaxing", "calm", "dreamy", "atmospheric", "dark",
    "angry", "aggressive", "uplifting", "beautiful", "emotional", "nostalgic", "sexy", "fun",
    "feel good", "epic", "haunting", "ethereal", "moody", "bittersweet", "heartbreak",
    "breakup", "lonely", "hopeful", "peaceful", "intense", "groovy", "funky", "soulful",
    "catchy", "sensual", "hypnotic", "trippy", "cinematic", "introspective", "anthemic",
    "playful", "quirky", "whimsical", "eerie", "tender", "gentle", "soft", "powerful",
    "triumphant", "euphoric", "rebellious", "carefree", "wistful", "longing", "dramatic",
    "sombre", "depressing", "cheerful", "smooth", "lush", "raw", "sentimental", "serene",
    "spooky", "desperate", "confident", "cute", "warm", "cold", "hazy", "tense", "joyful",
    # scenes and times
    "summer", "winter", "autumn", "spring", "night", "late night", "morning", "rainy day",
    "sleep", "driving", "road trip", "workout", "dance", "study", "sunset", "beach",
    "christmas", "lazy", "cozy",
    # genres
    "rock", "pop", "hip-hop", "rap", "rnb", "soul", "jazz", "blues", "country", "folk",
    "indie", "indie pop", "indie rock", "alternative", "electronic", "house", "techno",
    "ambient", "classical", "metal", "punk", "emo", "k-pop", "j-pop", "latin", "reggaeton",
    "reggae", "disco", "funk", "motown", "gospel", "singer-songwriter", "shoegaze",
    "dream pop", "synthpop", "new wave", "post-punk", "grunge", "britpop", "classic rock",
    "hard rock", "soft rock", "psychedelic", "trap", "lo-fi", "bedroom pop", "neo-soul",
    "afrobeats", "bossa nova", "art pop", "hyperpop", "pop punk", "garage rock",
    "rock and roll", "doo wop", "surf", "baroque pop", "trip-hop", "drum and bass", "edm",
    # decades
    "50s", "60s", "70s", "80s", "90s", "00s", "2010s", "oldies",
)  # fmt: skip
LASTFM_TAG_SET = frozenset(LASTFM_TAGS)
# A song with fewer tags than this from the bulk lists gets its own track.getTopTags call.
SONG_MIN_TAGS = 3
TAG_JUNK = re.compile(r"seen live|favou?rites?|\bmy\b|\bbest\b|spotify|albums? i own|^\d{4}$")


# Album titles that mark a compilation, live, or remix release rather than the original.
NOT_ORIGINAL = re.compile(
    r"\b(live|remix(es)?|greatest hits|best of|hits|collection|anthology|essentials?|"
    r"karaoke|tribute|compilation|now that'?s|in the style of|instrumental|covers?)\b",
    re.IGNORECASE,
)


def song_key(artist: str, title: str) -> tuple[str, str]:
    return norm(artist), norm(base_title(title))


def song_rank(track: dict, want_track: str) -> tuple[bool, bool]:
    """Sort key: original album first, then an exact title before a longer one."""
    album = track.get("album", {}).get("title", "")
    exact = norm(base_title(track.get("title", ""))) == want_track
    return (bool(NOT_ORIGINAL.search(album)), not exact)


def commons_url(file_url: str) -> str:
    """A Commons Special:FilePath URL for a 960 px thumbnail (a standard Wikimedia size)."""
    return file_url.replace("http://", "https://", 1) + "?width=960"


class Resolver:
    def __init__(self, cfg: ResolveConfig) -> None:
        self.cfg = cfg
        self.curate = CurateConfig()
        self.http = CachedClient(cfg)
        self.met_images: dict[str, str] = {}
        self.song_tags: dict[tuple[str, str], list[tuple[int, str]]] = {}

    def image(self, url: str, item_id: str) -> dict | None:
        name = item_id.replace(":", "-") + ".webp"
        path = IMG / name
        if path.exists():
            img = Image.open(path).convert("RGB")
        else:
            data = self.http.get_bytes(url)
            if not data:
                return None
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except OSError:
                return None
            if min(img.size) < 64:
                return None
            img.thumbnail(
                (self.cfg.image_long_edge, self.cfg.image_long_edge), Image.Resampling.LANCZOS
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path, "WEBP", quality=self.cfg.webp_quality, method=6)
        small = img.copy()
        small.thumbnail((64, 64))
        return {
            "src": f"/bundle/img/{name}",
            "url": url,
            "w": img.width,
            "h": img.height,
            "tone": image_tone(np.asarray(small)),
        }

    # batch lookups, once per run

    def load_met_images(self, art: list[Record]) -> None:
        """Commons images for Met objects, 200 Wikidata items per SPARQL query (P18)."""
        ids = sorted({r["source"]["wikidata"] for r in art if r["source"].get("wikidata")})
        files: dict[str, list[str]] = defaultdict(list)
        for start in range(0, len(ids), 200):
            values = " ".join(f"wd:{q}" for q in ids[start : start + 200])
            query = urlencode({"query": P18_QUERY.format(ids=values), "format": "json"})
            try:
                body = self.http.get_json(f"{WIKIDATA}?{query}") or {}
            except FetchError as e:
                log.warning(f"art: Wikidata failed for {len(ids[start : start + 200])} items ({e})")
                continue
            for b in body.get("results", {}).get("bindings", []):
                files[b["item"]["value"].rsplit("/", 1)[-1]].append(b["image"]["value"])
        # Prefer the Met's own photograph: its Commons file name carries "MET".
        self.met_images = {
            q: commons_url(min(urls, key=lambda u: ("MET" not in u, u)))
            for q, urls in files.items()
        }
        log.info(
            f"art: Commons images for {num(len(self.met_images))} of {num(len(ids))} Met "
            f"objects with a Wikidata item; the rest use the Met API"
        )

    def load_song_tags(self) -> None:
        """Mood and genre tags from the top-track list of each tag in LASTFM_TAGS."""
        found: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
        for tag in progress(LASTFM_TAGS, desc="last.fm tags", unit="tag"):
            params = {"method": "tag.gettoptracks", "tag": tag, "limit": 1000, "format": "json"}
            url = f"{LASTFM}?{urlencode(params)}"
            body = self.http.get_json(url, secret=keys.lastfm()) or {}
            for rank, t in enumerate((body.get("tracks") or {}).get("track", [])):
                found[song_key(t["artist"]["name"], t["name"])].append((rank, tag))
        self.song_tags = dict(found)
        log.info(
            f"song: tag lists for {len(LASTFM_TAGS)} Last.fm tags cover "
            f"{num(len(self.song_tags))} tracks"
        )

    # one item each

    def film(self, r: Record) -> Result:
        tmdb = r["source"]["tmdb"]
        d = tmdb_get(self.http, f"/movie/{tmdb}", {"append_to_response": "credits,keywords"})
        if d is None:
            return None, "tmdb_not_found"
        poster = d.get("poster_path")
        if not poster:
            return None, "no_tmdb_poster"
        image = self.image(f"{TMDB_IMAGES}{poster}", r["id"])
        if image is None:
            return None, "no_image"
        crew = (d.get("credits") or {}).get("crew", [])
        directors = unique((c["name"] for c in crew if c.get("job") == "Director"), 2)
        words = (d.get("keywords") or {}).get("keywords", [])
        imdb = d.get("imdb_id")
        signal = {
            **r["signal"],
            "genres": [g["name"] for g in d.get("genres", [])] or r["signal"]["genres"],
            "keywords": unique(
                (k["name"] for k in words if not KEYWORD_JUNK.search(k["name"])),
                self.curate.film_keywords,
            ),
            "tagline": (d.get("tagline") or "").strip() or None,
        }
        primary = f"https://www.imdb.com/title/{imdb}/" if imdb else r["links"]["primary"]
        return {
            **r,
            "creator": ", ".join(directors) or "Unknown director",
            "image": image,
            "signal": signal,
            "source": {**r["source"], "imdb": imdb},
            "links": {"primary": primary},
        }, "ok"

    def deezer_track(self, r: Record) -> tuple[dict | None, str]:
        """The Deezer track: by ISRC when MusicBrainz has one, else by search."""
        title = base_title(r["title"])
        want_artist, want_track = norm(r["creator"]), norm(title)

        def fits(t: dict) -> bool:
            return (
                norm(t.get("artist", {}).get("name", "")) == want_artist
                and norm(base_title(t.get("title", ""))).startswith(want_track)
                and bool(t.get("album", {}).get("cover_xl"))
            )

        isrc = r["source"].get("isrc")
        if isrc:
            t = self.http.get_json(f"{DEEZER}/track/isrc:{isrc}")
            if t and fits(t) and not VERSION.search(t["title"]):
                return t, "ok"
        query = f"{r['creator']} {title}"
        found = self.http.get_json(f"{DEEZER}/search?{urlencode({'q': query, 'limit': 25})}")
        matches = [t for t in (found or {}).get("data", []) if fits(t)]
        if not matches:
            return None, "no_deezer_match"
        if not VERSION.search(r["title"]):
            matches = [t for t in matches if not VERSION.search(t["title"])]
            if not matches:
                return None, "only_other_versions_on_deezer"
        return min(matches, key=lambda t: song_rank(t, want_track)), "ok"

    def lastfm_tags(self, r: Record) -> list[str]:
        """Tags from the bulk tag lists; a song with too few gets track.getTopTags."""
        ranked = sorted(self.song_tags.get(song_key(r["creator"], r["title"]), []))
        tags = unique((tag for _, tag in ranked), self.curate.song_tags)
        if len(tags) >= SONG_MIN_TAGS:
            return tags
        params = {
            "method": "track.gettoptags",
            "artist": r["creator"],
            "track": base_title(r["title"]),
            "autocorrect": 1,
            "format": "json",
        }
        body = self.http.get_json(f"{LASTFM}?{urlencode(params)}", secret=keys.lastfm()) or {}
        artist = r["creator"].lower()
        own = [
            t["name"].strip().lower()
            for t in (body.get("toptags") or {}).get("tag", [])
            if int(t.get("count") or 0) >= 5
            and t["name"].strip().lower() != artist
            and not TAG_JUNK.search(t["name"].lower())
        ]
        return unique(tags + own, self.curate.song_tags)

    def song(self, r: Record) -> Result:
        track, reason = self.deezer_track(r)
        if track is None:
            return None, reason
        image = self.image(track["album"]["cover_xl"], r["id"])
        if image is None:
            return None, "no_image"
        query = f"{r['creator']} {base_title(r['title'])}"
        links = {
            "primary": track["link"],
            "deezer": track["link"],
            "spotify": f"https://open.spotify.com/search/{quote(query)}",
            "apple": f"https://music.apple.com/us/search?term={quote(query)}",
            "youtube": f"https://www.youtube.com/results?search_query={quote_plus(query)}",
        }
        return {
            **r,
            "album": track["album"]["title"],
            "image": image,
            # Deezer preview URLs expire after minutes; the app's Worker serves this path.
            "preview": f"/api/preview/deezer/{track['id']}",
            "signal": {**r["signal"], "tags": self.lastfm_tags(r)},
            "source": {**r["source"], "deezer": track["id"]},
            "links": links,
        }, "ok"

    def book(self, r: Record) -> Result:
        image = None
        if r["source"].get("image"):
            image = self.image(r["source"]["image"], r["id"])
        if image is None:
            # Open Library has a cover for many books that Hardcover lacks one for.
            q = f"title={quote_plus(r['title'])}&author={quote_plus(r['creator'])}"
            found = self.http.get_json(f"{OPENLIBRARY}/search.json?{q}&limit=1&fields=cover_i")
            cover = next(iter((found or {}).get("docs", [])), {}).get("cover_i")
            if cover:
                url = f"https://covers.openlibrary.org/b/id/{cover}-L.jpg"
                image = self.image(url, r["id"])
        if image is None:
            return None, "no_cover"
        return {**r, "image": image}, "ok"

    def art(self, r: Record) -> Result:
        s = r["source"]
        if "aic" in s:
            url = f"{AIC_IIIF}/{s['image_id']}/full/843,/0/default.jpg"
        elif "cma" in s:
            url = s["image"]
        else:
            url = self.met_images.get(s.get("wikidata") or "")
            if url is None:
                obj = self.http.get_json(f"{MET}/{s['met']}")
                if obj is None:
                    return None, "met_not_found"
                url = obj.get("primaryImageSmall") or obj.get("primaryImage")
                if not url:
                    return None, "no_met_image"
        image = self.image(url, r["id"])
        if image is None:
            return None, "no_image"
        return {**r, "image": image}, "ok"

    def poem(self, r: Record) -> Result:
        return {**r, "image": None}, "ok"


def rank_key(r: Record) -> tuple[float, str]:
    """Candidate order: the most popular first, ties by id."""
    return (-r["rank"], r["id"])


def trim_to_targets(group_targets: dict[str, int]) -> None:
    """Keep exactly the `target` best-ranked resolved items of each group.

    Parallel workers can resolve a few more than the target. Every candidate that ranks
    above the kept ones was tried, so the result is the same as a one-by-one run.
    """
    rows = list(iter_jsonl(RESOLVED))
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in rows:
        if r.get("group") in group_targets:
            groups[r["group"]].append(r)
    extra: dict[str, Record] = {}
    for group, members in groups.items():
        for r in sorted(members, key=rank_key)[group_targets[group] :]:
            extra[r["id"]] = r
    if not extra:
        return
    for r in extra.values():
        if r.get("image"):
            (IMG / r["image"]["src"].rsplit("/", 1)[-1]).unlink(missing_ok=True)
    write_jsonl(RESOLVED, [r for r in rows if r["id"] not in extra])
    log.debug(f"trimmed {len(extra)} lower-ranked extras: {sorted(extra)}")


def log_targets(category: str, catalog: list[Record], group_targets: dict[str, int]) -> None:
    candidates = {r["id"]: r for r in catalog if r["category"] == category}
    kept = Counter(r.get("group") for r in iter_jsonl(RESOLVED) if r["id"] in candidates)
    drops = Counter(d["reason"] for d in iter_jsonl(RESOLVE_DROPPED) if d["id"] in candidates)
    reasons = ", ".join(f"{k} {num(n)}" for k, n in sorted(drops.items()))
    line = (
        f"{category}: {num(sum(kept.values()))} kept of {num(len(candidates))} candidates "
        f"({num(sum(drops.values()))} dropped{': ' + reasons if reasons else ''})"
    )
    (log.warning if drops else log.info)(line)
    for group in sorted(g for g in group_targets if g.startswith(f"{category}:")):
        if kept[group] < group_targets[group]:
            log.warning(
                f"{group}: {num(kept[group])} of {num(group_targets[group])} kept; the "
                "candidates ran out (raise candidate_factor in CurateConfig)"
            )


def run(categories: list[str] | None = None) -> None:
    start = time.perf_counter()
    cfg = ResolveConfig()
    if catalog_version() != CATALOG_VERSION:
        raise SystemExit(f"{CATALOG} is missing or in an old format; run curate first")
    catalog = list(iter_jsonl(CATALOG))
    wanted = [c for c in CATEGORIES if not categories or c in categories]
    keys.require([s for c, s in (("film", "tmdb"), ("song", "lastfm")) if c in wanted])
    resolved = list(iter_jsonl(RESOLVED))
    dropped_before = list(iter_jsonl(RESOLVE_DROPPED))
    done = {r["id"] for r in resolved} | {r["id"] for r in dropped_before}

    by_cat: dict[str, list[Record]] = defaultdict(list)
    for r in catalog:
        if r["id"] not in done and r["category"] in wanted:
            by_cat[r["category"]].append(r)
    for rows in by_cat.values():
        rows.sort(key=rank_key)
    pending = ", ".join(f"{c} {num(len(by_cat[c]))}" for c in wanted if by_cat[c])
    log.info(
        f"reading {CATALOG.name}: {num(len(catalog))} items; {num(len(resolved))} resolved and "
        f"{num(len(dropped_before))} dropped by earlier runs; to do: {pending or 'nothing'}"
    )

    resolver = Resolver(cfg)
    lock = threading.Lock()
    stats: dict[str, Counter[str]] = {c: Counter() for c in wanted}
    group_targets = targets(CurateConfig())
    kept_keys: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for r in resolved:
        if r.get("group") in group_targets:
            kept_keys[r["group"]].append(rank_key(r))

    def cutoffs() -> dict[str, tuple[float, str]]:
        """For each full group, the key of its last kept item."""
        with lock:
            return {
                g: sorted(k)[group_targets[g] - 1]
                for g, k in kept_keys.items()
                if len(k) >= group_targets[g]
            }

    def work(category: str) -> None:
        items = by_cat[category]
        if category == "song" and items:
            resolver.load_song_tags()
        if category == "art" and items:
            resolver.load_met_images([r for r in catalog if "met" in r["source"]])
        fn: Callable[[Record], Result] = getattr(resolver, category)
        counts = stats[category]
        bar = progress(
            total=len(items), desc=category, unit="item", position=CATEGORIES.index(category)
        )

        def one(r: Record) -> None:
            try:
                out, reason = fn(r)
            except Exception as e:  # one bad record must not stop a multi-hour run
                out, reason = None, f"error: {type(e).__name__}"
                log.debug(f"{r['id']}: {type(e).__name__}: {e}")
            with lock:
                counts[reason] += 1
                if out is not None:
                    append_jsonl(RESOLVED, [out])
                    if out.get("group") in group_targets:
                        kept_keys[out["group"]].append(rank_key(out))
                elif not reason.startswith("error"):
                    log.debug(f"drop {r['id']}: {reason}")
                    append_jsonl(RESOLVE_DROPPED, [{"id": r["id"], "reason": reason}])
                looked_up = resolver.http.hits + resolver.http.requests
                bar.update()
                bar.set_postfix(
                    ok=counts["ok"],
                    dropped=sum(n for k, n in counts.items() if k != "ok" and ":" not in k),
                    errors=sum(n for k, n in counts.items() if k.startswith("error")),
                    cached=f"{resolver.http.hits / max(looked_up, 1):.0%}",
                    refresh=False,
                )

        workers = cfg.workers.get(category, 1)
        chunk = workers * 8
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i in range(0, len(items), chunk):
                # Once a group keeps its target, only a candidate that ranks above the last
                # kept item can still change the result.
                cut = cutoffs()
                batch = [
                    r
                    for r in items[i : i + chunk]
                    if r.get("group") not in cut or rank_key(r) < cut[r["group"]]
                ]
                bar.update(len(items[i : i + chunk]) - len(batch))
                list(pool.map(one, batch))
        bar.close()

    with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
        list(pool.map(work, wanted))
    resolver.http.close()
    trim_to_targets(group_targets)
    sort_jsonl(RESOLVED, "id")
    sort_jsonl(RESOLVE_DROPPED, "id")

    for category in wanted:
        counts = stats[category]
        errors = {k: n for k, n in counts.items() if k.startswith("error")}
        if category == "poem":
            if counts:
                log.info(f"poem: {num(counts['ok'])} resolved")
        else:
            log_targets(category, catalog, group_targets)
        if errors:
            reasons = ", ".join(f"{k.split(': ')[1]} {num(n)}" for k, n in sorted(errors.items()))
            log.warning(
                f"{category}: {num(sum(errors.values()))} items failed ({reasons}); "
                "the next run tries them again (details in the log file)"
            )
    totals = Counter(r["category"] for r in iter_jsonl(RESOLVED))
    log.info(
        f"done in {elapsed(start)}: {num(sum(totals.values()))} items in "
        f"{RESOLVED.relative_to(ML_ROOT).as_posix()} "
        f"({', '.join(f'{c} {num(totals[c])}' for c in CATEGORIES)}); "
        f"{num(resolver.http.requests)} API calls, {num(resolver.http.hits)} answers from the cache"
    )
