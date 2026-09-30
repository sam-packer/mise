"""Resolve catalog candidates to media and links, then keep each group's highest-ranked items."""

import hashlib
import io
import json
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, quote_plus, urlencode, urlsplit

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
    RESOLVE_META,
    RESOLVE_VERSION,
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
from mise_ml.threads import run_all
from mise_ml.util import (
    append_jsonl,
    atomic_write,
    iter_jsonl,
    sort_jsonl,
    write_json,
    write_jsonl,
)

log = get(__name__)

Record = dict[str, Any]
Result = tuple[Record | None, str]

TMDB_IMAGES = "https://image.tmdb.org/t/p/w780"
DEEZER = "https://api.deezer.com"
# Deezer can return this JPEG even for a nonempty md5_image (track 185287).
# Include the WebP produced by image() to detect cached copies of the placeholder.
PLACEHOLDER_HASHES = {
    "b1fad669172f6263a50ca6d519a91d8cd16488014c1cb1623d823d2106c8586a",
    "5a4c35e31281988c6dbdeb16c22ab43f9745cc79afce62c9fa0f8d14b9c40180",
}
LASTFM_PLACEHOLDER = "2a96cbd8b46e442fc41c2b86b821562f"


def placeholder_url(url: str) -> bool:
    return "/images/cover//" in url or LASTFM_PLACEHOLDER in url


def placeholder_bytes(data: bytes) -> bool:
    return hashlib.sha256(data).hexdigest() in PLACEHOLDER_HASHES


WIKIDATA = "https://query.wikidata.org/sparql"
P18_QUERY = """SELECT ?item ?image WHERE {{
  VALUES ?item {{ {ids} }}
  ?item wdt:P18 ?image.
}}"""
# Art Institute of Chicago objects on Wikidata: P4610 is the "ARTIC artwork ID".
AIC_P18_QUERY = """SELECT ?item ?image WHERE {{
  VALUES ?item {{ {ids} }}
  ?object wdt:P4610 ?item; wdt:P18 ?image.
}}"""
OPENLIBRARY = "https://openlibrary.org"
MET = "https://collectionapi.metmuseum.org/public/collection/v1/objects"

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
# A song with fewer tags than this from the bulk lists gets its own track.getTopTags call.
SONG_MIN_TAGS = 3
TAG_JUNK = re.compile(r"seen live|favou?rites?|\bmy\b|\bbest\b|spotify|albums? i own|^\d{4}$")


# Title clues supplement Deezer's record_type and Various Artists album credit.
COMPILATION_TITLE = re.compile(
    r"\b(greatest hits|best of|hits|collection|anthology|essentials?|ultimate|solid gold|"
    r"soundtrack|motion picture|compilation|coffret|awards|radio \d+|vol(?:ume)?\.?\s*\d+|"
    r"now that'?s)\b",
    re.IGNORECASE,
)


def song_key(artist: str, title: str) -> tuple[str, str]:
    return norm(artist), norm(base_title(title))


def album_key(title: str) -> str:
    """Normalize edition suffixes, accents, and optional leading articles."""
    title = re.sub(r"^the\s+", "", base_title(title), flags=re.I)
    title = re.sub(
        r"\s+-\s+(?:(?:the )?\d+(?:st|nd|rd|th) (?:mini )?album|.*remaster.*)$",
        "",
        title,
        flags=re.I,
    )
    title = re.sub(
        r"\s*[([][^)\]]*(?:edition|remaster|deluxe|anniversary|expanded|bonus)[^)\]]*[)\]]",
        "",
        title,
        flags=re.I,
    )
    title = "".join(c for c in unicodedata.normalize("NFKD", title) if not unicodedata.combining(c))
    return "".join(c for c in title.casefold().replace("&", "and") if c.isalnum())


def release_keys(group: dict) -> set[str]:
    return {album_key(title) for title in [group["title"], *group.get("release_titles", [])]}


def compilation(album: dict) -> bool:
    return (
        album.get("record_type") == "compile"
        or norm(album.get("artist", {}).get("name", "")) == "variousartists"
        or bool(COMPILATION_TITLE.search(album.get("title", "")))
    )


# Joins in an artist credit: "A feat. B", "A ft. B", "A with B", "A x B", "A & B", "A, B".
CREDIT_SPLIT = re.compile(
    r"\s+(?:feat\.?|ft\.?|featuring|with|x|\u00d7)\s+|\s*(?:,|&|\+)\s*", re.IGNORECASE
)


def lead_artist(credit: str) -> str:
    """The first name of an artist credit: "Tyla feat. Gunna & Skillibeng" -> "Tyla"."""
    return CREDIT_SPLIT.split(credit, maxsplit=1)[0].strip() or credit


def commons_url(file_url: str) -> str:
    """A Commons Special:FilePath URL for a 960 px thumbnail (a standard Wikimedia size)."""
    return file_url.replace("http://", "https://", 1) + "?width=960"


class Resolver:
    def __init__(self, cfg: ResolveConfig) -> None:
        self.cfg = cfg
        # Extend this resolver's config without changing limits supplied by callers.
        for host, interval in {
            "lastfm.freetls.fastly.net": 0.2,
            "lastfm-img.freetls.fastly.net": 0.2,
            "itunes.apple.com": 3.1,
        }.items():
            cfg.min_interval.setdefault(host, interval)
        self.curate = CurateConfig()
        self.http = CachedClient(cfg)
        self.met_images: dict[str, str] = {}
        self.aic_images: dict[str, str] = {}
        self.song_tags: dict[tuple[str, str], list[tuple[int, str]]] = {}

    def image(
        self, url: str, item_id: str, *, min_edge: int = 64, refresh: bool = False
    ) -> dict | None:
        if not url or placeholder_url(url):
            return None
        name = item_id.replace(":", "-") + ".webp"
        path = IMG / name
        img = None
        if not refresh and path.exists() and not placeholder_bytes(path.read_bytes()):
            try:
                img = Image.open(path).convert("RGB")
            except OSError:
                # A file cut short by a hard kill: remove it and fetch the image again.
                log.warning(f"{name}: the image file is damaged; fetch it again")
                path.unlink()
        if img is None:
            data = self.http.get_bytes(url)
            if not data or placeholder_bytes(data):
                return None
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except OSError:
                return None
            if min(img.size) < min_edge:
                return None
            img.thumbnail(
                (self.cfg.image_long_edge, self.cfg.image_long_edge), Image.Resampling.LANCZOS
            )
            if min(img.size) < min_edge:
                return None
            out = io.BytesIO()
            img.save(out, "WEBP", quality=self.cfg.webp_quality, method=6)
            if placeholder_bytes(out.getvalue()):
                return None
            atomic_write(path, out.getvalue())
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

    def commons_files(self, query: str, values: list[str]) -> dict[str, list[str]]:
        """Commons files (P18) for Wikidata SPARQL VALUES, 200 per query."""
        files: dict[str, list[str]] = defaultdict(list)
        for start in range(0, len(values), 200):
            ids = " ".join(values[start : start + 200])
            params = urlencode({"query": query.format(ids=ids), "format": "json"})
            try:
                body = self.http.get_json(f"{WIKIDATA}?{params}") or {}
            except FetchError as e:
                log.warning(
                    f"art: Wikidata failed for {len(values[start : start + 200])} items ({e})"
                )
                continue
            for b in body.get("results", {}).get("bindings", []):
                files[b["item"]["value"].rsplit("/", 1)[-1]].append(b["image"]["value"])
        return files

    def load_art_images(self, art: list[Record]) -> None:
        """Find Commons images by Wikidata item for the Met and by ARTIC artwork ID for Chicago.

        Chicago's IIIF server returns Cloudflare 403 responses, so use Commons for its images.
        """
        met = sorted({r["source"]["wikidata"] for r in art if r["source"].get("wikidata")})
        files = self.commons_files(P18_QUERY, [f"wd:{q}" for q in met])
        # Prefer the Met's own photograph: its Commons file name carries "MET".
        self.met_images = {
            q: commons_url(min(urls, key=lambda u: ("MET" not in u, u)))
            for q, urls in files.items()
        }
        aic = sorted({str(r["source"]["aic"]) for r in art if "aic" in r["source"]}, key=int)
        files = self.commons_files(AIC_P18_QUERY, [f'"{a}"' for a in aic])
        self.aic_images = {a: commons_url(min(urls)) for a, urls in files.items()}
        log.info(
            f"art: Commons images for {num(len(self.met_images))} of {num(len(met))} Met "
            f"objects with a Wikidata item (the rest use the Met API) and for "
            f"{num(len(self.aic_images))} of {num(len(aic))} Art Institute of Chicago objects"
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
                lead = lead_artist(t["artist"]["name"])
                if lead != t["artist"]["name"]:
                    found[song_key(lead, t["name"])].append((rank, tag))
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

    def deezer_album(self, track: dict) -> dict:
        album = track.get("album") or {}
        if album.get("id") and "record_type" not in album:
            full = self.http.get_json(f"{DEEZER}/album/{album['id']}") or {}
            # Deezer can redirect a deleted compilation to an unrelated album.
            if full.get("id") == album["id"] or (
                full and album_key(full.get("title", "")) == album_key(album.get("title", ""))
            ):
                album = {
                    **album,
                    **{k: full[k] for k in ("record_type", "release_date", "artist") if k in full},
                }
                track["album"] = album
        return album

    def current_track(self, r: Record) -> dict | None:
        isrc = r["source"].get("isrc")
        track = self.http.get_json(f"{DEEZER}/track/isrc:{isrc}") if isrc else None
        if not track or track.get("id") != r["source"].get("deezer"):
            track = self.http.get_json(f"{DEEZER}/track/{r['source']['deezer']}")
        return track

    def release_groups(self, r: Record) -> list[dict]:
        """Browse every official release; recording lookups truncate their release list."""
        mbid = r["source"].get("mbid")
        if not mbid:
            return []
        groups = {}
        offset = 0
        while True:
            params = urlencode(
                {
                    "recording": mbid,
                    "inc": "release-groups",
                    "fmt": "json",
                    "limit": 100,
                    "offset": offset,
                }
            )
            body = self.http.get_json(f"https://musicbrainz.org/ws/2/release?{params}") or {}
            releases = body.get("releases", [])
            for release in releases:
                group = release.get("release-group") or {}
                if release.get("status") == "Official" and group.get("id"):
                    entry = groups.setdefault(group["id"], {**group, "release_titles": []})
                    if release.get("title") not in entry["release_titles"]:
                        entry["release_titles"].append(release.get("title") or group["title"])
            offset += len(releases)
            if not releases or offset >= body.get("release-count", 0):
                break
        return sorted(
            groups.values(), key=lambda g: (g.get("first-release-date") or "9999", g["id"])
        )

    def compilation_song(self, r: Record, album: dict) -> bool:
        if compilation(album) or compilation({"title": r.get("album", "")}):
            return True
        if not album.get("record_type"):
            matching = [
                g
                for g in self.release_groups(r)
                if album_key(r.get("album", "")) in release_keys(g)
            ]
            if matching:
                return any(
                    set(g.get("secondary-types", [])) & {"Compilation", "Soundtrack"}
                    for g in matching
                )
            # Deleted annual collections and mood playlists have no album type.
            return bool(
                re.search(
                    r"\b(?:(?:19|20)\d{2}|chill|playlist|classics?)\b", r.get("album", ""), re.I
                )
            )
        return False

    def deezer_track(self, r: Record, current: dict | None = None) -> tuple[dict | None, str]:
        """Prefer the original release without relaxing song or artist identity."""
        title = base_title(r["title"])
        lead = lead_artist(r["creator"])
        want_artists = {norm(r["creator"]), norm(lead)}
        want_track = norm(title)

        def fits(t: dict) -> bool:
            names = [t.get("artist", {}).get("name", "")]
            names += [c.get("name", "") for c in t.get("contributors") or []]
            return bool(want_artists & {norm(n) for n in names}) and norm(
                base_title(t.get("title", ""))
            ).startswith(want_track)

        def valid(t: dict) -> bool:
            return fits(t) and (bool(VERSION.search(r["title"])) or not VERSION.search(t["title"]))

        if current and not valid(current):
            current = None
        matches = []
        if current:
            matches.append(current)
        isrc = r["source"].get("isrc")
        if isrc:
            t = self.http.get_json(f"{DEEZER}/track/isrc:{isrc}")
            if t and fits(t) and not VERSION.search(t["title"]):
                matches.append(t)
        for artist in dict.fromkeys([r["creator"], lead]):
            query = urlencode({"q": f"{artist} {title}", "limit": 25})
            found = self.http.get_json(f"{DEEZER}/search?{query}")
            matches.extend(t for t in (found or {}).get("data", []) if valid(t))
        matches = list({t["id"]: t for t in matches}.values())
        for t in matches:
            self.deezer_album(t)

        # Deezer dates describe reissues. MusicBrainz ties the recording to its
        # original release group, including albums absent from track search.
        groups = self.release_groups(r)
        originals = [
            g
            for g in groups
            if g.get("primary-type") in ("Album", "Single", "EP")
            and not set(g.get("secondary-types", []))
            & {"Compilation", "Soundtrack", "Live", "Remix", "DJ-mix"}
        ]
        albums = [g for g in originals if g.get("primary-type") == "Album"]
        if not albums:
            # MusicBrainz can link a radio edit only to its single, while Deezer
            # has a matching track on the studio album. Keep the identity checks.
            studio = [
                t
                for t in matches
                if t["album"].get("record_type") == "album"
                and not compilation(t["album"])
                and not VERSION.search(t["album"].get("title", ""))
            ]
            if studio:
                return min(
                    studio,
                    key=lambda t: (
                        t["album"].get("release_date") or "9999",
                        norm(base_title(t["title"])) != want_track,
                        t["id"],
                    ),
                ), "Deezer studio album (MusicBrainz has no studio album for this recording)"
        # A recording can be a bonus track on an older album's reissue. If that
        # album has no matching track in Deezer, try the next official release.
        ordered = albums + [g for g in originals if g not in albums]
        for original in ordered:
            names = release_keys(original)
            if (
                current
                and album_key(r.get("album", "")) in names
                and album_key(current["album"]["title"]) in names
            ):
                return current, f"already on original release group {original['id']}"
            preferred = [t for t in matches if album_key(t["album"]["title"]) in names]
            if not preferred:
                query = urlencode({"q": f"{lead} {original['title']}", "limit": 100})
                found = self.http.get_json(f"{DEEZER}/search/album?{query}") or {}

                def album_lists(found: dict):
                    yield found.get("data", [])
                    artist_id = next(
                        (
                            t.get("artist", {}).get("id")
                            for t in matches
                            if t.get("artist", {}).get("id")
                        ),
                        None,
                    )
                    if not artist_id and current and fits(current):
                        artist_id = current.get("artist", {}).get("id")
                    if artist_id:
                        url = f"{DEEZER}/artist/{artist_id}/albums?limit=100"
                        while url:
                            page = self.http.get_json(url) or {}
                            yield page.get("data", [])
                            url = page.get("next")

                for candidates in album_lists(found):
                    for album in candidates:
                        if album_key(album["title"]) not in names:
                            continue
                        full = self.http.get_json(f"{DEEZER}/album/{album['id']}")
                        if not full or album_key(full["title"]) not in names:
                            continue
                        page = full.get("tracks") or {}
                        while True:
                            for t in page.get("data", []):
                                if valid(t):
                                    preferred.append({**t, "album": full})
                            if not page.get("next"):
                                break
                            page = self.http.get_json(page["next"]) or {}
                    if preferred:
                        break
            if preferred:
                chosen = min(
                    preferred,
                    key=lambda t: (
                        t["album"].get("record_type") == "compile",
                        not t.get("readable", True),
                        norm(base_title(t["title"])) != want_track,
                        not bool(t.get("preview")),
                        t["id"],
                    ),
                )
                return chosen, f"original release group {original['id']}"
        if current and groups:
            titles = ", ".join(repr(g["title"]) for g in ordered)
            return current, (
                f"no matching Deezer track on official original releases: {titles}"
                if ordered
                else "no official studio album, single, or EP for this recording"
            )
        if not matches:
            return None, "no_deezer_match"
        return min(
            matches,
            key=lambda t: (
                compilation(t["album"]) or bool(VERSION.search(t["album"].get("title", ""))),
                norm(base_title(t["title"])) != want_track,
                t["album"].get("release_date") or "9999",
                t["id"],
            ),
        ), "Deezer album type and date (no original release group)"

    def lastfm_tags(self, r: Record) -> list[str]:
        """Tags from the bulk tag lists; a song with too few gets track.getTopTags."""
        lead = lead_artist(r["creator"])
        ranked = sorted(
            self.song_tags.get(song_key(r["creator"], r["title"]))
            or self.song_tags.get(song_key(lead, r["title"]))
            or []
        )
        tags = unique((tag for _, tag in ranked), self.curate.song_tags)
        if len(tags) >= SONG_MIN_TAGS:
            return tags
        params = {
            "method": "track.gettoptags",
            "artist": lead,
            "track": base_title(r["title"]),
            "autocorrect": 1,
            "format": "json",
        }
        body = self.http.get_json(f"{LASTFM}?{urlencode(params)}", secret=keys.lastfm()) or {}
        artist = lead.lower()
        own = [
            t["name"].strip().lower()
            for t in (body.get("toptags") or {}).get("tag", [])
            if int(t.get("count") or 0) >= 5
            and t["name"].strip().lower() != artist
            and not TAG_JUNK.search(t["name"].lower())
        ]
        return unique(tags + own, self.curate.song_tags)

    def song_image(self, r: Record, track: dict) -> dict | None:
        """Accept only covers with at least 600 pixels on both edges, without upscaling."""
        album = track.get("album") or {}
        album_title = album.get("title") or r.get("album") or ""
        errors = []

        def cover(url: str) -> dict | None:
            if not url:
                return None
            host = urlsplit(url).hostname
            if host and host.endswith("mzstatic.com"):
                self.cfg.min_interval.setdefault(host, 0.2)
            try:
                result = self.image(url, r["id"], min_edge=600, refresh=True)
                if result:
                    log.info("%s: artwork from %s", r["id"], url)
                return result
            except FetchError as e:
                errors.append(str(e))
                return None

        if (
            track.get("md5_image") != ""
            and album.get("md5_image") != ""
            and (image := cover(album.get("cover_xl") or ""))
        ):
            return image
        params = {
            "term": f"{r['creator']} {base_title(r['title'])}",
            "entity": "song",
            "limit": 100,
        }
        try:
            body = self.http.get_json(f"https://itunes.apple.com/search?{urlencode(params)}") or {}
            for t in body.get("results", []):
                if song_key(t.get("artistName", ""), t.get("trackName", "")) == song_key(
                    r["creator"], r["title"]
                ) and album_key(t.get("collectionName", "")) == album_key(album_title):
                    url = (t.get("artworkUrl100") or "").replace("100x100bb", "1000x1000bb")
                    if image := cover(url):
                        return image
        except FetchError as e:
            errors.append(str(e))
        try:
            for group in self.release_groups(r):
                if album_key(album_title) not in release_keys(group):
                    continue
                body = (
                    self.http.get_json(f"https://coverartarchive.org/release-group/{group['id']}")
                    or {}
                )
                for art in body.get("images", []):
                    if art.get("front"):
                        for url in [(art.get("thumbnails") or {}).get("1200"), art.get("image")]:
                            if image := cover(url or ""):
                                return image
        except FetchError as e:
            errors.append(str(e))
        for method, field, value in (
            ("album.getinfo", "album", album_title),
            ("track.getinfo", "track", base_title(r["title"])),
        ):
            if not value:
                continue
            params = {
                "method": method,
                "artist": r["creator"],
                field: value,
                "autocorrect": 1,
                "format": "json",
            }
            try:
                body = (
                    self.http.get_json(f"{LASTFM}?{urlencode(params)}", secret=keys.lastfm()) or {}
                )
                info = (
                    body.get("album")
                    if field == "album"
                    else (body.get("track") or {}).get("album")
                )
                if field == "track" and album_key((info or {}).get("title", "")) != album_key(
                    album_title
                ):
                    continue
                urls = dict.fromkeys(
                    re.sub(r"/i/u/[^/]+/", "/i/u/", c.get("#text") or "")
                    for c in reversed((info or {}).get("image") or [])
                )
                for url in urls:
                    if image := cover(url):
                        return image
            except FetchError as e:
                errors.append(str(e))
        # Try the original-size URL of a saved Last.fm cover when its album
        # no longer appears in the services' search results.
        old_url = (r.get("image") or {}).get("url", "")
        if (
            album_key(album_title) == album_key(r.get("album", ""))
            and urlsplit(old_url).hostname
            in {"lastfm.freetls.fastly.net", "lastfm-img.freetls.fastly.net"}
            and (image := cover(re.sub(r"/i/u/[^/]+/", "/i/u/", old_url)))
        ):
            return image
        if errors:
            raise FetchError("; ".join(errors))
        return None

    def song(self, r: Record) -> Result:
        track, reason = self.deezer_track(r)
        if track is None:
            return None, reason
        image = self.song_image(r, track)
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
            url = self.aic_images.get(str(s["aic"]))
            if url is None:
                return None, "no_commons_image"
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

    Parallel workers can resolve more than the target.
    The resolver tries every higher-ranked candidate, so trimming matches sequential resolution.
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


def retry_fixable_drops() -> None:
    """On format mismatch, retry dropped songs and Chicago images that the resolver can recover."""
    version = 1
    if RESOLVE_META.exists():
        version = json.loads(RESOLVE_META.read_text(encoding="utf-8")).get("version", 1)
    if version >= RESOLVE_VERSION:
        return
    drops = list(iter_jsonl(RESOLVE_DROPPED))

    def fixable(d: Record) -> bool:
        song = d["id"].startswith("song:") and d["reason"] == "no_deezer_match"
        return song or (d["id"].startswith("art:") and "-aic" in d["id"])

    kept = [d for d in drops if not fixable(d)]
    if len(kept) < len(drops):
        write_jsonl(RESOLVE_DROPPED, kept)
    log.info(
        f"resolve format {version} -> {RESOLVE_VERSION}: {num(len(drops) - len(kept))} drop "
        "records (songs without a Deezer match, Chicago art) removed; they resolve again"
    )
    write_json(RESOLVE_META, {"version": RESOLVE_VERSION})


def repair_songs(resolver: Resolver, rows: list[Record]) -> None:
    """Inspect cached album types; change only compilations and inadequate artwork."""
    for r in progress(
        [r for r in rows if r["category"] == "song"], desc="check song albums", unit="song"
    ):
        track = resolver.current_track(r)
        album = resolver.deezer_album(track) if track else {"title": r.get("album", "")}
        old_image = r.get("image")
        bad_image = not old_image or min(old_image["w"], old_image["h"]) < 600
        if old_image:
            path = IMG / old_image["src"].rsplit("/", 1)[-1]
            if path.exists():
                try:
                    with Image.open(path) as im:
                        bad_image = (
                            bad_image or min(im.size) < 600 or placeholder_bytes(path.read_bytes())
                        )
                except OSError:
                    bad_image = True
            else:
                bad_image = True
            bad_image = bad_image or placeholder_url(old_image.get("url", ""))
        chosen = track
        reason = "artwork repair"
        suspect = resolver.compilation_song(r, album)
        if suspect:
            chosen, reason = resolver.deezer_track(r, track)
        changed = suspect and chosen and chosen["album"]["title"] != r.get("album")
        if not changed and not bad_image:
            if suspect:
                log.info("%s: kept %r: %s", r["id"], r.get("album"), reason)
            continue
        artwork_track = chosen or {"album": album}
        if not changed and album_key(artwork_track["album"].get("title", "")) != album_key(
            r.get("album", "")
        ):
            artwork_track = {"album": {"title": r.get("album", "")}}
        image = resolver.song_image(r, artwork_track)
        if image is None:
            raise FetchError(f"{r['id']}: no matching cover at least 600 px")
        if changed:
            log.info(
                "%s: album %r -> %r: %s", r["id"], r.get("album"), chosen["album"]["title"], reason
            )
            r["album"] = chosen["album"]["title"]
            r["source"]["deezer"] = chosen["id"]
            r["preview"] = f"/api/preview/deezer/{chosen['id']}"
            r["links"].update(primary=chosen["link"], deezer=chosen["link"])
        r["image"] = image
        write_jsonl(RESOLVED, rows)


def run() -> None:
    start = time.perf_counter()
    cfg = ResolveConfig()
    if catalog_version() != CATALOG_VERSION:
        raise SystemExit(f"{CATALOG} is missing or in an old format; run uv run download first")
    catalog = list(iter_jsonl(CATALOG))
    wanted = CATEGORIES
    keys.require([s for c, s in (("film", "tmdb"), ("song", "lastfm")) if c in wanted])
    retry_fixable_drops()
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
    repair_songs(resolver, resolved)
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
            resolver.load_art_images([r for r in catalog if r["category"] == "art"])
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
                bar.set_postfix(
                    ok=counts["ok"],
                    dropped=sum(n for k, n in counts.items() if k != "ok" and ":" not in k),
                    errors=sum(n for k, n in counts.items() if k.startswith("error")),
                    cached_all=f"{resolver.http.hits / max(looked_up, 1):.0%}",
                    refresh=False,
                )
                bar.update()

        workers = cfg.workers.get(category, 1)
        chunk = workers * 8
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
            run_all(one, batch, workers)
        bar.close()

    run_all(work, wanted, len(wanted))
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
    errors = sum(
        n for counts in stats.values() for reason, n in counts.items() if reason.startswith("error")
    )
    if errors:
        raise RuntimeError(f"{errors} unresolved requests failed; rerun download to retry them")
