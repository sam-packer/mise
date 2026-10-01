"""Fetch song facts for the labeler: a Wikipedia song article and lyrics from LRCLIB.

data/cache/http caches every API answer, so a rerun is fast and continues after a stop.
Lyrics are local data. They stay in facts.jsonl and never go to the log.
"""

import re
import threading
import time
from collections import Counter
from typing import Any
from urllib.parse import unquote, urlencode

from mise_ml.config import FACTS, ML_ROOT, RESOLVED, ResolveConfig
from mise_ml.curate import norm
from mise_ml.http import CachedClient, FetchError
from mise_ml.log import elapsed, get, num, progress
from mise_ml.threads import run_all
from mise_ml.util import iter_jsonl, write_jsonl

log = get(__name__)

Record = dict[str, Any]

WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
WIKIDATA = "https://query.wikidata.org/sparql"
LRCLIB = "https://lrclib.net/api"
DEEZER = "https://api.deezer.com"
# Works of our artists that have an English Wikipedia article. P434 is the MusicBrainz artist
# ID and P175 the performer.
WORKS_QUERY = """SELECT ?mb ?label ?article WHERE {{
  VALUES ?mb {{ {ids} }}
  ?artist wdt:P434 ?mb . ?item wdt:P175 ?artist .
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
  ?item rdfs:label ?label . FILTER(LANG(?label) = "en")
}}"""
ARTIST_BATCH = 50
# The Wikipedia text keeps at most this many characters and ends on a sentence.
CLIP = 1200
# Article sections after the intro that tell what a song means or how it came to be.
SONG_SECTIONS = re.compile(
    r"lyric|composition|background|meaning|theme|content|inspiration|writing", re.I
)
# Keep lyrics only when LRCLIB's track length is within this many seconds of our Deezer track.
# A larger gap means another recording or another song with the same name.
MAX_LENGTH_GAP = 8
# Songs in progress at one time. The per-host intervals still cap the rate; LRCLIB at
# 1 request/s sets the pace, and the other hosts run during its waits.
WORKERS = 4


def bare(title: str) -> str:
    """The title without parenthetical parts, for matching."""
    return re.sub(r"\s*[\(\[][^)\]]*[\)\]]", "", title).strip()


def sections(text: str) -> list[tuple[str, str]]:
    """The (heading, body) pairs of a plain-text article. The intro has the heading ""."""
    parts = re.split(r"\n\n?(={2,4}) ?([^=\n]+?) ?\1\n", "\n" + text)
    out = [("", parts[0].strip())]
    for i in range(1, len(parts) - 2, 3):
        out.append((parts[i + 1].strip(), parts[i + 2].strip()))
    return out


def clip(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text).strip()
    if len(text) <= CLIP:
        return text
    cut = text[:CLIP]
    return cut[: max(cut.rfind(". "), CLIP - 200) + 1].strip()


def song_article(text: str, artist: str) -> bool:
    """The intro names the artist, and its first sentence calls the page a song, not an album.

    This rejects an album article, a redirect to an album, and the original's article for a
    cover, because the original's intro does not name the cover artist.
    """
    intro = sections(text)[0][1][:900]
    lead = norm(re.split(r"\s*(?:&| feat| and the | x )\s*", artist)[0])
    first = re.split(r"(?<=[a-z0-9\)]{3})\. (?=[A-Z])", intro, maxsplit=1)[0]
    song = re.search(r"\b(song|single|track|instrumental|composition)\b", first, re.I)
    album = re.search(r"\b(album|extended play|EP|soundtrack|compilation)\b", first)
    return bool(song) and (not album or song.start() < album.start()) and lead[:8] in norm(intro)


class SongFacts:
    def __init__(self, http: CachedClient) -> None:
        self.http = http
        # (Wikidata label, article title) of each artist's works, by artist MBID.
        self.works: dict[str, list[tuple[str, str]]] = {}
        # Artists whose Wikidata batch failed. Their songs count as errors in this run.
        self.failed_artists: set[str] = set()

    def load_works(self, rows: list[Record]) -> None:
        """Read the works of all artists, ARTIST_BATCH artists per Wikidata query."""
        artists = sorted({r["source"]["artist_mbid"] for r in rows})
        for i in progress(
            range(0, len(artists), ARTIST_BATCH), desc="wikidata works", unit="batch"
        ):
            batch = artists[i : i + ARTIST_BATCH]
            query = WORKS_QUERY.format(ids=" ".join(f'"{a}"' for a in batch))
            try:
                body = self.http.get_json(
                    f"{WIKIDATA}?{urlencode({'query': query, 'format': 'json'})}"
                )
            except FetchError as e:
                log.warning("wikidata works: a batch failed; its songs count as errors: %s", e)
                self.failed_artists.update(batch)
                continue
            for x in (body or {}).get("results", {}).get("bindings", []):
                article = unquote(x["article"]["value"].rsplit("/", 1)[-1]).replace("_", " ")
                self.works.setdefault(x["mb"]["value"], []).append((x["label"]["value"], article))

    def page(self, title: str) -> tuple[str, str] | None:
        """(url, plain text) of an article, or None for a missing or disambiguation page."""
        params = {
            "action": "query",
            "prop": "extracts|pageprops|info",
            "inprop": "url",
            "ppprop": "disambiguation",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "redirects": 1,
        }
        body = self.http.get_json(f"{WIKIPEDIA}?{urlencode(params)}") or {}
        p = next(iter(body.get("query", {}).get("pages", {}).values()), {})
        text = p.get("extract") or ""
        if "missing" in p or not text or "disambiguation" in (p.get("pageprops") or {}):
            return None
        if re.search(r"\bmay (also )?refer to\b", text[:400]):
            return None
        return p.get("fullurl") or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}", text

    def search(self, query: str) -> list[str]:
        params = {
            "action": "query",
            "list": "search",
            "format": "json",
            "srsearch": query,
            "srlimit": 6,
        }
        body = self.http.get_json(f"{WIKIPEDIA}?{urlencode(params)}") or {}
        return [x["title"] for x in body.get("query", {}).get("search", [])]

    def wikipedia(self, r: Record) -> tuple[Record, str]:
        """The song article text and URL, and the route that found it ("" for none).

        Try the artist's Wikidata works with the same title first, then a Wikipedia search.
        """
        mbid = r["source"]["artist_mbid"]
        if mbid in self.failed_artists:
            raise FetchError(f"{r['id']}: the Wikidata batch of its artist failed")
        title = norm(bare(r["title"]))
        if not title:
            return {}, ""  # a title with no Latin letters or digits matches any other such title
        found = [
            (article, "wikidata")
            for label, article in self.works.get(mbid, [])
            if norm(bare(label)) == title or norm(bare(article)) == title
        ]
        searched = False
        while found or not searched:
            if not found:
                searched = True
                hits = self.search(f'"{bare(r["title"])}" {r["creator"]} song')
                found = [(x, "search") for x in hits if norm(bare(x)) == title][:2]
                if not found:
                    break
            article, route = found.pop(0)
            got = self.page(article)
            if not got:
                log.debug("%s: Wikipedia %r is missing or a disambiguation page", r["id"], article)
                continue
            url, text = got
            if not song_article(text, r["creator"]):
                log.debug("%s: Wikipedia %r is not about this song", r["id"], article)
                continue
            keep = [
                body for head, body in sections(text) if head == "" or SONG_SECTIONS.search(head)
            ]
            log.debug("%s: Wikipedia %r through %s", r["id"], article, route)
            return {"wikipedia": clip("\n".join(b for b in keep if b)), "wikipedia_url": url}, route
        return {}, ""

    def deezer_length(self, r: Record) -> float | None:
        track = self.http.get_json(f"{DEEZER}/track/{r['source']['deezer']}") or {}
        return track.get("duration") or None

    def lyrics(self, r: Record) -> tuple[Record, str]:
        """The plain lyrics from LRCLIB, and the result.

        The result is kept, matched (a search hit with our length), unchecked (no length to
        compare), gap, instrumental, or "".
        """
        title = bare(r["title"])
        lead = re.split(r"\s*(?:&| feat\.? |, )\s*", r["creator"])[0]
        query = urlencode({"artist_name": lead, "track_name": title})

        def search() -> list[Record]:
            """Search hits with the same artist and title that have lyrics or are instrumental."""
            if not (norm(lead) and norm(title)):
                return []  # an empty name matches any other empty name
            return [
                x
                for x in self.http.get_json(f"{LRCLIB}/search?{query}") or []
                if norm(lead)[:6] in norm(x.get("artistName") or "")
                and norm(title) == norm(bare(x.get("trackName") or ""))
                and (x.get("plainLyrics") or x.get("instrumental"))
            ]

        found = self.http.get_json(f"{LRCLIB}/get?{query}") or next(iter(search()), None)
        if not found:
            return {}, ""
        if found.get("instrumental"):
            return {}, "instrumental"
        text = (found.get("plainLyrics") or "").strip()
        if not text:
            return {}, ""
        theirs, ours = found.get("duration"), self.deezer_length(r)
        if theirs is None or ours is None:
            return {"lyrics": text}, "unchecked"
        if abs(theirs - ours) <= MAX_LENGTH_GAP:
            return {"lyrics": text}, "kept"
        # LRCLIB often has several versions of a song, such as a radio edit and the album
        # take. Look for the version with our length before the lyrics are dropped.
        for x in search():
            text = (x.get("plainLyrics") or "").strip()
            length = x.get("duration")
            if (
                text
                and not x.get("instrumental")
                and length
                and abs(length - ours) <= MAX_LENGTH_GAP
            ):
                log.debug(
                    "%s: lyrics of a search hit: LRCLIB %.0f s, Deezer %s s", r["id"], length, ours
                )
                return {"lyrics": text}, "matched"
        log.debug("%s: lyrics dropped: LRCLIB %.0f s, Deezer %s s", r["id"], theirs, ours)
        return {}, "gap"

    def song(self, r: Record) -> tuple[Record | None, list[str]]:
        """The facts row of one song (None without facts) and its outcomes for the counts."""
        wiki, route = self.wikipedia(r)
        words, result = self.lyrics(r)
        row = {"id": r["id"], **wiki, **words}
        outcomes = [f"wikipedia {route}" if route else "no wikipedia", f"lyrics {result or 'none'}"]
        return (row if len(row) > 1 else None), outcomes


def run() -> None:
    start = time.perf_counter()
    songs = [r for r in iter_jsonl(RESOLVED) if r["category"] == "song"]
    if not songs:
        raise SystemExit(f"{RESOLVED} has no songs; run uv run download first")
    log.info(f"reading {RESOLVED.name}: {num(len(songs))} songs")
    http = CachedClient(ResolveConfig())
    facts = SongFacts(http)
    facts.load_works(songs)

    lock = threading.Lock()
    rows: list[Record] = []
    counts: Counter[str] = Counter()
    bar = progress(total=len(songs), desc="song facts", unit="song")

    def one(r: Record) -> None:
        try:
            row, outcomes = facts.song(r)
        except Exception as e:  # one bad song must not stop a long run
            row, outcomes = None, ["error"]
            log.debug(f"{r['id']}: {type(e).__name__}: {e}")
        with lock:
            counts.update(outcomes)
            if row is not None:
                rows.append(row)
            looked_up = http.hits + http.requests
            bar.set_postfix(
                facts=len(rows),
                errors=counts["error"],
                cached_all=f"{http.hits / max(looked_up, 1):.0%}",
                refresh=False,
            )
            bar.update()

    run_all(one, songs, WORKERS)
    bar.close()
    http.close()
    write_jsonl(FACTS, sorted(rows, key=lambda row: row["id"]))

    wiki = counts["wikipedia wikidata"] + counts["wikipedia search"]
    kept = counts["lyrics kept"] + counts["lyrics matched"] + counts["lyrics unchecked"]
    log.info(
        f"Wikipedia: {num(wiki)} of {num(len(songs))} songs "
        f"({num(counts['wikipedia wikidata'])} through Wikidata, "
        f"{num(counts['wikipedia search'])} through search)"
    )
    log.info(
        f"lyrics: {num(kept)} kept ({num(counts['lyrics matched'])} from a search hit with "
        f"our length, {num(counts['lyrics unchecked'])} without a length to compare), "
        f"{num(counts['lyrics gap'])} dropped for a length gap over "
        f"{MAX_LENGTH_GAP} s, {num(counts['lyrics instrumental'])} instrumental"
    )
    log.info(
        f"done in {elapsed(start)}: {num(len(rows))} songs with facts in "
        f"{FACTS.relative_to(ML_ROOT).as_posix()}; "
        f"{num(http.requests)} API calls, {num(http.hits)} answers from the cache"
    )
    if counts["error"]:
        raise RuntimeError(
            f"{counts['error']} songs failed (details in the log file); rerun to retry them"
        )
