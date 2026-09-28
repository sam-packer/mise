import io
import re
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import quote_plus, urlencode

import numpy as np
from PIL import Image

from mise_ml.color import image_tone
from mise_ml.config import (
    CATALOG,
    CATEGORIES,
    IMG,
    ML_ROOT,
    RESOLVE_DROPPED,
    RESOLVED,
    ResolveConfig,
)
from mise_ml.http import CachedClient
from mise_ml.log import elapsed, get, num, progress
from mise_ml.util import append_jsonl, iter_jsonl, sort_jsonl

log = get(__name__)

Record = dict[str, Any]
Result = tuple[Record | None, str]

ITUNES = "https://itunes.apple.com"
IMDB_SUGGEST = "https://v3.sg.media-imdb.com/suggestion/x"
WIKIDATA = "https://query.wikidata.org/sparql"
DIRECTOR_QUERY = """SELECT ?imdb ?directorLabel WHERE {{
  VALUES ?imdb {{ {ids} }}
  ?film wdt:P345 ?imdb; wdt:P57 ?director.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""
OPENLIBRARY = "https://openlibrary.org"
MET = "https://collectionapi.metmuseum.org/public/collection/v1/objects"


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower().replace("&", "and"))


def itunes_year(result: dict) -> int | None:
    date = result.get("releaseDate") or ""
    return int(date[:4]) if date[:4].isdigit() else None


def hires_artwork(url: str) -> str:
    return re.sub(r"\d+x\d+bb", "1000x1000bb", url)


class Resolver:
    def __init__(self, cfg: ResolveConfig) -> None:
        self.cfg = cfg
        self.http = CachedClient(cfg)
        self.directors: dict[str, list[str]] = {}

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

    def load_directors(self, films: list[Record]) -> None:
        """Directors for all films at once, from Wikidata (IMDb ID P345 -> director P57)."""
        ids = sorted({r["source"]["imdb"] for r in films if r["source"].get("imdb")})
        found: dict[str, set[str]] = defaultdict(set)
        for start in range(0, len(ids), 200):
            values = " ".join(f'"{i}"' for i in ids[start : start + 200])
            query = urlencode({"query": DIRECTOR_QUERY.format(ids=values), "format": "json"})
            body = self.http.get_json(f"{WIKIDATA}?{query}") or {}
            for b in body.get("results", {}).get("bindings", []):
                name = b["directorLabel"]["value"]
                if not re.fullmatch(r"Q\d+", name):  # an entity without an English label
                    found[b["imdb"]["value"]].add(name)
        self.directors = {k: sorted(v) for k, v in found.items()}

    def film(self, r: Record) -> Result:
        imdb = r["source"].get("imdb")
        if not imdb:
            return None, "no_imdb_id"
        found = self.http.get_json(f"{IMDB_SUGGEST}/{imdb}.json") or {}
        hit = next((h for h in found.get("d", []) if h.get("id") == imdb), None)
        poster = ((hit or {}).get("i") or {}).get("imageUrl")
        if not poster:
            return None, "no_imdb_poster"
        image = self.image(poster.replace("._V1_.jpg", "._V1_UY1200_.jpg"), r["id"])
        if image is None:
            return None, "no_image"
        return {
            **r,
            "creator": ", ".join(self.directors.get(imdb, [])[:2]) or "Unknown director",
            "image": image,
            "links": {"primary": f"https://www.imdb.com/title/{imdb}/"},
        }, "ok"

    def song(self, r: Record) -> Result:
        term = quote_plus(f"{r['creator']} {r['title']}")
        found = self.http.get_json(
            f"{ITUNES}/search?term={term}&media=music&entity=song&limit=25&country=US"
        )
        want_artist, want_track = norm(r["creator"]), norm(r["title"])
        matches = [
            t
            for t in (found or {}).get("results", [])
            if norm(t.get("artistName", "")) == want_artist
            and norm(t.get("trackName", "")).startswith(want_track)
            and t.get("previewUrl")
            and t.get("artworkUrl100")
            and t.get("trackViewUrl")
        ]
        if not matches:
            return None, "no_itunes_match"
        track = matches[0]
        image = self.image(hires_artwork(track["artworkUrl100"]), r["id"])
        if image is None:
            return None, "no_image"
        youtube = "https://www.youtube.com/results?search_query=" + quote_plus(
            f"{r['creator']} {r['title']}"
        )
        spotify = f"https://open.spotify.com/track/{r['source']['spotify']}"
        links = {
            "primary": spotify,
            "spotify": spotify,
            "apple": track["trackViewUrl"],
            "youtube": youtube,
        }
        return {
            **r,
            "year": itunes_year(track),
            "album": track.get("collectionName"),
            "image": image,
            "preview": track["previewUrl"],
            "links": links,
        }, "ok"

    def book(self, r: Record) -> Result:
        cover, work = None, None
        isbn = r["source"].get("isbn")
        if isbn:
            ed = self.http.get_json(f"{OPENLIBRARY}/isbn/{isbn}.json") or {}
            cover = next((c for c in ed.get("covers", []) if isinstance(c, int) and c > 0), None)
            work = next((w.get("key") for w in ed.get("works", [])), None)
        if cover is None or work is None:
            q = f"title={quote_plus(r['title'])}&author={quote_plus(r['creator'])}"
            found = self.http.get_json(
                f"{OPENLIBRARY}/search.json?{q}&limit=1&fields=key,cover_i,first_publish_year"
            )
            doc = next(iter((found or {}).get("docs", [])), {})
            cover = cover or doc.get("cover_i")
            work = work or doc.get("key")
        if not cover or not work:
            return None, "no_openlibrary_match"
        image = self.image(f"https://covers.openlibrary.org/b/id/{cover}-L.jpg", r["id"])
        if image is None:
            return None, "no_image"
        return {**r, "image": image, "links": {"primary": f"{OPENLIBRARY}{work}"}}, "ok"

    def art(self, r: Record) -> Result:
        obj = self.http.get_json(f"{MET}/{r['source']['met']}") or {}
        if not obj.get("isPublicDomain"):
            return None, "not_public_domain"
        url = obj.get("primaryImageSmall") or obj.get("primaryImage")
        if not url:
            return None, "no_primary_image"
        image = self.image(url, r["id"])
        if image is None:
            return None, "no_image"
        links = {"primary": obj.get("objectURL") or r["links"]["primary"]}
        return {**r, "image": image, "links": links}, "ok"

    def poem(self, r: Record) -> Result:
        return {**r, "image": None}, "ok"


def run(categories: list[str] | None = None) -> None:
    start = time.perf_counter()
    cfg = ResolveConfig()
    catalog = list(iter_jsonl(CATALOG))
    if not catalog:
        raise SystemExit(f"no catalog at {CATALOG}; run curate first")
    resolved = list(iter_jsonl(RESOLVED))
    dropped_before = list(iter_jsonl(RESOLVE_DROPPED))
    done = {r["id"] for r in resolved} | {r["id"] for r in dropped_before}
    art_kept = Counter(r["signal"]["classification"] for r in resolved if r["category"] == "art")

    wanted = [c for c in CATEGORIES if not categories or c in categories]
    by_cat: dict[str, list[Record]] = defaultdict(list)
    for r in catalog:
        if r["id"] not in done and r["category"] in wanted:
            by_cat[r["category"]].append(r)
    for rows in by_cat.values():
        rows.sort(key=lambda r: (-r["rank"], r["id"]))
    pending = ", ".join(f"{c} {num(len(by_cat[c]))}" for c in wanted if by_cat[c])
    log.info(
        f"reading {CATALOG.name}: {num(len(catalog))} items; {num(len(resolved))} resolved and "
        f"{num(len(dropped_before))} dropped by earlier runs; to do: {pending or 'nothing'}"
    )

    resolver = Resolver(cfg)
    lock = threading.Lock()
    stats: dict[str, Counter[str]] = {c: Counter() for c in wanted}

    def work(category: str) -> None:
        fn: Callable[[Record], Result] = getattr(resolver, category)
        if category == "film" and by_cat["film"]:
            log.info(f"film: asking Wikidata for the directors of {num(len(by_cat['film']))} films")
            resolver.load_directors(by_cat["film"])
        counts = stats[category]
        bar = progress(
            by_cat[category], desc=category, unit="item", position=CATEGORIES.index(category)
        )
        for r in bar:
            if (
                category == "art"
                and art_kept[r["signal"]["classification"]] >= r["signal"]["quota"]
            ):
                counts["skipped: class quota full"] += 1
                continue
            try:
                out, reason = fn(r)
            except Exception as e:  # one bad record must not stop a multi-hour run
                out, reason = None, f"error: {type(e).__name__}"
                log.debug(f"{r['id']}: {type(e).__name__}: {e}")
            with lock:
                counts[reason] += 1
                if out is not None:
                    append_jsonl(RESOLVED, [out])
                    if category == "art":
                        art_kept[out["signal"]["classification"]] += 1
                elif not reason.startswith("error"):
                    log.debug(f"drop {r['id']}: {reason}")
                    append_jsonl(RESOLVE_DROPPED, [{"id": r["id"], "reason": reason}])
                looked_up = resolver.http.hits + resolver.http.requests
            bar.set_postfix(
                ok=counts["ok"],
                dropped=sum(n for k, n in counts.items() if k not in ("ok",) and ":" not in k),
                errors=sum(n for k, n in counts.items() if k.startswith("error")),
                cached=f"{resolver.http.hits / max(looked_up, 1):.0%}",
                refresh=False,
            )

    with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
        list(pool.map(work, wanted))
    resolver.http.close()
    sort_jsonl(RESOLVED, "id")
    sort_jsonl(RESOLVE_DROPPED, "id")

    for category in wanted:
        counts = stats[category]
        if not counts:
            continue
        drops = {
            k: n for k, n in counts.items() if k != "ok" and not k.startswith(("error", "skip"))
        }
        errors = {k: n for k, n in counts.items() if k.startswith("error")}
        line = f"{category}: {num(counts['ok'])} resolved"
        if drops:
            reasons = ", ".join(f"{k} {num(n)}" for k, n in sorted(drops.items()))
            log.warning(f"{line}, {num(sum(drops.values()))} dropped ({reasons})")
        else:
            log.info(line)
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
