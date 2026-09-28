from dataclasses import dataclass, field
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ML_ROOT.parent
DATA = ML_ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
IMG = DATA / "img"
FEATURES = DATA / "features"
MODELS = DATA / "models"
BATCHES = DATA / "batches"
CURATED = DATA / "curated"
OUT = ML_ROOT / "out"
BUNDLE = OUT / "bundle"
SOURCES = ML_ROOT / "sources.toml"
VOCAB_PATH = REPO_ROOT / "scripts" / "stub" / "vocab.json"
EVAL_FEELINGS = ML_ROOT / "eval_feelings.jsonl"
MET_CSV = RAW / "met" / "MetObjects.csv"

CATALOG = CURATED / "catalog.jsonl"
# The catalog format. curate starts the catalog fresh when this file holds another version.
CATALOG_META = CURATED / "catalog.meta.json"
CATALOG_VERSION = 2
PAT = CURATED / "pat.jsonl"
RESOLVED = CURATED / "resolved.jsonl"
RESOLVE_DROPPED = CURATED / "resolve_dropped.jsonl"
PROFILES = CURATED / "profiles.jsonl"
MOODS = CURATED / "moods.jsonl"
PAT_SENTENCES = CURATED / "pat_sentences.jsonl"
JUDGMENTS = CURATED / "judgments.jsonl"
EVAL_REPORT = OUT / "eval_report.json"
RUN_JSON = OUT / "run.json"

SEED = 1337
CATEGORIES = ("art", "film", "song", "poem", "book")


# An era is (first year, last year, items to keep). "Before 1950" starts at -5000, so
# ancient works with a negative release year count too.
Eras = tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class CurateConfig:
    film_eras: Eras = ((1920, 1969, 300), (1970, 1999, 600), (2000, 2019, 700), (2020, 2026, 400))
    # TMDB vote_count.gte for each era, in the order of film_eras.
    film_min_votes: tuple[int, ...] = (50, 100, 200, 100)
    film_keywords: int = 15
    book_eras: Eras = ((-5000, 1949, 300), (1950, 1999, 500), (2000, 2019, 700), (2020, 2026, 500))
    book_page: int = 500
    book_max_pages: int = 8
    book_tags: int = 8
    song_eras: Eras = ((1900, 1979, 450), (1980, 1999, 750), (2000, 2019, 1050), (2020, 2026, 750))
    song_per_artist: int = 6
    song_tags: int = 10
    # Candidates per kept item. resolve tries candidates in rank order and keeps the target
    # of each group (an era, or an art source), so a drop is replaced by the next candidate.
    candidate_factor: dict[str, float] = field(
        default_factory=lambda: {"film": 1.05, "book": 1.1, "song": 1.4, "art": 1.25}
    )
    # Spread within an era: a year gets at most this many times its even share at first.
    # Free slots then go to the best remaining candidates of the era.
    year_cap_factor: float = 2.0
    art_met: int = 1200
    art_aic: int = 500
    art_cma: int = 300
    # Met public-domain classes and their share of art_met.
    art_met_quota: dict[str, float] = field(
        default_factory=lambda: {
            "Paintings": 0.4,
            "Drawings": 0.2,
            "Prints": 0.2,
            "Photographs": 0.2,
        }
    )
    description_chars: int = 500
    poem_min_lines: int = 8
    poem_max_lines: int = 40
    poem_excerpt_lines: int = 14


@dataclass(frozen=True)
class ResolveConfig:
    image_long_edge: int = 800
    webp_quality: int = 82
    # Seconds between two requests to one host. Hosts run in parallel with each other.
    min_interval: dict[str, float] = field(
        default_factory=lambda: {
            # TMDB allows about 40 requests/s.
            "api.themoviedb.org": 0.03,
            "image.tmdb.org": 0.03,
            # Hardcover allows 60 requests/min.
            "api.hardcover.app": 1.05,
            "assets.hardcover.app": 0.05,
            # ListenBrainz allows 30 requests in 10 s; MusicBrainz 1 request/s.
            "api.listenbrainz.org": 0.34,
            "musicbrainz.org": 1.05,
            # Last.fm asks for at most 5 requests/s.
            "ws.audioscrobbler.com": 0.2,
            "api.deezer.com": 0.12,
            "cdn-images.dzcdn.net": 0.12,
            "query.wikidata.org": 1.0,
            "commons.wikimedia.org": 0.25,
            "api.artic.edu": 1.0,
            "www.artic.edu": 0.25,
            "openaccess-api.clevelandart.org": 0.5,
            "openaccess-cdn.clevelandart.org": 0.1,
            # 3 requests/s: Open Library's limit for a User-Agent with a contact email.
            "openlibrary.org": 0.34,
            # Covers by cover ID are not rate-limited.
            "covers.openlibrary.org": 0.05,
            "collectionapi.metmuseum.org": 0.1,
            "images.metmuseum.org": 0.1,
            "poetrydb.org": 1.0,
        }
    )
    default_interval: float = 1.0
    timeout: float = 30.0
    retries: int = 4
    # Parallel requests per category. The per-host intervals above still cap the rate.
    workers: dict[str, int] = field(
        default_factory=lambda: {"film": 8, "song": 4, "book": 4, "art": 4}
    )


@dataclass(frozen=True)
class ProfileConfig:
    model: str = "Qwen/Qwen3.5-9B"
    revision: str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    batch_size: int = 16
    image_batch_size: int = 4
    synthetic_moods: int = 6000
    moods_per_request: int = 25
    label_queries: int = 30000
    labels_per_request: int = 10
    pat_per_request: int = 25
    judge_pool_per_system: int = 20


@dataclass(frozen=True)
class TeacherConfig:
    backbone: str = "Qwen/Qwen3-Embedding-8B"
    revision: str = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
    max_length: int = 256
    encode_batch: int = 64
    dims: int = 384
    hidden: int = 1024
    dropout: float = 0.1
    epochs: int = 40
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.01
    temperature: float = 0.05
    hard_negatives: int = 8
    hard_negative_pool: int = 50
    hard_negative_warmup: int = 3
    palette_weight: float = 1.0
    lightness_weight: float = 0.5
    choice_weight: float = 0.5
    label_smoothing: float = 0.1
    val_fraction: float = 0.05
    heldout_paraphrase_fraction: float = 0.1


@dataclass(frozen=True)
class StudentConfig:
    backbone: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    max_length: int = 96
    item_max_length: int = 128
    dims: int = 384
    head_hidden: int = 256
    epochs: int = 4
    batch_size: int = 64
    encoder_lr: float = 3e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    temperature: float = 0.05
    teacher_topk: int = 16
    random_items: int = 128
    kl_weight: float = 1.0
    infonce_weight: float = 1.0
    palette_weight: float = 4.0
    choice_weight: float = 0.5
    choice_temperature: float = 2.0
    item_encode_batch: int = 256


@dataclass(frozen=True)
class ExportConfig:
    opset: int = 18
    max_model_bytes: int = 24 * 1024 * 1024
    max_tokens: int = 96
