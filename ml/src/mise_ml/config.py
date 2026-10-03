"""Define paths and settings shared by catalog, labeling, training, and export steps."""

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
CURATED = DATA / "curated"
OUT = ML_ROOT / "out"
BUNDLE = OUT / "bundle"
SOURCES = ML_ROOT / "sources.toml"
VOCAB_PATH = REPO_ROOT / "scripts" / "stub" / "vocab.json"
EVAL_FEELINGS = ML_ROOT / "eval_feelings.jsonl"
# Rated pairs over eval feelings. The tuning set selects settings; the gold set only reports.
TUNING_SET = ML_ROOT / "tuning_set.jsonl"
GOLD_SET = ML_ROOT / "gold_set.jsonl"
# A judge's 0-3 fit ratings of the teacher's top 10 per category for training feelings. The
# teacher learns to order each top 10 by them. No eval feeling has a fit rating.
FIT_LABELS = ML_ROOT / "fit_labels.jsonl"
MET_CSV = RAW / "met" / "MetObjects.csv"

CATALOG = CURATED / "catalog.jsonl"
# The catalog format. curate starts the catalog fresh when this file holds another version.
CATALOG_META = CURATED / "catalog.meta.json"
CATALOG_VERSION = 2
PAT = CURATED / "pat.jsonl"
RESOLVED = CURATED / "resolved.jsonl"
RESOLVE_DROPPED = CURATED / "resolve_dropped.jsonl"
# A format mismatch lets resolve retry songs and Chicago art that it can match.
RESOLVE_META = CURATED / "resolve.meta.json"
RESOLVE_VERSION = 3
PROFILES = CURATED / "profiles.jsonl"
# Song facts for the labeler: a Wikipedia song article and checked lyrics. Lyrics stay local;
# only the theme sentence that the labeler writes from them reaches a profile.
FACTS = CURATED / "facts.jsonl"
THEMES = CURATED / "themes.jsonl"
# Per category, the prompt words that leak into the leaks job's sample profiles. The items job
# rejects a profile that uses one.
LEAK_BLOCK = CURATED / "leak_block.jsonl"
# One line per graded profile: the refine round, the grade, and kept or dropped.
LEDGER = CURATED / "ledger.jsonl"
GRADES = CACHE / "grades.jsonl"
MOODS = CURATED / "moods.jsonl"
DISTILL = CURATED / "distill.jsonl"
PAT_SENTENCES = CURATED / "pat_sentences.jsonl"
EVAL_REPORT = OUT / "eval_report.json"
# The student's worst tuning pairs, for a person to read.
EVAL_FAILURES = OUT / "eval_failures.md"
# The words that too many profiles of one category share.
DATA_AUDIT = OUT / "data_audit.json"
RUN_JSON = OUT / "run.json"
# The TeacherConfig values from the last Optuna search. Git tracks this file.
TEACHER_PARAMS = ML_ROOT / "teacher_params.json"

SEED = 1337
CATEGORIES = ("art", "film", "song", "poem", "book")


# Each era contains the first year, last year, and item quota.
# Start ancient eras at -5000 to include works with negative release years.
Eras = tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class CurateConfig:
    film_eras: Eras = ((1920, 1969, 450), (1970, 1999, 900), (2000, 2019, 1050), (2020, 2026, 600))
    # TMDB vote_count.gte for each era, in the order of film_eras.
    film_min_votes: tuple[int, ...] = (50, 100, 200, 100)
    film_keywords: int = 15
    book_eras: Eras = ((-5000, 1949, 450), (1950, 1999, 750), (2000, 2019, 1050), (2020, 2026, 750))
    book_page: int = 500
    book_max_pages: int = 8
    book_tags: int = 8
    # 2020-2026 keeps 750 songs: equal listener cutoffs across eras left it only 125.
    song_eras: Eras = (
        (1900, 1979, 800),
        (1980, 1999, 1200),
        (2000, 2009, 2050),
        (2010, 2019, 825),
        (2020, 2026, 750),
    )
    song_per_artist: int = 3
    song_scene_floor: int = 10
    song_tags: int = 10
    # Candidates per kept item. Resolve fills each group's quota in rank order.
    # Replace each drop with the next candidate. The spare candidates also refill the works
    # that refine drops for a failed profile.
    candidate_factor: dict[str, float] = field(
        default_factory=lambda: {"film": 1.3, "book": 1.35, "song": 1.8, "art": 1.6}
    )
    # Spread within an era: a year gets at most this many times its even share at first.
    # Free slots then go to the best remaining candidates of the era.
    year_cap_factor: float = 2.0
    art_met: int = 1200
    art_aic: int = 500
    art_cma: int = 300
    art_nasa: int = 500
    art_si: int = 500
    # At most this many works per known maker in each museum source.
    art_maker_cap: int = 8
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
            "coverartarchive.org": 1.05,
            "archive.org": 1.05,
            # Last.fm asks for at most 5 requests/s.
            "ws.audioscrobbler.com": 0.2,
            "api.deezer.com": 0.12,
            "cdn-images.dzcdn.net": 0.12,
            "query.wikidata.org": 1.0,
            "commons.wikimedia.org": 0.25,
            # Wikimedia asks for serial API requests; there is no fixed limit.
            "en.wikisource.org": 0.25,
            "en.wikipedia.org": 0.25,
            # LRCLIB publishes no limit; 4 requests/s.
            "lrclib.net": 0.25,
            "api.artic.edu": 1.0,
            "openaccess-api.clevelandart.org": 0.5,
            "openaccess-cdn.clevelandart.org": 0.1,
            "images-api.nasa.gov": 0.5,
            "images-assets.nasa.gov": 0.1,
            # Registered api.data.gov keys allow about 1,000 requests/hour.
            "api.si.edu": 3.7,
            "ids.si.edu": 0.2,
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
    # Larger batches share the cost of grammar masks and generation steps.
    # With chunked prefill, 40 label requests peak at 24.9 GiB; 48 exceed a 32 GB GPU during decode.
    # After a memory failure, generation lowers the batch limit for the rest of the job.
    batch_size: int = 40
    image_batch_size: int = 8
    # Text batches also hold at most this many tokens: rows x (longest prompt + new tokens).
    # Long prompts then get fewer rows, and short ones keep the full batch.
    batch_token_budget: int = 100_000
    synthetic_moods: int = 6000
    moods_per_request: int = 25
    label_queries: int = 30000
    labels_per_request: int = 10
    pat_per_request: int = 25
    distill_feelings: int = 40000
    distill_per_request: int = 20
    # Extra requests that each name a few idioms or slang terms, so the LLM does not repeat
    # the same handful. Each request asks for distill_per_request feelings.
    distill_idiom_requests: int = 100
    distill_slang_requests: int = 100
    distill_terms_per_request: int = 5


@dataclass(frozen=True)
class RefineConfig:
    # The grader. In tests against Claude, Opus, and Astra it was the most consistent grader.
    model: str = "gpt-6.1-sol"
    reasoning_effort: str = "medium"
    # A grade takes about 7 s, almost all of it reasoning, and counts about 1,000 tokens against
    # the 2M tokens-per-minute limit. At the old 1M limit, 128 requests at a time hit it and 80
    # stayed near 65% of it, so 160 stay near 65% of the 2M limit.
    workers: int = 160
    # A profile fails when its emotion grade (0-3) is below this; refine drops the work.
    min_emotion: int = 2
    # The first download plus at most two refills. The last round drops failures without a
    # refill.
    rounds: int = 3


@dataclass(frozen=True)
class TeacherConfig:
    backbone: str = "Qwen/Qwen3-Embedding-8B"
    revision: str = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
    max_length: int = 256
    dims: int = 384
    # The share of the raw Qwen cosine in the teacher's similarity; the heads learn what the raw
    # features miss.
    raw_weight: float = 0.5
    hidden: int = 1024
    dropout: float = 0.1
    epochs: int = 6
    warmup_ratio: float = 0.06
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.01
    temperature: float = 0.05
    palette_weight: float = 3.0
    lightness_weight: float = 0.5
    choice_weight: float = 0.5
    label_smoothing: float = 0.1
    # The fit loss: the weight, and the temperature that turns the fit ratings into a target
    # distribution over each rated top 10. With ratings for 2,000 feelings and teacher_params.json,
    # weights 0, 12, 32, 64, 128, 256 gave tuning objectives 0.625, 0.687, 0.699, 0.704, 0.704,
    # 0.700; tau 0.5 and 2 scored lower than 1.
    fit_weight: float = 64.0
    fit_tau: float = 1.0
    # Rated groups per step, drawn at random. All groups in each step made an epoch twice as slow
    # for a tuning objective only 0.001 higher.
    fit_groups: int = 256
    val_fraction: float = 0.05
    heldout_paraphrase_fraction: float = 0.1
    # Training texts at or above this Qwen query cosine to an eval feeling are dropped. A read of
    # the pairs in each band showed rewordings of the same feeling from 0.88 up.
    near_eval_cosine: float = 0.88


@dataclass(frozen=True)
class StudentConfig:
    backbone: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    max_length: int = 96
    # The student encodes feelings only; only the untrained baseline in eval encodes catalog texts.
    item_max_length: int = 128
    dims: int = 384
    head_hidden: int = 256
    epochs: int = 40
    # Epochs without a better tuning-set objective before the run stops.
    patience: int = 10
    batch_size: int = 64
    encoder_lr: float = 3e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    temperature: float = 0.05
    # The teacher's top items per query and category in each step. The KL ranks each category
    # apart. A step holds at most batch_size x (5 x teacher_topk + 1) + random_items items.
    # Near misses from the teacher's ranks 10-60 lowered val fidelity and recall in tests.
    teacher_topk: int = 4
    random_items: int = 128
    # Share of train rows that the student reads with typing noise in each epoch.
    # The teacher target stays the one for the clean text.
    typo_share: float = 0.3
    # A strong KL keeps the student close to the teacher's ranking: weights 1, 4, 16 gave val
    # fidelity 0.22, 0.25, 0.26 in 2-epoch tests.
    kl_weight: float = 16.0
    infonce_weight: float = 1.0
    # Cosine loss to the teacher's feeling vector in the fixed item space. Weights 0, 1, 4, 16 gave
    # tuning objectives 0.548, 0.552, 0.559, 0.565 in 8-epoch tests.
    regression_weight: float = 16.0
    palette_weight: float = 3.0
    lightness_weight: float = 0.5
    choice_weight: float = 0.5
    choice_temperature: float = 2.0
    # Keep only the vocabulary tokens of the training, eval, and catalog texts, every character
    # piece, and the regular tokens with an id below prune_keep_below. The other rows of the word
    # embeddings leave the model. Needs a WordPiece tokenizer.
    prune_vocab: bool = False
    prune_keep_below: int = 8000


# The 12-layer encoder with the same hidden size and tokenizer as the default backbone.
MINILM_L12 = ("sentence-transformers/all-MiniLM-L12-v2", "a50ef00143b4d5391434df20ae11632588ac25be")


@dataclass(frozen=True)
class ExportConfig:
    opset: int = 18
    max_model_bytes: int = 24 * 1024 * 1024
    max_tokens: int = 96
    # The exposure penalty of feeling walls. Beta 0.02 cut the share of the top 1% of works from
    # 26.5% to 16.9% of the walls of the tuning feelings, with no loss in blind ratings of the walls
    # (paired mean change -0.011, interval -0.026 to +0.005). Damped rounds let the counts settle.
    exposure_beta: float = 0.02
    exposure_rounds: int = 6
