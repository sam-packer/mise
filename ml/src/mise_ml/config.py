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
MUSE_KAGGLE = RAW / "muse" / "muse_v3.csv"
MUSE_ZENODO = RAW / "muse" / "muse_dataset.csv"
SPOTIFY_TRACKS = RAW / "spotify" / "dataset.csv"

CATALOG = CURATED / "catalog.jsonl"
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


@dataclass(frozen=True)
class CurateConfig:
    film_min_ratings: int = 2000
    film_top: int = 2000
    film_genome_tags: int = 12
    film_genome_min_relevance: float = 0.5
    book_top: int = 2000
    book_shelf_tags: int = 15
    song_top: int = 3000
    # Some songs are not on Deezer; resolve keeps the top song_top of the candidates.
    song_candidate_factor: float = 1.4
    song_per_artist: int = 5
    poem_min_lines: int = 8
    poem_max_lines: int = 40
    poem_excerpt_lines: int = 14
    art_target: int = 2000
    art_candidate_factor: int = 3
    art_quota: dict[str, float] = field(
        default_factory=lambda: {
            "Paintings": 0.4,
            "Drawings": 0.2,
            "Prints": 0.2,
            "Photographs": 0.2,
        }
    )


@dataclass(frozen=True)
class ResolveConfig:
    image_long_edge: int = 800
    webp_quality: int = 82
    min_interval: dict[str, float] = field(
        default_factory=lambda: {
            "api.deezer.com": 0.12,
            "cdn-images.dzcdn.net": 0.12,
            "v3.sg.media-imdb.com": 0.4,
            "m.media-amazon.com": 0.2,
            "query.wikidata.org": 1.0,
            "openlibrary.org": 1.0,
            "covers.openlibrary.org": 0.5,
            "collectionapi.metmuseum.org": 0.1,
            "images.metmuseum.org": 0.1,
            "poetrydb.org": 1.0,
        }
    )
    default_interval: float = 1.0
    timeout: float = 30.0
    retries: int = 4
    # Parallel requests per category. The per-host intervals above still cap the rate.
    workers: dict[str, int] = field(default_factory=lambda: {"song": 4})


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
