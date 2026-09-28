import csv
import functools
import random
from typing import Any

from moodml.color import parse_hex
from moodml.config import (
    IMG,
    MOODS,
    PAT,
    PAT_SENTENCES,
    PROFILES,
    RAW,
    RESOLVED,
    SEED,
    ProfileConfig,
)
from moodml.data import load_eval_texts
from moodml.llm import Job, LocalLLM, Record, Request, Unit, sha
from moodml.util import hash_fraction, iter_jsonl, make_deterministic, sha256_file, word_count
from moodml.vocab import Vocab, labels_path, load_vocab

FEELING_RULES = """A feeling is a sentence about a scene or a moment, 6 to 30 words, lower case, \
casual, in the first person or as a scene. Examples:
- a snowy december and i just made warm hot chocolate
- rain on the window and nowhere to be
- driving home at 2am with the windows down
- the last warm evening of summer and everyone has already left the beach
Never write a bare list of mood words such as "calm, cozy, nostalgic". Never name a title, \
an artist, or a genre."""

ITEM_SYSTEM = f"""You write mood profiles for a moodboard app. A user types a feeling. \
The app answers with a film, a book, a song, a poem, and an artwork that fit it.

{FEELING_RULES}

For the work you get, write:
- vibe: one quiet line about the mood, lower case, at most 12 words, no title, no names, \
no final period.
- description: two or three plain sentences about the mood of the work: the feeling, the \
setting, the pace, and the colors and light it suggests. Do not retell the plot.
- q1, q2, q3: three different feelings a person might type when this work is the right \
answer. Follow the feeling rules. Use three different situations."""

MOODS_SYSTEM = f"""You write feelings that people type into a moodboard app.

{FEELING_RULES}

Write varied, specific, believable moments. Mix quiet and loud, happy and sad, ordinary and \
rare. Do not repeat a structure twice in a row."""

PAT_SYSTEM = f"""You turn short color-palette names into feelings for a moodboard app.

{FEELING_RULES}

For each numbered palette name, write one feeling that a person could type when that palette \
is the right answer. Keep the mood of the name, but write a moment, not a description of colors."""

LABEL_SYSTEM = """You design the look of a moodboard page for a feeling that a user typed.

For each numbered feeling, choose:
- c1 to c5: a palette of five colors as #rrggbb, in dominance order. c1 is the page \
ground and covers most of the page. c2 is the main ink or accent. c3 to c5 are supporting \
tones. Choose colors for the atmosphere of the moment, not only the literal objects in it.
- light: the room light that fits the moment.
- typeface: the typeface that fits the voice of the moment.
- scent: the scent note that fits the moment.
Use only ids from these lists.

"""

FACETS = {
    "season": ["deep winter", "early spring", "high summer", "late autumn", "any season"],
    "time": ["before dawn", "morning", "noon", "late afternoon", "dusk", "night", "3am"],
    "place": [
        "a small apartment",
        "a city street",
        "the countryside",
        "the sea",
        "a train or bus",
        "a car",
        "an office",
        "a school",
        "a kitchen",
        "a bed",
        "a café",
        "a party",
        "a hospital",
        "an airport",
        "a forest",
        "a rooftop",
        "a supermarket",
        "a library",
        "a garden",
        "a hotel room",
    ],
    "weather": ["rain", "snow", "fog", "heat", "wind", "a clear sky", "a storm", "no weather"],
    "company": ["alone", "with a partner", "with friends", "with family", "among strangers"],
    "undertone": [
        "tender",
        "restless",
        "grieving",
        "giddy",
        "bored",
        "hopeful",
        "anxious",
        "nostalgic",
        "triumphant",
        "lonely",
        "in love",
        "burnt out",
        "peaceful",
        "angry",
        "curious",
        "homesick",
        "celebrating",
        "embarrassed",
        "awestruck",
        "sleepy",
    ],
}


HEX = {"type": "string", "pattern": "^#[0-9a-f]{6}$"}


def obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def array_of(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": obj(properties)}


def is_feeling(text: Any) -> bool:
    return isinstance(text, str) and 5 <= word_count(text) <= 32


def numbered(texts: list[str]) -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))


def parse_numbered(keys: list[str], rows: list[dict[str, Any]], make: Any) -> list[Record]:
    out: dict[str, Record] = {}
    for row in rows:
        i = row["n"] - 1
        if 0 <= i < len(keys) and keys[i] not in out:
            rec = make(row)
            if rec is not None:
                out[keys[i]] = {"key": keys[i], **rec}
    return list(out.values())


def chunks(keys: list[str], size: int) -> list[list[str]]:
    return [keys[i : i + size] for i in range(0, len(keys), size)]


# items


def item_prompt(r: Record) -> str:
    lines = [f"category: {r['category']}", f"title: {r['title']}"]
    if r.get("creator"):
        lines.append(f"creator: {r['creator']}")
    if r.get("album"):
        lines.append(f"album: {r['album']}")
    if r.get("year"):
        lines.append(f"year: {r['year']}")
    for key, value in sorted(r.get("signal", {}).items()):
        if value in (None, [], "") or key == "quota":
            continue
        if isinstance(value, list):
            value = ", ".join(value)
        lines.append(f"{key}: {value}")
    if r.get("text"):
        lines.append(f"poem text:\n{r['text']}")
    if r["category"] == "song" and r["signal"].get("valence") is not None:
        lines.append("(valence, arousal, dominance are crowd ratings on a 1 to 9 scale)")
    return "\n".join(lines)


def item_image(r: Record):
    if r["category"] != "art" or not r.get("image"):
        return None
    return IMG / r["image"]["src"].rsplit("/", 1)[-1]


ITEM_SCHEMA = obj({k: {"type": "string"} for k in ("vibe", "description", "q1", "q2", "q3")})


def run_items(cfg: ProfileConfig, llm: Any) -> None:
    items = {r["id"]: r for r in iter_jsonl(RESOLVED)}

    def fingerprint(k: str) -> str:
        image = item_image(items[k])
        return sha(item_prompt(items[k]) + (sha256_file(image) if image else ""))

    def build(keys: list[str]) -> list[Unit]:
        return [
            Unit(
                [k],
                Request(ITEM_SYSTEM, item_prompt(items[k]), ITEM_SCHEMA, 400, item_image(items[k])),
            )
            for k in keys
        ]

    def parse(keys: list[str], data: dict[str, Any]) -> list[Record]:
        queries = [data[q].strip() for q in ("q1", "q2", "q3") if is_feeling(data[q])]
        vibe = data["vibe"].strip().lower().rstrip(".")
        if not vibe or not data["description"].strip() or len(queries) < 2:
            return []
        return [
            {
                "key": keys[0],
                "vibe": vibe,
                "description": data["description"].strip(),
                "queries": queries,
            }
        ]

    Job("items", PROFILES, cfg).run(list(items), build, parse, fingerprint, llm)


# synthetic moods


def moods_prompt(index: int, n: int) -> str:
    rng = random.Random(SEED * 100_003 + index)
    picks = {k: rng.sample(v, 3) for k, v in FACETS.items()}
    hints = "\n".join(f"- {k}: {', '.join(v)}" for k, v in picks.items())
    return f"Write {n} different feelings. Draw on these hints, and combine them freely:\n{hints}"


MOODS_SCHEMA = obj({"feelings": {"type": "array", "items": {"type": "string"}}})


def run_moods(cfg: ProfileConfig, llm: Any) -> None:
    n = cfg.moods_per_request
    keys = [f"moods-{i:05d}" for i in range(-(-cfg.synthetic_moods // n))]

    def prompt(k: str) -> str:
        return moods_prompt(int(k.split("-")[1]), n)

    def build(pending: list[str]) -> list[Unit]:
        return [Unit([k], Request(MOODS_SYSTEM, prompt(k), MOODS_SCHEMA, 40 * n)) for k in pending]

    def parse(keys: list[str], data: dict[str, Any]) -> list[Record]:
        feelings = [f.strip().lower() for f in data["feelings"] if is_feeling(f)]
        return [{"key": keys[0], "feelings": feelings}] if feelings else []

    Job("moods", MOODS, cfg).run(keys, build, parse, lambda k: sha(prompt(k)), llm)


# PAT palette names to feeling sentences


PAT_SCHEMA = obj({"sentences": array_of({"n": {"type": "integer"}, "text": {"type": "string"}})})


def run_pat(cfg: ProfileConfig, llm: Any) -> None:
    phrases = sorted({r["phrase"] for r in iter_jsonl(PAT)})
    if not phrases:
        print("[pat] no PAT palettes; skip")
        return

    def build(pending: list[str]) -> list[Unit]:
        return [
            Unit(chunk, Request(PAT_SYSTEM, numbered(chunk), PAT_SCHEMA, 50 * len(chunk)))
            for chunk in chunks(pending, cfg.pat_per_request)
        ]

    def parse(keys: list[str], data: dict[str, Any]) -> list[Record]:
        return parse_numbered(
            keys,
            data["sentences"],
            lambda row: {"text": row["text"].strip().lower()} if is_feeling(row["text"]) else None,
        )

    Job("pat", PAT_SENTENCES, cfg).run(phrases, build, parse, sha, llm)


# labels: palette, light, typeface, scent for each feeling


def artemis_feelings(limit: int) -> list[str]:
    path = RAW / "artemis" / "artemis_dataset_release_v0.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        texts = {
            row["utterance"].strip().lower()
            for row in csv.DictReader(f)
            if is_feeling(row.get("utterance", ""))
        }
    return sorted(texts, key=hash_fraction)[:limit]


def label_pool(cfg: ProfileConfig) -> list[str]:
    evals = load_eval_texts()
    moods = [f for r in iter_jsonl(MOODS) for f in r["feelings"]]
    extra = artemis_feelings(3000)
    paraphrases = sorted({q for r in iter_jsonl(PROFILES) for q in r["queries"]}, key=hash_fraction)
    pool = list(dict.fromkeys(evals + moods + extra))
    room = max(0, cfg.label_queries - len(pool))
    return list(dict.fromkeys(pool + paraphrases[:room]))


def label_schema(vocab: Vocab) -> dict[str, Any]:
    return obj(
        {
            "labels": array_of(
                {
                    "n": {"type": "integer"},
                    **{f"c{i}": HEX for i in range(1, 6)},
                    "light": {"type": "string", "enum": vocab.lights},
                    "typeface": {"type": "string", "enum": vocab.typeface_ids},
                    "scent": {"type": "string", "enum": vocab.scent_ids},
                }
            )
        }
    )


def run_labels(cfg: ProfileConfig, llm: Any) -> None:
    vocab = load_vocab()
    system = LABEL_SYSTEM + vocab.prompt_block()
    schema = label_schema(vocab)

    def build(pending: list[str]) -> list[Unit]:
        return [
            Unit(chunk, Request(system, numbered(chunk), schema, 90 * len(chunk)))
            for chunk in chunks(pending, cfg.labels_per_request)
        ]

    def make(row: dict[str, Any]) -> Record | None:
        palette = [row[f"c{i}"] for i in range(1, 6)]
        if any(parse_hex(c) is None for c in palette):
            return None
        return {k: row[k] for k in ("light", "typeface", "scent")} | {"palette": palette}

    def parse(keys: list[str], data: dict[str, Any]) -> list[Record]:
        return parse_numbered(keys, data["labels"], make)

    job = Job(f"labels-{vocab.digest[:8]}", labels_path(vocab), cfg)
    job.run(label_pool(cfg), build, parse, sha, llm)


def lazy_llm(cfg: ProfileConfig) -> Any:
    return functools.cache(lambda: LocalLLM(cfg))


def run() -> None:
    # Strict mode is untested with the Qwen3.5 generate path, so it only warns here.
    make_deterministic(SEED, warn_only=True)
    cfg = ProfileConfig()
    if not RESOLVED.exists():
        raise SystemExit(f"no resolved items at {RESOLVED}; run resolve first")
    llm = lazy_llm(cfg)
    run_items(cfg, llm)
    run_moods(cfg, llm)
    run_pat(cfg, llm)
    run_labels(cfg, llm)
