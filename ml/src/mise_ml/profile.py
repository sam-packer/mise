"""Define resumable LLM jobs for item profiles, mood queries, and visual labels."""

import functools
import json
import random
import re
from collections import Counter
from typing import Any

from mise_ml.audit import LEAK_GAIN, LEAK_P, LEAK_RATIO, SKIP, leaks
from mise_ml.color import parse_hex
from mise_ml.config import (
    DISTILL,
    FACTS,
    IMG,
    LEAK_BLOCK,
    MOODS,
    PAT,
    PAT_SENTENCES,
    PROFILES,
    RESOLVED,
    SEED,
    THEMES,
    ProfileConfig,
)
from mise_ml.data import MAX_FEELING, distill_rejection, load_eval_texts, normalize_feeling
from mise_ml.llm import Job, JobSpec, LocalLLM, Record, Request, Unit, sha
from mise_ml.log import get
from mise_ml.util import (
    hash_fraction,
    iter_jsonl,
    sha256_file,
    word_count,
)
from mise_ml.vocab import Vocab, labels_path, load_vocab

log = get(__name__)

FEELING_RULES = """A feeling is a thought about an emotional state or a moment, 6 to 30 words, \
lower case and casual. Use first person, a scene, or figurative everyday language. Examples:
- a snowy december and i just made warm hot chocolate
- rain on the window and nowhere to be
- driving home at 2am with the windows down
- the last warm evening of summer and everyone has already left the beach
Never write a bare list of mood words such as "calm, cozy, nostalgic". Never name a title, \
an artist, or a genre."""

# Tested against the old prompt in blind audits. Keep it neutral: a word in the prompt shows up
# in the profiles ("tension", "quiet", "irony", and "mix" all did).
ITEM_SYSTEM = """You write short mood profiles of works for mise. In mise, a person types how \
they feel, and the app answers with an artwork, a film, a book, a poem, and a song.

You get the facts about one work. Describe the emotion that most people feel from this work.

Accuracy comes first.
- Use your own knowledge of the work together with the facts.
- If the work is joyful, funny, calm, or warm, say so. Do not make it darker or more dramatic \
than it is.
- If you do not know the work and the facts do not show its emotion, write only what the facts \
support. Do not invent lyrics, sounds, colors, figures, or events.

Write:
- vibe: its main emotion in plain words, at most 10 words, lower case.
- description: two or three plain sentences about the emotion of the work and why it has that \
emotion. Use details from the work only when they carry the emotion.
- q1, q2, q3: three different feelings that a real person could type into the app when this \
work is a good answer. Write them the way people type: plain, first person or a simple \
situation, 6 to 30 words. Each feeling must fit the work's main emotion."""

EXAMPLE_SLOT = "{examples}"
# The pilot prompts (out/pilot/prompts.py) ask for a specific emotion, with three quoted examples.
# ITEM_SYSTEM stays as the pilot's baseline; the prompts share its "Write:" tail.
ITEM_HEAD = f"""You write short mood profiles of works for mise. In mise, a person types how \
they feel, and the app answers with an artwork, a film, a book, a poem, and a song.

You get the facts about one work. Describe the emotion that most people feel from this work.

Accuracy comes first.
- Use your own knowledge of the work together with the facts.
- If the work is joyful, funny, or warm, say so. Do not make it darker or more dramatic \
than it is.
- If you do not know the work and the facts do not show its emotion, write only what the facts \
support. Do not invent lyrics, sounds, colors, figures, or events.

Name the emotion of this work in precise, concrete words, for example {EXAMPLE_SLOT}. \
Do not use a broad mood word when a more exact emotion fits this work."""
ITEM_AVOID = """
Avoid these broad words unless nothing more precise fits: quiet, calm, peaceful, tense, dark, \
heavy, deep, warm, gentle."""
ITEM_TAIL = ITEM_SYSTEM[ITEM_SYSTEM.index("\n\nWrite:") :]
FIXED_EXAMPLES = '"solemn dignity", "wistful longing", or "awe at vast scale"'
ITEM_V1 = ITEM_HEAD.replace(EXAMPLE_SLOT, FIXED_EXAMPLES) + ITEM_TAIL
ITEM_V2 = ITEM_HEAD.replace(EXAMPLE_SLOT, FIXED_EXAMPLES) + ITEM_AVOID + ITEM_TAIL
# The item system prompt of each category, as the pilot chose them. A prompt with the
# EXAMPLE_SLOT gets EXAMPLE_COUNT phrases from its category's pool in ITEM_EXAMPLES.
ITEM_SYSTEMS = {
    "art": ITEM_V2,
    "poem": ITEM_HEAD + ITEM_AVOID + ITEM_TAIL,
    "song": ITEM_V1,
    "film": ITEM_V2,
    "book": ITEM_V1,
}
EXAMPLE_COUNT = 3
# Every phrase has its own head word, none is an umbrella word, and the pool spans the range.
POEM_EXAMPLES = (
    "giddy delight",
    "mischievous glee",
    "seething resentment",
    "jittery dread",
    "tender devotion",
    "eerie unease",
    "triumphant swagger",
    "raw bereavement",
    "bitter disillusion",
    "dumbstruck wonder",
    "restless yearning",
    "cringing embarrassment",
    "bittersweet homesickness",
    "cold menace",
    "deadpan absurdity",
    "lazy sunlit contentment",
    "devout reverence",
    "smug satisfaction",
    "frantic panic",
    "sly flirtation",
    "righteous fury",
    "abandoned desolation",
    "proud defiance",
    "sheepish relief",
    "feverish obsession",
    "drowsy coziness",
    "haunted guilt",
    "rowdy camaraderie",
    "aching regret",
    "breathless anticipation",
)
ITEM_EXAMPLES: dict[str, tuple[str, ...]] = {
    "art": (),
    "poem": POEM_EXAMPLES,
    "song": (),
    "film": (),
    "book": (),
}
# The leaks job labels this many works of each category twice: with the current prompts and with
# the neutral BASELINE_SYSTEM. A prompt word that the current prompt puts into the profiles (see
# audit.leaks) goes on the category's block list, and the items job discourages it.
LEAK_SAMPLE = {"art": 70, "poem": 40, "book": 40, "song": 25, "film": 25}
BASELINE_SYSTEM = ITEM_SYSTEM
BASELINE_KEY = "baseline:"

MOODS_SYSTEM = f"""You write feelings that people type into mise, a mood app.

{FEELING_RULES}

Write varied, specific, believable moments. Mix quiet and loud, happy and sad, ordinary and \
rare. Do not repeat a structure twice in a row."""

PAT_SYSTEM = f"""You turn short color-palette names into feelings for mise, a mood app.

{FEELING_RULES}

For each numbered palette name, write one feeling that a person could type when that palette \
is the right answer. Keep the mood of the name, but write a moment, not a description of colors."""

LABEL_SYSTEM = """You design the look of a page in mise, a mood app, for a feeling that a \
user typed.

Read idioms, slang, sarcasm, and mixed feelings for their intended emotion. Choose the \
room for that emotional core, not for a matching surface word.

For each numbered feeling, choose:
- c1 to c5: a palette of five colors as #rrggbb, in dominance order. c1 is the page \
ground and covers most of the page. c2 is the main ink or accent. c3 to c5 are supporting \
tones. Choose colors for the atmosphere of the moment, not only the literal objects in it.
- light: the room light that fits the moment.
- typeface: the typeface that fits the voice of the moment.
Use only ids from these lists.

"""

FACETS = {
    "voice": ["first-person confession", "casual first-person text", "a scene", "a wish"],
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
        "heartbroken",
        "missing an ex",
        "rejected love",
        "healing after a breakup",
    ],
}


HEX = {"type": "string", "pattern": "^#[0-9a-f]{6}$"}


def obj(properties: dict[str, Any]) -> dict[str, Any]:
    """Require every property and forbid extra properties in a JSON object schema."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def numbered_array(properties: dict[str, Any], count: int) -> dict[str, Any]:
    return {
        "type": "array",
        "prefixItems": [obj({"n": {"const": i + 1}, **properties}) for i in range(count)],
        "items": False,
        "minItems": count,
        "maxItems": count,
    }


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


PROMPT_EXAMPLES = {
    normalized(line[2:]) for line in FEELING_RULES.splitlines() if line.startswith("- ")
}
INSTRUCTION_FRAGMENTS = (
    "three different feelings a person might type",
    "follow the feeling rules",
    "two or three plain sentences about the mood",
    "one complete line about the emotional core",
    "at most 10 words",
    "never name a title",
    "never write a bare list of mood words",
    "6 to 30 words",
)


def copies_instruction(text: str) -> bool:
    text = normalized(text)
    return any(fragment in text for fragment in INSTRUCTION_FRAGMENTS)


def is_feeling(text: Any) -> bool:
    return (
        isinstance(text, str)
        and 6 <= word_count(text) <= 30
        and normalized(text) not in PROMPT_EXAMPLES
        and not copies_instruction(text)
    )


def numbered(texts: list[str]) -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))


def parse_numbered(keys: list[str], rows: list[dict[str, Any]], make: Any) -> list[Record]:
    """Map numbered rows to keys, rejecting duplicate numbers and records that make() rejects."""
    out: dict[str, Record] = {}
    repeated = {n for n, count in Counter(row["n"] for row in rows).items() if count > 1}
    for row in rows:
        i = row["n"] - 1
        if 0 <= i < len(keys) and row["n"] not in repeated:
            rec = make(row)
            if rec is not None:
                out[keys[i]] = {"key": keys[i], **rec}
    return list(out.values())


def chunks(keys: list[str], size: int) -> list[list[str]]:
    return [keys[i : i + size] for i in range(0, len(keys), size)]


# items


# Signal fields in prompt order, with the label the labeler sees. Tag lists come first.
SIGNAL_LABELS = {
    "genres": "genres",
    "moods": "reader mood tags",
    "tags": "listener tags",
    "keywords": "keywords",
    "subjects": "subjects",
    "styles": "styles",
    "terms": "museum terms",
    "tagline": "tagline",
    "overview": "synopsis",
    "description": "description",
    "classification": "type",
    "medium": "medium",
    "date": "date",
    "culture": "culture",
    "department": "museum department",
}


def item_prompt(r: Record) -> str:
    lines = [f"category: {r['category']}", f"title: {r['title']}"]
    if r.get("creator"):
        lines.append(f"creator: {r['creator']}")
    if r.get("album"):
        lines.append(f"album: {r['album']}")
    if r.get("year"):
        lines.append(f"year: {r['year']}")
    signal = r.get("signal", {})
    for key, label in SIGNAL_LABELS.items():
        value = signal.get(key)
        if value in (None, [], ""):
            continue
        # Hardcover's Casino Royale description contains an unrelated casino advertisement.
        if key == "description" and "https://chipz-finland.com/" in value:
            continue
        if isinstance(value, list):
            value = ", ".join(value)
        lines.append(f"{label}: {value}")
    if r.get("text"):
        lines.append(f"poem text:\n{r['text']}")
    if r.get("wikipedia"):
        lines.append(f"wikipedia: {r['wikipedia']}")
    if r.get("lyrics_theme"):
        lines.append(f"lyrics theme: {r['lyrics_theme']}")
    return "\n".join(lines)


def item_system(r: Record) -> str:
    """The category's system prompt, with this work's quoted example phrases in the slot."""
    system = ITEM_SYSTEMS[r["category"]]
    if EXAMPLE_SLOT not in system:
        return system
    pool = ITEM_EXAMPLES[r["category"]]
    if len(pool) < EXAMPLE_COUNT:
        raise ValueError(f"{r['category']} has fewer than {EXAMPLE_COUNT} example phrases")
    # The pilot's selection, so the labels reproduce the pilot's distribution.
    picks = random.Random(int(sha(f"examples:{r['id']}")[:8], 16)).sample(pool, EXAMPLE_COUNT)
    quoted = [f'"{p}"' for p in picks]
    return system.replace(EXAMPLE_SLOT, ", ".join(quoted[:-1]) + f", or {quoted[-1]}")


def item_records() -> dict[str, Record]:
    """Resolved works by id, with the song facts that the labeler and the grader read."""
    facts = {r["id"]: r for r in iter_jsonl(FACTS)} if FACTS.is_file() else {}
    themes = {r["key"]: r["theme"] for r in iter_jsonl(THEMES)} if THEMES.is_file() else {}
    out = {}
    for r in iter_jsonl(RESOLVED):
        extra = {}
        if facts.get(r["id"], {}).get("wikipedia"):
            extra["wikipedia"] = facts[r["id"]]["wikipedia"]
        if r["id"] in themes:
            extra["lyrics_theme"] = themes[r["id"]]
        out[r["id"]] = {**r, **extra}
    return out


def item_fields(r: Record) -> dict[str, Any]:
    # Digest the rendered values: omitted fields and ignored descriptions stay omitted.
    fields = {"category": r["category"], "title": r["title"], "system": sha(item_system(r))}
    fields.update(
        {
            k: r[k]
            for k in ("creator", "album", "year", "text", "wikipedia", "lyrics_theme")
            if r.get(k)
        }
    )
    for key in SIGNAL_LABELS:
        value = r.get("signal", {}).get(key)
        if value in (None, [], ""):
            continue
        if key == "description" and "https://chipz-finland.com/" in value:
            continue
        fields[f"signal.{key}"] = ", ".join(value) if isinstance(value, list) else value
    image = item_image(r)
    if image:
        fields["image"] = sha256_file(image)
    return fields


def item_image(r: Record):
    if r["category"] != "art" or not r.get("image"):
        return None
    return IMG / r["image"]["src"].rsplit("/", 1)[-1]


ITEM_SCHEMA = obj({k: {"type": "string"} for k in ("vibe", "description", "q1", "q2", "q3")})
# Keep word boundaries explicit so a token budget cannot encourage glued words.
MAX_WORD = 14
WORD = rf"[A-Za-z0-9'-]{{1,{MAX_WORD}}}"
PLAIN_WORD = re.compile(WORD)
# Prose allows longer words ("misunderstandings", "black-and-white"); only the vibe line is capped.
PROSE_WORD = r"[A-Za-z0-9'-]{1,24}[,.!?;:]?"
APOSTROPHE_ENDINGS = {"s", "t", "d", "m", "re", "ve", "ll", "clock", "am", "all", "mon"}
FUNCTION_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "nor",
        "for",
        "so",
        "yet",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "into",
        "onto",
        "upon",
        "through",
        "between",
        "among",
        "during",
        "without",
        "within",
        "as",
        "than",
        "that",
        "which",
        "whose",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "because",
        "although",
        "though",
        "if",
        "unless",
        "until",
        "against",
        "not",
        "whether",
        "while",
        "since",
        "toward",
        "towards",
        "despite",
        "either",
        "neither",
    ]
)
FEELING_TEXT = {"type": "string", "pattern": f"^{PROSE_WORD}( {PROSE_WORD}){{5,29}}$"}
ITEM_GENERATION_SCHEMA = obj(
    {
        "vibe": {"type": "string", "pattern": f"^{WORD}( {WORD}){{0,9}}$"},
        "description": {"type": "string", "pattern": f"^{PROSE_WORD}( {PROSE_WORD}){{0,89}}$"},
        "q1": FEELING_TEXT,
        "q2": FEELING_TEXT,
        "q3": FEELING_TEXT,
    }
)


def parse_item(
    keys: list[str], data: dict[str, Any], blocked: re.Pattern[str] | None = None
) -> list[Record]:
    if any(copies_instruction(data[k]) for k in ITEM_SCHEMA["properties"]):
        raise ValueError("answer copies prompt instructions instead of describing the work")
    vibe = normalized(data["vibe"]).rstrip(".")
    if not 1 <= word_count(vibe) <= 10:
        raise ValueError("vibe must contain 1 to 10 words")
    # The grammar cannot forbid a dangling last word, so trim it ("a secret kept for a" becomes
    # "a secret kept") rather than fail the key.
    words = vibe.rstrip(",").split()
    while len(words) > 1 and words[-1].rstrip(",") in FUNCTION_WORDS:
        words.pop()
    vibe = " ".join(words).rstrip(",")
    if vibe.endswith(",") or vibe.split()[-1] in FUNCTION_WORDS:
        raise ValueError("vibe must finish a thought, not end with a function word or comma")
    for word in vibe.split():
        if "'" in word.strip("'") and word.rsplit("'", 1)[1] not in APOSTROPHE_ENDINGS:
            raise ValueError("put a space after a possessive or contraction; do not glue words")
    if not data["description"].strip():
        raise ValueError("description is empty")
    queries = [normalized(data[q]) for q in ("q1", "q2", "q3")]
    for name, query in zip(("q1", "q2", "q3"), queries, strict=True):
        # Name the copied example: a general reason lets the retry copy it again.
        if query in PROMPT_EXAMPLES:
            raise ValueError(
                f"{name} copies the prompt example {query!r}; write a new feeling about this work"
            )
        if not is_feeling(query):
            raise ValueError(f"{name} must use 6 to 30 words")
    if not all(PLAIN_WORD.fullmatch(w) for w in vibe.split()) or not all(
        re.fullmatch(PROSE_WORD, w) for t in (data["description"], *queries) for w in t.split()
    ):
        raise ValueError(
            f"put a space between words; a word has at most {MAX_WORD} characters "
            "using ASCII letters, digits, apostrophes, or hyphens, with prose punctuation only"
        )
    if len(set(queries)) != 3:
        raise ValueError("q1, q2, and q3 must be three different feelings")
    # Last, so an answer that fails only this check passes every other check.
    if blocked and (match := blocked.search(f"{data['vibe']} {data['description']}")):
        # Name the word: a general reason lets the retry use it again.
        raise ValueError(
            f"{match.group().lower()!r} is overused in this catalog; use it only if no other "
            "word fits this work, otherwise describe the emotion in other words"
        )
    return [
        {
            "key": keys[0],
            "vibe": vibe,
            "description": data["description"].strip(),
            "queries": queries,
        }
    ]


def item_request(r: Record, system: str | None = None) -> Request:
    return Request(
        system or item_system(r),
        item_prompt(r),
        ITEM_SCHEMA,
        400,
        item_image(r),
        generation_schema=ITEM_GENERATION_SCHEMA,
        seed=int(sha(r["id"])[:8], 16),
        per_key_system=True,
    )


def item_print(r: Record, *extra: Any, system: str | None = None) -> str:
    # The signature leaves out the per-work system prompt, so the fingerprint covers it.
    request = item_request(r, system)
    image = sha256_file(request.image) if request.image else ""
    return sha(json.dumps([request.system, request.user, image, request.seed, *extra]))


def leak_prompt(category: str) -> str:
    """The category's items prompt with its whole example pool, for leak detection."""
    return ITEM_SYSTEMS[category].replace(EXAMPLE_SLOT, " ".join(ITEM_EXAMPLES[category]))


def leak_sample(items: dict[str, Record]) -> list[str]:
    # Rank by a hash of the id: a dropped work changes only its own place in the sample.
    keys = []
    for category, n in LEAK_SAMPLE.items():
        ids = [k for k, r in items.items() if r["category"] == category]
        keys += sorted(ids, key=lambda k: sha(f"leak:{k}"))[:n]
    return keys


def leak_prints(items: dict[str, Record], job: Job) -> dict[str, str]:
    """Per category, a digest of everything that decides its block list except the catalog:
    its prompt and example pool, the baseline prompt, the sample size, the leak rule with its
    word exclusions, and the model and decoding settings."""
    signature = job.signature(item_request(next(iter(items.values()))))
    return {
        category: sha(
            json.dumps(
                [
                    ITEM_SYSTEMS[category],
                    ITEM_EXAMPLES[category],
                    EXAMPLE_COUNT,
                    BASELINE_SYSTEM,
                    n,
                    LEAK_GAIN,
                    LEAK_RATIO,
                    LEAK_P,
                    sorted(SKIP),
                    signature,
                ]
            )
        )
        for category, n in LEAK_SAMPLE.items()
    }


def leaks_job() -> JobSpec:
    """Label a fixed sample with the current prompts and with the baseline prompt, and block
    the prompt words that the current prompts put into the profiles.

    A category keeps its block list while its leak_prints() digest is unchanged, so catalog
    changes (refine drops and refills) do not change the list. When the digest changes, the job
    draws a new sample for that category from the current catalog.
    """
    cfg = ProfileConfig()
    items = item_records()
    job = Job("leaks", LEAK_BLOCK, cfg)
    prints = leak_prints(items, job)
    kept = {
        r["category"]: r
        for r in (iter_jsonl(LEAK_BLOCK) if LEAK_BLOCK.is_file() else [])
        if r.get("prompt") == prints.get(r["category"])
    }
    sample = [k for k in leak_sample(items) if items[k]["category"] not in kept]

    def split(k: str) -> tuple[Record, str | None]:
        """The work of a key, and the baseline system prompt for a baseline key."""
        if k.startswith(BASELINE_KEY):
            return items[k.removeprefix(BASELINE_KEY)], BASELINE_SYSTEM
        return items[k], None

    def build(keys: list[str]) -> list[Unit]:
        return [Unit([k], item_request(*split(k))) for k in keys]

    def project(rows: list[Record]) -> list[Record]:
        texts: dict[str, list[str]] = {category: [] for category in LEAK_SAMPLE}
        control: dict[str, list[str]] = {category: [] for category in LEAK_SAMPLE}
        for row in rows:
            r, baseline = split(row["key"])
            (control if baseline else texts)[r["category"]].append(
                f"{row['vibe']} {row['description']}"
            )
        found = leaks({category: leak_prompt(category) for category in texts}, texts, control)
        accepted = {row["key"] for row in rows}
        out = []
        for category in texts:
            if category in kept:
                out.append(kept[category])
                continue
            ids = sorted(k for k in sample if items[k]["category"] == category)
            row = {
                "category": category,
                "words": sorted(found[category]),
                "shares": {
                    w: {name: round(s, 4) for name, s in pair.items()}
                    for w, pair in sorted(found[category].items())
                },
                "sample": ids,
            }
            # Freeze a list only when every sample key has an answer under both prompts. Without
            # the digest, the next run labels the category again; Job.run stops this run.
            if all(k in accepted and BASELINE_KEY + k in accepted for k in ids):
                row["prompt"] = prints[category]
            out.append(row)
        return out

    return JobSpec(
        job,
        sample + [BASELINE_KEY + k for k in sample],
        build,
        parse_item,
        lambda k: item_print(split(k)[0], system=split(k)[1]),
        lambda k: {
            **item_fields(split(k)[0]),
            **({"system": sha(BASELINE_SYSTEM)} if split(k)[1] else {}),
        },
        project,
    )


def block_pattern(words: list[str]) -> re.Pattern[str] | None:
    """Match the words and their plural and -ly forms, ignoring case."""
    if not words:
        return None
    alternatives = "|".join(map(re.escape, words))
    return re.compile(rf"\b(?:{alternatives})(?:s|es|ly|ally)?\b", re.IGNORECASE)


def items_job() -> JobSpec:
    cfg = ProfileConfig()
    items = item_records()
    blocked = {r["category"]: r["words"] for r in iter_jsonl(LEAK_BLOCK)}
    patterns = {category: block_pattern(words) for category, words in blocked.items()}

    def build(keys: list[str]) -> list[Unit]:
        return [Unit([k], item_request(items[k])) for k in keys]

    # A blocked word fails the first answer and the first retry. The last retry and cached
    # answers parse without the block list, so a work keeps the word when the model still
    # chooses it, and the block list never fails a key.
    def strict(keys: list[str], data: dict[str, Any]) -> list[Record]:
        return parse_item(keys, data, patterns.get(items[keys[0]]["category"]))

    def blocks(k: str) -> list[str]:
        return blocked.get(items[k]["category"], [])

    return JobSpec(
        Job("items", PROFILES, cfg),
        list(items),
        build,
        parse_item,
        # A change to a category's block list regenerates the works of that category.
        lambda k: item_print(items[k], blocks(k)),
        lambda k: {**item_fields(items[k]), "blocked": blocks(k)},
        strict=strict,
    )


# Neutral on purpose: "double meaning or irony" in an earlier wording put irony in 56 of 70
# themes.
THEME_SYSTEM = """Read the lyrics of one song. In one or two plain sentences, say what the \
song is about and the feeling it carries. Do not quote the lyrics."""
THEME_SCHEMA = obj({"theme": {"type": "string", "maxLength": 300}})


def themes_job() -> JobSpec:
    """One theme sentence per song with checked lyrics. The lyrics never leave this machine."""
    cfg = ProfileConfig()
    lyrics = (
        {r["id"]: r["lyrics"] for r in iter_jsonl(FACTS) if r.get("lyrics")}
        if FACTS.is_file()
        else {}
    )

    def build(keys: list[str]) -> list[Unit]:
        return [Unit([k], Request(THEME_SYSTEM, lyrics[k][:4000], THEME_SCHEMA, 120)) for k in keys]

    def parse(keys: list[str], data: dict[str, Any]) -> list[Record]:
        theme = " ".join(data["theme"].split())
        if not theme:
            raise ValueError("theme is empty")
        return [{"key": keys[0], "theme": theme}]

    return JobSpec(
        Job("themes", THEMES, cfg),
        list(lyrics),
        build,
        parse,
        lambda k: sha(lyrics[k]),
        lambda k: {"lyrics": sha(lyrics[k])},
    )


# synthetic moods


def moods_prompt(index: int, n: int) -> str:
    rng = random.Random(SEED * 100_003 + index)
    picks = {k: rng.sample(v, 3) for k, v in FACETS.items()}
    hints = "\n".join(f"- {k}: {', '.join(v)}" for k, v in picks.items())
    return f"Write {n} different feelings. Draw on these hints, and combine them freely:\n{hints}"


def moods_schema(n: int) -> dict[str, Any]:
    return obj(
        {"feelings": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n}}
    )


def moods_job() -> JobSpec:
    cfg = ProfileConfig()
    n = cfg.moods_per_request
    schema = moods_schema(n)
    keys = [f"moods-{i:05d}" for i in range(-(-cfg.synthetic_moods // n))]

    def prompt(k: str) -> str:
        return moods_prompt(int(k.split("-")[1]), n)

    def build(pending: list[str]) -> list[Unit]:
        return [Unit([k], Request(MOODS_SYSTEM, prompt(k), schema, 40 * n)) for k in pending]

    def parse(keys: list[str], data: dict[str, Any]) -> list[Record]:
        # Keep valid feelings when one feeling is invalid or repeated.
        # Reject the answer only when too few valid feelings remain.
        feelings = list(dict.fromkeys(normalized(f) for f in data["feelings"] if is_feeling(f)))
        if len(feelings) < n * 4 // 5:
            raise ValueError(
                f"only {len(feelings)} of {n} feelings are different and valid; write {n} "
                "different feelings of 6 to 30 words and do not copy the prompt examples"
            )
        return [{"key": keys[0], "feelings": feelings}]

    return JobSpec(
        Job("moods", MOODS, cfg),
        keys,
        build,
        parse,
        lambda k: sha(prompt(k)),
        lambda k: {"count": n, "hints": prompt(k).split("\n", 1)[1]},
    )


# Unlabeled student queries. These seeds are independent of all evaluation text.
DISTILL_SITUATIONS = (
    "moving into a first apartment",
    "leaving home for the first time",
    "starting over after a breakup",
    "finishing a long project",
    "a birthday alone",
    "a reunion after years apart",
    "waiting for important news",
    "retiring from a familiar routine",
    "learning a difficult new skill",
    "packing up a childhood bedroom",
    "a quiet library",
    "a crowded laundromat",
    "an empty swimming pool",
    "a corner cafe",
    "a dusty attic",
    "a hospital waiting room",
    "a small balcony",
    "a secondhand shop",
    "a kitchen after everyone leaves",
    "an unfamiliar hotel room",
    "the first frost",
    "spring thaw",
    "pollen in warm air",
    "midsummer heat",
    "the end of summer",
    "falling autumn leaves",
    "a dark winter afternoon",
    "a thunderstorm approaching",
    "fog over water",
    "rain after a drought",
    "awake before sunrise",
    "a slow weekend morning",
    "a rushed weekday breakfast",
    "the midday lull",
    "late afternoon sunlight",
    "the blue hour",
    "a long evening alone",
    "midnight with friends",
    "unable to sleep at three am",
    "the morning after a celebration",
    "missing an old friend",
    "a comfortable silence with a partner",
    "an argument with a sibling",
    "a first date",
    "feeling left out of a group",
    "caring for an aging parent",
    "a baby finally asleep",
    "a pet curled up nearby",
    "making a new friend",
    "saying goodbye at a station",
    "a first day at work",
    "a deadline getting closer",
    "burnout after too many meetings",
    "a small success at work",
    "waiting for exam results",
    "studying in an empty classroom",
    "the last day of school",
    "a boring commute",
    "working a night shift",
    "lunch alone between classes",
    "a delayed flight",
    "a train through unfamiliar countryside",
    "a long drive with no schedule",
    "getting lost in a new city",
    "returning home from a trip",
    "a ferry crossing",
    "a roadside diner",
    "a tent in the rain",
    "a suitcase waiting by the door",
    "hearing an unfamiliar language",
    "a mossy forest floor",
    "a windswept beach",
    "mountains in the distance",
    "a river at dusk",
    "a field full of insects",
    "a garden after rain",
    "stars far from city lights",
    "a frozen lake",
    "a desert road",
    "birds outside a window",
    "neon reflected in wet pavement",
    "a crowded subway platform",
    "an empty parking garage",
    "a rooftop above traffic",
    "a city waking up",
    "grief arriving without warning",
    "unexpected joy over something small",
    "boredom on a day with no plans",
    "anxiety with no clear cause",
    "nostalgia for a place that changed",
    "dark academia",
    "cottagecore",
    "an industrial concrete landscape",
    "a pastel seaside town",
    "a faded retro diner",
    "a moonlit gothic garden",
    "a minimalist sunlit room",
    "a cluttered bohemian studio",
    "a futuristic rainy city",
    "a rustic cabin in winter",
)
DISTILL_STYLES = (
    "a short fragment",
    "a very short mood in one to three words",
    "one full sentence",
    "a run-on thought",
    "casual typing as a text to a close friend",
    "no capital letters and little punctuation",
    "second person, addressing yourself as you",
    "a question",
    "starting with i feel",
    "a place or aesthetic name",
    "one sensory detail",
    "a brief emotional confession",
    "a wish starting with i want",
    "two contrasting feelings together",
    "a physical sensation",
    "a fragment of a memory",
    "a casual text message with an abbreviation",
    "a thought interrupted by an ellipsis",
    "a simple comparison using like",
    "a blunt everyday statement",
    "sarcasm or irony where the intended feeling differs from the literal words",
    "a common idiom in EVERY feeling, used for its emotional meaning, not a literal scene",
    "gen-z and internet slang used naturally",
    "heartbreak after losing or leaving a relationship",
    "envy of someone who has what you want",
    "spite and wanting someone to regret what they did",
    "shame about something you said or did",
    "dark humor about a difficult feeling",
    "a second-person accusation or comfort addressed to another person",
    "a mixed feeling with a contrast, such as sunny out and i can't get out of bed",
)
DISTILL_IDIOM_STYLE = (
    "each feeling uses one of these idioms, in any tense or person, for its emotional meaning, "
    "not a literal scene; use each idiom about equally: "
)
DISTILL_IDIOMS = (
    "walking on eggshells",
    "on cloud nine",
    "down in the dumps",
    "at the end of my rope",
    "a weight off my shoulders",
    "butterflies in my stomach",
    "over the moon",
    "under the weather",
    "at my wits' end",
    "a lump in my throat",
    "on pins and needles",
    "in a rut",
    "out of my depth",
    "the last straw",
    "burning the candle at both ends",
    "hit rock bottom",
    "on top of the world",
    "feeling blue",
    "green with envy",
    "a chip on my shoulder",
    "a heavy heart",
    "cold feet",
    "a shot in the dark",
    "coming out of my shell",
    "the elephant in the room",
    "running on fumes",
    "bent out of shape",
    "a fish out of water",
    "treading water",
    "keeping my head above water",
    "pulling my hair out",
    "tip of my tongue",
    "wearing my heart on my sleeve",
    "a bitter pill to swallow",
    "cry over spilled milk",
    "can't see the forest for the trees",
    "the calm before the storm",
    "a storm in a teacup",
    "in the same boat",
    "on the fence",
    "sitting on a powder keg",
    "skating on thin ice",
    "a blessing in disguise",
    "salt in the wound",
    "throw in the towel",
    "back to square one",
    "at a crossroads",
    "light at the end of the tunnel",
    "every cloud has a silver lining",
    "head in the clouds",
    "down to earth",
    "a breath of fresh air",
    "walking on air",
    "tickled pink",
    "jumping for joy",
    "happy as a clam",
    "fit as a fiddle",
    "like a kid in a candy store",
    "on edge",
    "at sixes and sevens",
    "all over the place",
    "falling apart at the seams",
    "coming apart",
    "keeping it together",
    "biting my tongue",
    "bottling it up",
    "letting off steam",
    "blowing off steam",
    "flew off the handle",
    "seeing red",
    "hot under the collar",
    "gets under my skin",
    "drives me up the wall",
    "fed up",
    "sick and tired",
    "had it up to here",
    "wear thin",
    "running out of steam",
    "a second wind",
    "a new lease on life",
    "turn over a new leaf",
    "a clean slate",
    "water under the bridge",
    "burned my bridges",
    "bury the hatchet",
    "cut ties",
    "the cold shoulder",
    "left out in the cold",
    "a third wheel",
    "odd one out",
    "on the outside looking in",
    "a shoulder to cry on",
    "in good hands",
    "a safe harbor",
    "home is where the heart is",
    "a far cry from home",
    "a stranger in a strange land",
    "the grass is always greener",
    "a trip down memory lane",
    "the good old days",
    "a blast from the past",
    "time flies",
    "stuck in the past",
    "living on borrowed time",
    "in limbo",
    "twiddling my thumbs",
    "bored to tears",
    "watching paint dry",
    "wide awake",
    "dead on my feet",
    "a night owl",
    "burning the midnight oil",
    "on a roll",
    "in the zone",
    "firing on all cylinders",
    "riding high",
    "hanging by a thread",
    "on thin ice",
    "in hot water",
    "between a rock and a hard place",
    "the world on my shoulders",
    "a monkey on my back",
    "skeletons in the closet",
    "egg on my face",
    "wanted the ground to swallow me",
    "red in the face",
    "my heart sank",
    "heart in my mouth",
    "heart of gold",
    "heartstrings",
    "broke my heart",
    "head over heels",
    "swept off my feet",
    "the one that got away",
    "plenty of fish in the sea",
    "a match made in heaven",
    "love is blind",
    "butter wouldn't melt",
    "a wolf in sheep's clothing",
    "crocodile tears",
    "tongue in cheek",
    "take it with a grain of salt",
    "whistling in the dark",
    "whistling past the graveyard",
    "putting on a brave face",
    "grin and bear it",
    "keep a stiff upper lip",
    "chin up",
    "roll with the punches",
    "go with the flow",
)
DISTILL_SLANG_STYLE = (
    "casual internet messages; each feeling uses one of these slang terms naturally, with the "
    "meaning a young person would give it; use each term about equally: "
)
DISTILL_SLANG = (
    "ngl",
    "lowkey",
    "highkey",
    "fr",
    "no cap",
    "deadass",
    "rn",
    "tbh",
    "idk",
    "imo",
    "smh",
    "istg",
    "iykyk",
    "afk",
    "brb",
    "irl",
    "delulu",
    "bestie",
    "bestie vibes",
    "vibe check",
    "it's giving",
    "main character energy",
    "npc",
    "touch grass",
    "chronically online",
    "rent free",
    "living rent free",
    "slay",
    "ate",
    "no crumbs",
    "mid",
    "bussin",
    "sus",
    "salty",
    "pressed",
    "shook",
    "sending me",
    "i'm dead",
    "crying",
    "screaming",
    "bruh",
    "oof",
    "yikes",
    "cringe",
    "big mood",
    "mood",
    "same",
    "it's the little things for me",
    "the ick",
    "situationship",
    "ghosted",
    "left on read",
    "soft launch",
    "hard launch",
    "red flag",
    "green flag",
    "beige flag",
    "rizz",
    "unhinged",
    "feral",
    "gremlin mode",
    "goblin mode",
    "rotting",
    "bed rot",
    "brain rot",
    "doomscrolling",
    "burnt out",
    "cooked",
    "we're so back",
    "it's so over",
    "copium",
    "hopium",
    "sleep is for the weak",
    "core memory",
    "emotional damage",
    "the vibes are off",
    "vibes",
    "understood the assignment",
    "and that's on period",
    "periodt",
    "tea",
    "spill",
    "receipts",
    "stan",
    "simp",
    "lives in my head",
    "on god",
    "bet",
    "say less",
    "valid",
    "real",
    "so real",
    "not me",
    "why am i like this",
    "send help",
    "help",
    "gatekeep",
    "girl dinner",
    "romanticize",
    "main character",
    "villain era",
    "healing era",
    "flop era",
    "in my feels",
    "feels",
)
DISTILL_SYSTEM = """Write what people type into a mood-board app about how they feel right now.
Follow the requested situation and writing style. Vary the length from 1 to about 25 words
where the style allows it. Lower case is allowed. Use ordinary human language, including
fragments and imperfect typing. Do not name real people or titles of works. Make each
feeling distinct. Every line must use the requested style. The situation is background,
not an instruction to describe scenery. For sarcasm, the intended feeling must differ
from the literal claim. For idioms, use familiar expressions for their nonliteral meaning.
For internet slang, write like a casual message, not a polished description.
Return the numbered feelings as JSON."""


def distill_seeds(cfg: ProfileConfig) -> dict[str, tuple[str, str, int]]:
    if cfg.distill_feelings < 1 or cfg.distill_per_request < 1:
        raise ValueError("distill counts must be positive")
    # Cycle through all styles so counts differ by at most one request.
    situations = list(DISTILL_SITUATIONS)
    random.Random(SEED).shuffle(situations)
    grid = [(s, style) for s in situations for style in DISTILL_STYLES]
    count = -(-cfg.distill_feelings // cfg.distill_per_request)
    if count > len(grid):
        raise ValueError("distill_feelings exceeds the seed grid capacity")
    seeds = {
        f"distill-{i:04d}": (
            *grid[i],
            min(cfg.distill_per_request, cfg.distill_feelings - i * cfg.distill_per_request),
        )
        for i in range(count)
    }
    # Name the terms in the extra requests, so each request uses different idioms and slang.
    for kind, terms, requests, style in (
        ("idiom", DISTILL_IDIOMS, cfg.distill_idiom_requests, DISTILL_IDIOM_STYLE),
        ("slang", DISTILL_SLANG, cfg.distill_slang_requests, DISTILL_SLANG_STYLE),
    ):
        for i in range(requests):
            picked = random.Random(f"{SEED}:{kind}:{i}").sample(
                terms, cfg.distill_terms_per_request
            )
            seeds[f"distill-{kind}-{i:03d}"] = (
                situations[i % len(situations)],
                style + "; ".join(picked),
                cfg.distill_per_request,
            )
    return seeds


def distill_job() -> JobSpec:
    cfg = ProfileConfig()
    seeds = distill_seeds(cfg)

    def prompt(k: str) -> str:
        situation, style, count = seeds[k]
        return f"Write {count} distinct feelings.\nSituation: {situation}\nStyle: {style}"

    def build(keys: list[str]) -> list[Unit]:
        return [
            Unit(
                [k],
                Request(
                    DISTILL_SYSTEM,
                    prompt(k),
                    obj({"feelings": numbered_array({"text": {"type": "string"}}, seeds[k][2])}),
                    60 * cfg.distill_per_request,
                    generation_schema=obj(
                        {
                            "feelings": numbered_array(
                                {
                                    "text": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": MAX_FEELING,
                                    }
                                },
                                seeds[k][2],
                            )
                        }
                    ),
                ),
            )
            for k in keys
        ]

    def parse(keys: list[str], data: dict[str, Any]) -> list[Record]:
        # Keep parsing stateless: Job parses answers several times during generation and replay.
        return [{"key": keys[0], "feelings": [r["text"] for r in data["feelings"]]}]

    return JobSpec(
        Job("distill", DISTILL, cfg),
        list(seeds),
        build,
        parse,
        lambda k: sha(prompt(k)),
        lambda k: dict(zip(("situation", "style", "count"), seeds[k], strict=True)),
        distill_rows,
    )


def distill_rows(records: list[Record]) -> list[Record]:
    evals = {normalize_feeling(t) for t in load_eval_texts()}
    seen: set[str] = set()
    rows = []
    for record in records:
        for raw in record["feelings"]:
            text = raw.strip()
            if distill_rejection(text, evals, seen):
                continue
            seen.add(normalize_feeling(text))
            rows.append({"key": record["key"], "text": text})
    return rows


# PAT palette names to feeling sentences


def pat_schema(count: int) -> dict[str, Any]:
    return obj({"sentences": numbered_array({"text": {"type": "string"}}, count)})


def pat_job() -> JobSpec:
    cfg = ProfileConfig()
    phrases = sorted({r["phrase"] for r in iter_jsonl(PAT)})

    # The grammar forces 6 to 30 words, so the model cannot echo a short palette name.
    def build(pending: list[str]) -> list[Unit]:
        return [
            Unit(
                chunk,
                Request(
                    PAT_SYSTEM,
                    numbered(chunk),
                    pat_schema(len(chunk)),
                    50 * len(chunk),
                    generation_schema=obj(
                        {"sentences": numbered_array({"text": FEELING_TEXT}, len(chunk))}
                    ),
                ),
            )
            for chunk in chunks(pending, cfg.pat_per_request)
        ]

    def parse(keys: list[str], data: dict[str, Any]) -> list[Record]:
        return parse_numbered(
            keys,
            data["sentences"],
            lambda row: {"text": normalized(row["text"])} if is_feeling(row["text"]) else None,
        )

    return JobSpec(
        Job("pat", PAT_SENTENCES, cfg), phrases, build, parse, sha, lambda k: {"phrase": k}
    )


# labels: palette, light, and typeface for each feeling


def label_pool(cfg: ProfileConfig) -> list[str]:
    evals = load_eval_texts()
    moods = [f for r in iter_jsonl(MOODS) for f in r["feelings"]]
    paraphrases = sorted({q for r in iter_jsonl(PROFILES) for q in r["queries"]}, key=hash_fraction)
    pool = list(dict.fromkeys(evals + moods))
    room = max(0, cfg.label_queries - len(pool))
    present = set(pool)
    return pool + [q for q in paraphrases if q not in present][:room]


def label_schema(vocab: Vocab, count: int) -> dict[str, Any]:
    return obj(
        {
            "labels": numbered_array(
                {
                    **{f"c{i}": HEX for i in range(1, 6)},
                    "light": {"type": "string", "enum": vocab.lights},
                    "typeface": {"type": "string", "enum": vocab.typeface_ids},
                },
                count,
            )
        }
    )


def labels_job() -> JobSpec:
    cfg = ProfileConfig()
    vocab = load_vocab()
    system = LABEL_SYSTEM + vocab.prompt_block()

    def build(pending: list[str]) -> list[Unit]:
        return [
            Unit(
                chunk,
                Request(system, numbered(chunk), label_schema(vocab, len(chunk)), 160 * len(chunk)),
            )
            for chunk in chunks(pending, cfg.labels_per_request)
        ]

    def make(row: dict[str, Any]) -> Record | None:
        palette = [row[f"c{i}"] for i in range(1, 6)]
        if any(parse_hex(c) is None for c in palette):
            return None
        return {k: row[k] for k in ("light", "typeface")} | {"palette": palette}

    def parse(keys: list[str], data: dict[str, Any]) -> list[Record]:
        return parse_numbered(keys, data["labels"], make)

    job = Job(f"labels-{vocab.digest[:8]}", labels_path(vocab), cfg)
    return JobSpec(job, label_pool(cfg), build, parse, sha, lambda k: {"text": k})


def lazy_llm(cfg: ProfileConfig) -> Any:
    return functools.cache(lambda: LocalLLM(cfg))
