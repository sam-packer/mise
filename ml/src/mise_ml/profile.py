import functools
import random
import re
from collections import Counter
from typing import Any

from mise_ml.color import parse_hex
from mise_ml.config import (
    DISTILL,
    IMG,
    MOODS,
    PAT,
    PAT_SENTENCES,
    PROFILES,
    RESOLVED,
    SEED,
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

FEELING_RULES = """A feeling is a sentence about a scene or a moment, 6 to 30 words, lower case, \
casual, in the first person or as a scene. Examples:
- a snowy december and i just made warm hot chocolate
- rain on the window and nowhere to be
- driving home at 2am with the windows down
- the last warm evening of summer and everyone has already left the beach
Never write a bare list of mood words such as "calm, cozy, nostalgic". Never name a title, \
an artist, or a genre."""

ITEM_SYSTEM = f"""You write mood profiles for mise, a mood app. A user types a feeling. \
The app answers with a film, a book, a song, a poem, and an artwork that fit it.

{FEELING_RULES}

The work comes with facts and crowd tags: film genres and keywords, reader moods and \
genres for books, listener tags for songs, subjects and styles for art. Use them as hints \
for the mood. Crowd tags can be noisy; ignore a tag that does not fit the rest.

For the work you get, write:
- vibe: one quiet line about the mood, lower case, at most 12 words, no title, no names, \
no final period.
- description: two or three plain sentences about the mood of the work: the feeling, the \
setting, the pace, and the colors and light it suggests. Do not retell the plot.
- q1, q2, q3: three different feelings a person might type when this work is the right \
answer. Follow the feeling rules. Use three different situations."""

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
    "one quiet line about the mood",
    "at most 12 words",
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
    return "\n".join(lines)


def item_fields(r: Record) -> dict[str, Any]:
    # Digest the rendered values: omitted fields and ignored descriptions stay omitted.
    fields = {"category": r["category"], "title": r["title"]}
    fields.update({k: r[k] for k in ("creator", "album", "year", "text") if r.get(k)})
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
# Exclude Python's whitespace characters so the grammar agrees with word_count(). A word has
# a length limit and no comma inside, so the model cannot glue words to pass the word limit
# or repeat inside one word until the token limit.
MAX_WORD = 20
WORD = (
    r"[^\u0000-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000,;]"
    rf"{{1,{MAX_WORD}}}[,;]?"
)
PLAIN_WORD = re.compile(rf"[^\s,;]{{1,{MAX_WORD}}}[,;]?")
FEELING_TEXT = {"type": "string", "pattern": f"^{WORD}( {WORD}){{5,29}}$"}
ITEM_GENERATION_SCHEMA = obj(
    {
        "vibe": {"type": "string", "pattern": f"^{WORD}( {WORD}){{0,11}}$"},
        "description": {"type": "string"},
        **{q: FEELING_TEXT for q in ("q1", "q2", "q3")},
    }
)


def parse_item(keys: list[str], data: dict[str, Any]) -> list[Record]:
    if any(copies_instruction(data[k]) for k in ITEM_SCHEMA["properties"]):
        raise ValueError("answer copies prompt instructions instead of describing the work")
    vibe = normalized(data["vibe"]).rstrip(".")
    if not 1 <= word_count(vibe) <= 12:
        raise ValueError("vibe must contain 1 to 12 words")
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
    if not all(PLAIN_WORD.fullmatch(w) for t in (vibe, *queries) for w in t.split()):
        raise ValueError(
            f"put a space between words; a word has at most {MAX_WORD} characters "
            "and no comma inside"
        )
    if len(set(queries)) != 3:
        raise ValueError("q1, q2, and q3 must be three different feelings")
    return [
        {
            "key": keys[0],
            "vibe": vibe,
            "description": data["description"].strip(),
            "queries": queries,
        }
    ]


def items_job() -> JobSpec:
    cfg = ProfileConfig()
    items = {r["id"]: r for r in iter_jsonl(RESOLVED)}

    def fingerprint(k: str) -> str:
        image = item_image(items[k])
        return sha(item_prompt(items[k]) + (sha256_file(image) if image else ""))

    def build(keys: list[str]) -> list[Unit]:
        return [
            Unit(
                [k],
                Request(
                    ITEM_SYSTEM,
                    item_prompt(items[k]),
                    ITEM_SCHEMA,
                    400,
                    item_image(items[k]),
                    generation_schema=ITEM_GENERATION_SCHEMA,
                ),
            )
            for k in keys
        ]

    return JobSpec(
        Job("items", PROFILES, cfg),
        list(items),
        build,
        parse_item,
        fingerprint,
        lambda k: item_fields(items[k]),
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
        # Drop a bad or repeated feeling and keep the others: one repeat in 25 feelings
        # must not reject the whole answer. Too few good feelings means a bad answer.
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
    "just one or two words",
    "one full sentence",
    "a run-on thought",
    "casual typing with a small typo",
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
)
DISTILL_SYSTEM = """Write what people type into a mood-board app about how they feel right now.
Follow the requested situation and writing style. Vary the length from 1 to about 25 words
where the style allows it. Lower case is allowed. Use ordinary human language, including
fragments and imperfect typing. Do not name real people or titles of works. Make each
feeling distinct. Return the numbered feelings as JSON."""


def distill_seeds(cfg: ProfileConfig) -> dict[str, tuple[str, str, int]]:
    if cfg.distill_feelings < 1 or cfg.distill_per_request < 1:
        raise ValueError("distill counts must be positive")
    grid = [(s, style) for s in DISTILL_SITUATIONS for style in DISTILL_STYLES]
    count = -(-cfg.distill_feelings // cfg.distill_per_request)
    if count > len(grid):
        raise ValueError("distill_feelings exceeds the seed grid capacity")
    return {
        f"distill-{i:04d}": (
            *grid[i],
            min(cfg.distill_per_request, cfg.distill_feelings - i * cfg.distill_per_request),
        )
        for i in range(count)
    }


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


# labels: palette, light, typeface, scent for each feeling


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
                    "scent": {"type": "string", "enum": vocab.scent_ids},
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
        return {k: row[k] for k in ("light", "typeface", "scent")} | {"palette": palette}

    def parse(keys: list[str], data: dict[str, Any]) -> list[Record]:
        return parse_numbered(keys, data["labels"], make)

    job = Job(f"labels-{vocab.digest[:8]}", labels_path(vocab), cfg)
    return JobSpec(job, label_pool(cfg), build, parse, sha, lambda k: {"text": k})


def lazy_llm(cfg: ProfileConfig) -> Any:
    return functools.cache(lambda: LocalLLM(cfg))
