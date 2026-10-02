"""Grade every profile with GPT-6.1 Sol, drop the works it fails, and refill them.

One round: label the new works, grade each profile that the ledger does not have yet, drop the
failures, and let resolve fill their places from the next candidates. A run stops after
RefineConfig.rounds rounds or when no profile fails. The last round drops failures without a
refill; the next download or refine fills those places.

A dropped work goes to resolve_dropped.jsonl with the reason "label_quality", so a later
download does not try it again. ledger.jsonl keeps every grade with its round and verdict.
"""

import base64
import json
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from openai import OpenAI

from mise_ml import keys
from mise_ml.config import (
    CATEGORIES,
    GRADES,
    IMG,
    LEDGER,
    PROFILES,
    RESOLVE_DROPPED,
    RESOLVED,
    RefineConfig,
)
from mise_ml.llm import sha
from mise_ml.log import get, num, progress
from mise_ml.profile import item_image, item_prompt, item_records
from mise_ml.util import append_jsonl, iter_jsonl, sort_jsonl, write_jsonl

log = get(__name__)

# The rubric that Claude, Opus, Sonnet, Luna, Sol, and Astra used in the grader comparison.
SYSTEM = """You grade one profile of a work that a labeler wrote for a mood app.
Use your own knowledge of the real work together with the facts.

mise is a mood app: a person types a feeling and gets works whose feeling matches. A labeler \
wrote a profile for the work: vibe, description, and three queries (feelings a person might \
type for which this work is a great answer). The retrieval models learn only from these texts.

1. Read the source facts. If an image is given, look at it before you grade.
2. Decide for yourself what the work's real central emotion is, using your own knowledge of \
the work and the facts or image.
3. Grade the profile with these fields:
   - factual: 0-2. 2 = nothing contradicts the real work (or the image). 1 = a minor \
misreading. 0 = describes a different work or invents wrong content or mood.
   - emotion: 0-3. How well vibe + description capture the real central emotion. 3 = exactly, \
2 = mostly with a skewed tone, 1 = a side emotion or generic mood, 0 = wrong or opposite.
   - specific: 0-2. 2 = could only describe this work; 1 = fits this kind of work; 0 = generic.
   - queries_good: 0-3. How many queries a real person would type AND this work answers well.
   - real_emotion: the work's real central emotion, in a few words.
Grade against the real work. Never quote lyrics or other text of the work."""

SCHEMA = {
    "type": "object",
    "properties": {
        "real_emotion": {"type": "string"},
        "factual": {"type": "integer", "enum": [0, 1, 2]},
        "emotion": {"type": "integer", "enum": [0, 1, 2, 3]},
        "specific": {"type": "integer", "enum": [0, 1, 2]},
        "queries_good": {"type": "integer", "enum": [0, 1, 2, 3]},
    },
    "required": ["real_emotion", "factual", "emotion", "specific", "queries_good"],
    "additionalProperties": False,
}


def profile_text(p: dict[str, Any]) -> str:
    queries = "\n".join(f"- {q}" for q in p["queries"])
    return f"vibe: {p['vibe']}\ndescription: {p['description']}\nqueries:\n{queries}"


def label_digest(p: dict[str, Any]) -> str:
    return sha(profile_text(p))[:16]


class Grader:
    """Grade profiles with the OpenAI Responses API. Answers are cached by request."""

    def __init__(self, cfg: RefineConfig) -> None:
        self.cfg = cfg
        # The client reads OPENAI_API_KEY. It retries rate limits, server errors, and connection
        # errors with backoff.
        self.client = OpenAI(max_retries=6, timeout=600)
        self.lock = threading.Lock()
        self.cache: dict[str, dict[str, Any]] = {}
        if GRADES.is_file():
            self.cache = {r["request"]: r["answer"] for r in iter_jsonl(GRADES)}
        self.usage = Counter()

    def body(self, record: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        text = f"Source facts:\n{item_prompt(record)}\n\nProfile:\n{profile_text(profile)}"
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        image = item_image(record)
        if image and image.exists():
            data = base64.b64encode(image.read_bytes()).decode()
            content.append({"type": "input_image", "image_url": f"data:image/webp;base64,{data}"})
        return {
            "model": self.cfg.model,
            "reasoning": {"effort": self.cfg.reasoning_effort},
            "instructions": SYSTEM,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {"type": "json_schema", "name": "grade", "strict": True, "schema": SCHEMA}
            },
        }

    def request(self, record: dict[str, Any], profile: dict[str, Any]) -> tuple[dict, str]:
        """The request body and its digest. The digest covers the model, the rubric, the facts,
        the image, and the profile, so a change to any of them grades the profile again."""
        body = self.body(record, profile)
        return body, sha(json.dumps(body, sort_keys=True))

    def grade(self, record: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        body, request = self.request(record, profile)
        if request in self.cache:
            return self.cache[request]
        response = self.client.responses.create(**body)
        if response.status != "completed":
            raise RuntimeError(f"{record['id']}: grader response is {response.status}")
        answer = json.loads(response.output_text)
        with self.lock:
            if response.usage:
                self.usage.update(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            self.cache[request] = answer
            append_jsonl(GRADES, [{"request": request, "answer": answer}])
        return answer


def drop(ids: list[str]) -> None:
    """Remove failed works from resolved.jsonl and record them as resolve drops."""
    gone = set(ids)
    rows = list(iter_jsonl(RESOLVED))
    for r in rows:
        if r["id"] in gone and r.get("image"):
            (IMG / r["image"]["src"].rsplit("/", 1)[-1]).unlink(missing_ok=True)
    write_jsonl(RESOLVED, [r for r in rows if r["id"] not in gone])
    append_jsonl(RESOLVE_DROPPED, [{"id": i, "reason": "label_quality"} for i in sorted(gone)])
    sort_jsonl(RESOLVE_DROPPED, "id")


def pending_drops(grader: Grader) -> list[str]:
    """Works whose grade for their current grader request fails. A work without a grade for its
    current request is not dropped; the next round grades it. Computed from the ledger, so an
    interrupted run and a changed threshold both take effect on the next run."""
    if not LEDGER.is_file():
        return []
    by_request = {e["request"]: e for e in iter_jsonl(LEDGER)}
    records = item_records()
    failed = []
    for p in iter_jsonl(PROFILES):
        if p["key"] not in records:
            continue
        entry = by_request.get(grader.request(records[p["key"]], p)[1])
        if entry and entry["grade"]["emotion"] < grader.cfg.min_emotion:
            failed.append(p["key"])
    return sorted(failed)


def grade_round(grader: Grader, round_: int) -> list[dict[str, Any]]:
    """Grade each current profile whose grader request the ledger does not have yet."""
    records = item_records()
    profiles = {p["key"]: p for p in iter_jsonl(PROFILES) if p["key"] in records}
    seen = {e["request"] for e in iter_jsonl(LEDGER)} if LEDGER.is_file() else set()
    todo = [p for k, p in profiles.items() if grader.request(records[k], p)[1] not in seen]
    log.info(
        f"round {round_}: {num(len(todo))} of {num(len(profiles))} profiles to grade with "
        f"{grader.cfg.model}"
    )
    entries: list[dict[str, Any]] = []
    bar = progress(total=len(todo), desc=f"grade round {round_}", unit="profile")

    def one(p: dict[str, Any]) -> None:
        answer = grader.grade(records[p["key"]], p)
        failed = answer["emotion"] < grader.cfg.min_emotion
        entry = {
            "id": p["key"],
            "round": round_,
            "request": grader.request(records[p["key"]], p)[1],
            "label": label_digest(p),
            "model": grader.cfg.model,
            "grade": {k: answer[k] for k in ("factual", "emotion", "specific", "queries_good")},
            "real_emotion": answer["real_emotion"],
            "verdict": "dropped" if failed else "kept",
        }
        with grader.lock:
            entries.append(entry)
            append_jsonl(LEDGER, [entry])
            bar.update()

    with ThreadPoolExecutor(grader.cfg.workers) as pool:
        list(pool.map(one, todo))
    bar.close()
    return entries


def report(entries: list[dict[str, Any]], round_: int) -> None:
    by_cat = Counter(e["id"].split(":")[0] for e in entries)
    failed = Counter(e["id"].split(":")[0] for e in entries if e["verdict"] == "dropped")
    parts = ", ".join(f"{c} {num(failed[c])} of {num(by_cat[c])}" for c in CATEGORIES if by_cat[c])
    log.info(f"round {round_}: failed {num(sum(failed.values()))} ({parts or 'none graded'})")


def loop(step: Callable[[str], Any]) -> None:
    """Run up to RefineConfig.rounds grading rounds. `step` runs a pipeline step by name and
    skips it when it is current."""
    cfg = RefineConfig()
    keys.require(["openai"])
    grader = Grader(cfg)
    ledger = list(iter_jsonl(LEDGER)) if LEDGER.is_file() else []
    round_ = max((e["round"] for e in ledger), default=0) + 1
    # Apply drops that an interrupted run graded but did not apply, then fill every free place.
    if leftover := pending_drops(grader):
        log.info(f"{num(len(leftover))} failed works from an earlier run are dropped first")
        drop(leftover)
    step("resolve")
    for attempt in range(cfg.rounds):
        for name in ("facts", "themes", "leaks", "items"):
            step(name)
        report(grade_round(grader, round_), round_)
        failed = pending_drops(grader)
        if not failed:
            break
        drop(failed)
        if attempt == cfg.rounds - 1:
            log.warning(
                f"round {round_} was the last of this run: {num(len(failed))} failed works are "
                "dropped; the next download or refine fills their places"
            )
            break
        log.info(f"round {round_}: refill {num(len(failed))} places from the next candidates")
        step("resolve")
        round_ += 1
    log.info(
        f"grader tokens: {num(grader.usage['input_tokens'])} in, "
        f"{num(grader.usage['output_tokens'])} out"
    )
