"""Run local Qwen3.5 labeling jobs with transformers and resumable caches.

xgrammar restricts generation to the requested JSON schema.
jsonschema checks each answer because the token limit can cut generation short.
"""

import gc
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jsonschema
import torch
import xgrammar as xgr
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)

from mise_ml.cache_io import cache_lock
from mise_ml.config import DATA, ML_ROOT, ProfileConfig
from mise_ml.log import elapsed, get, num, progress
from mise_ml.util import append_jsonl, iter_jsonl, write_jsonl

Record = dict[str, Any]
log = get(__name__)
LLM_CACHE = DATA / "llm"
# A multiple of the 64-token Gated DeltaNet block, so chunks keep its block boundaries.
PREFILL_CHUNK = 512
# Decoding settings. Every job signature includes them, so a change regenerates the answers.
# Seeded requests and all retries sample with TEMPERATURE and TOP_P.
TEMPERATURE = 0.7
TOP_P = 0.95
THINKING = False
ANY_WHITESPACE = False
# A failed answer gets one sampled retry per seed prefix, with more tokens and this feedback.
RETRY_ATTEMPTS = ("retry:", "retry2:")
RETRY_TOKEN_FACTOR = 2
RETRY_NOTE = (
    "\n\nThe previous answer failed validation: {error}. "
    "Return complete JSON. Keep text concise and follow the requested counts "
    "and word limits. Ignore unrelated source text; use the relevant facts "
    "and mood hints. Do not copy lists of metadata into the answer."
)


@dataclass(frozen=True)
class Request:
    system: str
    user: str
    schema: dict[str, Any]
    max_new_tokens: int
    image: Path | None = None
    # Constrains decoding in place of schema. The signature includes it.
    generation_schema: dict[str, Any] | None = None
    seed: int | None = None
    # The system prompt differs by key. The key fingerprint covers it; the signature does not.
    per_key_system: bool = False


@dataclass
class Unit:
    keys: list[str]
    request: Request


Build = Callable[[list[str]], list[Unit]]
Parse = Callable[[list[str], dict[str, Any]], list[Record]]
Fingerprint = Callable[[str], str]
Fields = Callable[[str], dict[str, Any]]


@dataclass
class JobSpec:
    job: "Job"
    keys: list[str]
    build: Build
    parse: Parse
    fingerprint: Fingerprint
    fields: Fields
    project: Callable[[list[Record]], list[Record]] | None = None
    # A stricter parse for new answers. The last retry and cached answers use parse.
    strict: Parse | None = None

    def run(self, llm: Callable[[], "LocalLLM"]) -> None:
        self.job.run(
            self.keys,
            self.build,
            self.parse,
            self.fingerprint,
            llm,
            self.fields,
            self.project,
            self.strict,
        )


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def limit_gpu_memory() -> None:
    # Force allocator OOM before Windows can spill CUDA allocations into system RAM.
    free, total = torch.cuda.mem_get_info()
    budget = min(int(total * 0.9), free - 2 * 2**30)
    if budget <= 0:
        raise RuntimeError("not enough free GPU memory to leave 2 GiB of headroom")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    log.info(
        f"GPU allocation budget {budget / 2**30:.1f} GiB; "
        f"{(free - budget) / 2**30:.1f} GiB free headroom"
    )


class GrammarProcessor(LogitsProcessor):
    """Masks, row by row, the tokens that the row's JSON grammar does not allow next."""

    def __init__(self, grammars: list[xgr.CompiledGrammar], vocab_size: int) -> None:
        self.matchers = [xgr.GrammarMatcher(g) for g in grammars]
        self.bitmask = xgr.allocate_token_bitmask(len(grammars), vocab_size)
        self.started = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        last = input_ids[:, -1].tolist() if self.started else []
        for i, matcher in enumerate(self.matchers):
            if matcher.is_terminated():
                continue
            if self.started and not matcher.accept_token(last[i]):
                raise RuntimeError("grammar rejected a token it allowed")
            if not matcher.is_terminated():
                matcher.fill_next_token_bitmask(self.bitmask, i)
        self.started = True
        words = self.bitmask.to(scores.device)
        shifts = torch.arange(32, device=scores.device, dtype=torch.int32)
        allowed = ((words.unsqueeze(-1) >> shifts) & 1).view(words.shape[0], -1).bool()
        return scores.masked_fill(~allowed[:, : scores.shape[-1]], float("-inf"))


class SeededSamplingProcessor(LogitsProcessor):
    """Sample each row with its own random stream, independent of other requests."""

    def __init__(self, seeds: list[int]) -> None:
        self.generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]
        self.temperature = TemperatureLogitsWarper(TEMPERATURE)
        self.top_p = TopPLogitsWarper(TOP_P)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        scores = self.top_p(input_ids, self.temperature(input_ids, scores))
        probabilities = scores.softmax(dim=-1)
        tokens = torch.stack(
            [
                torch.multinomial(row, 1, generator=generator)
                for row, generator in zip(probabilities, self.generators, strict=True)
            ]
        )
        # Generation takes argmax after this processor; only the sampled token remains.
        return torch.full_like(scores, float("-inf")).scatter_(1, tokens, 0.0)


class LocalLLM:
    def __init__(self, cfg: ProfileConfig) -> None:
        start = time.perf_counter()
        log.info(f"loading {cfg.model} at {cfg.revision[:12]} in bf16 (about 19 GB)")
        self.cfg = cfg
        limit_gpu_memory()
        self.processor = AutoProcessor.from_pretrained(cfg.model, revision=cfg.revision)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.model, revision=cfg.revision, dtype=torch.bfloat16
        )
        self.model = self.model.cuda().eval()
        # The chat turn ends with the tokenizer EOS (<|im_end|>); the config names another.
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.eos = sorted({self.processor.tokenizer.eos_token_id, *eos})
        self.vocab_size = self.model.config.get_text_config().vocab_size
        info = xgr.TokenizerInfo.from_huggingface(
            self.processor.tokenizer, vocab_size=self.vocab_size, stop_token_ids=self.eos
        )
        self.compiler = xgr.GrammarCompiler(info)
        self.grammars: dict[str, xgr.CompiledGrammar] = {}
        # The largest batch that fits, by token limit, learned from out-of-memory errors.
        # Image batches use it; text batches use the token budget, lowered the same way.
        self.batch_limits: dict[int, int] = {}
        self.token_budget = cfg.batch_token_budget
        self.prompt_lengths: dict[tuple[str, str], int] = {}
        used = torch.cuda.memory_allocated() / 2**30
        log.info(f"model ready in {elapsed(start)}, {used:.1f} GiB on the GPU")

    def grammar(self, schema: dict[str, Any]) -> xgr.CompiledGrammar:
        # Preserve property order so numbered answers identify the input before its labels.
        key = json.dumps(schema)
        if key not in self.grammars:
            self.grammars[key] = self.compiler.compile_json_schema(
                key, any_whitespace=ANY_WHITESPACE
            )
        return self.grammars[key]

    @staticmethod
    def messages(r: Request) -> list[dict[str, Any]]:
        user: list[dict[str, Any]] = []
        if r.image is not None:
            user.append({"type": "image", "image": Image.open(r.image).convert("RGB")})
        user.append({"type": "text", "text": r.user})
        return [
            {"role": "system", "content": [{"type": "text", "text": r.system}]},
            {"role": "user", "content": user},
        ]

    def generate(
        self, requests: list[Request], max_new_tokens: int, sample: bool = False
    ) -> tuple[list[str], int]:
        """Return answers and the number of generated tokens.

        On a GPU memory failure, retry with three quarters of the batch, down to one request.
        Reuse that lower limit for later batches: the token budget for text, the row limit for
        images.
        """
        limit = self.batch_size(requests, max_new_tokens)
        if len(requests) > limit:
            texts: list[str] = []
            tokens = 0
            for i in range(0, len(requests), limit):
                part, n = self.generate(requests[i : i + limit], max_new_tokens, sample)
                texts += part
                tokens += n
            return texts, tokens
        try:
            return self._generate(requests, max_new_tokens, sample)
        except torch.OutOfMemoryError:
            if len(requests) == 1:
                raise
        # Leave the exception handler first: its traceback retains the failed batch's tensors.
        gc.collect()
        torch.cuda.empty_cache()
        # A small step keeps batches large: a failed attempt costs seconds, in prefill.
        if all(r.image is None for r in requests):
            self.token_budget = max(1, self.cells(requests, max_new_tokens) * 3 // 4)
            log.warning(
                f"out of GPU memory on a batch of {len(requests)}; text batches now hold at "
                f"most {self.token_budget:,} tokens"
            )
        else:
            self.batch_limits[max_new_tokens] = max(1, len(requests) * 3 // 4)
            log.warning(
                f"out of GPU memory on a batch of {len(requests)}; batches with up to "
                f"{max_new_tokens} new tokens now hold at most "
                f"{self.batch_limits[max_new_tokens]} requests"
            )
        return self.generate(requests, max_new_tokens, sample)

    def prompt_tokens(self, r: Request) -> int:
        """Tokens of the system and user text; the chat template adds a few more."""
        key = (r.system, r.user)
        if key not in self.prompt_lengths:
            ids = self.processor.tokenizer(r.system + r.user, add_special_tokens=False)
            self.prompt_lengths[key] = len(ids["input_ids"]) + 32
        return self.prompt_lengths[key]

    def cells(self, requests: list[Request], max_new_tokens: int) -> int:
        """Rows times the padded length a batch of these requests reaches."""
        return len(requests) * (max(map(self.prompt_tokens, requests)) + max_new_tokens)

    def batch_size(self, requests: list[Request], max_new_tokens: int) -> int:
        """How many of `requests`, from the front, one batch holds."""
        size = len(requests)
        if any(r.image is not None for r in requests):
            return min(size, self.batch_limits.get(max_new_tokens, size))
        rows = 1
        while rows < size and self.cells(requests[: rows + 1], max_new_tokens) <= self.token_budget:
            rows += 1
        return rows

    @torch.inference_mode()
    def _generate(
        self, requests: list[Request], max_new_tokens: int, sample: bool
    ) -> tuple[list[str], int]:
        inputs = self.processor.apply_chat_template(
            [self.messages(r) for r in requests],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # Transformers 5 takes processor options here; template variables stay in kwargs.
            processor_kwargs={"padding": True},
            enable_thinking=THINKING,
        ).to("cuda")
        grammars = [self.grammar(r.generation_schema or r.schema) for r in requests]
        prefill: dict[str, Any] = {}
        if all(r.image is None for r in requests):
            # Chunked prefill bounds memory use as the prompt grows.
            # The chunk loop slices Qwen's 3D mrope positions on the batch axis.
            # Supply exact 2D text positions; Qwen expands them to the required shape.
            positions = (inputs["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
            prefill = {"position_ids": positions, "prefill_chunk_size": PREFILL_CHUNK}
        processors = [GrammarProcessor(grammars, self.vocab_size)]
        seeded = any(r.seed is not None for r in requests)
        if seeded:
            if not all(r.seed is not None for r in requests):
                raise ValueError("a batch must use either seeded or unseeded requests")
            processors.append(SeededSamplingProcessor([r.seed for r in requests]))
        out = self.model.generate(
            **inputs,
            **prefill,
            max_new_tokens=max_new_tokens,
            do_sample=sample and not seeded,
            temperature=TEMPERATURE if sample and not seeded else None,
            top_p=TOP_P if sample and not seeded else None,
            top_k=None,
            eos_token_id=self.eos,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            logits_processor=processors,
        )
        new = out[:, inputs["input_ids"].shape[1] :]
        tokens = int((new != self.processor.tokenizer.pad_token_id).sum())
        return self.processor.batch_decode(new, skip_special_tokens=True), tokens


def parse_answer(text: str, unit: Unit, parse: Parse) -> tuple[dict[str, Any] | None, str | None]:
    """Validate an answer and retain accepted rows when other rows need a retry."""
    try:
        data = json.loads(text)
        jsonschema.validate(data, unit.request.schema)
    except json.JSONDecodeError as exc:
        return None, f"incomplete or invalid JSON at character {exc.pos}: {exc.msg}"
    except jsonschema.ValidationError as exc:
        return None, f"schema mismatch: {exc.message}"
    try:
        accepted = {r["key"] for r in parse(unit.keys, data)}
    except ValueError as exc:
        return None, str(exc)
    missing = [k for k in unit.keys if k not in accepted]
    if missing:
        # Return the data too: the accepted rows of a numbered answer stay valid records.
        rows = ", ".join(str(unit.keys.index(k) + 1) for k in missing)
        return data, f"content rejected for {len(missing)} of {len(unit.keys)} keys (rows {rows})"
    return data, None


def cached_records(keys: list[str], data: dict[str, Any], parse: Parse) -> list[Record]:
    try:
        return parse(keys, data)
    except ValueError:
        return []


class Job:
    """A resumable labeling job.

    Store one cache line per answered request.
    The signature identifies the model, revision, system prompt, schema, token limit, and the
    decoding settings: the generation grammar, the sampler, thinking, and the retries.
    Schemas keep their property order, because the grammar follows it.
    Each key has a fingerprint of its own prompt content.
    Reuse accepted records only when both the signature and fingerprint match.
    Keep accepted rows from a partly rejected numbered answer and retry the remaining keys.
    """

    def __init__(self, name: str, output: Path, cfg: ProfileConfig) -> None:
        self.name = name
        self.output = output
        self.cfg = cfg
        self.cache = LLM_CACHE / f"{name}.jsonl"
        self.strict: Parse | None = None

    def signature(self, r: Request) -> str:
        return sha(json.dumps(self.signature_parts(r), sort_keys=True))

    def signature_parts(self, r: Request) -> dict[str, Any]:
        return {
            "model": self.cfg.model,
            "revision": self.cfg.revision,
            "prompt": "per-key" if r.per_key_system else sha(r.system),
            "schema": sha(json.dumps(r.schema)),
            "max_new_tokens": r.max_new_tokens,
            "decoding": decoding(r),
        }

    def run(
        self,
        keys: list[str],
        build: Build,
        parse: Parse,
        fingerprint: Fingerprint,
        llm: Callable[[], LocalLLM],
        fields: Fields,
        project: Callable[[list[Record]], list[Record]] | None = None,
        strict: Parse | None = None,
    ) -> None:
        self.strict = strict
        keys = sorted(set(keys))
        prints = {k: fingerprint(k) for k in keys}
        # Use a one-key request for stable identity across chunk sizes. Cached answers are
        # parsed against their original keys, including their original row numbers.
        probe = build(keys[:1])[0].request if keys else None
        sig = self.signature(probe) if probe else ""
        self.parts = self.signature_parts(probe) if probe else {}
        self.field_digests = {k: field_digests(fields(k)) for k in keys}
        records = self.records(sig, prints, parse)
        pending = [k for k in keys if k not in records]
        log.info(
            f"{self.name}: {num(len(keys))} keys; {num(len(keys) - len(pending))} in the cache, "
            f"{num(len(pending))} to generate"
        )
        if pending:
            units = build(pending)
            model = llm()
            start = time.perf_counter()
            stats = self._generate(units, sig, prints, model, parse)
            records = self.records(sig, prints, parse)
            # Keys rejected inside a partly accepted numbered answer get one more pass, in a
            # new request with other rows around them.
            again = [k for u in units if len(u.keys) > 1 for k in u.keys if k not in records]
            if again:
                log.info(f"{self.name}: {num(len(again))} rejected rows; one more pass")
                more = build(again)
                stats += self._generate(more, sig, prints, model, parse)
                units += more
                records = self.records(sig, prints, parse)
            rate = stats["tokens"] / max(time.perf_counter() - start, 1e-9)
            log.info(
                f"{self.name}: {num(len(units))} requests in {elapsed(start)}, "
                f"{num(stats['valid'])} valid, {num(stats['retried'])} retried, "
                f"{num(stats['skipped'])} skipped, {rate:.0f} tokens/s"
            )
            if strict:
                log.info(
                    f"{self.name}: {num(stats['kept'])} answers pass only the lenient check "
                    "on the last retry, and stay"
                )
            if stats["skipped"]:
                log.warning(
                    f"{self.name}: {num(stats['skipped'])} requests failed validation after a "
                    "retry; failed answers and reasons are saved in the cache; "
                    "a rerun retries these failures"
                )
        rows = [records[k] for k in sorted(records)]
        write_jsonl(self.output, project(rows) if project else rows)
        missing = len(keys) - len(records)
        log.info(
            f"{self.name}: {num(len(records))} of {num(len(keys))} keys have a record"
            + (f" ({num(missing)} without: skipped or rejected by parse)" if missing else "")
            + f" -> {self.output.relative_to(ML_ROOT).as_posix()}"
        )
        if missing:
            raise RuntimeError(
                f"{self.name}: {missing} keys have no accepted record; "
                "stopping before downstream steps use incomplete data"
            )

    def records(self, sig: str, prints: dict[str, str], parse: Parse) -> dict[str, Record]:
        # Retain valid records even when the same cached answer also contains unwanted keys.
        records: dict[str, Record] = {}
        for e in iter_jsonl(self.cache):
            if e["sig"] != sig or e.get("data") is None:
                continue
            for rec in cached_records(e["keys"], e["data"], parse):
                k = rec["key"]
                if prints.get(k) is not None and prints[k] == e["prints"].get(k):
                    records[k] = rec
        return records

    def _generate(
        self, units: list[Unit], sig: str, prints: dict[str, str], llm: LocalLLM, parse: Parse
    ) -> Counter[str]:
        stats: Counter[str] = Counter()
        by_kind: dict[bool, list[Unit]] = {False: [], True: []}
        for u in units:
            by_kind[u.request.image is not None].append(u)
        for has_image, group in by_kind.items():
            if not group:
                continue
            group.sort(key=lambda u: (len(u.request.user), u.keys))
            size = self.cfg.image_batch_size if has_image else self.cfg.batch_size
            desc = f"{self.name}{' (images)' if has_image else ''}"
            starting_stats = stats.copy()
            start = time.perf_counter()
            with progress(total=len(group), desc=desc, unit="req") as bar:
                i = 0
                while i < len(group):
                    # Size each batch to the limit learned from out-of-memory errors.
                    window = [u.request for u in group[i : i + size]]
                    tokens = max(r.max_new_tokens for r in window)
                    batch = group[i : i + llm.batch_size(window, tokens)]
                    i += len(batch)
                    for _ in self._run_batch(batch, sig, prints, llm, stats, parse):
                        group_stats = stats - starting_stats
                        rate = group_stats["tokens"] / max(time.perf_counter() - start, 1e-9)
                        bar.set_postfix(
                            valid=group_stats["valid"],
                            retried=group_stats["retried"],
                            skipped=group_stats["skipped"],
                            tok_s=f"{rate:.0f}",
                            refresh=False,
                        )
                        bar.update(1)
                        bar.refresh()
        return stats

    def _lenient(self, text: str, unit: Unit, parse: Parse) -> tuple[str, dict[str, Any]] | None:
        """The answer and its data when only the strict check rejects it, else None."""
        if self.strict is None:
            return None
        data, error = parse_answer(text, unit, parse)
        return (text, data) if not error and data is not None else None

    def _run_batch(
        self,
        batch: list[Unit],
        sig: str,
        prints: dict[str, str],
        llm: LocalLLM,
        stats: Counter[str],
        parse: Parse,
    ) -> Iterator[None]:
        limit = max(u.request.max_new_tokens for u in batch)
        texts, tokens = llm.generate([u.request for u in batch], limit)
        stats["tokens"] += tokens
        strict = self.strict or parse
        answers = [
            (unit, text, *parse_answer(text, unit, strict))
            for unit, text in zip(batch, texts, strict=True)
        ]
        # Save and count completed answers before starting the slower retries.
        answers.sort(key=lambda answer: answer[3] is not None)
        for unit, text, data, error in answers:
            if error:
                # The latest answer that fails only the strict check, for when no retry passes.
                fallback = self._lenient(text, unit, parse)
                # Give the model validation feedback, rather than repeat the same failed answer.
                stats["retried"] += 1
                log.info(
                    f"{self.name}: retrying {unit.keys[0]!r}: {error} "
                    f"(up to {RETRY_TOKEN_FACTOR * unit.request.max_new_tokens} tokens)"
                )
                log.debug(f"{self.name}: retry {unit.keys[0]!r}: {text[-200:]!r}")
                # Two sampled retries, each with its own seed: one key that fails every attempt
                # stops the whole command, so a second try is cheap insurance on long jobs.
                for attempt in RETRY_ATTEMPTS:
                    retry_request = replace(
                        unit.request,
                        seed=(
                            int(sha(attempt + unit.keys[0])[:8], 16)
                            if unit.request.seed is not None
                            else None
                        ),
                        user=unit.request.user + RETRY_NOTE.format(error=error),
                    )
                    # Sample the retry: greedy decoding often repeats the failed answer. A seed
                    # from the key keeps each retry repeatable.
                    torch.manual_seed(int(sha(attempt + unit.keys[0])[:8], 16))
                    retry, tokens = llm.generate(
                        [retry_request],
                        RETRY_TOKEN_FACTOR * unit.request.max_new_tokens,
                        sample=True,
                    )
                    stats["tokens"] += tokens
                    text = retry[0]
                    # The last retry uses the lenient parse, so the strict check alone never
                    # fails a key.
                    last = attempt == RETRY_ATTEMPTS[-1]
                    data, error = parse_answer(text, unit, parse if last else strict)
                    if not error and last and parse_answer(text, unit, strict)[1]:
                        stats["kept"] += 1
                        log.info(f"{self.name}: kept {unit.keys[0]!r} on the lenient check")
                    if not error:
                        break
                    if not last:
                        fallback = self._lenient(text, unit, parse) or fallback
                if error and fallback:
                    text, data = fallback
                    error = None
                    stats["kept"] += 1
                    log.info(
                        f"{self.name}: kept {unit.keys[0]!r} on the lenient check, from an "
                        "earlier attempt"
                    )
                if error:
                    stats["skipped"] += 1
                    log.warning(f"{self.name}: failed {unit.keys}: {error}")
            if not error:
                stats["valid"] += 1
            with cache_lock(self.cache):
                append_jsonl(
                    self.cache,
                    [
                        {
                            "sig": sig,
                            "keys": unit.keys,
                            "prints": {k: prints[k] for k in unit.keys},
                            "fields": {k: self.field_digests[k] for k in unit.keys},
                            "signature": self.parts,
                            "data": data,
                            **({"error": error, "response": text} if error else {}),
                        }
                    ],
                )
            yield None


def decoding(r: Request) -> dict[str, Any]:
    """The settings besides the prompt that decide a request's answers, retries included."""
    return {
        # Seeded requests sample the first answer; other requests decode it greedily.
        "sampled": r.seed is not None,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "thinking": THINKING,
        # The grammar follows the property order, so keep it: no sort_keys.
        "grammar": sha(json.dumps(r.generation_schema or r.schema)),
        "any_whitespace": ANY_WHITESPACE,
        "retries": [list(RETRY_ATTEMPTS), RETRY_TOKEN_FACTOR, sha(RETRY_NOTE)],
    }


def field_digests(fields: dict[str, Any]) -> dict[str, str]:
    return {
        name: sha(json.dumps(value, ensure_ascii=True, sort_keys=True))
        for name, value in fields.items()
    }
