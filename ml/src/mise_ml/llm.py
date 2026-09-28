"""The local labeling LLM: Qwen3.5 in-process with transformers, JSON forced by a grammar.

xgrammar compiles each JSON schema (with the vocab-id enums) into a grammar. A logits
processor masks every token that would break the schema, so each finished answer is valid
JSON. jsonschema checks it again, because a hit on max_new_tokens can still cut an answer.
"""

import gc
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import torch
import xgrammar as xgr
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessor

from mise_ml.config import DATA, ML_ROOT, ProfileConfig
from mise_ml.log import elapsed, get, num, progress
from mise_ml.util import append_jsonl, iter_jsonl, write_jsonl

Record = dict[str, Any]
log = get(__name__)
LLM_CACHE = DATA / "llm"


@dataclass(frozen=True)
class Request:
    system: str
    user: str
    schema: dict[str, Any]
    max_new_tokens: int
    image: Path | None = None


@dataclass
class Unit:
    keys: list[str]
    request: Request


Build = Callable[[list[str]], list[Unit]]
Parse = Callable[[list[str], dict[str, Any]], list[Record]]
Fingerprint = Callable[[str], str]


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
        used = torch.cuda.memory_allocated() / 2**30
        log.info(f"model ready in {elapsed(start)}, {used:.1f} GiB on the GPU")

    def grammar(self, schema: dict[str, Any]) -> xgr.CompiledGrammar:
        key = json.dumps(schema, sort_keys=True)
        if key not in self.grammars:
            self.grammars[key] = self.compiler.compile_json_schema(key, any_whitespace=False)
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

    def generate(self, requests: list[Request], max_new_tokens: int) -> tuple[list[str], int]:
        """The answers and the number of new tokens generated.

        A batch that runs out of GPU memory is split in half and retried, down to one request,
        so the longest batches at the end of a job slow down instead of stopping the run.
        """
        try:
            return self._generate(requests, max_new_tokens)
        except torch.OutOfMemoryError:
            if len(requests) == 1:
                raise
        # Leave the exception handler first: its traceback retains the failed batch's tensors.
        gc.collect()
        torch.cuda.empty_cache()
        half = len(requests) // 2
        log.warning(f"out of GPU memory on a batch of {len(requests)}; retrying as two halves")
        first, t1 = self.generate(requests[:half], max_new_tokens)
        second, t2 = self.generate(requests[half:], max_new_tokens)
        return first + second, t1 + t2

    @torch.inference_mode()
    def _generate(self, requests: list[Request], max_new_tokens: int) -> tuple[list[str], int]:
        inputs = self.processor.apply_chat_template(
            [self.messages(r) for r in requests],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # Transformers 5 takes processor options here; template variables stay in kwargs.
            processor_kwargs={"padding": True},
            enable_thinking=False,
        ).to("cuda")
        grammars = [self.grammar(r.schema) for r in requests]
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=self.eos,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            logits_processor=[GrammarProcessor(grammars, self.vocab_size)],
        )
        new = out[:, inputs["input_ids"].shape[1] :]
        tokens = int((new != self.processor.tokenizer.pad_token_id).sum())
        return self.processor.batch_decode(new, skip_special_tokens=True), tokens


def valid_json(text: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        jsonschema.validate(data, schema)
    except (json.JSONDecodeError, jsonschema.ValidationError):
        return None
    return data


class Job:
    """A resumable labeling job.

    The cache holds one line per answered request, with the job signature (model,
    revision, system prompt, schema, token limit) and one fingerprint per key (the hash
    of that key's own prompt content). A key is done when a cache line with the current
    signature and fingerprint covers it. A new model, revision, or prompt therefore
    redoes exactly the affected keys, and a rerun never redoes finished work.
    """

    def __init__(self, name: str, output: Path, cfg: ProfileConfig) -> None:
        self.name = name
        self.output = output
        self.cfg = cfg
        self.cache = LLM_CACHE / f"{name}.jsonl"

    def signature(self, r: Request) -> str:
        return sha(
            json.dumps(
                [self.cfg.model, self.cfg.revision, r.system, r.schema, r.max_new_tokens],
                sort_keys=True,
            )
        )

    def run(
        self,
        keys: list[str],
        build: Build,
        parse: Parse,
        fingerprint: Fingerprint,
        llm: Callable[[], LocalLLM],
    ) -> None:
        keys = sorted(set(keys))
        prints = {k: fingerprint(k) for k in keys}
        probe = build(keys[:1])[0].request if keys else None
        sig = self.signature(probe) if probe else ""
        entries = [
            e for e in iter_jsonl(self.cache) if e["sig"] == sig and e.get("data") is not None
        ]
        attempted = {
            k
            for e in iter_jsonl(self.cache)
            if e["sig"] == sig
            for k, p in e["prints"].items()
            if prints.get(k) == p
        }
        pending = [k for k in keys if k not in attempted]
        log.info(
            f"{self.name}: {num(len(keys))} keys; {num(len(keys) - len(pending))} in the cache, "
            f"{num(len(pending))} to generate"
        )
        if pending:
            units = build(pending)
            start = time.perf_counter()
            stats = self._generate(units, sig, prints, llm())
            rate = stats["tokens"] / max(time.perf_counter() - start, 1e-9)
            log.info(
                f"{self.name}: {num(len(units))} requests in {elapsed(start)}, "
                f"{num(stats['valid'])} valid, {num(stats['retried'])} retried, "
                f"{num(stats['skipped'])} skipped, {rate:.0f} tokens/s"
            )
            if stats["skipped"]:
                log.warning(
                    f"{self.name}: {num(stats['skipped'])} requests gave no valid JSON after a "
                    "retry and were skipped (keys in the log file); a rerun tries them again"
                )
            entries = [
                e for e in iter_jsonl(self.cache) if e["sig"] == sig and e.get("data") is not None
            ]
        # Keep each parsed record on its own merits: a cached answer can cover keys that are
        # no longer wanted, and its other keys must not be lost with them.
        records: dict[str, Record] = {}
        for e in entries:
            for rec in parse(e["keys"], e["data"]):
                k = rec["key"]
                if prints.get(k) is not None and prints[k] == e["prints"].get(k):
                    records[k] = rec
        write_jsonl(self.output, [records[k] for k in sorted(records)])
        missing = len(keys) - len(records)
        log.info(
            f"{self.name}: {num(len(records))} of {num(len(keys))} keys have a record"
            + (f" ({num(missing)} without: skipped or rejected by parse)" if missing else "")
            + f" -> {self.output.relative_to(ML_ROOT).as_posix()}"
        )

    def _generate(
        self, units: list[Unit], sig: str, prints: dict[str, str], llm: LocalLLM
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
            bar = progress(total=len(group), desc=desc, unit="req")
            start = time.perf_counter()
            for i in range(0, len(group), size):
                batch = group[i : i + size]
                self._run_batch(batch, sig, prints, llm, stats)
                bar.update(len(batch))
                bar.set_postfix(
                    valid=stats["valid"],
                    retried=stats["retried"],
                    skipped=stats["skipped"],
                    tok_s=f"{stats['tokens'] / max(time.perf_counter() - start, 1e-9):.0f}",
                    refresh=False,
                )
            bar.close()
        return stats

    def _run_batch(
        self,
        batch: list[Unit],
        sig: str,
        prints: dict[str, str],
        llm: LocalLLM,
        stats: Counter[str],
    ) -> None:
        limit = max(u.request.max_new_tokens for u in batch)
        texts, tokens = llm.generate([u.request for u in batch], limit)
        stats["tokens"] += tokens
        rows = []
        for unit, text in zip(batch, texts, strict=True):
            data = valid_json(text, unit.request.schema)
            if data is None:
                # One retry alone, with room for a longer answer.
                stats["retried"] += 1
                log.debug(f"{self.name}: retry {unit.keys[0]!r}: {text[-200:]!r}")
                retry, tokens = llm.generate([unit.request], 2 * unit.request.max_new_tokens)
                stats["tokens"] += tokens
                data = valid_json(retry[0], unit.request.schema)
                if data is None:
                    stats["skipped"] += 1
                    log.debug(f"{self.name}: skip {unit.keys}: no valid JSON after a retry")
            if data is not None:
                stats["valid"] += 1
            rows.append(
                {
                    "sig": sig,
                    "keys": unit.keys,
                    "prints": {k: prints[k] for k in unit.keys},
                    "data": data,
                }
            )
        append_jsonl(self.cache, rows)
