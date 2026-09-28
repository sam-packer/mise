"""The local labeling LLM: Qwen3.5 in-process with transformers, JSON forced by a grammar.

xgrammar compiles each JSON schema (with the vocab-id enums) into a grammar. A logits
processor masks every token that would break the schema, so each finished answer is valid
JSON. jsonschema checks it again, because a hit on max_new_tokens can still cut an answer.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import torch
import xgrammar as xgr
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessor

from moodml.config import DATA, ProfileConfig
from moodml.util import append_jsonl, iter_jsonl, write_jsonl

Record = dict[str, Any]
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


class GrammarProcessor(LogitsProcessor):
    """Masks, row by row, the tokens that the row's JSON grammar does not allow next."""

    def __init__(self, grammars: list[xgr.CompiledGrammar], vocab_size: int) -> None:
        self.matchers = [xgr.GrammarMatcher(g) for g in grammars]
        self.bitmask = xgr.allocate_token_bitmask(len(grammars), vocab_size)
        self.started = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        for i, matcher in enumerate(self.matchers):
            if matcher.is_terminated():
                continue
            if self.started and not matcher.accept_token(int(input_ids[i, -1])):
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
        self.cfg = cfg
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

    @torch.inference_mode()
    def generate(self, requests: list[Request], max_new_tokens: int) -> list[str]:
        inputs = self.processor.apply_chat_template(
            [self.messages(r) for r in requests],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
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
        return self.processor.batch_decode(new, skip_special_tokens=True)


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
        print(f"[{self.name}] {len(keys) - len(pending)} done, {len(pending)} to generate")
        if pending:
            self._generate(build(pending), sig, prints, llm())
            entries = [
                e for e in iter_jsonl(self.cache) if e["sig"] == sig and e.get("data") is not None
            ]
        records: dict[str, Record] = {}
        for e in entries:
            fresh = [k for k in e["keys"] if prints.get(k) == e["prints"].get(k)]
            if len(fresh) == len(e["keys"]):
                for rec in parse(e["keys"], e["data"]):
                    records[rec["key"]] = rec
        write_jsonl(self.output, [records[k] for k in sorted(records)])
        print(f"[{self.name}] {len(records)} of {len(keys)} keys have a record -> {self.output}")

    def _generate(self, units: list[Unit], sig: str, prints: dict[str, str], llm: LocalLLM) -> None:
        by_kind: dict[bool, list[Unit]] = {False: [], True: []}
        for u in units:
            by_kind[u.request.image is not None].append(u)
        for has_image, group in by_kind.items():
            group.sort(key=lambda u: (len(u.request.user), u.keys))
            size = self.cfg.image_batch_size if has_image else self.cfg.batch_size
            bar = tqdm(total=len(group), desc=f"{self.name}{' (images)' if has_image else ''}")
            for start in range(0, len(group), size):
                batch = group[start : start + size]
                self._run_batch(batch, sig, prints, llm)
                bar.update(len(batch))
            bar.close()

    def _run_batch(
        self, batch: list[Unit], sig: str, prints: dict[str, str], llm: LocalLLM
    ) -> None:
        limit = max(u.request.max_new_tokens for u in batch)
        texts = llm.generate([u.request for u in batch], limit)
        rows = []
        for unit, text in zip(batch, texts, strict=True):
            data = valid_json(text, unit.request.schema)
            if data is None:
                # One retry alone, with room for a longer answer.
                retry = llm.generate([unit.request], 2 * unit.request.max_new_tokens)[0]
                data = valid_json(retry, unit.request.schema)
                if data is None:
                    print(f"[{self.name}] skip {unit.keys[0]!r}: no valid JSON after a retry")
            rows.append(
                {
                    "sig": sig,
                    "keys": unit.keys,
                    "prints": {k: prints[k] for k in unit.keys},
                    "data": data,
                }
            )
        append_jsonl(self.cache, rows)
