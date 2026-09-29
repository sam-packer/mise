"""Encode length-sorted token batches and restore the input order."""

import gc
from collections.abc import Callable

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from mise_ml.log import get, progress

log = get(__name__)


def encode_batches(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
    encode: Callable[[dict[str, torch.Tensor]], np.ndarray],
    model: torch.nn.Module | None = None,
    device: str = "cpu",
) -> np.ndarray:
    if not texts:
        raise ValueError("cannot encode an empty text list")
    tokens = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    lengths = [len(row) for row in tokens["input_ids"]]
    order = sorted(range(len(texts)), key=lambda i: (lengths[i], i))
    budget = 8192
    if torch.device(device).type == "cuda":
        if model is None:
            raise ValueError("GPU batching needs the model to estimate activation memory")
        free, _ = torch.cuda.mem_get_info(device)
        weights = sum(p.numel() * p.element_size() for p in model.parameters())
        cfg = model.config
        # Reserve workspace and estimate attention and hidden activations per padded token.
        per_token = (
            4
            * cfg.num_hidden_layers
            * (16 * cfg.hidden_size + cfg.num_attention_heads * max_length)
        )
        available = max(0, free - max(512 * 2**20, weights // 10))
        budget = max(max(lengths), int(available * 0.5) // per_token)
    log.info("inference token budget %d on %s for %d texts", budget, device, len(texts))
    result = None
    start = 0
    with progress(total=len(texts), desc="encode", unit="text") as bar:
        while start < len(order):
            end = start + 1
            while end < len(order) and (end - start + 1) * lengths[order[end]] <= budget:
                end += 1
            idx = order[start:end]
            batch = tokenizer.pad(
                [{key: values[i] for key, values in tokens.items()} for i in idx],
                padding=True,
                return_tensors="pt",
            )
            try:
                values = encode(batch)
            except torch.OutOfMemoryError:
                if len(idx) == 1:
                    raise
            else:
                if result is None:
                    result = np.empty((len(texts), *values.shape[1:]), dtype=values.dtype)
                result[idx] = values
                start = end
                bar.update(len(idx))
                continue
            # Leave the handler so its traceback releases failed GPU tensors.
            del batch
            gc.collect()
            torch.cuda.empty_cache()
            budget = max(1, budget // 2)
            log.warning("out of memory; retry at token budget %d", budget)
    return result
