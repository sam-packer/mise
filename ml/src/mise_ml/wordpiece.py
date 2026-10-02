"""Prune a WordPiece vocabulary and the matching rows of the encoder's word embeddings.

WordPiece takes the longest vocabulary piece at each position of a word. A word whose pieces all
stay in the vocabulary gets the same pieces after pruning. Other words fall back to shorter kept
pieces, at worst one character each, because pruning keeps every character piece.
"""

import json
import tempfile

from tokenizers import Tokenizer
from torch import nn
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def kept_ids(tokenizer: PreTrainedTokenizerBase, texts: list[str], keep_below: int) -> list[int]:
    """Sorted original ids of the tokens to keep.

    Keep the special tokens, every character piece ("a" and "##a"), the tokens of `texts`, and
    every regular token with an id below `keep_below`. BERT vocabularies list words in about
    their corpus frequency order, so a low id marks a common token.
    """
    vocab = tokenizer.get_vocab()
    kept = set(tokenizer.added_tokens_decoder)
    kept.update(i for t, i in vocab.items() if len(t.removeprefix("##")) == 1)
    kept.update(i for t, i in vocab.items() if i < keep_below and not t.startswith("[unused"))
    for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
        kept.update(ids)
    return sorted(kept)


def prune_tokenizer(tokenizer: PreTrainedTokenizerBase, kept: list[int]) -> PreTrainedTokenizerBase:
    """The tokenizer with only the `kept` tokens. New ids follow the order of `kept`."""
    new_id = {old: new for new, old in enumerate(kept)}
    data = json.loads(tokenizer.backend_tokenizer.to_str())
    model = data["model"]
    if model["type"] != "WordPiece":
        raise ValueError(f"vocabulary pruning needs a WordPiece tokenizer, not {model['type']}")
    model["vocab"] = {t: new_id[i] for t, i in model["vocab"].items() if i in new_id}
    for token in data["added_tokens"]:
        token["id"] = new_id[token["id"]]
    post = data["post_processor"]
    if post["type"] != "TemplateProcessing":
        raise ValueError(f"vocabulary pruning needs TemplateProcessing, not {post['type']}")
    for special in post["special_tokens"].values():
        special["ids"] = [new_id[i] for i in special["ids"]]
    with tempfile.TemporaryDirectory(prefix="mise-wordpiece-") as directory:
        tokenizer.save_pretrained(directory)
        Tokenizer.from_str(json.dumps(data)).save(f"{directory}/tokenizer.json")
        pruned = AutoTokenizer.from_pretrained(directory)
    if len(pruned) != len(kept):
        raise RuntimeError(f"pruned tokenizer has {len(pruned)} tokens, expected {len(kept)}")
    return pruned


def prune_encoder(encoder: PreTrainedModel, kept: list[int]) -> None:
    """Keep only the `kept` rows of the word embeddings, in the order of `kept`."""
    old = encoder.get_input_embeddings()
    pad = encoder.config.pad_token_id
    pad = kept.index(pad) if pad is not None else None
    new = nn.Embedding(
        len(kept),
        old.embedding_dim,
        padding_idx=pad,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    new.weight.data.copy_(old.weight.data[kept])
    encoder.set_input_embeddings(new)
    encoder.config.vocab_size = len(kept)
    encoder.config.pad_token_id = pad


def prune(
    tokenizer: PreTrainedTokenizerBase, encoder: PreTrainedModel, texts: list[str], keep_below: int
) -> PreTrainedTokenizerBase:
    """Prune the encoder in place and return the matching tokenizer."""
    kept = kept_ids(tokenizer, texts, keep_below)
    prune_encoder(encoder, kept)
    return prune_tokenizer(tokenizer, kept)
