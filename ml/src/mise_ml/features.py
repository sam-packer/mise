"""Frozen Qwen3-Embedding features, computed once and cached on disk by text.

The encoding follows the Qwen3-Embedding model card: queries get a one-sentence
instruction, documents (items) get none, the tokenizer appends <|endoftext|>, padding is on
the left, the feature is the last-token hidden state, and it is L2-normalized.
"""

import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from mise_ml.config import FEATURES, TeacherConfig
from mise_ml.util import slugify

QUERY_TASK = (
    "Given a feeling someone describes as a scene or a moment, retrieve films, books, songs, "
    "poems, and artworks that share its mood"
)
QUERY_TEMPLATE = f"Instruct: {QUERY_TASK}\nQuery:{{text}}"
ITEM_TEMPLATE = "{text}"


class QwenEncoder:
    def __init__(self, cfg: TeacherConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.backbone, revision=cfg.revision, padding_side="left"
        )
        self.model = AutoModel.from_pretrained(
            cfg.backbone, revision=cfg.revision, dtype=torch.bfloat16
        )
        self.model = self.model.cuda().eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts: list[str], template: str) -> np.ndarray:
        """Last-token pooling with left padding, then L2 normalization."""
        order = sorted(range(len(texts)), key=lambda i: (len(texts[i]), texts[i]))
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float16)
        step = self.cfg.encode_batch
        for start in tqdm(range(0, len(order), step), desc="embedding features"):
            idx = order[start : start + step]
            batch = self.tokenizer(
                [template.format(text=texts[i]) for i in idx],
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to("cuda")
            hidden = self.model(**batch, use_cache=False).last_hidden_state
            last = F.normalize(hidden[:, -1].float(), p=2, dim=-1)
            out[idx] = last.cpu().numpy().astype(np.float16)
        return out


@functools.cache
def shared_encoder(cfg: TeacherConfig) -> QwenEncoder:
    return QwenEncoder(cfg)


class FeatureStore:
    """One array file plus a text list per template. Missing texts are encoded on demand.

    The file name holds the model id and revision, so features of another backbone are
    never reused.
    """

    def __init__(self, name: str, template: str, cfg: TeacherConfig) -> None:
        self.template = template
        self.cfg = cfg
        # The template and token limit change the features too, so they are part of the name.
        setup = hashlib.sha256(f"{template}\0{cfg.max_length}".encode()).hexdigest()[:8]
        stem = f"{slugify(cfg.backbone)}-{cfg.revision[:12]}-{name}-{setup}"
        self.array_path: Path = FEATURES / f"{stem}.npy"
        self.text_path: Path = FEATURES / f"{stem}.json"

    def _load(self) -> tuple[list[str], np.ndarray | None]:
        if not self.array_path.exists():
            return [], None
        texts = json.loads(self.text_path.read_text(encoding="utf-8"))
        return texts, np.load(self.array_path)

    def get(self, texts: list[str]) -> np.ndarray:
        known, array = self._load()
        index = {t: i for i, t in enumerate(known)}
        missing = list(dict.fromkeys(t for t in texts if t not in index))
        if missing:
            new = shared_encoder(self.cfg).encode(missing, self.template)
            array = new if array is None else np.concatenate([array, new])
            for t in missing:
                index[t] = len(index)
            FEATURES.mkdir(parents=True, exist_ok=True)
            np.save(self.array_path, array)
            self.text_path.write_text(json.dumps(list(index)), encoding="utf-8")
        assert array is not None
        return array[[index[t] for t in texts]]
