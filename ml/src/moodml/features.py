"""Frozen Qwen3 features, computed once and cached on disk by text."""

import functools
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from moodml.config import FEATURES, TeacherConfig
from moodml.util import warn_only_determinism

QUERY_TEMPLATE = (
    "Instruct: Imagine this moment. Think of its colors, its light, its sound, and the films, "
    "books, songs, poems and paintings that feel like it.\nMoment: {text}\nFeeling:"
)
ITEM_TEMPLATE = (
    "Instruct: Imagine this work. Think of its colors, its light, its sound, and the moments "
    "in a life that feel like it.\nWork: {text}\nFeeling:"
)


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
        """Mean of the final hidden states concatenated with the last-token state."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        dim = self.model.config.hidden_size * 2
        out = np.zeros((len(texts), dim), dtype=np.float16)
        step = self.cfg.encode_batch
        for start in tqdm(range(0, len(order), step), desc="qwen features"):
            idx = order[start : start + step]
            batch = self.tokenizer(
                [template.format(text=texts[i]) for i in idx],
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to("cuda")
            hidden = self.model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            mean = (hidden * mask).sum(1) / mask.sum(1)
            last = hidden[:, -1]
            out[idx] = torch.cat([mean, last], dim=-1).cpu().numpy().astype(np.float16)
        return out


@functools.cache
def shared_encoder(cfg: TeacherConfig) -> QwenEncoder:
    return QwenEncoder(cfg)


class FeatureStore:
    """One array file plus a text list per template. Missing texts are encoded on demand."""

    def __init__(self, name: str, template: str, cfg: TeacherConfig) -> None:
        self.template = template
        self.cfg = cfg
        self.array_path: Path = FEATURES / f"{name}.npy"
        self.text_path: Path = FEATURES / f"{name}.json"

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
            with warn_only_determinism():
                new = shared_encoder(self.cfg).encode(missing, self.template)
            array = new if array is None else np.concatenate([array, new])
            for t in missing:
                index[t] = len(index)
            FEATURES.mkdir(parents=True, exist_ok=True)
            np.save(self.array_path, array)
            self.text_path.write_text(json.dumps(list(index)), encoding="utf-8")
        assert array is not None
        return array[[index[t] for t in texts]]
