"""Cache frozen Qwen3-Embedding features on disk for teacher training.

Queries receive an instruction; items do not.
Use left padding and normalize the last token's hidden state, as the model card specifies.
"""

import functools
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from mise_ml.config import FEATURES, TeacherConfig
from mise_ml.inference import encode_batches
from mise_ml.log import elapsed, get, num
from mise_ml.util import slugify

log = get(__name__)

QUERY_TASK = (
    "Given a feeling someone describes as a scene or a moment, retrieve films, books, songs, "
    "poems, and artworks that share its mood"
)
QUERY_TEMPLATE = f"Instruct: {QUERY_TASK}\nQuery:{{text}}"
ITEM_TEMPLATE = "{text}"


class QwenEncoder:
    def __init__(self, cfg: TeacherConfig) -> None:
        start = time.perf_counter()
        log.info(f"loading {cfg.backbone} at {cfg.revision[:12]} in bf16 (about 15 GB)")
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.backbone, revision=cfg.revision, padding_side="left"
        )
        # The model's tokenizer appends <|endoftext|>, whose hidden state represents the text.
        self.model = AutoModel.from_pretrained(
            cfg.backbone, revision=cfg.revision, dtype=torch.bfloat16
        )
        self.model = self.model.cuda().eval()
        self.model.requires_grad_(False)
        used = torch.cuda.memory_allocated() / 2**30
        log.info(f"model ready in {elapsed(start)}, {used:.1f} GiB on the GPU")

    @torch.inference_mode()
    def encode(self, texts: list[str], template: str) -> np.ndarray:
        """Last-token pooling with left padding, then L2 normalization."""

        def encode(batch):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            hidden = self.model(**batch, use_cache=False).last_hidden_state
            last = F.normalize(hidden[:, -1].float(), p=2, dim=-1)
            return last.cpu().numpy().astype(np.float16)

        return encode_batches(
            self.tokenizer,
            [template.format(text=t) for t in texts],
            self.cfg.max_length,
            encode,
            "cuda",
        )


@functools.cache
def shared_encoder(cfg: TeacherConfig) -> QwenEncoder:
    return QwenEncoder(cfg)


class FeatureStore:
    """One array file plus a text list per template. Missing texts are encoded on demand.

    Put the model ID and revision in the file name to separate features from different backbones.
    Labeled and distillation feelings share the queries store and template.
    """

    def __init__(self, name: str, template: str, cfg: TeacherConfig) -> None:
        self.name = name
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
        unique = list(dict.fromkeys(texts))
        missing = [t for t in unique if t not in index]
        log.info(
            f"{self.name} features: {num(len(texts))} texts ({num(len(unique))} unique), "
            f"{num(len(unique) - len(missing))} unique texts from the cache, "
            f"{num(len(missing))} to encode"
        )
        if missing:
            encoder = shared_encoder(self.cfg)
            start = time.perf_counter()
            new = encoder.encode(missing, self.template)
            rate = len(missing) / max(time.perf_counter() - start, 1e-9)
            log.info(
                f"{self.name} features: encoded {num(len(missing))} texts in {elapsed(start)} "
                f"({rate:.0f} texts/s) -> {self.array_path.name}"
            )
            array = new if array is None else np.concatenate([array, new])
            for t in missing:
                index[t] = len(index)
            FEATURES.mkdir(parents=True, exist_ok=True)
            np.save(self.array_path, array)
            self.text_path.write_text(json.dumps(list(index)), encoding="utf-8")
        assert array is not None
        return array[[index[t] for t in texts]]
