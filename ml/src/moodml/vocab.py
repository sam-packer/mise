import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from moodml.config import CURATED, VOCAB_PATH

LIGHT_NOTES = {
    "dawn": "pale first light, soft pink and blue",
    "golden-hour": "low warm sun, long shadows",
    "overcast": "flat, cool, diffuse grey daylight",
    "neon": "magenta and cyan signs at night",
    "candle": "small warm flickering flame",
    "moonlight": "cool faint silver light from above",
    "desk-lamp": "one warm pool of light in a dark room",
    "fluorescent": "flat greenish office or store light",
}


@dataclass(frozen=True)
class Vocab:
    lights: list[str]
    typefaces: list[dict[str, str]]
    scents: list[dict[str, str]]
    digest: str

    @property
    def typeface_ids(self) -> list[str]:
        return [t["id"] for t in self.typefaces]

    @property
    def scent_ids(self) -> list[str]:
        return [s["id"] for s in self.scents]

    def sizes(self) -> tuple[int, int, int]:
        return len(self.lights), len(self.typefaces), len(self.scents)

    def prompt_block(self) -> str:
        lights = "\n".join(f"- {x}: {LIGHT_NOTES.get(x, x)}" for x in self.lights)
        faces = "\n".join(f"- {t['id']}: {t['family']}" for t in self.typefaces)
        scents = "\n".join(f"- {s['id']}: {s['text']}" for s in self.scents)
        return f"LIGHTS\n{lights}\n\nTYPEFACES\n{faces}\n\nSCENTS\n{scents}"


def load_vocab() -> Vocab:
    if not VOCAB_PATH.exists():
        raise SystemExit(f"vocab not found: {VOCAB_PATH}. The stub bundle sources own this file.")
    raw = VOCAB_PATH.read_bytes()
    data = json.loads(raw)
    return Vocab(
        lights=list(data["lights"]),
        typefaces=list(data["typefaces"]),
        scents=list(data["scents"]),
        digest=hashlib.sha1(raw).hexdigest(),
    )


def labels_path(vocab: Vocab) -> Path:
    """Labels depend on the vocab, so each vocab version has its own label file."""
    return CURATED / f"labels-{vocab.digest[:8]}.jsonl"
