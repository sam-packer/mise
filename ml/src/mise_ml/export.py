import logging
import re
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import PreTrainedTokenizerBase

from mise_ml.config import (
    BUNDLE,
    CATALOG_BUNDLE,
    IMG,
    ML_ROOT,
    OUT,
    SEED,
    VOCAB_PATH,
    ExportConfig,
    StudentConfig,
    TeacherConfig,
)
from mise_ml.data import Catalog, load_catalog, load_queries, recall_at_k
from mise_ml.inference import encode_batches
from mise_ml.log import elapsed, get, num, progress
from mise_ml.student import OUTPUT_NAMES, STUDENT_DIR, Student, load_student
from mise_ml.util import make_deterministic, sha256_file, write_json
from mise_ml.vocab import load_vocab

log = get(__name__)
INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")
ITEM_FIELDS = (
    "id",
    "category",
    "title",
    "creator",
    "year",
    "vibe",
    "image",
    "text",
    "preview",
    "album",
)


def export_onnx(model: Student, path: Path, opset: int) -> None:
    # The exporter warns that torchvision and triton are absent; neither is used here.
    for name in ("torch.onnx", "torch.utils.flop_counter"):
        logging.getLogger(name).setLevel(logging.ERROR)
    model = model.float().cpu().eval()
    example = (
        torch.ones((2, 16), dtype=torch.int64),
        torch.ones((2, 16), dtype=torch.int64),
        torch.zeros((2, 16), dtype=torch.int64),
    )
    dims = {0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        opset_version=opset,
        dynamo=True,
        dynamic_shapes={name: dims for name in INPUT_NAMES},
        external_data=False,
        verbose=False,
    )
    rename_shadowed_values(path)


def rename_shadowed_values(path: Path) -> None:
    """Give internal values that reuse a graph output name their own name.

    The dynamo exporter can name an internal value after an output, for example
    the word-embedding Gather also becomes "embedding". ONNX needs each name once.
    """
    model = onnx.load(str(path))
    producers = Counter(name for node in model.graph.node for name in node.output)
    seen: Counter[str] = Counter()
    current: dict[str, str] = {}
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = current.get(name, name)
        for i, name in enumerate(node.output):
            if producers[name] < 2:
                continue
            seen[name] += 1
            if seen[name] < producers[name]:
                node.output[i] = current[name] = f"{name}__shadow{seen[name]}"
            else:
                current.pop(name, None)
    model.graph.ClearField("value_info")
    onnx.save(model, str(path))


def quantize(
    fp32: Path, int8: Path, *, per_channel: bool = True, exclude: list[str] | None = None
) -> None:
    # onnxruntime logs a pre-processing hint on the root logger on every call; hide it.
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        quantize_dynamic(
            fp32,
            int8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            nodes_to_exclude=exclude,
        )
    finally:
        root.setLevel(level)


def onnx_session(path: Path, threads: int = 0) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def onnx_run(
    session: ort.InferenceSession,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
) -> dict[str, np.ndarray]:
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="np"
    )
    feeds = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
        "token_type_ids": enc.get("token_type_ids", np.zeros_like(enc["input_ids"])).astype(
            np.int64
        ),
    }
    feeds = {node.name: feeds[node.name] for node in session.get_inputs()}
    return dict(zip(OUTPUT_NAMES, session.run(list(OUTPUT_NAMES), feeds), strict=True))


def onnx_embed(
    session: ort.InferenceSession,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
    desc: str | None = None,
) -> np.ndarray:
    def encode(batch):
        batch.setdefault("token_type_ids", torch.zeros_like(batch["input_ids"]))
        feeds = {node.name: batch[node.name].numpy() for node in session.get_inputs()}
        return session.run(["embedding"], feeds)[0]

    if desc:
        log.info(desc)
    emb = encode_batches(tokenizer, texts, max_length, encode).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def quantization_recipes(fp32: Path) -> dict[str, tuple[bool, list[str]]]:
    graph = onnx.load(fp32).graph
    metadata = {n.name: " ".join(p.value for p in n.metadata_props) for n in graph.node}
    heads = [
        n.name
        for n in graph.node
        if any(tag in metadata[n.name] for tag in ("/heads", "/projection", "/pooler"))
    ]
    attention = [
        n.name
        for n in graph.node
        if n.op_type in ("MatMul", "Gemm") and "attention.output.dense" in metadata[n.name]
    ]
    linear = [
        n.name
        for n in graph.node
        if n.op_type in ("MatMul", "Gemm")
        and "aten.linear" in metadata[n.name]
        and n.name not in heads
    ]
    return {
        "per_tensor": (False, []),
        "per_channel": (True, []),
        "heads_fp32": (True, heads),
        "last_attention_fp32": (True, heads + attention[-1:]),
        "last_linear_fp32": (True, heads + linear[-1:]),
        "embeddings_fp32": (True, heads + [n.name for n in graph.node if n.op_type == "Gather"]),
    }


def retrieval_measurement(
    path: Path, tokenizer, catalog: Catalog, qs, cfg: StudentConfig, split: str = "val"
) -> float:
    rows = qs.where(split)
    rows = rows[qs.pos[rows] >= 0]
    if not len(rows):
        raise ValueError(f"no retrieval feelings in {split}")
    session = onnx_session(path, threads=4)
    items = onnx_embed(session, tokenizer, catalog.texts, cfg.item_max_length)
    # Browser queries run alone; dynamic int8 ranges depend on the other rows in a batch.
    queries = np.concatenate(
        [onnx_run(session, tokenizer, [qs.texts[i]], cfg.max_length)["embedding"] for i in rows]
    )
    return recall_at_k(queries, qs.pos[rows], items, catalog.categories)


def select_quantization(
    fp32: Path,
    target: Path,
    tokenizer,
    catalog: Catalog,
    cfg: StudentConfig,
    export_cfg: ExportConfig,
) -> dict:
    qs = load_queries(catalog, load_vocab(), TeacherConfig())
    reference = retrieval_measurement(fp32, tokenizer, catalog, qs, cfg)
    log.info("fp32 val recall@10 %.6f", reference)
    measurements = {"fp32": {"val_recall@10": reference, "bytes": fp32.stat().st_size}}
    best = None
    scores: dict[str, float] = {}
    for name, (per_channel, exclude) in quantization_recipes(fp32).items():
        path = target.with_name(f"{name}.onnx")
        quantize(fp32, path, per_channel=per_channel, exclude=exclude)
        size = path.stat().st_size
        row = {"bytes": size, "per_channel": per_channel, "excluded": exclude}
        measurements[name] = row
        if size > export_cfg.max_model_bytes:
            row["rejected"] = "over size cap"
            log.info("%s: %.2f MiB exceeds the size cap", name, size / 2**20)
            continue
        digest = sha256_file(path)
        if digest not in scores:
            scores[digest] = retrieval_measurement(path, tokenizer, catalog, qs, cfg)
        else:
            log.info("%s produces an identical graph; reuse its measured recall", name)
        recall = scores[digest]
        row["val_recall@10"] = recall
        log.info(
            "%s: %.2f MiB, val recall@10 %.6f, fp32 gap %.2f points",
            name,
            size / 2**20,
            recall,
            (reference - recall) * 100,
        )
        rank = (recall, -size)
        if best is None or rank > best[0]:
            best = (rank, name, path)
    if best is None:
        raise RuntimeError("no quantization recipe fits the model size cap")
    shutil.copyfile(best[2], target)
    measurements["selected"] = best[1]
    write_json(target.with_suffix(".json"), measurements)
    log.info("selected %s by validation recall@10", best[1])
    return measurements


def bundle_item(item: dict[str, Any]) -> dict[str, Any]:
    out = {k: item[k] for k in ITEM_FIELDS if item.get(k) is not None or k in ("year", "image")}
    out["links"] = {k: v for k, v in item["links"].items() if v}
    if out["image"]:
        out["image"] = {k: out["image"][k] for k in ("src", "w", "h", "tone")}
    return out


def validate_items(items: list[dict[str, Any]]) -> list[str]:
    problems = []
    for it in items:
        if it["category"] != "poem" and it["image"] is None:
            problems.append(f"{it['id']}: no image")
        if not it["links"].get("primary"):
            problems.append(f"{it['id']}: no primary link")
        if it["category"] == "song" and not (it["links"].get("deezer") and it.get("preview")):
            problems.append(f"{it['id']}: no Deezer link or preview")
        if it["category"] == "poem" and not it.get("text"):
            problems.append(f"{it['id']}: no poem text")
    return problems


def common_words(tokenizer: PreTrainedTokenizerBase, items: list[dict[str, Any]]) -> list[str]:
    """Check vocabulary words and each name word with the exported tokenizer."""
    candidates = {word for word in tokenizer.get_vocab() if re.fullmatch(r"[a-z0-9]+", word)}
    for item in items:
        for field in ("title", "creator", "album"):
            text = unicodedata.normalize("NFD", item.get(field) or "")
            text = re.sub(r"[\u0300-\u036f]", "", text).lower().replace("&", " and ")
            text = re.sub(r"['\u2019]", "", text)
            candidates.update(filter(None, re.split(r"[^a-z0-9]+", text)))
    return sorted(
        word for word in candidates if len(tokenizer.encode(word, add_special_tokens=False)) == 1
    )


def write_bundle(
    model_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    encoder_dir: Path,
    catalog: Catalog,
    student_cfg: StudentConfig,
) -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    if CATALOG_BUNDLE.exists():
        shutil.rmtree(CATALOG_BUNDLE)
    CATALOG_BUNDLE.mkdir(parents=True)
    model_dir = BUNDLE / "model"
    model_dir.mkdir(parents=True)
    shutil.copy2(model_path, model_dir / "model.onnx")
    tokenizer.save_pretrained(model_dir)
    shutil.copy2(encoder_dir / "config.json", model_dir / "config.json")

    items = [bundle_item(it) for it in catalog.items]
    names = [it["image"]["src"].rsplit("/", 1)[-1] for it in items if it["image"]]
    absent = [n for n in names if not (IMG / n).exists()]
    if absent:
        raise SystemExit(f"{len(absent)} images are missing from {IMG}; run `uv run download`")
    problems = validate_items(items)
    if problems:
        for p in problems:
            log.debug(f"item problem: {p}")
        log.error(f"{num(len(problems))} item problems, for example: {'; '.join(problems[:5])}")
        raise SystemExit(1)

    session = onnx_session(model_path)
    log.info(f"item vectors: encoding {num(len(catalog.texts))} items with the int8 graph")
    vectors = onnx_embed(
        session, tokenizer, catalog.texts, student_cfg.item_max_length, desc="item vectors"
    )
    (CATALOG_BUNDLE / "vectors.bin").write_bytes(vectors.astype("<f4").tobytes())
    write_json(CATALOG_BUNDLE / "items.json", items)
    shutil.copy2(VOCAB_PATH, BUNDLE / "vocab.json")

    img_dir = BUNDLE / "img"
    img_dir.mkdir()
    for name in progress(names, desc="images", unit="file"):
        shutil.copy2(IMG / name, img_dir / name)

    manifest = {
        "version": "ml-"
        + sha256_file(model_dir / "model.onnx")[:8]
        + sha256_file(CATALOG_BUNDLE / "vectors.bin")[:8],
        "encoder": {
            "model": "model/model.onnx",
            "tokenizer": "model/",
            "dims": int(vectors.shape[1]),
            "maxTokens": student_cfg.max_length,
            "pooling": "none",
            "normalize": True,
            "outputs": {name: name for name in OUTPUT_NAMES},
        },
        "heads": {"kind": "onnx"},
        "counts": {"items": len(items)},
    }
    write_json(BUNDLE / "manifest.json", manifest)
    write_json(
        CATALOG_BUNDLE / "catalog.json",
        {
            "version": manifest["version"],
            "dims": manifest["encoder"]["dims"],
            "counts": manifest["counts"],
            "heads": manifest["heads"],
            "words": common_words(tokenizer, items),
        },
    )
    counts = Counter(it["category"] for it in items)
    size = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file())
    log.info(
        f"bundle {manifest['version']}: {num(len(items))} items "
        f"({', '.join(f'{k} {num(v)}' for k, v in sorted(counts.items()))}), "
        f"{num(len(names))} images, {size / 2**20:.0f} MiB -> "
        f"{BUNDLE.relative_to(ML_ROOT).as_posix()}"
    )


def check_outputs(
    model_path: Path, tokenizer: PreTrainedTokenizerBase, sizes: tuple, dims: int
) -> None:
    session = onnx_session(model_path)
    out = onnx_run(session, tokenizer, ["a snowy december and i just made warm hot chocolate"], 96)
    expected = {
        "embedding": (1, dims),
        "palette": (1, 5, 3),
        "light": (1, sizes[0]),
        "typeface": (1, sizes[1]),
        "scent": (1, sizes[2]),
    }
    for name, shape in expected.items():
        if out[name].shape != shape:
            log.error(f"output {name} has shape {out[name].shape}, expected {shape}")
            raise SystemExit(1)
    log.info("output shapes ok: " + ", ".join(f"{k} {v}" for k, v in expected.items()))


def run() -> None:
    start = time.perf_counter()
    make_deterministic(SEED)
    cfg = ExportConfig()
    vocab = load_vocab()
    catalog = load_catalog()
    model, tokenizer, meta = load_student()
    if meta["vocab"] != vocab.digest:
        raise SystemExit("vocab changed since train-student; retrain")
    if meta["item_ids"] != [it["id"] for it in catalog.items]:
        raise SystemExit("catalog changed since train-student; retrain")
    log.info(
        f"reading {STUDENT_DIR.relative_to(ML_ROOT).as_posix()} "
        f"(val recall@10 {meta.get('val_recall@10', float('nan')):.3f}) and "
        f"{num(len(catalog.items))} items; opset {cfg.opset}, "
        f"limit {cfg.max_model_bytes / 2**20:.0f} MiB"
    )

    work = OUT / "onnx"
    fp32, int8 = work / "student.fp32.onnx", work / "student.int8.onnx"
    log.info("exporting the ONNX graph (fp32)")
    export_onnx(model, fp32, cfg.opset)
    log.info("dynamic int8 quantization")
    student_cfg = StudentConfig(
        **{k: v for k, v in meta["cfg"].items() if k != "item_encode_batch"}
    )
    select_quantization(fp32, int8, tokenizer, catalog, student_cfg, cfg)
    size = int8.stat().st_size
    log.info(f"int8 model: {size / 2**20:.2f} MiB (fp32 {fp32.stat().st_size / 2**20:.1f} MiB)")
    if size > cfg.max_model_bytes:
        log.error(
            f"the model is {size / 2**20:.2f} MiB, "
            f"over the {cfg.max_model_bytes / 2**20:.0f} MiB limit"
        )
        raise SystemExit(1)
    check_outputs(int8, tokenizer, vocab.sizes(), student_cfg.dims)

    sample = [it["queries"][0] for it in catalog.items[:: max(1, len(catalog.items) // 200)]]
    a = onnx_embed(onnx_session(fp32), tokenizer, sample, cfg.max_tokens)
    b = onnx_embed(onnx_session(int8), tokenizer, sample, cfg.max_tokens)
    cosine = (a * b).sum(1)
    log.info(
        f"int8 vs fp32 embeddings on {len(sample)} feelings: cosine mean {cosine.mean():.4f}, "
        f"min {cosine.min():.4f}"
    )
    if cosine.min() < 0.9:
        log.warning("some int8 embeddings differ a lot from fp32 (cosine below 0.9)")

    write_bundle(int8, tokenizer, STUDENT_DIR / "encoder", catalog, student_cfg)
    log.info(f"done in {elapsed(start)}")
