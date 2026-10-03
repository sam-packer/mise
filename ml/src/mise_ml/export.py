"""Export the student to ONNX and assemble its catalog, vectors, and metadata for the browser."""

import json
import logging
import re
import shutil
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import PreTrainedTokenizerBase

from mise_ml.config import (
    BUNDLE,
    IMG,
    ML_ROOT,
    OUT,
    REPO_ROOT,
    SEED,
    TUNING_SET,
    VOCAB_PATH,
    ExportConfig,
    StudentConfig,
    TeacherConfig,
)
from mise_ml.data import Catalog, load_catalog, load_queries
from mise_ml.inference import encode_batches
from mise_ml.log import elapsed, get, num, progress
from mise_ml.rated import RatedSet, brief, load_rated, rated_scores
from mise_ml.student import OUTPUT_NAMES, STUDENT_DIR, Student, load_student
from mise_ml.teacher import OUTPUTS as TEACHER_OUTPUTS
from mise_ml.teacher import drop_near_eval, query_store
from mise_ml.util import make_deterministic, sha256_file, write_json
from mise_ml.vocab import Vocab, load_vocab

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

    The dynamo exporter can assign "embedding" to both an internal Gather value and a graph output.
    ONNX requires unique names.
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


def tuning_measurement(
    path: Path, tokenizer, items: np.ndarray, tuning: RatedSet, cfg: StudentConfig
) -> dict[str, float]:
    """Tuning-set scores of a graph against the fixed item vectors."""
    session = onnx_session(path, threads=4)
    # Browser queries run alone; dynamic int8 ranges depend on the other rows in a batch.
    queries = np.concatenate(
        [onnx_run(session, tokenizer, [f], cfg.max_length)["embedding"] for f in tuning.feelings]
    )
    return rated_scores(tuning, queries, items)


def select_quantization(
    fp32: Path,
    target: Path,
    tokenizer,
    catalog: Catalog,
    items: np.ndarray,
    cfg: StudentConfig,
    export_cfg: ExportConfig,
) -> dict:
    """Pick the int8 recipe by the tuning-set objective and return the measurements. Only the
    feeling graph varies; the item vectors are fixed."""
    tuning = load_rated(TUNING_SET, catalog)
    reference = tuning_measurement(fp32, tokenizer, items, tuning, cfg)
    log.info("fp32 tuning %s", brief(reference))
    measurements = {"fp32": {"tuning": reference, "bytes": fp32.stat().st_size}}
    best = None
    scores: dict[str, dict[str, float]] = {}
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
            scores[digest] = tuning_measurement(path, tokenizer, items, tuning, cfg)
        else:
            log.info("%s produces an identical graph; reuse its measured scores", name)
        tuned = scores[digest]
        row["tuning"] = tuned
        log.info(
            "%s: %.2f MiB, tuning %s, fp32 gap %.2f points",
            name,
            size / 2**20,
            brief(tuned),
            (reference["objective"] - tuned["objective"]) * 100,
        )
        rank = (tuned["objective"], -size)
        if best is None or rank > best[0]:
            best = (rank, name, path)
    if best is None:
        raise RuntimeError("no quantization recipe fits the model size cap")
    shutil.copyfile(best[2], target)
    measurements["selected"] = best[1]
    write_json(target.with_suffix(".json"), measurements)
    log.info("selected %s by the tuning-set objective", best[1])
    return measurements


def wall_wins(
    queries: torch.Tensor, items: torch.Tensor, rows: list[torch.Tensor], penalty: torch.Tensor
) -> np.ndarray:
    """Count how often each work is in its category's top 2 (the app's PICKS_EACH) for the queries.

    The score is query . item - penalty. The count leaves out the app's one-per-creator rule, as the
    rated test did.
    """
    wins = torch.zeros(len(items), device=items.device)
    for s in range(0, len(queries), 4096):
        scores = queries[s : s + 4096] @ items.T - penalty
        for r in rows:
            top = r[scores[:, r].topk(2, dim=1).indices].flatten()
            wins.index_add_(0, top, torch.ones_like(top, dtype=torch.float32))
    return wins.cpu().numpy()


def log_exposure(label: str, wins: np.ndarray, rows: list[np.ndarray]) -> None:
    top = np.mean([np.sort(wins[r])[::-1][: len(r) // 100].sum() / wins[r].sum() for r in rows])
    log.info(
        f"exposure {label}: the top 1% of works take {top:.1%} of the top-2 places, "
        f"max {num(int(wins.max()))}"
    )


def exposure_penalty(
    model_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    catalog: Catalog,
    vectors: np.ndarray,
    student_cfg: StudentConfig,
    cfg: ExportConfig,
) -> np.ndarray:
    """Fit a per-work penalty that lowers the score of works that win too many feeling walls.

    The queries are the training texts of teacher_outputs.pt. Each round counts the top-2 wins of
    each work, then moves the penalty halfway to beta * log(share). The share is the work's wins
    over its category's mean wins, each plus one. A work at or below the mean gets no penalty.
    """
    texts = torch.load(TEACHER_OUTPUTS, weights_only=False)["texts"]
    session = onnx_session(model_path)
    # Browser queries run alone; dynamic int8 ranges depend on the other rows in a batch.
    queries = np.concatenate(
        [
            onnx_run(session, tokenizer, [text], student_cfg.max_length)["embedding"]
            for text in progress(texts, desc="exposure queries", unit="text")
        ]
    )
    q = torch.tensor(queries, dtype=torch.float32, device="cuda")
    v = torch.tensor(vectors, dtype=torch.float32, device="cuda")
    category = np.array([it["category"] for it in catalog.items])
    rows = [np.flatnonzero(category == c) for c in sorted(set(category))]
    rows_cuda = [torch.tensor(r, device="cuda") for r in rows]

    def wins(penalty: np.ndarray) -> np.ndarray:
        return wall_wins(q, v, rows_cuda, torch.tensor(penalty, device="cuda"))

    penalty = np.zeros(len(vectors), dtype=np.float32)
    log_exposure(f"on {num(len(texts))} training texts, before", wins(penalty), rows)
    for _ in range(cfg.exposure_rounds):
        counts = wins(penalty)
        step = np.zeros_like(penalty)
        for r in rows:
            share = (counts[r] + 1) / (counts[r].sum() / len(r) + 1)
            step[r] = cfg.exposure_beta * np.log(np.maximum(share, 1.0))
        penalty = 0.5 * penalty + 0.5 * step
    # penalty.bin holds fp16 values; measure exactly those.
    penalty = penalty.astype(np.float16).astype(np.float32)
    log_exposure(
        f"after (beta {cfg.exposure_beta}, max penalty {penalty.max():.3f})", wins(penalty), rows
    )
    return penalty


def bundle_item(item: dict[str, Any]) -> dict[str, Any]:
    out = {k: item[k] for k in ITEM_FIELDS if item.get(k) is not None or k in ("year", "image")}
    out["links"] = {k: v for k, v in item["links"].items() if v}
    if out["image"]:
        out["image"] = {k: out["image"][k] for k in ("src", "w", "h", "tone")}
    return out


def calibrate_heads(
    session: ort.InferenceSession,
    tokenizer: PreTrainedTokenizerBase,
    catalog: Catalog,
    vocab: Vocab,
    cfg: StudentConfig,
) -> dict[str, dict[str, Any]]:
    """Fit training-label priors and choose each correction on validation feelings, without the
    texts that train-teacher drops for being near an eval feeling."""
    teacher_cfg = TeacherConfig()
    qs = load_queries(catalog, vocab, teacher_cfg)
    fq = torch.tensor(query_store(teacher_cfg).get(qs.texts), dtype=torch.float32, device="cuda")
    drop_near_eval(qs, fq, teacher_cfg.near_eval_cosine)
    del fq
    names = ("light", "typeface")
    train = qs.where("train")
    val = qs.where("val")
    val = val[np.any([getattr(qs, name)[val] >= 0 for name in names], axis=0)]
    if not len(val):
        raise ValueError("no labeled validation feelings for room correction")
    # Match browser inference: dynamic int8 ranges depend on the other rows in a batch.
    outputs = [
        onnx_run(session, tokenizer, [qs.texts[i]], cfg.max_length)
        for i in progress(val, desc="room correction", unit="feeling")
    ]
    corrections = {}
    for name, size in zip(names, vocab.sizes(), strict=True):
        labels = getattr(qs, name)
        training = labels[train]
        training = training[training >= 0]
        labeled = labels[val] >= 0
        if not len(training) or not labeled.any():
            raise ValueError(f"no training or validation labels for {name}")
        # Add one count per class so an unused label has a finite log prior.
        counts = np.bincount(training, minlength=size).astype(np.float64) + 1
        prior = counts / counts.sum()
        logits = np.concatenate([out[name] for out in outputs])[labeled].astype(np.float64)
        target = labels[val][labeled]
        accuracy = {
            tau: float(((logits - tau * np.log(prior)).argmax(axis=1) == target).mean())
            for tau in (0, 0.25, 0.5, 0.75)
        }
        tau = max(tau for tau, score in accuracy.items() if score >= accuracy[0] - 0.02)
        corrections[name] = {"prior": prior.tolist(), "tau": tau}
        log.info(
            "%s correction: tau %.2f, val choice accuracy %.4f -> %.4f",
            name,
            tau,
            accuracy[0],
            accuracy[tau],
        )
    return corrections


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
    """Return sorted vocabulary and item-name words that encode as one token."""
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


def bun() -> str:
    path = shutil.which("bun")
    if not path:
        raise SystemExit("install Bun before exporting the browser bundle")
    return path


def write_name_data(bundle: Path, vectors: np.ndarray, words: list[str]) -> None:
    """Choose representatives with the shared search code before fp16 conversion."""
    with TemporaryDirectory(prefix="mise-names-") as directory:
        tmp = Path(directory)
        (tmp / "vectors.bin").write_bytes(vectors.astype("<f4").tobytes())
        write_json(tmp / "words.json", words)
        subprocess.run(
            [
                bun(),
                "--no-env-file",
                str(REPO_ROOT / "scripts" / "build-name-data.ts"),
                str(bundle / "items.json"),
                str(tmp / "vectors.bin"),
                str(tmp / "words.json"),
                str(bundle / "search-index.json"),
            ],
            cwd=REPO_ROOT,
            check=True,
        )


def write_samples(bundle: Path) -> None:
    """Run the app's sample feelings through the finished bundle with the browser engine."""
    subprocess.run(
        [bun(), "--no-env-file", str(REPO_ROOT / "scripts" / "build-samples.ts"), str(bundle)],
        cwd=REPO_ROOT,
        check=True,
    )


def write_bundle(
    model_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    encoder_dir: Path,
    catalog: Catalog,
    student_cfg: StudentConfig,
    vectors: np.ndarray,
    penalty: np.ndarray,
    bundle: Path = BUNDLE,
    images: bool = True,
) -> None:
    """Write the public bundle. `vectors` are the catalog's fixed item vectors from training, and
    `penalty` the exposure penalty of each item.

    Without `images`, the bundle has no img folder; it is then only for scoring.
    """
    if bundle.exists():
        shutil.rmtree(bundle)
    model_dir = bundle / "model"
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
    (bundle / "vectors.bin").write_bytes(vectors.astype("<f2").tobytes())
    (bundle / "penalty.bin").write_bytes(penalty.astype("<f2").tobytes())
    (bundle / "items.json").write_text(
        json.dumps(items, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n"
    )
    write_name_data(bundle, vectors, common_words(tokenizer, items))
    shutil.copy2(VOCAB_PATH, bundle / "vocab.json")

    if images:
        img_dir = bundle / "img"
        img_dir.mkdir()
        for name in progress(names, desc="images", unit="file"):
            shutil.copy2(IMG / name, img_dir / name)

    manifest = {
        "version": "ml-"
        + sha256_file(model_dir / "model.onnx")[:8]
        + sha256_file(bundle / "vectors.bin")[:8],
        "encoder": {
            "model": "model/model.onnx",
            "tokenizer": "model/",
            "dims": int(vectors.shape[1]),
            "maxTokens": student_cfg.max_length,
            "pooling": "none",
            "normalize": True,
            "outputs": {name: name for name in OUTPUT_NAMES},
        },
        "files": {
            "items": {"path": "items.json", "format": "json"},
            "vectors": {"path": "vectors.bin", "format": "fp16-le"},
            "penalty": {"path": "penalty.bin", "format": "fp16-le"},
            "names": {"path": "search-index.json", "format": "json"},
            "vocab": {"path": "vocab.json", "format": "json"},
        },
        "heads": {
            "kind": "onnx",
            "corrections": calibrate_heads(session, tokenizer, catalog, load_vocab(), student_cfg),
        },
        "counts": {"items": len(items)},
    }
    write_json(bundle / "manifest.json", manifest)
    write_samples(bundle)
    counts = Counter(it["category"] for it in items)
    size = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    log.info(
        f"bundle {manifest['version']}: {num(len(items))} items "
        f"({', '.join(f'{k} {num(v)}' for k, v in sorted(counts.items()))}), "
        f"{num(len(names) if images else 0)} images, {size / 2**20:.0f} MiB -> "
        f"{bundle.relative_to(ML_ROOT).as_posix()}"
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
    }
    for name, shape in expected.items():
        if out[name].shape != shape:
            log.error(f"output {name} has shape {out[name].shape}, expected {shape}")
            raise SystemExit(1)
    log.info("output shapes ok: " + ", ".join(f"{k} {v}" for k, v in expected.items()))


def run(
    student_dir: Path = STUDENT_DIR,
    bundle: Path = BUNDLE,
    work: Path = OUT / "onnx",
    cfg: ExportConfig | None = None,
    images: bool = True,
) -> None:
    start = time.perf_counter()
    make_deterministic(SEED)
    cfg = cfg or ExportConfig()
    vocab = load_vocab()
    catalog = load_catalog()
    model, tokenizer, meta = load_student(student_dir)
    if meta["vocab"] != vocab.digest:
        raise SystemExit("vocab changed since train-student; retrain")
    if meta["item_ids"] != [it["id"] for it in catalog.items]:
        raise SystemExit("catalog changed since train-student; retrain")
    log.info(
        f"reading {student_dir.relative_to(ML_ROOT).as_posix()} "
        f"(tuning objective {meta['tuning']['objective']:.4f}) and "
        f"{num(len(catalog.items))} items; opset {cfg.opset}, "
        f"limit {cfg.max_model_bytes / 2**20:.0f} MiB"
    )

    fp32, int8 = work / "student.fp32.onnx", work / "student.int8.onnx"
    log.info("exporting the ONNX graph (fp32)")
    export_onnx(model, fp32, cfg.opset)
    log.info("dynamic int8 quantization")
    student_cfg = StudentConfig(**meta["cfg"])
    # vectors.bin holds fp16 values; selection and the name data use exactly those values.
    vectors = np.load(student_dir / "items.npy").astype(np.float16).astype(np.float32)
    select_quantization(fp32, int8, tokenizer, catalog, vectors, student_cfg, cfg)
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

    penalty = exposure_penalty(int8, tokenizer, catalog, vectors, student_cfg, cfg)
    write_bundle(
        int8,
        tokenizer,
        student_dir / "encoder",
        catalog,
        student_cfg,
        vectors,
        penalty,
        bundle,
        images,
    )
    log.info(f"done in {elapsed(start)}")
