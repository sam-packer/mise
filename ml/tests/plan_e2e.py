"""Exercise the CLI against a disposable pipeline with recorded LLM answers.

Run with `uv run python tests/plan_e2e.py`. No model, service, or real data is used.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ML = Path(__file__).resolve().parents[1]


def seed(root: Path) -> None:
    sys.path.insert(0, str(root / "ml" / "src"))
    from mise_ml import config as c
    from mise_ml.evaluate import SEP, judge_job
    from mise_ml.llm import field_digests
    from mise_ml.provenance import write_stamp
    from mise_ml.steps import JUDGE_POOL, command_steps, judge_pool_state
    from mise_ml.util import write_json, write_jsonl

    write_json(
        c.VOCAB_PATH,
        {
            "lights": ["dawn"],
            "typefaces": [{"id": "face", "family": "Family"}],
            "scents": [{"id": "scent", "text": "rain"}],
        },
    )
    feeling = "i listen to the rain before dawn"
    write_jsonl(c.EVAL_FEELINGS, [{"text": feeling}])
    items = [
        {
            "id": f"{cat}:one",
            "category": cat,
            "title": "One",
            "creator": "Person",
            "year": 2000,
            "album": "First",
            "signal": {"tags": ["quiet"]},
            "image": None,
            "links": {"primary": "https://example.test/"},
        }
        for cat in ("song", "poem")
    ]
    items[1]["text"] = "A quiet poem."
    write_jsonl(c.RESOLVED, items)
    write_jsonl(c.PAT, [{"phrase": "rain", "rgb": [[1, 2, 3]] * 5}])
    (c.REPO_ROOT / "scripts" / "build-name-data.ts").write_text("// fixture builder\n")

    def record(spec, answer):
        keys = sorted(set(spec.keys))
        units = spec.build(keys)
        probe = spec.build(keys[:1])[0].request
        entries, rows = [], []
        for unit in units:
            data = answer(unit.keys)
            entries.append(
                {
                    "keys": unit.keys,
                    "sig": spec.job.signature(probe),
                    "data": data,
                    "prints": {k: spec.fingerprint(k) for k in unit.keys},
                    "signature": spec.job.signature_parts(probe),
                    "fields": {k: field_digests(spec.fields(k)) for k in unit.keys},
                }
            )
            rows.extend(spec.parse(unit.keys, data))
        write_jsonl(spec.job.cache, entries)
        rows.sort(key=lambda r: r["key"])
        write_jsonl(spec.job.output, spec.project(rows) if spec.project else rows)

    def answer(name, keys):
        if name == "items":
            return {
                "vibe": "quiet water",
                "description": "A quiet place.",
                "q1": f"i watch the quiet {keys[0].split(':')[0]} before sunrise",
                "q2": "i sit by the window after midnight",
                "q3": "the warm light settles on my shoulders",
            }
        if name == "moods":
            return {"feelings": ["i rest beneath the trees after work"]}
        if name == "distill":
            return {"feelings": [{"n": 1, "text": "sleepy after lunch"}]}
        if name == "pat":
            return {
                "sentences": [
                    {"n": i + 1, "text": "i walk through the rain after work"}
                    for i, _ in enumerate(keys)
                ]
            }
        if name == "judge":
            return {"fits": [{"n": i + 1, "fit": True} for i, _ in enumerate(keys)]}
        return {
            "labels": [
                {
                    "n": i + 1,
                    **{f"c{j}": "#112233" for j in range(1, 6)},
                    "light": "dawn",
                    "typeface": "face",
                    "scent": "scent",
                }
                for i, _ in enumerate(keys)
            ]
        }

    for step in command_steps("label"):
        record(step.call(), lambda keys, name=step.name: answer(name, keys))
    # An obsolete request is a complete cache record; prune must remove just its bytes.
    item_cache = c.DATA / "llm" / "items.jsonl"
    first = json.loads(item_cache.read_text().splitlines()[0])
    first["keys"] = ["song:retired"]
    first["prints"] = {"song:retired": "obsolete"}
    with item_cache.open("ab") as stream:
        stream.write((json.dumps(first) + "\n").encode())
    # Add a superseded batch. All keys in the newer batch must remain usable after prune.
    label_cache = next((c.DATA / "llm").glob("labels-*.jsonl"))
    label_cache.write_bytes(label_cache.read_bytes() * 2)
    for step in command_steps("train"):
        if step.name == "judge":
            continue
        for path in step.outputs():
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture output\n")
    write_json(c.EVAL_REPORT, {"ship": {"ok": True, "reasons": []}})
    keys = [SEP.join((feeling, it["category"], it["id"])) for it in items]
    record(judge_job(keys), lambda keys: answer("judge", keys))
    write_json(JUDGE_POOL, {"state": judge_pool_state(), "keys": keys})
    for step in command_steps("train"):
        write_stamp(step.name, step.state())


def run(root: Path, name: str, flag: str = "--plan") -> str:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    code = (
        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
        "from mise_ml.__main__ import command; command(sys.argv.pop(1))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(root / "ml" / "src"), name, flag],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert not result.stderr, result.stderr
    return result.stdout


def tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def check() -> None:
    with tempfile.TemporaryDirectory(prefix="mise-plan-e2e-") as directory:
        root = Path(directory)
        shutil.copytree(
            ML / "src", root / "ml" / "src", ignore=shutil.ignore_patterns("__pycache__")
        )
        config = root / "ml" / "src" / "mise_ml" / "config.py"
        source = config.read_text()
        for old, new in (
            ("synthetic_moods: int = 6000", "synthetic_moods: int = 1"),
            ("moods_per_request: int = 25", "moods_per_request: int = 1"),
            ("distill_feelings: int = 40000", "distill_feelings: int = 1"),
            ("distill_per_request: int = 20", "distill_per_request: int = 1"),
        ):
            source = source.replace(old, new)
        config.write_text(source, encoding="utf-8")
        subprocess.run([sys.executable, "-B", __file__, "seed", str(root)], check=True)
        before = tree(root)
        baseline = run(root, "train")
        assert "will run" not in baseline, baseline
        label = run(root, "label")
        assert "will run" not in label, label
        assert "records no longer used: 1" in label, label
        assert tree(root) == before, "--plan wrote files"

        config.write_text(source.replace("epochs: int = 12", "epochs: int = 13"))
        changed = run(root, "train")
        assert "train-teacher  | up to date" in changed, changed
        assert "config.epochs changed: 12 -> 13" in changed, changed
        assert "upstream train-student will run" in changed, changed
        config.write_text(source)

        feelings = root / "ml" / "eval_feelings.jsonl"
        original = feelings.read_bytes()
        with feelings.open("ab") as stream:
            stream.write(b'{"text":"i wait for the first snow after sunset"}\n')
        changed = run(root, "label")
        assert "new keys: 1;" in changed, changed
        assert "new keys: at least 2 for 1 unranked feelings" in changed, changed
        assert "eval_feelings.jsonl#text changed" in run(root, "train")
        feelings.write_bytes(original)

        resolved = root / "ml" / "data" / "curated" / "resolved.jsonl"
        original = resolved.read_bytes()
        resolved.write_bytes(original.replace(b'"album": "First"', b'"album": "Second"', 1))
        changed = run(root, "label")
        assert "fingerprint changed: album: 1;" in changed, changed
        resolved.write_bytes(original)

        before = tree(root)
        report = run(root, "label", "--prune")
        assert "pruned 2 unused request records" in report, report
        after = tree(root)
        modified = {p for p in before if before[p] != after[p]}
        assert len(modified) == 2 and all(p.startswith("ml/data/llm/") for p in modified), modified
        replay = run(root, "label")
        assert "will run" not in replay, replay
        assert "records no longer used: 1" not in replay, replay
        moods = root / "ml" / "data" / "curated" / "moods.jsonl"
        mood_bytes = moods.read_bytes()
        moods.unlink()
        before = tree(root)
        try:
            run(root, "label", "--prune")
        except subprocess.CalledProcessError as exc:
            assert "prune stopped: incomplete cache inspection" in exc.stderr, exc.stderr
        else:
            raise AssertionError("prune must refuse an incomplete current key set")
        assert tree(root) == before, "prune changed caches with missing prerequisites"
        moods.write_bytes(mood_bytes)
        item_cache = root / "ml" / "data" / "llm" / "items.jsonl"
        with item_cache.open("ab") as stream:
            stream.write(b'{"sig":')
        before = tree(root)
        replay = run(root, "label")
        assert "incomplete final cache line" in replay and "will run" not in replay, replay
        assert tree(root) == before, "plan changed an interrupted append"
        run(root, "label", "--prune")
        assert tree(root) == before, "prune removed bytes outside unused request records"
        print(
            "PASS: read-only plans; config and input changes; field reasons; exact prune and replay"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        seed(Path(sys.argv[2]))
    else:
        check()
