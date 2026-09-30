# The mise model pipeline

This folder builds everything the app loads from the CDN: the catalog of about 11,000 works, their
images, and the small model that runs in the browser. It runs on one local NVIDIA GPU.

The pipeline has four commands. Run them from `ml/`, in this order:

```sh
uv sync
uv run download   # fetch the sources, choose the catalog, find images and links
uv run label      # a local LLM describes every work and writes training feelings
uv run train      # train the teacher, distill the student, export, evaluate, install
uv run publish    # upload the installed bundle to R2
```

Add `--help` to any command for its details. Add `--plan` to see what it would do without doing it.

## What you need

- An NVIDIA GPU with 32 GB of memory. The defaults suit an RTX 5090.
- About 60 GB of free disk, [uv](https://docs.astral.sh/uv/), and [Bun](https://bun.sh) (the export
  step runs the app's TypeScript search code).
- Four free API keys in `ml/.env`. Copy [.env.example](.env.example) to `.env` and follow its
  comments. Git ignores `.env`. A variable already set in your shell wins over the file.

| Variable                       | Where to get it                 | What it is for                        |
| ------------------------------ | ------------------------------- | ------------------------------------- |
| `TMDB_TOKEN` or `TMDB_API_KEY` | themoviedb.org > Settings > API | films                                 |
| `HARDCOVER_TOKEN`              | hardcover.app/account/api       | books                                 |
| `LISTENBRAINZ_TOKEN`           | listenbrainz.org/settings       | songs: the top recordings per artist  |
| `LASTFM_API_KEY`               | last.fm/api/account/create      | songs: older artists and mood tags    |

`publish` also needs the R2 variables in `.env.example`.

A full first run takes most of a night. You can stop any command with Ctrl+C. A rerun continues
where it stopped.

## The three models

| Model                                    | Job                                                                          | Trained?   |
| ---------------------------------------- | ---------------------------------------------------------------------------- | ---------- |
| `Qwen/Qwen3.5-9B`                        | the labeler: describes each work and writes feelings, palettes, and labels | no         |
| `Qwen/Qwen3-Embedding-8B`                | the teacher: a large embedding model with small trained heads               | heads only |
| `sentence-transformers/all-MiniLM-L6-v2` | the student: the small model that ships to the browser                      | yes        |

An embedding model turns a text into a list of numbers (a vector). Texts with similar meaning get
vectors that point the same way, so a search is a comparison of vectors. The teacher is too large
for a browser (15 GB), so the student learns to copy the teacher's rankings in 23 MB.

`src/mise_ml/config.py` pins each model to an exact revision.

## download

`download` runs three steps.

1. **fetch** downloads the fixed source files in `sources.toml` (the Met collection and a palette
   data set) and checks each checksum. PoetryDB is an API, so fetch reads it poem by poem.
2. **curate** chooses the catalog: 2,000 films (TMDB), 2,000 books (Hardcover), 3,000 songs
   (ListenBrainz, Last.fm, MusicBrainz), 2,000 artworks (the Met, the Art Institute of Chicago, and
   the Cleveland Museum of Art), and about 2,000 public-domain poems. Films, books, and songs are
   split into eras, and each era gets a fixed share, so the catalog is not only recent hits. Songs
   rank by the number of listeners, not the number of plays.
3. **resolve** finds an image and links for each work. Posters come from TMDB, covers from
   Hardcover or Open Library, album art and previews from Deezer, and museum images from Wikimedia
   Commons or the museum. Each image becomes an 800 px WebP file.

Every API answer goes into `data/cache/http/`, so a second run needs almost no network. After ten
failures in a row, the pipeline stops calling that host for the rest of the run. The next run tries
again.

## label

`label` runs Qwen3.5-9B on the GPU. Constrained decoding (xgrammar) makes every answer valid JSON
for its schema, so the model can't return a broken answer. It runs five jobs:

| Job     | What the LLM writes                                                                     |
| ------- | --------------------------------------------------------------------------------------- |
| items   | a short vibe line, a mood description, and three example feelings for each work        |
| moods   | about 6,000 invented feelings, from random scene hints                                  |
| pat     | a feeling for each named palette in the PAT data set                                    |
| labels  | five colors, a light, a typeface, and a scent for about 30,000 feelings                 |
| distill | about 36,000 more feelings in many styles (short, long, casual, typos), with no labels |

An answer that fails its checks gets one retry. Each job keeps its answers in `data/llm/<job>.jsonl`.
The key of each answer is a hash of its exact prompt. So a rerun only asks for what is new or
changed, and a change to a prompt or the model redoes only the answers it affects.

## train

`train` runs these steps and stops at the first failure.

1. **train-teacher.** The frozen Qwen3-Embedding-8B encodes every text once. Small heads learn
   to rank works for a feeling (InfoNCE: the matching work must score above the other works in the
   batch), and to predict the palette, light, typeface, and scent. 6 epochs at a learning rate of
   3e-4, with warmup and a cosine decay. The run keeps the epoch with the best validation score.
2. **train-student.** MiniLM learns from the teacher. For each feeling it copies the teacher's
   ranking of works, its palette, and its label choices. The distillation feelings from `label`
   have no answers of their own. They only teach the student to rank the way the teacher does, on
   a wider range of writing.
3. **export** writes one ONNX file with the encoder and all heads, then quantizes it to int8 (8-bit
   weights instead of 32-bit). It tries a few quantization recipes and keeps the one with the best
   validation score under 24 MiB. It also writes the item vectors as 16-bit floats and the name
   data for the "in the key of" matcher.
4. **eval-judge** and **eval** score the models (see below).
5. **install** copies the bundle to `../static/bundle/`, but only when the ship gate passes.

`train` never uploads anything.

### How the model is scored

There are two tests.

- **Held-out recall@10.** 10% of the item feelings never go into training. For each one, the test
  asks whether its source work lands in the top 10 of its category.
- **Judged recall@10.** For the 290 feelings in `eval_feelings.jsonl`, the teacher, the student, and
  the untrained MiniLM each return their top 20 works per category. The LLM marks each work in that
  pool as a fit or not. A model's score is the fits in its top 10, divided by the fits it could
  have found (at most 10).

The ship gate needs all of these to pass:

- The student is no more than 5 points below the teacher on both tests.
- The student is at least 10 points above the untrained MiniLM on held-out recall.
- At least 100 eval feelings have complete judgments.

The report gives a 95% confidence interval for each gap. The current model:

| Model                   | Judged recall@10 | Held-out recall@10 |
| ----------------------- | ---------------- | ------------------ |
| teacher                 | 0.717            | 0.329              |
| student (int8, shipped) | 0.673            | 0.307              |
| untrained MiniLM        | 0.505            | 0.131              |

The student trails the teacher by 4.4 points on judged recall (interval 2.8 to 5.9) and by 2.2
points on held-out recall. It runs in about 1 ms per feeling on one CPU thread.

`eval_feelings.jsonl` holds one JSON object per line, like
`{ "text": "a snowy december and i just made warm hot chocolate" }`. These feelings never go into
training. If you add some, run `uv run label` again. It labels only the new lines.

## publish

`publish` uploads `../static/bundle/` to the R2 bucket `mise`, served at `cdn.mise.art`.

- Images go to `img/<hash>.webp`, so an image uploads once and every release shares it.
- All other files go to `bundles/<date>-<hash>/`. The prefix hash covers the file contents, so a
  release never changes after upload. Files are cached for a year.
- The manifest uploads last, so a release is never half there.
- At the end, `publish` writes the new URL into `../src/lib/bundle.ts`.

Then commit `src/lib/bundle.ts` and run `bun run deploy` from the repository root.

The app's web worker loads the bundle from another origin, so the bucket needs a CORS rule.
`publish` adds one if it can. An "Object Read & Write" token usually can't, so `publish` prints the
JSON for you to paste into R2 > `mise` > Settings > CORS policy.

## What reruns, and why

Each step records what it used: a hash of each input file, the config, and its source code. A rerun
skips a step whose record still matches. `src/mise_ml/steps.py` lists every step with its inputs
and outputs.

`--plan` shows what a command would do, and why, without a GPU, a model, or an upload. For example,
after a change of `StudentConfig.epochs` from 12 to 13:

```text
$ uv run train --plan
train plan (no models; no writes)
step           | status     | reasons
train-teacher  | up to date | current inputs and outputs match
train-student  | will run   | config.epochs changed: 12 -> 13
export         | will run   | upstream train-student will run; recheck after its outputs change
...
```

`uv run label --plan` also lists cache records that no current prompt uses. `uv run label --prune`
deletes exactly those. Nothing else deletes cached answers.

## Logs and outputs

The console shows one short line per event, such as `12:04:31 resolve  INFO  film: 1,843 resolved`,
and a table of step times at the end. Each command also writes a full debug log to
`out/logs/<time>-<command>.log`.

| Path                   | What it holds                                                    |
| ---------------------- | ---------------------------------------------------------------- |
| `data/raw/`            | the fixed source files                                           |
| `data/cache/http/`     | every API answer                                                 |
| `data/curated/`        | the catalog, the resolved works, and the LLM outputs             |
| `data/llm/`            | the LLM answer caches                                            |
| `data/models/`         | the teacher, the student, and the step records                   |
| `out/bundle/`          | the exported bundle                                              |
| `out/eval_report.json` | the scores and the ship decision                                 |
| `out/run.json`         | the seeds, package versions, GPU, model revisions, and file hashes |

Git ignores `data/` and `out/`.

## Reproducibility

- Every step writes sorted output, and the API cache makes `curate` repeatable. To take newer
  data, delete the cache folder for that host.
- Training uses fixed seeds and PyTorch's deterministic mode.
- The LLM decodes greedily in a fixed order, so labels repeat on the same GPU and driver. Other
  hardware can round differently.

## Configuration

Every setting is a dataclass in `src/mise_ml/config.py`. Change a value there to change a run. The
list of lights, typefaces, and scents is `../scripts/stub/vocab.json`.

## Licenses

The Met, Art Institute of Chicago, and Cleveland Museum of Art images are public domain or CC0.
ListenBrainz and MusicBrainz data is CC0. The PAT palettes come from Bahng et al., "Coloring with
Words", ECCV 2018. TMDB posters, book covers, and Deezer album art and previews belong to their
owners. The app shows them next to links to their sources and credits each one on its attribution
page.
