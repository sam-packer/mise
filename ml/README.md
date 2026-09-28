# mise-ml

This package builds the real mood bundle for mise. It downloads public datasets, curates a
catalog, finds images and links, labels everything with a local LLM, trains a teacher, distills
it into a small student, and writes the bundle to `out/bundle/` in the format of spec §4.

The pipeline uses three models. All three run on your GPU and are pinned to a revision:

| Model | Role | Trained? |
|---|---|---|
| `Qwen/Qwen3.5-9B` | the labeler: writes item profiles, feelings, palettes, and labels | no |
| `Qwen/Qwen3-Embedding-8B` | the teacher backbone: turns each text into a 4,096-value embedding | no, frozen |
| `sentence-transformers/all-MiniLM-L6-v2` | the student: the only model that ships to the browser | yes, fully |

The teacher is the embedding backbone plus small trained heads. Similarity ranking is the job
the embedding model is built for. The student learns to copy the teacher and fits in 24 MiB.

## Reproduce

You need an NVIDIA GPU with 32 GB (the defaults suit an RTX 5090), uv, about 60 GB of free
disk (estimate), and an internet connection. You need no API key and no account.

```
uv sync
uv run mise-ml all
```

`all` runs every step in order. Each step skips work that is already done, so you can stop
`all` at any time and start it again. No step needs an argument.

Before `all` starts, it checks for a CUDA GPU, for `../scripts/stub/vocab.json`, and for
`eval_feelings.jsonl`. It names each missing item and stops before any long work.

At the end, `all` checks the ship rule (see "eval" below):

- If the student passes, `all` runs `install`, which puts the new bundle into
  `../static/bundle/`. The web app then uses it.
- If the student fails, `all` stops. It prints why and the path of `out/eval_report.json`. To
  use the student anyway, run `uv run mise-ml install`.

To run one step, use `uv run mise-ml <step>`. To see the steps in run order, run
`uv run mise-ml` with no step.

## Logs

Each command prints short lines to the console, such as
`12:04:31 resolve  INFO  film: 1,843 resolved`. Each step prints what it reads and how much work
is left, then what it wrote, what it dropped and why, and how long it took. Progress bars show
the long loops. WARN lines name what you should know: dropped items (counted by reason),
retries, and failures. A step that fails ends with an ERROR line.

Every command also writes a full log with DEBUG detail, such as each dropped item and each
checksum, to `out/logs/<time>-<step>.log`. One `all` run writes one file. `all` ends with a
table of the steps and their times.

## Steps
The times are estimates for an RTX 5090, except where the text says "measured".

| Step | Input | Output | Time |
|---|---|---|---|
| `fetch` | `sources.toml` | `data/raw/` | 10–20 min |
| `curate` | `data/raw/` | `data/curated/catalog.jsonl`, `pat.jsonl` | 12 s (measured) |
| `resolve` | `catalog.jsonl` | `data/curated/resolved.jsonl`, `data/img/` | about 1 h (books are the slowest; songs 5–10 min) |
| `profile` | `resolved.jsonl`, `eval_feelings.jsonl`, vocab | `data/curated/profiles.jsonl`, `moods.jsonl`, `pat_sentences.jsonl`, `labels-<vocab>.jsonl` | 4–10 h |
| `train-teacher` | the curated files | `data/models/teacher.pt`, `teacher_outputs.pt` | 30–60 min |
| `train-student` | teacher outputs | `data/models/student/` | 20–40 min |
| `export` | student, curated files, images | `out/bundle/` | 5 min |
| `eval --judge` | teacher, student, bundle | `data/curated/judgments.jsonl` | 1–2 h |
| `eval` | all of the above | `out/eval_report.json`, `out/run.json` | 5 min |
| `install` | `out/bundle/` | `../static/bundle/` | seconds |

### fetch

`fetch` downloads each file in `sources.toml` and checks its size and SHA-256. If a file does not
match, `fetch` stops with a "CHECKSUM MISMATCH" message. That means the source changed upstream.
Review the new file, then update `sources.toml`.

| Source | Use | License |
|---|---|---|
| MovieLens ml-latest, with the Tag Genome | films and their mood tags | research and non-commercial use only |
| goodbooks-10k | books and their shelf tags | CC BY-SA 4.0 |
| MuSe, Zenodo release | songs with valence, arousal, dominance, and a Spotify ID | CC BY 4.0 |
| Spotify tracks dataset (Hugging Face) | a popularity value to rank the MuSe songs | BSD |
| Met Open Access CSV | public-domain artworks | CC0 |
| PoetryDB | public-domain poems | poems are public domain |
| Text2Colors PAT (mirror) | 10,183 name-to-palette pairs | see below |
| ArtEmis (optional, manual) | extra feeling sentences | research use only |

- PoetryDB is an API, not a file. `fetch` reads every author in sorted order and writes
  `data/raw/poetrydb/poems.jsonl` sorted by author and title. PoetryDB answers HTTP 503 for a
  few large authors, such as Byron. For those, `fetch` asks for the titles and gets each poem of
  8 to 40 lines on its own. The file has no fixed checksum, so `run.json` records its hash.
- The Text2Colors repository never held the PAT data, and its Google Drive file is gone. The
  pipeline uses a mirror of `data/hexcolor_vf/`. The original code is MIT. The mirror has no
  license. When you use PAT, cite Bahng et al., "Coloring with Words: Guiding Image
  Colorization Through Text-based Palette Generation", ECCV 2018.
- The Kaggle release of MuSe has more columns (tags, genre), but it needs a login. It is
  optional. If you put `muse_v3.csv` in `data/raw/muse/`, `curate` uses it instead of the
  Zenodo file.
- ArtEmis is gated and optional. If you put `artemis_dataset_release_v0.csv` in
  `data/raw/artemis/`, `profile` uses up to 3,000 of its utterances as extra feelings.

### curate (§9.1)

`curate` reads the raw files and writes one catalog. It needs no network. The measured result
on 2026-09-27 was 2,000 films, 2,000 books, 3,000 songs, 1,993 poems, and 6,000 art candidates.

- **film:** MovieLens movies with at least 2,000 ratings, the top 2,000 by rating count. The
  mood signal is the top 12 Tag Genome tags with relevance 0.5 or more.
- **book:** the top 2,000 goodbooks-10k titles by ratings count. The signal is the shelf tags,
  without shelving noise such as "to-read" or "owned".
- **song:** MuSe tracks with a Spotify ID. The Zenodo release has no popularity column, and only
  about 2,000 of its tracks match a track in the Spotify tracks dataset. So `curate` ranks by
  artist popularity (the highest popularity of any track by that artist), then by track
  popularity, and keeps at most 5 tracks per artist. It keeps the top 4,200 as candidates (1.4 times
  3,000), because some songs are not on Deezer. With the Kaggle file, it ranks by Last.fm
  listeners or tag count instead.
- **poem:** PoetryDB poems with 8 to 40 lines. The excerpt is the first 14 lines.
- **art:** Met public-domain Paintings, Drawings, Prints, and Photographs, highlights first. It
  keeps three candidates for each slot, because some objects have no image.
- **PAT:** each palette name with its five RGB colors, in `pat.jsonl`.

### resolve

`resolve` adds an image and links to each catalog item. It caches every API answer in
`data/cache/http/`, waits between calls to each host, and appends each result as it goes. It
drops items without an image, except poems.

- **film:** the IMDb suggestion endpoint gives the poster for the IMDb ID from MovieLens.
  Wikidata (IMDb ID P345, director P57) gives the director, 200 films for each query. The
  primary link is the IMDb title page.
- **song:** the Deezer search API (no key) finds the track by artist and title. A match needs
  the same artist and a title that starts with the catalog title, ignoring "(feat. ...)" parts.
  `resolve` prefers the original album over a live, remix, or compilation album, and it rejects
  another version of the song (remix, live, karaoke, and so on) unless the catalog title names
  that version too. Deezer gives the album cover, the album name, and the track link. The
  Spotify link comes from the MuSe Spotify ID. The Apple Music and YouTube links are search
  URLs for the artist and title. The preview is the path `/api/preview/deezer/<id>`, which the
  web app's Worker serves, because Deezer preview URLs expire after about 15 minutes. Songs
  run with 4 parallel requests; the 0.12 s gap for each Deezer host keeps them under Deezer's
  limit of about 50 requests in 5 s.
- `resolve` keeps exactly 3,000 songs: the best-ranked candidates that Deezer has. It tries the
  candidates in rank order, in chunks, and stops when no untried candidate can rank above the
  3,000th kept song. Then it removes any extra lower-ranked songs. So 1 or 4 workers, and a run
  that resumes, keep the same 3,000. It logs `song: 3,000 kept of 4,200 candidates (N dropped:
  reasons)`.
- Songs that an earlier version of `resolve` found on iTunes have no Deezer id. `resolve`
  removes them and their images at the start, so they resolve again from Deezer.
- **book:** Open Library gives the cover and the work page.
- **art:** the Met Collection API gives the image and the museum page. `resolve` checks
  `isPublicDomain` for each object and keeps 2,000: 40% paintings and 20% of each other class.
- **poem:** no media. The primary link is a Poetry Foundation search.

Each image becomes a WebP file with 800 px on the long edge. `resolve` records `w`, `h`, and the
average OKLab `tone`. To resolve some categories only, name them: `uv run mise-ml resolve song`.

### profile (§9.2)

`profile` runs `Qwen/Qwen3.5-9B` (Apache-2.0, pinned revision) in-process with transformers, in
bf16, with thinking mode off. It runs four jobs:

1. **items:** a vibe line, a mood description, and three example feelings for each item. Art
   items also send the image.
2. **moods:** 6,000 synthetic feelings, 25 for each prompt, from random scene hints.
3. **pat:** one feeling sentence for each PAT palette name. The PAT palette stays the label.
4. **labels:** five colors, a light, a typeface, and a scent for about 30,000 feelings: your eval
   feelings, the synthetic moods, the ArtEmis sentences, and item feelings to fill the rest.

Every prompt asks for feelings as a sentence about a scene or a moment, 6 to 30 words. It never
asks for a list of mood words.

**Valid JSON.** `profile` uses constrained decoding with xgrammar. xgrammar compiles each JSON
schema, including the enums of vocab ids and a `#rrggbb` pattern for colors, into a grammar. A
logits processor then blocks every token that would break the schema. jsonschema checks each
answer again. If an answer is cut off at the token limit, `profile` retries it once alone with
twice the limit, and then logs and skips it. The xgrammar CUDA kernel needs Triton, which is
not available on Windows, so the processor applies the token mask with plain PyTorch.

**Resume.** Each job keeps its answers in `data/llm/<job>.jsonl`. A line holds a signature
(model, revision, system prompt, schema, token limit) and a hash of each key's own prompt. A
rerun generates only the keys without a matching line. A new model, revision, or prompt redoes
exactly the keys it affects.

**Repeatability.** Decoding is greedy, the batch size is fixed, and the prompts run in a fixed
sorted order with left padding. bf16 batching makes the results repeatable on the same GPU,
driver, and torch version. The results are not bit-identical on other hardware. A run that
resumes after an interruption can put different prompts in one batch, so a few answers can
differ from an uninterrupted run.

### train-teacher (§9.3)

- The backbone is `Qwen/Qwen3-Embedding-8B`, frozen, in bf16 (about 15 GB of weights), at a
  pinned revision. It loads with `AutoModel`. The encoding follows the model card:
  - A query (a feeling) gets a one-sentence instruction:
    `Instruct: Given a feeling someone describes as a scene or a moment, retrieve films, books,
    songs, poems, and artworks that share its mood\nQuery:<feeling>`.
  - An item text gets no instruction.
  - The tokenizer adds `<|endoftext|>` at the end and pads on the left. The feature is the
    hidden state of that last token (4,096 values), L2-normalized.
- `data/features/` caches the features by text. The file names hold the model id and revision,
  so features from another backbone are never reused. A second run encodes only new texts.
- The feature pass encodes about 60,000–75,000 texts, in batches of 64, at most 256 tokens
  each. Estimate: 15–40 min on an RTX 5090. The head training after it takes 5–15 min.
- The retrieval heads project queries and items to 384 dims. The loss is InfoNCE with in-batch
  negatives. After the first 3 epochs, each query also gets 8 hard negatives from the top 50
  items of its own category. The step mines them again at the start of each epoch.
- The palette head predicts 5 OKLab colors. The loss is the slot-wise squared OKLab distance plus
  a term on the sorted lightness values. The data is the LLM palettes and the PAT palettes.
- The light, typeface, and scent heads use cross-entropy on the LLM labels.
- The step keeps the checkpoint with the best validation loss. Then it writes the teacher outputs
  for distillation.

### train-student (§9.4)

- The backbone is `sentence-transformers/all-MiniLM-L6-v2`, fully fine-tuned, with mean pooling,
  at a pinned revision.
- The losses are: KL divergence between the teacher and student query-to-item similarity
  distributions (temperature 0.05), InfoNCE on (item feeling, item) pairs, MSE to the teacher
  palette, and KL to the teacher choice logits.
- Each step encodes, with gradients, the teacher's top 16 items for each query, 128 random
  items, and the positives. The step keeps the checkpoint with the best validation recall@10.

### export

- One ONNX graph with the inputs `input_ids`, `attention_mask`, and `token_type_ids`, and the
  outputs `embedding` [N, 384], `palette` [N, 5, 3], `light`, `typeface`, and `scent`.
- Dynamic int8 quantization. The step stops if the model is larger than 24 MiB. The int8 MiniLM
  graph is about 23 MiB.
- The bundle has `manifest.json` (`heads.kind = "onnx"`, `pooling = "none"`, `maxTokens = 96`),
  `model/` (the model and the tokenizer files), `items.json`, `vectors.bin`, `vocab.json`, and
  `img/`. The int8 graph computes the item vectors, so items and queries use one code path.
  Songs carry `album`. The manifest version is a hash of the model and the vectors.

### install

`install` copies `out/bundle/` to a new folder next to `../static/bundle/`. When the copy is
complete, it swaps the new folder in and deletes the old one. So a failed copy never leaves half
a bundle. It prints the manifest version and the item count of each category.

### eval (§9.5)

`eval` reports, for the teacher and for the shipped student (the int8 ONNX graph and
`vectors.bin`):

- **recall@10 on held-out item feelings.** 10% of the item feelings never go into training. A
  hit is the source item in the top 10 of its category.
- **recall@10 on your eval feelings.** `eval --judge` pools the top 20 items for each category
  from the teacher, the student, and the untrained MiniLM. The local LLM marks each pooled item
  as a fit or not. The score is the fits in the top 10 divided by the smaller of the fit count
  and 10.
- **palette ΔE:** the mean Euclidean OKLab distance for each color against the LLM palettes.
- **choice accuracy:** top-1 accuracy of light, typeface, and scent against the LLM labels.
- **student latency:** onnxruntime on the CPU with one thread, batch size 1.

The ship rule uses the judged recall@10 when it exists, and the held-out recall@10 otherwise.
The student ships only if it is within 5 points of the teacher.

## Eval feelings

`eval_feelings.jsonl` (in this folder, committed) holds the human-written eval set. Write one
JSON object per line with one field, `text`:

```json
{"text": "a snowy december and i just made warm hot chocolate"}
```

- Write about 300 lines. The file has 10 examples to start from.
- Write each feeling as a sentence about a scene or a moment, 6 to 30 words. Do not write lists
  of mood words.
- Write the file before `profile`, because `profile` labels these texts. If you add lines
  later, run `profile` again. It labels only the new lines.
- The eval feelings never go into training.

## Reproducibility

- Nothing under `data/` or `out/` is committed. `all` rebuilds everything from `sources.toml`.
- Every step writes sorted JSONL with sorted keys, so two runs give the same files.
- `curate` gave the same `catalog.jsonl` hash on two runs (measured).
- Training uses fixed seeds (1337), `torch.use_deterministic_algorithms(True)`, deterministic
  cuDNN, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. A small check ran every training op (MiniLM
  forward and backward with SDPA attention, the teacher heads, hard-negative mining, the losses)
  twice in strict mode. No op raised an error, and the gradient sums matched exactly.
- The teacher feature pass runs in strict mode too. Qwen3-Embedding-8B is a standard
  transformer with SDPA attention and no linear-attention layers, so the Triton kernel caveat
  of Qwen3.5 does not apply. A strict-mode forward of a small random Qwen3 model, with left
  padding and last-token pooling, ran twice and gave identical results.
- The Qwen3.5 generate path (`profile`, `eval --judge`) uses warn-only mode, because nobody has
  run it in strict mode yet. Warn-only mode keeps the deterministic kernels and prints a warning
  for an op that has none.
- The Hugging Face models are pinned to a revision in `src/mise_ml/config.py`.
- `out/run.json` records the seed, the Python and package versions, the GPU, the model ids and
  revisions, all configs, the SHA-256 of each source and each curated file, the PoetryDB content
  hash, the bundle files, and the ship decision.
- `all` writes a stamp in `data/models/` after each training step. The stamp holds a hash of the
  step's inputs and config, so `all` skips the step when nothing changed.

## Configuration

The defaults are dataclasses in `src/mise_ml/config.py`: `CurateConfig`, `ResolveConfig`,
`ProfileConfig`, `TeacherConfig`, `StudentConfig`, and `ExportConfig`. Change a value there to
change a run. If `profile` runs out of GPU memory, lower `ProfileConfig.batch_size`.

The vocab source of truth is `../scripts/stub/vocab.json`. The labels and the heads use its
order. The label file name contains a hash of the vocab, so a new vocab gets new labels.

## Licenses

MovieLens and ArtEmis are for research or non-commercial use only. A class project is allowed.
A public commercial launch is not. The Met Open Access images are CC0. The IMDb posters and the
Deezer album covers and previews belong to their owners. The app shows them next to links to the
source pages.
