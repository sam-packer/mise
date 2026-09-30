# ML pipeline

This package builds the real mood bundle for mise. It selects a current catalog from public APIs
and museum data, finds images and links, labels everything with a local LLM, trains a teacher,
distills it into a small student, and writes the public bundle to `out/bundle/`.

The pipeline uses three models. All three run on your GPU and are pinned to a revision:

| Model                                    | Role                                                               | Trained?   |
| ---------------------------------------- | ------------------------------------------------------------------ | ---------- |
| `Qwen/Qwen3.5-9B`                        | the labeler: writes item profiles, feelings, palettes, and labels  | no         |
| `Qwen/Qwen3-Embedding-8B`                | the teacher backbone: turns each text into a 4,096-value embedding | no, frozen |
| `sentence-transformers/all-MiniLM-L6-v2` | the student: the only model that ships to the browser              | yes, fully |

The teacher is the embedding backbone plus small trained heads. Similarity ranking is the job
the embedding model is built for. The student learns to copy the teacher and fits in 24 MiB.

## Reproduce

You need an NVIDIA GPU with 32 GB (the defaults suit an RTX 5090), uv, about 60 GB of free
disk (estimate), an internet connection, and four free API keys.

Copy `.env.example` to `.env` and fill in the keys. `.env` is gitignored. A variable that is set
in the shell wins over `.env`.

| Variable                       | Get it at                       | Used for                                 |
| ------------------------------ | ------------------------------- | ---------------------------------------- |
| `TMDB_TOKEN` or `TMDB_API_KEY` | themoviedb.org > Settings > API | films: selection and details             |
| `HARDCOVER_TOKEN`              | hardcover.app/account/api       | books: selection, tags, and covers       |
| `LISTENBRAINZ_TOKEN`           | listenbrainz.org/settings       | songs: the top recordings of each artist |
| `LASTFM_API_KEY`               | last.fm/api/account/create      | songs: older artists and listener tags   |

Set one of the two TMDB variables. The bearer token wins when both are set. The pipeline logs
only "set" or "missing" for a key, never its value. It sends the keys in headers or as secret
query parameters, which never go into the HTTP cache.

```
uv sync
uv run download
uv run label
uv run train
uv run publish
```

Run these commands from `ml/`. Use `--help` with any command to see its purpose.

| Command    | Internal steps                                                  | Preflight                       |
| ---------- | --------------------------------------------------------------- | ------------------------------- |
| `download` | fetch, curate, resolve                                          | catalog API keys                |
| `label`    | profile                                                         | CUDA, vocab, eval feelings file |
| `train`    | train-teacher, train-student, export, eval-judge, eval, install | CUDA, vocab, eval feelings file |
| `publish`  | publish                                                         | R2 keys                         |

Each command skips completed work. Training and export use provenance stamps. Fetch,
resolve, and labeling use their own resumable caches. A failed step stops its command.

`train` installs the public bundle in `../static/bundle/` only after the ship gate passes. If it fails, the command exits with an error and names
`out/eval_report.json`. There is no override. `train` never publishes.
Run `publish` separately to upload the installed public bundle.

## Logs

Each command prints short lines to the console, such as
`12:04:31 resolve  INFO  film: 1,843 resolved`. Each step prints what it reads and how much work
is left, then what it wrote, what it dropped and why, and how long it took. Progress bars show
the long loops. WARN lines name what you should know: dropped items (counted by reason),
retries, and failures. A step that fails ends with an ERROR line.

Every command also writes a full log with DEBUG detail, such as each dropped item and each
checksum, to `out/logs/<time>-<command>.log`. Each command writes one file and ends with a
table of its steps and their times, including a failed step.

## Steps

The times are estimates for an RTX 5090, except where the text says "measured".

| Step            | Input                                          | Output                                                                                                       | Time                                                                               |
| --------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `fetch`         | `sources.toml`                                 | `data/raw/`                                                                                                  | 5–10 min                                                                           |
| `curate`        | APIs, `data/raw/`                              | `data/curated/catalog.jsonl`, `catalog.meta.json`, `pat.jsonl`                                               | about 21 min the first time (song selection, measured); under 1 min from the cache |
| `resolve`       | `catalog.jsonl`, APIs                          | `data/curated/resolved.jsonl`, `data/img/`                                                                   | about 15 min (songs are the slowest)                                               |
| `profile`       | `resolved.jsonl`, `eval_feelings.jsonl`, vocab | `data/curated/profiles.jsonl`, `moods.jsonl`, `pat_sentences.jsonl`, `labels-<vocab>.jsonl`, `distill.jsonl` | 4–10 h plus distillation feelings                                                  |
| `train-teacher` | the curated files                              | `data/models/teacher.pt`, `teacher_outputs.pt`                                                               | 30–60 min                                                                          |
| `train-student` | teacher outputs                                | `data/models/student/`                                                                                       | 20–40 min                                                                          |
| `export`        | student, curated files, images                 | `out/bundle/`                                                                                                | 5 min                                                                              |
| `eval-judge`    | teacher, student, bundle                       | `data/curated/judgments.jsonl`                                                                               | 1–2 h                                                                              |
| `eval`          | all of the above                               | `out/eval_report.json`, `out/run.json`                                                                       | 5 min                                                                              |
| `install`       | `out/bundle/`                                  | `../static/bundle/`                                                                                          | seconds                                                                            |
| `publish`       | `../static/bundle/`                            | public R2 bucket, `bundle.ts`                                                                                | minutes the first time                                                             |

### fetch

`fetch` downloads each file in `sources.toml` and checks its size and SHA-256. If a file does not
match, `fetch` stops with a "CHECKSUM MISMATCH" message. That means the source changed upstream.
Review the new file, then update `sources.toml`.

| Source                   | Use                                           | License                 |
| ------------------------ | --------------------------------------------- | ----------------------- |
| Met Open Access CSV      | public-domain artworks and their subject tags | CC0                     |
| PoetryDB                 | public-domain poems                           | poems are public domain |
| Text2Colors PAT (mirror) | 10,183 name-to-palette pairs                  | see below               |

- PoetryDB is an API, not a file. `fetch` reads every author in sorted order and writes
  `data/raw/poetrydb/poems.jsonl` sorted by author and title. PoetryDB answers HTTP 503 for a
  few large authors, such as Byron. For those, `fetch` asks for the titles and gets each poem of
  8 to 40 lines on its own. The file has no fixed checksum, so `run.json` records its hash.
- The Text2Colors repository never held the PAT data, and its Google Drive file is gone. The
  pipeline uses a mirror of `data/hexcolor_vf/`. The original code is MIT. The mirror has no
  license. When you use PAT, cite Bahng et al., "Coloring with Words: Guiding Image
  Colorization Through Text-based Palette Generation", ECCV 2018.

### curate (§9.1)

`curate` selects the catalog. Films, books, songs, and the Chicago and Cleveland art come from
APIs. `data/cache/http/` caches every API answer, so a second run needs almost no network. The
four categories run in parallel, because each one uses its own hosts.

| Source                   | What `curate` asks                                                                                                  | Calls for a full run |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------- | -------------------- |
| TMDB                     | `/3/discover/movie` for each year, 20 films a page                                                                  | about 300            |
| Hardcover                | one GraphQL query of 500 books for each page                                                                        | about 10             |
| ListenBrainz             | sitewide recording and artist stats (1,000 a call), popularity in batches of 500, the top recordings of each artist | about 2,100          |
| Last.fm                  | top artists of 14 older tags, 100 a call                                                                            | 14                   |
| MusicBrainz              | recording search `rid:(a OR b ...)`, 100 recordings a call                                                          | about 90             |
| Art Institute of Chicago | artwork search, 100 a page                                                                                          | 7                    |
| Cleveland Museum of Art  | artworks, 1,000 a page                                                                                              | 4                    |

**Eras.** Films, books, and songs are chosen by era. Each era keeps its quota:

| Category     | Eras and items                                                        |
| ------------ | --------------------------------------------------------------------- |
| film (2,000) | 1920–1969: 300 · 1970–1999: 600 · 2000–2019: 700 · 2020–2026: 400     |
| book (2,000) | before 1950: 300 · 1950–1999: 500 · 2000–2019: 700 · 2020–2026: 500   |
| song (3,000) | before 1980: 450 · 1980–1999: 750 · 2000–2019: 1,050 · 2020–2026: 750 |

Within an era, a year first gets at most two times its even share. The free slots then go to
the best items left in the era. So a strong year cannot fill the whole era, and a weak year
does not take items that nobody knows.

**Groups and candidates.** Each era, and each art source, is a group. `curate` writes more
candidates than a group keeps: 1.05 times for films, 1.1 for books, 1.4 for songs, and 1.25
for art. `resolve` keeps the target of each group (see below).

- **film:** TMDB discover for each release year from 1920, ranked by vote count. The minimum
  vote count is 50 before 1970, 100 for 1970–1999, 200 for 2000–2019, and 100 from 2020. A film
  needs a poster. The signal has the genres and the TMDB overview.
- **book:** Hardcover books for each era, ranked by readers (`users_read_count`), without
  compilations and duplicate records. `curate` asks the Hardcover schema for the fields it uses
  before the first query, and it stops with the names of any fields that are gone. It also
  removes box sets and a second record with the same title and author. The signal has the top 8
  reader mood tags, the top 8 genre tags, and the description.
- **song:** the ListenBrainz sitewide recording stats (9 ranges) give recordings and artists.
  The sitewide artist stats give more artists. ListenBrainz listeners play mostly recent
  music, so Last.fm's top 100 artists of 14 older decade and genre tags (50s, 60s, 70s, oldies,
  motown, soul, jazz, and so on) fill the era before 1980. The top recordings of each artist
  come next.
  `curate` ranks the recordings by the number of users who listened, not by listen count,
  because a few fans who stream a song all day make listen counts unfair. It keeps at most 6
  songs for each artist, one recording of each song, and no live, remix, karaoke, or other
  version. MusicBrainz gives the first release year and the ISRCs.
- **poem:** PoetryDB poems with 8 to 40 lines. The excerpt is the first 14 lines.
- **art:** 1,200 from the Met, 500 from the Art Institute of Chicago, and 300 from the
  Cleveland Museum of Art, all public domain.
  - The Met CSV gives Paintings (40%), Drawings, Prints, and Photographs (20% each). Highlights
    come first, then objects with a Wikidata item, then objects with subject tags. The CSV tags
    are the signal.
  - Chicago gives public-domain paintings with an image, the museum's boosted works first. The
    subjects, styles, and terms are the signal.
  - Cleveland gives CC0 paintings with an image: highlights first, then works with a curator's
    description. The description is the signal.
- **PAT:** each palette name with its five RGB colors, in `pat.jsonl`.

**Format.** `curate` writes `catalog.meta.json` with the catalog format. If `data/curated/`
holds a catalog in an older format (MovieLens, goodbooks-10k, MuSe), `curate` logs a warning and
starts fresh. It removes `catalog.jsonl`, `resolved.jsonl`, `resolve_dropped.jsonl`, and the
images in `data/img/`. It keeps `data/raw/` and `data/cache/http/`.

### resolve

`resolve` adds an image and links to each catalog item, and more signal where the source has
it. It caches every API answer in `data/cache/http/`, waits between calls to each host, and
appends each result as it goes. The categories run in parallel. It drops items without an
image, except poems.

- **Targets.** `resolve` keeps exactly the target of each group: the best-ranked candidates that
  resolve. It tries the candidates in rank order, in chunks, and skips a candidate that cannot
  rank above the last kept item of its full group. Then it removes any extra lower-ranked items.
  So the number of workers, and a run that resumes, do not change the result. It logs, for
  example, `song: 3,000 kept of 4,200 candidates (N dropped: reasons)`, and a warning for a
  group whose candidates ran out.
- **film:** one TMDB call for each film, `/3/movie/{id}?append_to_response=credits,keywords`.
  It gives the directors, the genres, up to 15 keywords, and the tagline. The poster is the
  TMDB `w780` image. The primary link is the IMDb page, or the TMDB page for a film without an
  IMDb id. Films run with 8 workers.
- **song:** Deezer finds the track by ISRC. If that fails, the Deezer search API finds it by
  artist and title. A match needs the same artist and a title that starts with the catalog
  title, ignoring "(feat. ...)" parts. An artist credit such as "Dave feat. Stormzy" or
  "Galantis, JVKE" matches when the full credit or the lead artist is the Deezer artist or one
  of its contributors. So "Simon & Garfunkel" stays whole, and "Tyla feat. Gunna" matches
  "Tyla". The catalog keeps the full credit as the creator. `resolve` prefers the original album over a live, remix,
  or compilation album, and it rejects another version of the song. Deezer gives the album
  cover, the album name, and the track link, which is the primary link. The Spotify, Apple
  Music, and YouTube links are search URLs for the artist and title. The preview is the path
  `/api/preview/deezer/<id>`, which the web app's Worker serves, because Deezer preview URLs
  expire after about 15 minutes.
- **song tags:** Last.fm has no batch tag lookup. So `resolve` first reads the top 1,000
  tracks of 165 mood, scene, genre, and decade tags (165 calls), and gives each song the tags
  whose lists hold it. A song with fewer than 3 tags from these lists gets one
  `track.getTopTags` call. Measured on 1,126 candidates: 40% have 3 or more tags from the lists
  alone, and the extra call gives 3 or more to 60% of the rest. So about 76% of songs have 3 or
  more tags, with about 40% fewer calls than one call for each song.
- **book:** the Hardcover cover. For a book without one, Open Library gives the cover. The
  primary link is the Hardcover book page.
- **art:** for Met objects, Wikidata gives the Commons image (P18), 200 objects for each
  SPARQL query. `resolve` prefers a file with "MET" in its name, and it downloads the 960 px
  Commons thumbnail. An object without a Commons image gets its image from the Met Collection
  API. Chicago images come from Commons too: Wikidata maps the Chicago artwork ID (P4610) to
  the image, 200 ids for each query. Chicago's own IIIF image server answers this pipeline with
  a Cloudflare 403, so a Chicago object without a Commons image is dropped
  (`no_commons_image`). Measured: 546 of the 625 Chicago candidates have one, for a target of 500. Cleveland images are the `web` images from its API.
- **poem:** no media. The primary link is a Poetry Foundation search.

**A blocked host.** After 10 failed requests in a row to one host, `resolve` makes no more
calls to that host in this run, and it logs one warning. Those items count as errors, and the
next run tries them again. So one blocked host cannot stall a run.

**Stop and resume.** Press Ctrl+C to stop any step, on Windows too. The step stops in about a
second, and a rerun continues from there. Every file write is safe to cut: a new file replaces
the old one only when it is complete. A progress file (`resolved.jsonl`, the LLM caches) can
end in half a line after a hard kill. The next run skips that line with a warning and redoes
its work. A damaged image file is fetched again.

**Format.** `data/curated/resolve.meta.json` holds the resolve format. When the format is
newer than the file, `resolve` removes the drop records that the new code can fix, once. For
format 2, these are songs without a Deezer match and Chicago art.

Each image becomes a WebP file with 800 px on the long edge. `resolve` records `w`, `h`, and the
average OKLab `tone`. Run `uv run download` to resolve all categories.

### profile (§9.2)

`profile` runs `Qwen/Qwen3.5-9B` (Apache-2.0, pinned revision) in-process with transformers, in
bf16, with thinking mode off. It runs five jobs:

1. **items:** a vibe line, a mood description, and three example feelings for each item. Art
   items also send the image.
2. **moods:** up to 6,000 synthetic feelings, 25 for each prompt, from random scene hints.
   The parser drops invalid or repeated feelings. It rejects an answer with fewer than 20
   good feelings.
3. **pat:** one feeling sentence for each PAT palette name. The PAT palette stays the label.
4. **labels:** five colors, a light, a typeface, and a scent for about 30,000 feelings: your eval
   feelings, the synthetic moods, and item feelings to fill the rest.

5. **distill:** 40,000 requested unlabeled feelings. A fixed grid crosses 100 situations with
   20 writing styles. Each of the 2,000 seed keys requests 20 feelings. Set the total with
   `ProfileConfig.distill_feelings` and the request size with `distill_per_request`.
   The grid covers life events, places, seasons, weather, relationships, work, travel,
   nature, city life, emotions, and aesthetics. Styles include one or two words, fragments,
   questions, typos, sensory details, and sentences of up to about 25 words.

The existing feeling jobs request scenes of 6 to 30 words. The separate `distill` prompt
allows short and informal text. It never includes eval feelings, including in retries.
After cache replay, it drops empty text, text over the app's 500 UTF-16-unit limit,
duplicates across the job, and eval matches. Matching uses lower case, collapsed spaces,
and punctuation removed from both ends. It logs generated, kept, and dropped counts by reason.
`data/curated/distill.jsonl` holds one `text` and its seed `key` per row. The cache preserves
completed seed requests even when some feelings are dropped. A rerun generates only missing
keys. This job does not change the existing prompts, caches, or label pool.

**Valid JSON.** `profile` uses constrained decoding with xgrammar. xgrammar compiles each JSON
schema, including the enums of vocab ids and a `#rrggbb` pattern for colors, into a grammar.
Numbered jobs require exactly one row per input, with fixed row numbers. A logits processor
blocks tokens that would break the schema and applies the token mask with plain PyTorch.
jsonschema checks each answer again. Content checks reject copied prompt instructions,
copied prompt examples, repeated feelings within a request, and text outside the word limits.
An item must have three distinct feelings and a vibe of at most 12 words. The item grammar
limits a word to 20 characters with no comma inside, so the model cannot glue words together
to pass the word limit.

If an answer fails validation, `profile` retries it once with the reason and twice the token
limit. The retry samples at temperature 0.7 with a seed from the key, because a greedy retry
often repeats the failed answer. A numbered answer keeps its accepted rows. Its rejected rows
get one more pass in a new request in the same run. `profile` saves persistent failures with the response and reason. It stops before downstream
steps if any key still lacks an accepted record.

**Resume.** Each job keeps its answers in `data/llm/<job>.jsonl`. A line holds a signature
(model, revision, system prompt, schema, token limit) and a hash of each key's own prompt. A
rerun validates matching cached answers and generates only missing or rejected records. Failed
answers remain eligible for retry. A new model, revision, or prompt redoes the keys it affects.
Accepted records retain their original request keys and row numbers during cache replay.

When one feeling refers to several works, the query loader does not choose an arbitrary work
as its retrieval target. It keeps available palette and choice labels for that feeling. It
excludes non-evaluation rows that have no remaining training label.

**Prefill.** Text-only batches prefill the prompt in 512-token chunks with explicit text
positions. This keeps prefill memory low, so 40 `labels` requests fit in one batch. Chunked
prefill changes bf16 rounding the same way a change of batch size does.

**Repeatability.** Decoding is greedy except for retries, the batch size is fixed, and the prompts run in a fixed
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
- The feature pass encodes about 60,000 to 75,000 existing texts plus the kept distillation
  feelings, at most 256 tokens each. Distillation feelings use the same query template and
  feature cache. They do not train the teacher heads. The best teacher checkpoint writes
  their embeddings, palettes, and choice logits with split `distill` and positive item `-1`.
  The loader excludes normalized matches to existing queries before adding these rows.
  It sorts by token length and packs batches by padded token count. It derives the starting
  budget from free GPU memory and model size. Each memory failure halves the budget and
  retries the pending batch. It restores input order after encoding.
- The retrieval heads project queries and items to 384 dims. The loss is InfoNCE with in-batch
  negatives. Training does not use hard-negative mining.
- The palette head predicts 5 OKLab colors. Both models use the same loss:
  `(mean squared slot distance + 0.5 * sorted-lightness MSE) / 3`.
  OKLab uses L in [0, 1] and unscaled a and b. The data is the LLM and PAT palettes.
  The teacher palette weight is 3.0. The student palette weight is also 3.0.
  The teacher keeps its previous weighted loss exactly. On 2,101 validation targets from
  the saved student, the old MSE is 0.00112443 and the shared loss is 0.00149211.
  The weighted student term is 0.00447633, within 0.5% of its old value, 0.00449771.
- The light, typeface, and scent heads use cross-entropy on the LLM labels.
- Training runs all 6 epochs at a learning rate of `3e-4`, without early stopping. It keeps
  the checkpoint with the best validation recall@10. The schedule uses linear warmup for
  6% of the steps, then cosine decay. It writes distillation targets from the best checkpoint.
  The training batch size stays 512.
- A sweep on cached features used the full cosine schedule and selected checkpoints by
  validation recall@10. At every epoch count, `3e-4` beat `1e-3`: validation recall was
  0.29–0.31 versus 0.25–0.30. Runs without hard-negative mining matched or beat runs with it.
  The best recipe by validation score was 6 epochs at `3e-4` without hard negatives:
  validation recall@10 was 0.309 and held-out recall@10 was 0.329. The old teacher scored
  0.283 and 0.321. An early-stopped 10-epoch run reached only 0.276 on validation because
  it stopped before the learning rate decayed.

### train-student (§9.4)

The student previously trained on about 40,000 LLM-written queries. The measured baseline
in `out/eval_report.json` uses 290 human-style eval feelings:

| Model                                 | Held-out recall@10 (LLM paraphrases) | Judged recall@10 (human-style feelings) |
| ------------------------------------- | ------------------------------------ | --------------------------------------- |
| Qwen3-Embedding-8B teacher with heads | 0.329                                | 0.718                                   |
| MiniLM-L6 student, int8               | 0.312                                | 0.644                                   |

The held-out gap is about 1.6 percentage points with unrounded scores. The human-style gap
is 7.4 points, with a confidence interval of [5.8, 9.1]. The student is at the 24 MiB cap.
Unlabeled distillation adds varied text without a larger student or another labeling pass.
These are baseline measurements, not results from training with the new feelings.

- Set `StudentConfig.backbone` and `revision` to choose the encoder. The default is
  `sentence-transformers/all-MiniLM-L6-v2`, fully fine-tuned, at a pinned revision.
  The model uses masked mean pooling. A learned projection maps other hidden sizes to
  384 dimensions. It passes token type IDs only when the encoder supports them.
- Training uses SDPA attention, fused AdamW, and TF32 matmul. The encoder runs eager:
  with `torch.compile` an epoch took 131 s, against 94 s eager, on an RTX 5090.
- Each step encodes about 1,030-1,120 items with gradients, a different count each step. The
  commands set `PYTORCH_ALLOC_CONF=roundup_power2_divisions:4`, so the CUDA allocator reuses
  blocks across these sizes. Without it, reserved memory grew past the GPU and a kernel launch
  failed in epoch 3. With it, peak reserved memory is about 15 GB. A value you set in the shell
  wins.
- The losses are: KL divergence between the teacher and student query-to-item similarity
  distributions (temperature 0.05), InfoNCE on (item feeling, item) pairs, the shared palette
  loss against teacher colors, and KL to the teacher choice logits.
  Distillation rows use only the three teacher-derived losses. They have no InfoNCE target.
- Each step encodes, with gradients, the teacher's top 16 items for each query, 128 random
  draws, and the positives. After deduplication, full batches keep 1,030–1,120 items by adding
  or trimming random extras. They always keep the teacher's neighbors and the positives.
  A final partial batch can use fewer items. Training keeps the checkpoint with the best
  validation recall@10.
- The maximum is 12 epochs, with patience 3 and the same warmup/cosine schedule as the
  teacher. The training batch size stays 64. Item and validation encoding use the shared
  token-budget helper. The helper never changes the training batch size.
- One epoch is one pass over `train` plus `distill` rows. Training logs both row counts.
  Validation and checkpoint selection use only `val`. Held-out and val splits stay unchanged.
  Both training stamps include `data/curated/distill.jsonl`. If that file is missing,
  `uv run train` stops and asks you to run `uv run label`.

### export

- One ONNX graph with the inputs `input_ids`, `attention_mask`, and `token_type_ids`, and the
  outputs `embedding` [N, 384], `palette` [N, 5, 3], `light`, `typeface`, and `scent`.
- Export measures fp32 validation recall@10, then compares dynamic int8 recipes.
  Candidates use per-tensor or per-channel weights, with optional exclusions for heads,
  the retrieval projection, the last attention output, the last linear layer, or embeddings.
  It rejects candidates above 24 MiB before scoring them. It selects the highest validation
  recall, with smaller size as the tie-breaker. It logs each fp32 gap and writes measurements
  beside the selected graph in `out/onnx/`. Held-out scores do not select the recipe.
- The public bundle in `out/bundle/` has `manifest.json`, `model/`, `vocab.json`, `img/`,
  `items.json`, `vectors.bin`, and `search-index.json`. The manifest lists each file and format.
  Item vectors use IEEE fp16, little-endian, in catalog row order. The int8 graph computes
  the vectors. The browser decodes them once to float32, without renormalizing them.
  Name data contains common words that occur in names and representative rows for creators
  and albums. Export selects representatives from the original float32 vectors with the
  shared TypeScript search code. Export needs Bun.
  The stub writes to `out/stub/bundle/` and adds `anchors.json` and fp32 `anchors.bin` for
  its heads. It keeps the trained export. Songs carry `album`.
  The manifest version is a hash of the model and the fp16 vectors.

The saved MiniLM-L6 student measured as follows on 1,335 validation feelings and 3,111
held-out feelings. Each query runs alone, as it does in the browser. Item vectors use the
shared length-sorted token batches. Selection uses only the validation column.

| Recipe                                     | MiB   | Val recall@10 | Held-out recall@10        |
| ------------------------------------------ | ----- | ------------- | ------------------------- |
| fp32 reference                             | 88.54 | 0.278652      | 0.298939                  |
| per-tensor int8                            | 22.77 | 0.271910      | 0.287689                  |
| per-channel int8                           | 22.88 | 0.274906      | 0.297011                  |
| per-channel, heads in fp32                 | 22.88 | 0.274906      | same graph as per-channel |
| per-channel, last attention output in fp32 | 23.30 | 0.273408      | not used for selection    |
| per-channel, last linear layer in fp32     | 24.56 | exceeds cap   | exceeds cap               |
| per-channel, embeddings in fp32            | 56.97 | exceeds cap   | exceeds cap               |

Per-channel int8 loses 0.19 percentage points of held-out recall against fp32.
Export measures the candidates again for each new checkpoint.

### install

This internal step runs only after the ship gate in `uv run train`.

`install` copies `out/bundle/` to `../static/bundle/` through a staging directory.
Use `PUBLIC_BUNDLE_URL=/bundle/` in the dev server environment to select these files.
Install needs no Cloudflare credentials.

### publish

The app runs on Cloudflare Workers. The public bundle lives in the R2 bucket `mise`,
served at `cdn.mise.art`. `publish` reads the installed bundle from `../static/bundle/`.
Run `uv run train` to export, check the ship gate, and install a trained model.
The Bun stub installs its own bundle without Python. After a stub build, install the
trained bundle again before publishing a trained release.

Use an R2 API token with Object Read & Write access to the public bucket.
Set `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`
(default `mise`), and `R2_PUBLIC_URL` (such as `https://cdn.mise.art`) in `.env`.
See `.env.example`. Inspect the bucket and its custom domain before the first publish.

Run `uv run publish` from this directory.

- Images go to `img/<first 20 hex of the SHA-256>.webp`. Publish uploads missing images,
  16 at a time, and rewrites each record's image URL to the public CDN URL.
- All other files go to `bundles/<date>-<hash8>/`, including records, fp16 vectors, and
  name data. `items.json` contains the published image URLs. Publish uploads the manifest last.
- The prefix hash covers the file names and bytes, including the rewritten records.
  Publish reuses an existing prefix with the same content hash.
- Objects use `Cache-Control: public, max-age=31536000, immutable`.
  JSON uses `application/json`; ONNX and binary files use `application/octet-stream`.

After the upload completes, publish writes the bundle URL to `../src/lib/bundle.ts`.
Commit that file, then run `bun run deploy` from the repository root.

**CORS.** The app's web worker fetches the model and the JSON from another origin, so the bucket
needs a CORS rule. `publish` adds the rule if no rule covers it yet, and keeps the other rules.
An "Object Read & Write" token usually cannot change the CORS policy. Then `publish` logs a
WARN with the JSON below and continues. Paste the JSON into R2 > `mise` > Settings > CORS
policy:

```json
[
	{
		"AllowedOrigins": [
			"https://mise.art",
			"https://www.mise.art",
			"http://localhost:5173",
			"http://localhost:4173"
		],
		"AllowedMethods": ["GET", "HEAD"],
		"AllowedHeaders": ["*"],
		"MaxAgeSeconds": 86400
	}
]
```

S3 CORS rules do not accept a wildcard in the middle of an origin. To use the app on
workers.dev, add `https://mise.<your subdomain>.workers.dev` to `AllowedOrigins` yourself.
Your subdomain is in Workers & Pages > your account's workers.dev subdomain. Preview URLs get a
new host for each version, so they need their own origin line.

### eval (§9.5)

`eval` reports, for the teacher and for the shipped student (the int8 ONNX graph and
`vectors.bin`):

- **recall@10 on held-out item feelings.** 10% of the item feelings never go into training. A
  hit is the source item in the top 10 of its category.
- **recall@10 on your eval feelings.** `eval-judge` pools the top 20 items for each category
  from the teacher, the student, and the untrained student backbone. The local LLM marks each pooled item
  as a fit or not. The score is the fits in the top 10 divided by the smaller of the fit count
  and 10.
- **palette ΔE:** the mean Euclidean OKLab distance for each color against the LLM palettes.
- **choice accuracy:** top-1 accuracy of light, typeface, and scent against the LLM labels.
- **student latency:** onnxruntime on the CPU with one thread, batch size 1.

The ship gate requires all of these conditions:

- At least 100 distinct human-reviewed eval feelings and complete judgments for the current pool.
- Student judged recall@10 no more than 5 percentage points below the teacher.
- Student held-out recall@10 no more than 5 percentage points below the teacher.
- Student held-out recall@10 at least 10 percentage points above the untrained backbone.

Judged recall averages the categories with relevant works within each feeling, then averages
feelings. The report and log include a paired bootstrap 95% interval for each gap. The
bootstrap resamples feelings 10,000 times with seed 1337. The gate uses the point estimates;
it reports intervals to show uncertainty. Missing scores fail the gate.

## Browser catalog search

The browser worker loads the model, records, vectors, and name data once. It runs the encoder,
heads, anchor matcher, and exact scan locally. Each request returns one complete mood.
Picks follow category order, with ties resolved by catalog row. A zero vector selects the
first eligible row in each category. Named works use a normalized 70% item / 30% query blend.
Picks exclude the anchor's exact creator and title. Creator and album queries use the
representatives selected at export. The anchors-kind stub uses the same search path.

Run `bun run query "a quiet evening"` from the repository root to use the browser engine
with the installed bundle and the native ONNX runtime.

## Eval feelings

`eval_feelings.jsonl` (in this folder, committed) holds the human-written eval set. Write one
JSON object per line with one field, `text`:

```json
{ "text": "a snowy december and i just made warm hot chocolate" }
```

- Write about 300 distinct feelings. The ship gate requires at least 100.
- Write each feeling as a sentence about a scene or a moment, 6 to 30 words. Do not write lists
  of mood words.
- Write the file before `uv run label`. If you add lines later, run `uv run label` again.
  It labels only the new lines.
- The eval feelings never go into training.

## Reproducibility

- Nothing under `data/` or `out/` is committed. Run `download`, `label`, then `train`
  to build from `sources.toml` and the APIs.
- Every step writes sorted JSONL with sorted keys, so two runs give the same files.
- `curate` reads the API answers from the cache, so a rerun gives the same `catalog.jsonl`.
  To take newer data, remove `data/cache/http/` for that host.
- Training uses fixed seeds (1337), `torch.use_deterministic_algorithms(True)`, deterministic
  cuDNN, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. A small check ran every training op (MiniLM
  forward and backward with SDPA attention, the teacher heads, the losses)
  twice in strict mode. No op raised an error, and the gradient sums matched exactly.
- The teacher feature pass runs in strict mode too. Qwen3-Embedding-8B is a standard
  transformer with SDPA attention and no linear-attention layers, so the Triton kernel caveat
  of Qwen3.5 does not apply. A strict-mode forward of a small random Qwen3 model, with left
  padding and last-token pooling, ran twice and gave identical results.
- The Qwen3.5 generate path (`profile`, `eval-judge`) uses warn-only mode, because nobody has
  run it in strict mode yet. Warn-only mode keeps the deterministic kernels and prints a warning
  for an op that has none.
- The Hugging Face models are pinned to a revision in `src/mise_ml/config.py`.
- `out/run.json` records the seed, the Python and package versions, the GPU, the model ids and
  revisions, all configs, the SHA-256 of each source and each curated file, the PoetryDB content
  hash, the bundle files, and the ship decision.
- Training writes a stamp in `data/models/` after each training and export step. The stamp
  hashes the inputs, relevant source files, and config. Each command skips a step when its
  stamp matches and its required outputs exist. Label caches keep their own prompt signatures.

## What reruns, and why

Use `uv run download --plan`, `uv run label --plan`, `uv run train --plan`, or
`uv run publish --plan` to inspect a command. A plan does not load a model, use the GPU,
generate labels, upload files, or write pipeline files. It exits with status 0.
Publish compares the exact upload bytes with a read-only R2 object listing. It reports
unknown upload counts if it cannot read the bucket.

`src/mise_ml/steps.py` declares each step's files, content fields, config, source modules,
outputs, and dependencies. Training hashes the catalog fields it consumes. A change to a
song's album cover does not change its training text. Stamps record each digest and config
value. Plans name changed inputs, config fields, source files, and outputs. A missing or
unverified stamp requires a run. Downstream steps are marked for recheck when an upstream
step will run; execution checks the new output bytes before deciding to run them.

LLM plans count new keys, changed prompt fields, changed signatures, and rejected answers.
They show up to five example keys for each reason and the initial request count, before
retries. New cache records include field digests. For an older record with a changed prompt,
the reason is `prompt changed (fields not recorded)`. Older records are not rewritten.
Labels use the current profile and mood files; their key count can change after those jobs.

The judge saves the actual retrieval pool with its ranking inputs. A plan can inspect that
pool without inference. New feelings or changed ranking inputs require model ranking first.
The plan then reports a lower bound for new keys and an unknown final request count.
It does not invent exact counts from older judgments.

`uv run label --plan` also reports unused request records and their bytes per job.
Use `uv run label --prune` to print the same report and remove those records without
generating answers. A request that still supplies one current accepted key stays intact.
Prune keeps records whose use depends on unfinished upstream work or an unverified judge
pool. It stops if required inputs are missing. Cache writers and prune use file locks;
prune also checks that each cache still matches the inspected digest.

For example, change `StudentConfig.epochs` from `12` to `13`. The train plan names
`config.epochs`, keeps the teacher current, and marks the student, export, judge, eval,
and install for work or recheck. Restore `12` to reuse the student stamp. Adding an eval
feeling adds a label key and requires new judge rankings. It also invalidates the teacher:
`load_queries` uses eval membership to assign splits and exclude distillation text.

`out/run.json` includes the step stamps, LLM cache digests, accepted record counts, and
whether each cache key set was verified. Jobs without their required inputs have zero
records used and an explanation. HTTP read receipts track the cached API files used by
download steps. Failed resolve requests prevent a completed stamp, so a rerun retries them.

## Configuration

The defaults are dataclasses in `src/mise_ml/config.py`: `CurateConfig`, `ResolveConfig`,
`ProfileConfig`, `TeacherConfig`, `StudentConfig`, and `ExportConfig`. Change a value there to
change a run. If a `profile` batch runs out of GPU memory, `profile` lowers the batch size to
three quarters and keeps that limit for the rest of the job.

The vocab source of truth is `../scripts/stub/vocab.json`. The labels and the heads use its
order. The label file name contains a hash of the vocab, so a new vocab gets new labels.

## Licenses

Licenses and attribution are not final for this draft. The Met, Art Institute of Chicago, and Cleveland Museum of Art images used here are public domain
or CC0. ListenBrainz and MusicBrainz data is CC0. The TMDB posters, the Hardcover and Open
Library covers, and the Deezer album covers and previews belong to their owners. The app shows
them next to links to the source pages. TMDB asks for an attribution notice in the app.
