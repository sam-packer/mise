# The mise model pipeline

This folder builds everything the app loads from the CDN: the catalog of about 16,000 works, their
images, and the small model that runs in the browser. It runs on one local NVIDIA GPU.

The pipeline has five commands. Run them from `ml/`, in this order:

```sh
uv sync
uv run download   # fetch the sources, choose the catalog, find images, links, and song facts
uv run label      # a local LLM describes every work and writes training feelings
uv run refine     # GPT-6.1 Sol grades every description; failed works are replaced
uv run train      # train the teacher, distill the student, export, evaluate, install
uv run publish    # upload the installed bundle to R2
```

Add `--help` to any command for its details. Add `--plan` to see what it would do without doing it
(all commands except `refine`).

## What you need

- An NVIDIA GPU with 32 GB of memory. The defaults suit an RTX 5090.
- About 60 GB of free disk, [uv](https://docs.astral.sh/uv/), and [Bun](https://bun.sh) (the export
  step runs the app's TypeScript search code and engine). Run `bun install` in the repository root
  first.
- Four free API keys and an OpenAI key in `ml/.env`. Copy [.env.example](.env.example) to `.env` and follow its
  comments. Git ignores `.env`. A variable already set in your shell wins over the file.

| Variable                       | Where to get it                 | What it is for                        |
| ------------------------------ | ------------------------------- | ------------------------------------- |
| `TMDB_TOKEN` or `TMDB_API_KEY` | themoviedb.org > Settings > API | films                                 |
| `HARDCOVER_TOKEN`              | hardcover.app/account/api       | books                                 |
| `LISTENBRAINZ_TOKEN`           | listenbrainz.org/settings       | songs: the top recordings per artist  |
| `LASTFM_API_KEY`               | last.fm/api/account/create      | songs: older artists and mood tags    |
| `SMITHSONIAN_API_KEY`          | api.data.gov/signup             | optional: Smithsonian CC0 art         |
| `OPENAI_API_KEY`               | platform.openai.com/api-keys    | `refine`: the description grader      |

NASA needs no key. Without a Smithsonian key, the Met, Chicago, and Cleveland fill its 500 slots
in a 60/25/15 split. Set the optional key to include Smithsonian works on the next download.

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

`download` runs four steps.

1. **fetch** downloads the fixed source files in `sources.toml` (the Met collection and a palette
   data set) and checks each checksum. PoetryDB is an API, so fetch reads it poem by poem.
2. **curate** chooses the catalog: 3,000 films (TMDB), 3,000 books (Hardcover), 5,000 songs
   (ListenBrainz, Last.fm, MusicBrainz), 3,000 artworks (the Met, the Art Institute of Chicago,
   Cleveland, NASA, and Smithsonian), and all eligible public-domain poems. PoetryDB currently
   yields 1,993 poems with 8 to 40 lines, for a target of 15,993 works. Films, books, and songs are
   split into eras, and each era gets a fixed share, so the catalog is not only recent hits. Songs
   rank by the number of listeners, not the number of plays.
3. **resolve** finds an image and links for each work. Posters come from TMDB, covers from
   Hardcover or Open Library, album art and previews from Deezer, and museum images from Wikimedia
   Commons or the museum. Each image becomes an 800 px WebP file.
4. **facts** finds two facts for each song: the song's Wikipedia article (the intro and its
   lyrics, composition, background, and meaning sections, at most 1,200 characters) and its
   lyrics from LRCLIB. The article must be about this song by this artist, not an album, a
   disambiguation page, or the original of a cover. Lyrics count only when LRCLIB's track length
   is within 8 seconds of the Deezer track; otherwise facts looks for a version that matches. The
   lyrics stay on this machine in `curated/facts.jsonl`. Only a one-sentence theme that the labeler
   writes from them goes into a description, and no lyrics go into the bundle.

### Song breadth and scene balance

The artist pool combines ListenBrainz sitewide recordings and artist lists with up to 200 artists
per Last.fm tag (two pages of 100). The 14 older tags stay. Another 29 distinct tags cover indie,
alternative R&B, neo-soul, bedroom pop, dream pop, art pop, shoegaze, post-punk, singer-songwriter,
lo-fi, alternative rock, electronic, ambient, trip-hop, hip-hop, K-pop, J-pop, city pop,
afrobeats, Latin, bossa nova, reggae, country, Americana, classical, soundtrack, indie rock,
synthpop, and indie pop. Jazz is already in the older tags.

Each artist keeps the first Last.fm tag that found it, in the fixed tag and page order. Match by
artist ID, then by normalized name when the services use different IDs. Artists with no such tag
use `sitewide`. This is a discovery group, not a claim about an artist's whole
genre. Each artist supplies at most three songs. The cap applies to both artist IDs and display
names, so duplicate service IDs cannot double the allowance. Name matching preserves non-Latin
characters. Title deduplication also keeps non-Latin letters instead of merging those titles
into an empty key. Alternate versions are excluded as before.
When titles share a slug, each record gets its recording ID as a suffix. This keeps distinct
works separate and prevents a later selection from reusing another work's saved media.

Within each era, selection reserves 14 candidates per discovery group for a floor of 10 songs
(set by the song candidate factor). If the era is too small, the reserve shrinks to its even
share. Each reserve uses listener rank. The rest use listener rank with the existing year cap.
Resolve keeps this selection order, including when it reuses saved images and repaired facts.
An unavailable scene releases its slots to the rest. Failed media can reduce a scene's final
count; the extra candidates provide replacements. There is no artist boost or personal list.

The song eras separate the 2000s and 2010s. Selection logs each era's quota, pool size, candidate
count, last candidate's listener count, and minimum including the scene reserves.
Scene reserves can have fewer listeners than the main cutoff.
The candidate factors are 1.3 for films, 1.35 for books, 1.8 for songs, and 1.6 for art. The spare
candidates also refill the works that `refine` drops.

See [the expansion measurements](catalog-expansion.md) for the quota, scene-share, cutoff,
image-download, size, and time tables from this change.

### Art sources

The Met keeps 1,200 slots, Chicago 500, and Cleveland 300. NASA and Smithsonian each add 500.
NASA uses 18 nature and space topics. Smithsonian uses 12 topics across photographs and
paintings from CHNDM, SAAM, NASM, EEPA, NMAAHC, and Smithsonian Gardens. Each selector sorts by
ID, shuffles with the fixed seed, and takes turns across topics. A topic supplies at most twice
its even share. Duplicate source IDs count once.

NASA selection excludes explicit third-party copyright notices and technical, staff, ceremony,
logo, and test metadata. Smithsonian selection requires CC0 on each selected image, excludes
metadata-marked portraits, and retains the credit line in `source.credit`. Images come from
NASA's medium rendition and Smithsonian IDS at 1,200 pixels. IDS gets a plain User-Agent.
If a new source fails or supplies too few candidates, the other museums fill the missing slots
in the same 60/25/15 split. Each art row records the effective group target for resolve.

Every API answer goes into `data/cache/http/`, so a second run needs almost no network. After ten
failures in a row, the pipeline stops calling that host for the rest of the run. The next run tries
again.

Resolve checks compilation and live-album names against official releases. It uses a song's
original release year and repairs recording identities that point only to bootlegs. For books,
it uses the source's first-publication year. A year from 1 to 999 needs source evidence of an
ancient work. A resolve format change also repairs existing song and book rows. It keeps unrelated
resolved works and reuses cached API answers.

## label

`label` runs Qwen3.5-9B on the GPU. Constrained decoding (xgrammar) makes every answer valid JSON
for its schema, so the model can't return a broken answer. It runs seven jobs:

| Job     | What the LLM writes                                                                     |
| ------- | --------------------------------------------------------------------------------------- |
| themes  | one sentence about what a song is about and how it feels, from its checked lyrics      |
| leaks   | item profiles for a fixed sample of 200 works, to find prompt words that leak           |
| items   | the work's main emotion as a vibe, a description, and three feelings a person might type |
| moods   | about 6,000 invented feelings, with scene, first-person, and heartbreak hints |
| pat     | a feeling for each named palette in the PAT data set                                    |
| labels  | five colors, a light, and a typeface for about 30,000 feelings                 |
| distill | up to 40,000 feelings across 30 styles, with no labels |

Blind audits by Claude showed that a word in the prompt shows up in the descriptions: "tension"
made works darker, and "irony" put irony into most lyric themes. The items prompt asks for the
emotion most people feel from the work, in precise words with three quoted examples. It says to
keep light works light, and forbids invented lyrics, sounds, colors, figures, and events. For
songs, the prompt also gets the Wikipedia text and the lyric theme from `facts`. The **audit**
step of `train` looks for such words in the finished profiles. It also gives the catalog shares
of the words on the leaks block list.

The **leaks** job runs before items. It labels a fixed sample of works two times: 70 art works, 40
poems, 40 books, 25 songs, and 25 films. A hash of the work id selects them. The first time, it
uses the current prompts. The second time, it uses the neutral baseline prompt (`BASELINE_SYSTEM`).
A word of a category's prompt leaks when its share of the vibes and descriptions under the current
prompt is at least 5 points higher than under the baseline, and at least two times the baseline
share. The baseline share counts as at least one work. The job writes the leaked words of each
category, with both shares, to `curated/leak_block.jsonl`. Each row also keeps its sample and a
digest of the category's prompt, the baseline prompt, the leak rule, and the decoding settings.
A category keeps its block list while that digest stays the same, so refine drops and refills do
not change it. When the digest changes, the job draws a new sample for that category.

The items job discourages the words on its category's block list, and their plural and -ly forms.
If a vibe or description uses one, the first answer and the first retry fail with a message that
names the word. The last retry accepts the word, so a work keeps it when no other word fits, and the
block list never stops the job. The log gives the number of works that kept a blocked word. The
block list is part of the key of each item, so a change to a category's list labels that category
again.

Each category has its own items prompt in `ITEM_SYSTEMS` in `profile.py`, as a pilot selected
them. Art and film prompts also list broad words to avoid; book and song prompts do not. The poem
prompt adds the avoid list and has an `{examples}` slot in place of the fixed examples. The labeler
fills the slot with `EXAMPLE_COUNT` phrases from the poem pool in `ITEM_EXAMPLES`. A seed from a
hash of the work id selects the phrases, so a work always gets the same phrases.

The items job asks for at most 10 words in a vibe. The grammar and the parser also allow at most 10
words. Words use ASCII letters, digits, apostrophes, and hyphens, with at most 14
characters. The parser rejects a vibe that ends with a function word or comma. Descriptions and
example feelings use the same word rule, with ordinary punctuation. Each example feeling has 6 to
30 words.

Distill styles include idioms, sarcasm, internet slang, heartbreak, envy, spite, shame, dark humor,
second-person lines, mixed feelings, and one-to-three-word moods. Another 200 requests each name
five idioms or slang terms from a fixed list, so the texts do not repeat the same few. Duplicate
texts and eval feelings are excluded from distillation.

An answer that fails its checks gets one retry. Each job keeps its answers in `data/llm/<job>.jsonl`.
The key of each answer is a hash of its exact prompt. So a rerun only asks for what is new or
changed, and a change to a prompt or the model redoes only the answers it affects. The key also
covers the decoding settings of the job: the generation grammar with its property order, the
sampler, thinking, and the retries. For the items job, the key also covers the work's system
prompt and seed.

## refine

`refine` checks every description and replaces the works that the labeler cannot describe well.
It needs `OPENAI_API_KEY` and the download keys, because a refill downloads new works.

One round:

1. Run facts, themes, leaks, and items for works that do not have a description yet.
2. GPT-6.1 Sol grades each description that `curated/ledger.jsonl` does not have yet, one at a
   time, with the source facts and, for art, the image. It gives the work's real emotion in a few
   words and grades the description: factual 0-2, emotion 0-3, specific 0-2, and good queries
   0-3.
3. A description with an emotion grade below 2 fails. Its work goes to `resolve_dropped.jsonl` with
   the reason `label_quality`, so no later download tries it again.
4. Resolve fills each free place with the next candidate of the same group.

One run of refine does at most three rounds: the first grading and two refills. In the last round,
failed works are dropped without a refill, and the next download or refine fills their places. A
run stops early when no description fails. A run first applies any failed grades that an
interrupted run did not apply, and fills any free places.

`curated/ledger.jsonl` keeps one line per graded description: the round, the grades, the grader's
note on the real emotion, and the verdict. A rerun grades a description again only when its grader
request changes: the description, the facts, the image, the model, or the rubric.
`data/cache/grades.jsonl` keeps each answer, so a repeated request costs nothing. A changed
`min_emotion` applies to the grades already in the ledger.

The grader came from a comparison on 830 descriptions that Claude had graded. Each candidate graded
the same labels, one at a time. GPT-6.1 Sol agreed with itself on 93% of the right-or-wrong calls
and with the Claude graders' consensus on 91%. GPT-6 Astra did no better at five times the price.
Jev, a second local labeler, and a re-run of the labeler itself caught fewer than half of the bad
descriptions.

## train

`train` runs these steps and stops at the first failure.

1. **audit** checks the work profiles for words that the labeler overuses. For each category, it
   counts the share of works whose vibe contains a word, and the share whose vibe and description
   contain it. It skips function words and words about the form of a work, such as "film" or
   "narrator". A word gets a warning when its share is above 25% in the vibes or above 40% in the
   vibes and descriptions. The warning also gives the word's share in the other categories, so you
   can see a habit of one category ("quiet" in art) apart from a general one. The step writes
   `out/data_audit.json`. It only warns; it never stops the run. A change to the profiles runs it
   again. To run it alone, use `uv run python -m mise_ml.audit`. It needs no GPU.
2. **train-teacher.** The frozen Qwen3-Embedding-8B encodes every text once. For each work, it
   encodes the source facts (the facts that the raters read, cut to about 700 characters) and then
   the profile (vibe, description, category, title, and creator). With raw Qwen vectors on the
   tuning set, facts and profile score 0.641, the profile alone 0.620, and the facts alone 0.620.
   Small heads learn
   to rank works for a feeling (InfoNCE: the matching work must score above the other works in the
   batch), and to predict the palette, light, and typeface, with warmup and a cosine decay. The
   teacher's score for a feeling and a work mixes the cosine of the head vectors with the cosine
   of the raw Qwen vectors (`raw_weight`, 0.5 by default), so the heads learn what the raw
   vectors miss. Before training, the step drops each train, val, and distill text whose Qwen
   query cosine to any line of `eval_feelings.jsonl` is 0.88 or higher (`near_eval_cosine`), so
   rewordings of the rated feelings do not train either model. The log gives the count.

   The heads also learn from fit ratings. `fit_labels.jsonl` holds 100,000 ratings from 0 to 3:
   for 2,000 training feelings, a teacher's top 10 works in each category, one work per creator.
   GPT-6.1 Sol rated each pair from the work's source facts with the rubric of the rated sets. On
   the Opus-rated top 10 of 100 tuning feelings, its order of the works agreed with Opus
   (Spearman 0.77). The **fit loss** is ListNet: for each feeling and category, a softmax
   cross-entropy pulls the teacher's scores of the 10 works toward targets proportional to
   exp(rating / `fit_tau`). A shift of all ratings of a group does not change the targets, so a
   lenient judge does no harm. Each step adds the fit loss of 256 random groups (`fit_groups`)
   with the weight `fit_weight`. Only feelings in the train split use their ratings, so a feeling
   that the near-eval rule drops never trains. No eval feeling has a fit rating. A change to
   `fit_labels.jsonl` runs train-teacher and the steps after it again. The fit loss raised the
   teacher's tuning objective from 0.648 to 0.704. The ratings of 500 and 1,000 of the 2,000
   feelings gave 0.663 and 0.677, so more ratings still help.

   The run keeps the epoch with the best tuning objective (see "The rated sets"
   below). `teacher_params.json` holds the learning rate, weight decay, dropout, epochs, hidden
   size, batch size, temperature, label smoothing, warmup, and raw weight. Without that file, the
   step first runs an Optuna search of 50 trials (about 25 minutes). Each trial trains the heads,
   and the search keeps the trial with the best tuning objective. The step writes the settings of
   that trial to the file. Git tracks the file.
   `uv run train --tune` runs a new search and replaces the file. A change to the file runs
   train-teacher and the steps after it again.
3. **train-student.** MiniLM learns from the teacher. The student encodes feelings only. The
   work vectors are fixed: the teacher's work vectors on their top 384 singular vectors (the
   384-dim space that best keeps the teacher's work-to-work scores). On the tuning set, the
   teacher scores 0.701 in this space and 0.704 in its full space. The step saves these vectors
   as `items.npy`, and export ships them as `vectors.bin`. For each feeling the student copies the
   teacher's ranking of works inside each category (the teacher's top 4 per category, with a
   strong KL weight of 16), the teacher's feeling vector in the fixed space, its palette, and its
   label choices. The distillation feelings from `label`
   have no answers of their own. They only teach the student to rank the way the teacher does, on
   a wider range of writing. In each epoch, 30% of the training feelings get typing noise (casual
   spellings such as "im" and "wanna", and one or two letter slips); the teacher's target stays
   the one for the clean text. The run trains for up to 40 epochs and keeps the epoch with the
   best tuning objective. It stops early after 10 epochs without a better tuning objective.
4. **export** writes one ONNX file with the encoder and all heads, then quantizes it to int8 (8-bit
   weights instead of 32-bit). It tries a few quantization recipes and keeps the one with the best
   tuning objective under 24 MiB. It also writes the fixed item vectors as 16-bit floats and the name
   data for the "in the key of" matcher. Last, it runs `../scripts/build-samples.ts` on the
   finished bundle. That script runs each sample feeling of the app (`../src/lib/examples.ts`)
   through the browser's engine and ONNX Runtime Web. It writes `samples/<code>.json` for each
   sample: the mood, and the world behind each of its tiles. `samples/index.json` gives the file
   and the palette of each sample. The page shows a sample's room from its file before the model
   loads. A change to the samples or to the app's search code runs export again.
5. **eval** scores the models and applies the ship gate (see below).
6. **install** copies the bundle to `../static/bundle/`, but only when the ship gate passes.

`train` never uploads anything.

Export also saves training-label priors for light and typeface in the manifest's `heads`
object. It adds one count per choice so each prior stays positive. For each head, it selects the
largest correction strength from 0, 0.25, 0.5, and 0.75 that loses at most two percentage points
of choice accuracy on validation feelings. The browser subtracts `tau * log(prior)` from each
score before it chooses the highest. Bundles without priors use the raw scores.

Export also writes an exposure penalty for each work to `penalty.bin` (16-bit floats, in catalog
order). Without it, a few works fill many walls: the top 1% of works took 26.5% of the wall places
of the tuning feelings, and one work showed for 75 of the 1,061 feelings. Export runs the int8
graph on each training text in `teacher_outputs.pt`, one text at a time as the browser does. It
counts how often each work is in the top 2 of its category. A work's share is its count over the
mean count of its category, each plus one. Each round moves the penalty halfway to
`beta * log(share)`, and a work at or below the mean gets no penalty. Export runs six rounds with
a beta of 0.02 (`ExportConfig.exposure_beta` and `exposure_rounds`) and logs the top 1% share and
the largest count before and after. The count skips the app's rule of one work per creator.

The app subtracts the penalty from each score when it chooses the works for a feeling. The world
of a work does not use it. Eval also subtracts it from a bundle's feeling scores, so the ship gate
scores the walls that the app shows. The teacher, the baseline, and the world fidelity use no
penalty. On the tuning feelings, a beta of 0.02 lowered the top 1% share to
16.9% and the largest count to 32. Blind ratings of the walls did not change: the paired mean
change was -0.011 (interval -0.026 to +0.005). Bundles without the file use no penalty.

### The rated sets

Two files hold rated pairs. Each pair is an eval feeling, a work, and a rating from 0 to 3. The
rating is the mean of two independent Claude Opus raters that read the work's source facts and the
rubric. A pair fits when its rating is 2 or higher.

- `tuning_set.jsonl` holds 14,360 pairs over 1,061 feelings. It selects the settings.
- `gold_set.jsonl` holds 12,202 pairs over 941 other feelings. It is the final check only. No step
  selects by it, and the ship gate does not read it.

No feeling is in both files. Training drops every text within a Qwen cosine of 0.88 of any eval
feeling, so near-copies of rated feelings never train.

A model's score on a set has two parts:

- **Within-feeling AUC.** For each feeling, take every pair of one fit and one non-fit. The AUC is
  the share of those pairs that the model ranks in the correct order.
- **Spearman.** The rank correlation between the ratings and the model's scores over all pairs.

The **objective** is the mean of the AUC and the Spearman. The tuning objective selects these:

- the Optuna trial and the epoch of the teacher, and the fit-loss weight and `fit_tau`,
- the epoch of the student, and the early stop,
- the int8 quantization recipe.

### How the model is scored

`eval` writes `out/eval_report.json`. For the teacher, the student, and the untrained MiniLM, the
report gives these numbers:

- **Held-out recall@10.** 10% of the item feelings never go into training. For each one, the test
  asks whether its source work lands in the top 10 of its category.
- **Tuning and gold scores.** The AUC, the Spearman, and the objective on each rated set.
- **Label scores.** The palette error and the light and typeface accuracy against the LLM labels.
  Only the teacher and the student get them, because the baseline has no palette or choice heads.

For the student only, the report also gives the fidelity, the latency, and the room usage. The
fidelity is the share of the teacher's top 10 per category that the student also ranks in its
top 10. The latency is the time for one feeling on one CPU thread. The room usage is the number of
distinct student lights and typefaces, and the share of each head's most common choice on eval
feelings. The world fidelity copies the app's "world" of a work: its 2 nearest works in its own
category and 2 in each other category, by the similarity of the work vectors. The rule skips the
work itself, works with the same creator or title, and a second work by one creator in a category.
For 1,000 works from a fixed sample, the world fidelity is the share of the teacher's world
neighbors that the student's world also holds. The student's work vectors are the fixed vectors
in `vectors.bin`, so the world fidelity measures only the loss of the 384-dim space. These numbers
are for the report only. The ship
gate does not use them.

Eval also writes `out/eval_failures.md`, a list for a person to read. It gives the student's 25
worst tuning pairs of two kinds:

- **Missed fits:** pairs with a rating of 2 or more that the student ranks lowest among the rated
  pairs of their feeling.
- **False fits:** pairs with a rating of 0.5 or less that the student ranks highest in their
  feeling.

Each entry gives the feeling, the work, the rating, the student's similarity and its rank, and the
work's vibe and description. Use it to see if a bad description causes the error.

The ship gate reads the tuning set only. It needs these checks to pass:

- The student's tuning objective is at least 0.05 above the tuning objective of the untrained
  MiniLM. This check catches a broken student: a trained one leads by about 0.09.
- The student's tuning objective is no more than 0.01 below the installed bundle in
  `../static/bundle/`. Eval runs the installed bundle the same way as the new student. This check
  uses only the tuning pairs whose works are in both the installed bundle and the current
  catalog. When no installed bundle exists, eval skips this check and the report says so.

The report gives the result, the reason for each failed check, and the numbers of each check. A
change to the installed bundle runs eval again.

`eval_feelings.jsonl` holds one JSON object per line, like
`{ "text": "a snowy december and i just made warm hot chocolate", "set": "scene" }`.
It has 290 scene feelings, 80 casual first-person feelings, and 70 each of idioms and metaphors,
sarcasm and irony lines, slang and typo lines, mixed feelings, and heavy feelings such as grief,
envy, and shame. The report also gives the student's fidelity for each set: the share of the
teacher's top 10 per category that the student also ranks in its top 10. These feelings never go
into training. If you add some, run `uv run label` again to label the new lines, then
`uv run train` to update the report.

## publish

`publish` uploads `../static/bundle/` to the R2 bucket `mise`, served at `cdn.mise.art`.

- Images go to `img/<hash>.webp`, so an image uploads once and every release shares it.
- All other files go to `bundles/<date>-<hash>/`. The prefix hash covers the file contents, so a
  release never changes after upload. Files are cached for a year.
- The manifest uploads last, so a release is never half there.
- The sample files hold copies of items, so their image links change to the public URLs too.
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
| `out/eval_failures.md` | the student's worst tuning pairs                                 |
| `out/data_audit.json`  | the word shares per category and the overused words              |
| `out/run.json`         | the seeds, package versions, GPU, model revisions, and file hashes |

Git ignores `data/` and `out/`.

## Reproducibility

- Every step writes sorted output, and the API cache makes `curate` repeatable. To take newer
  data, delete the cache folder for that host.
- Training uses fixed seeds and PyTorch's deterministic mode.
- The items job samples at temperature 0.7 and top-p 0.95, with a separate seed for each work.
  Other jobs decode greedily on the first pass. Retries sample with a seed from the key. Fixed
  seeds make the random choices repeatable; hardware and batch rounding can still change output.

## Configuration

Every setting is a dataclass in `src/mise_ml/config.py`. Change a value there to change a run. The
list of lights and typefaces is `../scripts/stub/vocab.json`.

## Licenses

The Met, Art Institute of Chicago, and Cleveland Museum of Art images are public domain or CC0.
Smithsonian images are CC0. NASA images follow the
[NASA media guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/).
NASA is credited as the image source. Feelings and descriptions generated by mise are not NASA
content. The app does not use NASA logos.
ListenBrainz and MusicBrainz data is CC0. The PAT palettes come from Bahng et al., "Coloring with
Words", ECCV 2018. TMDB posters, book covers, and Deezer album art and previews belong to their
owners. The app shows them next to links to their sources and credits each one on its attribution
page.
