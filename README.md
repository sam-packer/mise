# mise

Describe a feeling in a sentence. The page takes its colors, light, and typeface, and shows a small wall
to match: an artwork, a film, a song, a poem, a book, and a scent. The model runs in the browser.
Search runs in the browser worker. It loads the public catalog once and scans every item locally.

## Run the app

```powershell
bun install
bun run stub
$env:PUBLIC_BUNDLE_URL = '/bundle/'
bun run dev
```

Run `bun run stub` for a local sample. It needs Bun and network access, but no Python.
The stub writes to `ml/out/stub/bundle/` and copies the bundle to `static/bundle/`.
It keeps the trained export in `ml/out/bundle/`.
For a trained export, run `uv run train` from `ml/`.
Use `PUBLIC_BUNDLE_URL=/bundle/` to select local files. Without an override, the app uses
`src/lib/bundle.ts` to select the published bundle.

Try a feeling in the terminal: `bun run query "a dark and stormy night"`.
The command uses the same engine as the browser with the installed local bundle.

## Train the real model

You need an NVIDIA GPU with about 32 GB (an RTX 5090), about 60 GB of free disk, and keys for TMDB,
Hardcover, ListenBrainz, and Last.fm in `ml/.env` (see `ml/.env.example`).

```sh
cd ml
uv sync
uv run download
uv run label
uv run train
uv run publish
```

Run `download` to fetch and resolve the catalog. Run `label` for the local LLM pass.
Run `train` to train, export, evaluate, and install only when the ship gate passes.
Run `publish` separately to upload the installed public bundle to R2. It updates
`src/lib/bundle.ts` to select the release.
The full pipeline can run overnight. Stop at any time; a rerun skips finished work.
Use `--help` with any command. See [ml/README.md](ml/README.md) for details.

Before the first run, add about 300 feelings to `ml/eval_feelings.jsonl`, one sentence per line. The eval
step measures the model against them.

After publish, commit `src/lib/bundle.ts`. Then run
`bun run deploy`. Keep this release order.

## Deploy

```sh
bun run deploy
```

It builds the site and deploys the app to Cloudflare Workers. The public bundle lives in the
`mise` R2 bucket, served at `cdn.mise.art`. See [ml/README.md](ml/README.md#publish).

The bundle contains the ONNX model, tokenizer, vocab, images, item records, fp16 item vectors,
and compact name data. Its manifest lists all files and formats. The worker decodes vectors
once to float32. Each search runs the encoder, heads, name matching, and an exact catalog scan.
Named works use a 70% item / 30% query blend and exclude the same title and creator from picks.
The stub also includes palette and label anchors for its heads.

A link to a feeling looks like `mise.art/<code>`. The Worker keeps each code's feeling in the KV
namespace `MOODS`. To make a new one, run `bunx wrangler kv namespace create moods` and put its id in
`wrangler.jsonc`.

## Layout

| Path       | Contents                                         |
| ---------- | ------------------------------------------------ |
| `src/`     | the SvelteKit app                                |
| `scripts/` | the stub bundle build, from hand-curated sources |
| `ml/`      | the training pipeline                            |

## Credit

The idea of turning a described feeling into picks across media was sparked by
[Wave](https://github.com/SophiaYifei/wave-recsys). mise shares no code or design with it.
