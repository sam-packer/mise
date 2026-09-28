# mise

Describe a feeling in a sentence. The page takes its colors, light, and typeface, and shows a small wall
to match: an artwork, a film, a song, a poem, a book, and a scent. The model runs in the browser.

## Run the app

```sh
bun install
bun run dev
```

This loads the published bundle from `cdn.mise.art`, once one is published (see "Train" below). To
use an offline stub instead, run `bun run stub` (builds it into `static/bundle/`, about 3 min the
first time) and set `PUBLIC_BUNDLE_URL=/bundle/` in `.env`.

Try a feeling in the terminal: `bun run query "a dark and stormy night"`.

## Train the real model

You need an NVIDIA GPU with about 32 GB (an RTX 5090), about 60 GB of free disk, and keys for TMDB,
Hardcover, ListenBrainz, and Last.fm in `ml/.env` (see `ml/.env.example`).

```sh
cd ml
uv sync
uv run mise-ml all
```

`all` downloads the datasets, labels them with a local LLM, trains, evaluates, and installs the
result locally. If Cloudflare R2 is configured (see "Publish" in [ml/README.md](ml/README.md)),
it also publishes the bundle there and updates `src/lib/bundle.ts` to point at it. It runs
overnight (an estimate). Stop it at any time: a rerun skips finished work. `uv run mise-ml` lists
each step; [ml/README.md](ml/README.md) explains them.

Before the first run, add about 300 feelings to `ml/eval_feelings.jsonl`, one sentence per line. The eval
step measures the model against them.

Commit the updated `src/lib/bundle.ts`, then deploy (see "Deploy" below) or push.

## Deploy

```sh
bun run deploy
```

It builds the site and deploys it to Cloudflare Workers. This ships the app only: the bundle lives
in a Cloudflare R2 bucket, not in the build.

## Layout

| Path       | Contents                                         |
| ---------- | ------------------------------------------------ |
| `src/`     | the SvelteKit app                                |
| `scripts/` | the stub bundle build, from hand-curated sources |
| `ml/`      | the training pipeline (`mise-ml`)                |

## Credit

The idea of turning a described feeling into picks across media was sparked by
[Wave](https://github.com/SophiaYifei/wave-recsys). mise shares no code or design with it.
