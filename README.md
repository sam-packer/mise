# mise

Describe a feeling in a sentence. The page takes its colors, light, and typeface, and shows a small wall
to match: an artwork, a film, a song, a poem, a book, and a scent. The model runs in the browser.

## Run the app

```sh
bun install
bun run stub    # build the stub bundle into static/bundle/ (about 3 min the first time)
bun run dev
```

The app loads its model and catalog from `static/bundle/`. That folder is not committed, so build it
first: either the stub above, or the trained model below.

Try a feeling in the terminal: `bun run query "a dark and stormy night"`.

## Train the real model

You need an NVIDIA GPU with about 32 GB (an RTX 5090) and about 60 GB of free disk. No API keys.

```sh
cd ml
uv sync
uv run mise-ml all
```

`all` downloads the datasets, labels them with a local LLM, trains, evaluates, and installs the result
into `static/bundle/`. It runs overnight (an estimate). Stop it at any time: a rerun skips finished work.
`uv run mise-ml` lists each step. [ml/README.md](ml/README.md) explains them.

Before the first run, add about 300 feelings to `ml/eval_feelings.jsonl`, one sentence per line. The eval
step measures the model against them.

## Deploy

In production the bundle lives in a Cloudflare R2 bucket, not `static/bundle/`. Publish it with
`cd ml && uv run mise-ml publish`, then set `PUBLIC_BUNDLE_URL` in `wrangler.jsonc` to the URL it
prints. Then run:

```sh
bun run deploy
```

It builds the site and deploys it to Cloudflare Workers. The bundle no longer needs to be in the
build, so a plain push works too.

## Layout

| Path       | Contents                                         |
| ---------- | ------------------------------------------------ |
| `src/`     | the SvelteKit app                                |
| `scripts/` | the stub bundle build, from hand-curated sources |
| `ml/`      | the training pipeline (`mise-ml`)                |
| `docs/`    | the design spec                                  |

## Credit

The idea of turning a described feeling into picks across media was sparked by
[Wave](https://github.com/SophiaYifei/wave-recsys). mise shares no code or design with it.
