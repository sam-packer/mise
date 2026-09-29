# mise

Describe a feeling in a sentence. The page takes its colors, light, and typeface, and shows a small wall
to match: an artwork, a film, a song, a poem, a book, and a scent. The model runs in the browser.
The Worker searches a private catalog through the `CATALOG` R2 binding. The browser sends the
query and its embedding to `POST /api/match`. The server returns only the selected items.

## Run the app

```sh
bun install
bun run stub
bun run dev
```

A fresh clone needs a public model and a matching catalog in local R2 before search works.
Run `bun run stub` for a local sample. This needs Bun and network access, but no Python.
The stub writes to `ml/out/stub/bundle/` and `ml/out/stub/catalog/`. It leaves the trained
export in `ml/out/bundle/` and `ml/out/catalog/` unchanged. It copies its public files to
`static/bundle/`, puts its catalog in local R2, and sets `CATALOG_PREFIX` in `.dev.vars`.
For a trained export, run `uv run train` from `ml/` instead.
Set `PUBLIC_BUNDLE_URL=/bundle/` in the dev server environment to use the local public files.

Git ignores `.dev.vars`. Install keeps its other settings and does not change
`src/lib/server/catalog.ts`. Only publish writes the published prefix to that file.
Publish also copies the exact published catalog to local R2 and selects it in `.dev.vars`.
To return from an installed local catalog to the published catalog, remove the
`CATALOG_PREFIX` line from `.dev.vars`. Remove `PUBLIC_BUNDLE_URL` from the dev server
environment to use the published public model. The published catalog must already be in
local R2, for example from a publish on this machine. A clone does not contain R2 data.

The Cloudflare adapter in `vite.config.ts` loads `.dev.vars` through Wrangler and supplies
the values in `platform.env` during `bun run dev`. Vite and `wrangler dev` use the same
`.wrangler/state` storage. Restart the dev server after you change the catalog selection.
The server returns HTTP 409 if the public model version and private catalog version differ.

Try a feeling in the terminal: `bun run query "a dark and stormy night"`.

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
Run `publish` separately to upload the installed bundle and catalog to R2. It updates
`src/lib/bundle.ts` and `src/lib/server/catalog.ts` to select the release.
The full pipeline can run overnight. Stop at any time; a rerun skips finished work.
Use `--help` with any command. See [ml/README.md](ml/README.md) for details.

Before the first run, add about 300 feelings to `ml/eval_feelings.jsonl`, one sentence per line. The eval
step measures the model against them.

After publish, commit `src/lib/bundle.ts` and `src/lib/server/catalog.ts`. Then run
`bun run deploy`. Keep this release order.

## Deploy

```sh
bun run deploy
```

It builds the site and deploys it to Cloudflare Workers. This ships the app only: the bundle lives
in public R2. The catalog lives in a separate private R2 bucket.

- Public: `ml/out/bundle/` contains `manifest.json`, `model/`, `vocab.json`, and `img/`.
- Private: `ml/out/catalog/` contains `catalog.json`, `items.json`, and `vectors.bin`.
  The stub also has `anchors.json` and `anchors.bin`. Never put these files in `static/`.

Create the private bucket once with `bunx wrangler r2 bucket create mise-catalog`. Do not
connect a public domain or enable public access. Set `R2_CATALOG_BUCKET` in `ml/.env`
(default `mise-catalog`). If you use another name, set the same name for `CATALOG` in
`wrangler.jsonc`. The R2 API token needs Object Read & Write on both buckets. See
[ml/README.md](ml/README.md#publish) for the public bucket settings and publish steps.

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
