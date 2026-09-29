# Vectorize verification, 2026-09-29

The 99% top-1 gate passed before the API was connected to Vectorize.

## Live data and design

- Wrangler: installed version 4.142.0, authenticated with the existing login.
- Index: `mise-catalog`, 384 dimensions, cosine distance.
- Namespace: `catalog/2026-09-29-e6a24991/`.
- Vector IDs: namespace plus the original catalog row. IDs cannot collide across versions.
- Records: all 10,993 published R2 records, with CDN image URLs, in Vectorize metadata.
- Metadata indexes: `category` (string), `creator` and `title` (number).
- Exclusion keys: the first catalog row for each exact creator/title string. No string
  truncation or hash collision affects exclusions.
- Largest complete metadata record: 1,775 UTF-8 bytes, for the poem
  `poem:algernon-charles-swinbur-nephelidia`. The limit is 10,240 bytes.
- Private R2 files added: `search-index.json` and `vectorize-v1.json` under that namespace.

The original published R2 catalog metadata matches the local metadata. The published
vectors are byte-identical to `ml/out/catalog/vectors.bin`. The three original private
files have content hash `e6a24991`. No original R2 files were changed.

The compact anchor file is 968,434 bytes. It preserves the common-word list and catalog
order. It contains the title, creator and album fields, and album/creator representative
rows calculated with the original JavaScript Float32 accumulation. Local Bun JSON
parsing, over 100 measured iterations after one warmup, took 2.91 ms median and 3.58 ms
at the 95th percentile. These are local parse times, not R2 or Worker latency.

## Agreement

The saved query embeddings are teacher embeddings with 4,096 dimensions. They cannot
query this 384-dimensional index. The verification used 200 item vectors sampled without
replacement, with NumPy seed 1337. It added Gaussian noise with standard deviation 0.03
per component and normalized each vector. Each category query requested 50 candidates
with values and full metadata. The verifier checked returned Float32 values against
the source, then reranked by the exact dot product and catalog row. The reference ranks
every item in the category with NumPy float64 accumulation over the Float32 values.

| Category | Top-1 agreement | Top-5 overlap | Exact ordered top-5 | Top-10 overlap |
| --- | ---: | ---: | ---: | ---: |
| Art | 100% | 100% | 100% | 99.90% |
| Book | 100% | 100% | 100% | 100% |
| Film | 100% | 100% | 100% | 100% |
| Poem | 100% | 100% | 100% | 99.95% |
| Song | 100% | 99.80% | 99% | 99.75% |

Overlap is the mean share of the exact top-k found in the reranked top-k. Exact ordered
top-5 requires all five rows in the same order. Lower-rank differences come from items
omitted by approximate candidate retrieval. Reranking cannot recover an omitted item.
The user explicitly authorized approximate retrieval subject to the top-1 gate. All
five categories exceeded that gate; this measurement does not prove universal parity.
All eight missing top-10 rows were absent from the complete 50-candidate sets: two art,
one poem, and five song rows. There were no reranking-order errors among those candidates.

Three additional queries used unmodified rows 0, 4,000 and 9,000. All 15 per-category
top results matched the exact reference. Wrangler returned `count` and `matches`; it
did not print a service query latency. Measured CLI process times, including process
startup, authentication, network and JSON output, were:

| Source row | Per-category CLI time range |
| --- | ---: |
| 0 | 0.958–1.382 s |
| 4,000 | 0.986–1.093 s |
| 9,000 | 0.891–1.299 s |

Do not compare these process times with the production Worker request time. After a
deployment, use the API's `Server-Timing` header to measure `catalog` and `search`.

## API and publish verification

A one-shot SvelteKit request dispatcher exercised the built `/api/match` route against
real private R2 and Vectorize through Wrangler. It did not listen on a port. Full JSON
responses matched the original search for an ordinary query, title+creator, creator,
album+creator, song title+creator, a common word, and a zero embedding. The checks also
covered input validation, version mismatch (409), and both timing fields. Across those
requests, the Vectorize path read only `search-index.json` once from R2.

The actual stub builder ran with output redirected to the scratchpad. It built 374 items
and 492 anchors and passed its validation. The built API's fallback response matched the
original search, including all anchors-kind heads. The installed bundle was not replaced.

The publisher reads every uploaded vector and record before marking a namespace complete.
Live verification established two API details: `getByIds` accepts at most 20 IDs, and
returned JSON decimal strings must be compared at Float32 precision. The verifier uses
those rules. A completed namespace is read and verified on rerun rather than uploaded again.

Local tools select the `local` Wrangler environment, which has no Vectorize binding.
This preserves offline search for unpublished installed models and the stub. Production
uses the Vectorize binding. Both implementations share anchor rules, representatives,
blending and head calculation through one catalog interface.

The scripts and raw results are in the session scratchpad's `vectorize-check` directory:
`prepare.py`, `check.py`, `report.json`, `exact-*.json`, `measure.ts`, `api-check.ts`,
`verify-upload.py`, and `reuse.py`. No unit tests were added.

The real publish helper's reuse path passed. It verified the existing namespace and
returned the identical compact file without upserting vectors. During verification,
Cloudflare rejected one read token with code 10000. A fresh Wrangler read succeeded.
The helper now retries that read with a fresh Wrangler process, up to three attempts;
it does not change credentials or permissions. The retry path also passed in the reuse run.

## Required checks

- Passed: `bun run check` (zero errors and warnings), `bunx eslint src scripts`,
  `bunx prettier --check src scripts`, and `bun run build`.
- Passed: Ruff lint and formatting for `delivery.py`, `publish.py`, and `vectorize.py`.
- Fixed by the concurrent agent: the two long expressions in `resolve.py`. The final
  complete-tree `uv run ruff check` and `uv run ruff format --check` both passed, with
  31 Python files formatted. This task did not edit `resolve.py`.

## Operations

No deployment or training command ran. The current catalog is ready for a later deployment.
The existing Wrangler login worked; no new token is required. See
[the one-time setup and publish instructions](README.md#vectorize-catalog-search).

## Cloudflare references

- [Index creation](https://developers.cloudflare.com/vectorize/best-practices/create-indexes/)
- [NDJSON uploads and namespaces](https://developers.cloudflare.com/vectorize/best-practices/insert-vectors/)
- [Query options and getByIds](https://developers.cloudflare.com/vectorize/reference/client-api/)
- [Scoring precision](https://developers.cloudflare.com/vectorize/best-practices/query-vectors/)
- [Metadata filtering](https://developers.cloudflare.com/vectorize/reference/metadata-filtering/)
- [Limits](https://developers.cloudflare.com/vectorize/platform/limits/)
- [Wrangler commands](https://developers.cloudflare.com/vectorize/reference/wrangler-commands/)
- [Binding configuration](https://developers.cloudflare.com/workers/wrangler/configuration/#vectorize-indexes)
