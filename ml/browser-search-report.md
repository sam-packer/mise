# Browser search verification

Measured on 2026-09-29 on Windows with Bun 1.4.2 and an AMD Ryzen 9 9950X3D.
The catalog contains 10,993 items with 384 dimensions.

## Agreement

The reference is the float32 exact search from commit `e9bbca0`, including its anchor
matcher, blend, exclusions, and representative selection. The candidate runs the shared
browser search code with fp16 vectors decoded to float32. Decoded vectors are not normalized again.

The research sample contains 1,000 real validation texts, selected with seed 20260929.
The saved embeddings were computed by the exported student, not the teacher.

| Category | Saved student embeddings | Fresh browser-engine encoding |
| -------- | ------------------------ | ----------------------------- |
| Art      | 1,000 / 1,000            | 1,000 / 1,000                 |
| Film     | 1,000 / 1,000            | 1,000 / 1,000                 |
| Song     | 1,000 / 1,000            | 1,000 / 1,000                 |
| Poem     | 1,000 / 1,000            | 1,000 / 1,000                 |
| Book     | 999 / 1,000              | 1,000 / 1,000                 |
| Overall  | 99.98%                   | 100%                          |

Fresh encoding uses `createMoodEngine` with the native ONNX runtime and one CPU thread.
Each text runs alone. Both sides of each comparison use the same query embedding.
These results measure the effect of item-vector rounding, not model relevance.

All 176 additional title, creator, title-plus-creator, and album queries matched the
reference anchor and all five picks. Of these, 166 selected an anchor. Anchor agreement
was also 100% on the 1,000 feeling queries. The zero-vector result matched, including
category order and row-order ties. Representatives are selected from float32 vectors
at export, before rounding.

## Timing

The engine received preloaded file bytes. These measurements exclude downloads and disk reads.

| Operation                                                                      | Time                     |
| ------------------------------------------------------------------------------ | ------------------------ |
| Complete engine load, including ONNX session, JSON parsing, names, and vectors | 176.1 ms                 |
| fp16 decode alone                                                              | 40.4 ms                  |
| Complete inference, 1,000 real texts                                           | 7.14 ms p50; 8.82 ms p95 |
| Scan a decoded float32 buffer                                                  | 8.47 ms p50              |
| Scan fp16 directly with DataView reads                                         | 28.04 ms p50             |

The two scan variants use the same loop and query. They have 20 warmups and 100 timed runs.
Their generic benchmark loop is separate from the optimized category scan in the engine.
Decode once was selected. Bun uses JavaScriptCore and native ONNX here; these times are
not mobile-browser or WebAssembly performance claims.

## Public search files

| File                | Format                                    |      Bytes |    MiB |
| ------------------- | ----------------------------------------- | ---------: | -----: |
| `vectors.bin`       | IEEE fp16, little-endian, row-major       |  8,442,624 |  8.052 |
| `items.json`        | Compact UTF-8 JSON                        |  6,213,906 |  5.926 |
| `search-index.json` | Common name words and representative rows |    134,433 |  0.128 |
| Total               |                                           | 14,790,963 | 14.106 |

These sizes exclude the model, tokenizer, vocab, images, and manifest. Publish rewrites
image paths to CDN URLs, so the published record size depends on the configured domain.

## Release resources

The deployment dry run lists only `MOODS` KV and `ASSETS`. After the owner publishes the
new bundle, deploys this app, and verifies the release, these configured resources will
have no caller in this app:

- R2 bucket `mise-catalog`, including its versioned catalog files.
- Vectorize index `mise-catalog`, including its namespaces and category, creator, and title metadata indexes.

Keep the public R2 bucket `mise`, its `cdn.mise.art` domain, the app Worker, and `MOODS` KV.
No deployment, upload, or Cloudflare resource deletion was performed for this change.
Inspect live resources and active releases before approving deletion.

## Checks

- `bun run stub`: 374 items and 492 head anchors; validation passed.
- Real trained-model export packaging and local install passed, without running training.
- Chromium with the real worker and WASM: five picks, anchor display, hint spacing, Tab focus,
  Esc focus return, and stale-query protection passed. Catalog files loaded once. No
  `/api/match` request occurred. The harness served the installed ORT files in place of CDN downloads.
- Both edited Svelte files passed the autofixer CLI with no issues or suggestions.
- Svelte check, ESLint, Prettier, Ruff check, Ruff format check, build, and deployment dry run passed.

The dry run emits an `ignored-bare-import` warning for SvelteKit's generated `import "devalue"`.
The installed devalue package declares `sideEffects: false`, so the bundler drops that unused import.
The build and browser checks pass. Changing generated framework code is not a durable fix.
