# Catalog expansion on `world`

The new target is **15,993 works**, up from 10,993 (+5,000, 45.5%).
These are selection measurements. No real pipeline stage or tests ran. The owner will run
resolve, label, and training overnight. The measured song candidates include the existing
1.4 replacement factor; they are not a claim that all media has resolved.

## Changes

- Increase film and book targets to 3,000 each, song to 5,000, and art to 3,000.
- Add 29 distinct scene tags beside the 14 older tags; fetch up to 200 artists per tag.
- Cap songs at three per artist and preserve scene reserves through resolve.
- Split the 2000s and 2010s; balance modern quotas against measured listener cutoffs.
- Add NASA and Smithsonian selectors, image resolution, attribution, and art fallback quotas.
- Keep the ASCII grammar, original-release fact repairs, and error-tolerant lookups.

## Quotas

| Category | Era or source | Target | Candidate factor |
| --- | --- | ---: | ---: |
| film | 1920-1969 | 450 | 1.05 |
| film | 1970-1999 | 900 | 1.05 |
| film | 2000-2019 | 1,050 | 1.05 |
| film | 2020-2026 | 600 | 1.05 |
| book | -5000-1949 | 450 | 1.1 |
| book | 1950-1999 | 750 | 1.1 |
| book | 2000-2019 | 1,050 | 1.1 |
| book | 2020-2026 | 750 | 1.1 |
| song | 1900-1979 | 800 | 1.4 |
| song | 1980-1999 | 1,200 | 1.4 |
| song | 2000-2009 | 2,050 | 1.4 |
| song | 2010-2019 | 825 | 1.4 |
| song | 2020-2026 | 125 | 1.4 |
| art | Met | 1,200 | 1.25 |
| art | Chicago | 500 | 1.25 |
| art | Cleveland | 300 | 1.25 |
| art | NASA | 500 | 1.25 |
| art | Smithsonian | 500 | 1.25 |
| poem | PoetryDB, 8-40 lines | 1,993 | all eligible |

The Met split remains 40% paintings and 20% each drawings, prints, and photographs.
Missing Smithsonian credentials move 500 slots to Met +300, Chicago +125, Cleveland +75.
Source outages or short results use the same 60/25/15 fallback. Effective group quotas travel
with the art records so resolve uses the same totals.

## Song measurements

Discovery returned 6,523 artist IDs. Of those, 6,387 supplied recordings, for a raw pool of
1,834,416 recordings. Selection then applied the version filter, identity caps, dates, scene
reserves, and era quotas. The nominal 5,000-song selection contains 2,321 artist names before
media checks. The table below includes all replacement candidates.

| Measure | Before | After |
| --- | ---: | ---: |
| Candidates | 4,200 | 7,000 |
| Distinct artist names | 1,330 | 3,191 |
| Distinct artist IDs | 1,154 | 2,986 |

| Songs per artist | Artists before | Artists after |
| ---: | ---: | ---: |
| 1 | 504 | 1,044 |
| 2 | 170 | 485 |
| 3 | 101 | 1,662 |
| 4 | 91 | 0 |
| 5 | 95 | 0 |
| 6 | 369 | 0 |

### Era cutoffs

The last-candidate column is the listener count of the final candidate in selection order.
The minimum column also includes the scene reserves. Small scenes can have a lower minimum
than the main listener cutoff. Counts come from the same cached API snapshot used for selection.

| Selection | Era | Target | Candidates | Last candidate listeners | Minimum with scene reserves |
| --- | --- | ---: | ---: | ---: | ---: |
| Before | 1900-1979 | 450 | 630 | 13,110 | 13,110 |
| Before | 1980-1999 | 750 | 1,050 | 26,288 | 26,288 |
| Before | 2000-2019 | 1,050 | 1,470 | 42,294 | 42,294 |
| Before | 2020-2026 | 750 | 1,050 | 876 | 876 |
| After | 1900-1979 | 800 | 1,120 | 5,843 | 4 |
| After | 1980-1999 | 1,200 | 1,680 | 7,432 | 74 |
| After | 2000-2009 | 2,050 | 2,870 | 3,494 | 1 |
| After | 2010-2019 | 825 | 1,155 | 3,576 | 13 |
| After | 2020-2026 | 125 | 175 | 3,554 | 3 |

The fixed modern quotas were chosen in increments of 25 against this measured pool, while
keeping 800 older and 1,200 1980s/1990s slots. The highest modern candidate cutoff is only
2.3% above the lowest. This puts more slots in the 2000s, which have many more high-listener
recordings, and reduces the 2020s target to 125. These are fixed config values, not a quota
search during download. No artist gets a separate weight.

The current resolved catalog has 190 of 3,000 songs from the 2010s (6.3%). The new era target
is 825 of 5,000 (16.5%). Each decade from the 1920s through the 2020s has candidates. The
capped, dated pool contains no songs from 1900-1919; no dates or recordings were invented.

The reserve is 14 candidates per available discovery group in the first four eras and two
in the smaller 2020s era, or all available candidates when a group has fewer. Selection ranks
by listeners within each reserve, then fills the remaining slots by listeners with a year cap.

### Discovery-tag shares

Each song has one discovery tag. Before rows are assigned the same tag mapping for a fair
comparison. Match by artist ID, then normalized name when the services use different IDs.
`sitewide` means no discovery-tag match. These groups overlap in musical meaning, but each
song counts once. They are not a claim that each artist belongs to only one genre.

| Discovery tag | Before songs | Before share | After songs | After share |
| --- | ---: | ---: | ---: | ---: |
| 50s | 28 | 0.67% | 98 | 1.40% |
| 60s | 143 | 3.40% | 219 | 3.13% |
| 70s | 115 | 2.74% | 206 | 2.94% |
| afrobeats | 1 | 0.02% | 23 | 0.33% |
| alternative r&b | 0 | 0.00% | 18 | 0.26% |
| alternative rock | 294 | 7.00% | 395 | 5.64% |
| ambient | 41 | 0.98% | 175 | 2.50% |
| americana | 0 | 0.00% | 110 | 1.57% |
| art pop | 33 | 0.79% | 54 | 0.77% |
| bedroom pop | 25 | 0.60% | 29 | 0.41% |
| blues | 74 | 1.76% | 182 | 2.60% |
| bossa nova | 0 | 0.00% | 88 | 1.26% |
| city pop | 0 | 0.00% | 57 | 0.81% |
| classic rock | 297 | 7.07% | 300 | 4.29% |
| classical | 2 | 0.05% | 93 | 1.33% |
| country | 10 | 0.24% | 136 | 1.94% |
| disco | 20 | 0.48% | 82 | 1.17% |
| doo wop | 1 | 0.02% | 59 | 0.84% |
| dream pop | 37 | 0.88% | 81 | 1.16% |
| electronic | 269 | 6.40% | 248 | 3.54% |
| folk | 202 | 4.81% | 251 | 3.59% |
| funk | 49 | 1.17% | 107 | 1.53% |
| hip-hop | 171 | 4.07% | 258 | 3.69% |
| indie | 353 | 8.40% | 279 | 3.99% |
| indie pop | 107 | 2.55% | 109 | 1.56% |
| indie rock | 115 | 2.74% | 135 | 1.93% |
| j-pop | 32 | 0.76% | 93 | 1.33% |
| jazz | 85 | 2.02% | 193 | 2.76% |
| k-pop | 112 | 2.67% | 54 | 0.77% |
| latin | 25 | 0.60% | 142 | 2.03% |
| lo-fi | 14 | 0.33% | 127 | 1.81% |
| motown | 28 | 0.67% | 85 | 1.21% |
| neo-soul | 12 | 0.29% | 97 | 1.39% |
| oldies | 48 | 1.14% | 88 | 1.26% |
| post-punk | 92 | 2.19% | 184 | 2.63% |
| reggae | 13 | 0.31% | 204 | 2.91% |
| rock and roll | 6 | 0.14% | 63 | 0.90% |
| shoegaze | 32 | 0.76% | 88 | 1.26% |
| singer-songwriter | 79 | 1.88% | 181 | 2.59% |
| sitewide | 998 | 23.76% | 846 | 12.09% |
| soul | 142 | 3.38% | 190 | 2.71% |
| soundtrack | 13 | 0.31% | 135 | 1.93% |
| synthpop | 30 | 0.71% | 138 | 1.97% |
| trip-hop | 52 | 1.24% | 300 | 4.29% |

### Blood Orange

Blood Orange is in the generic pool. Last.fm supplies artist ID
`c74877e0-58e0-41dd-a022-29361095d753`, which has no recordings in ListenBrainz.
ListenBrainz sitewide stats use `3d254f1e-7d18-431b-baeb-68e57695dfdb`.
The generic name fallback preserves the `indie` discovery tag across those IDs.
No artist name, ID, allowlist, or personal library appears in the selection rules.

| Song | Candidate year | Listeners | Listener rank in decade pool | Selection rank in era |
| --- | ---: | ---: | ---: | ---: |
| Sutphin Boulevard | 2011 | 22,149 | 456 | 804 |
| Champagne Coast | 2011 | 15,590 | 581 | 890 |

Candidate years are MusicBrainz recording metadata. Resolve still checks original release facts.
Sutphin Boulevard falls within the nominal 825-slot target. Champagne Coast is a replacement
candidate. The capped pool also contains Forget It (11,933 listeners); the year spread excludes
it from the final candidates. The same rules apply to every artist.

## Art smoke calls

Each small selector call requested 8 slots and returned 10 candidates. Each full selector
returned 625 candidates for 500 slots. The Smithsonian key was present. The missing-key call
returned `[]` with one warning. A small `select_art` call without the key retained all 56 target
slots: Met 29, Chicago 12, Cleveland 7, NASA 8.
The download credential preflight also passed with the Smithsonian key removed from the scratch
process. It still requires the four original service keys.

The NASA filter was tightened after metadata exposed cargo-handling and aircraft images in
nature searches. It excludes explicit third-party copyright, technical, staff, ceremony,
logo, and test terms. Smithsonian images require per-image CC0 and exclude portrait metadata.
The credit line is retained in `source.credit`.

### NASA

| Title | Image URL |
| --- | --- |
| Martian Aurora 25 Times Brighter Than Prior Brightest | [image](https://images-assets.nasa.gov/image/PIA21857/PIA21857~medium.jpg) |
| Earth Observation taken during the Expedition 37 mission | [image](https://images-assets.nasa.gov/image/iss037e004654/iss037e004654~medium.jpg) |
| An orbital sunset silhouettes the cloud tops above the Pacific Ocean | [image](https://images-assets.nasa.gov/image/iss074e0046371/iss074e0046371~medium.jpg) |
| The Large Magellanic Cloud and the Small Magellanic Cloud | [image](https://images-assets.nasa.gov/image/iss071e418742/iss071e418742~medium.jpg) |
| The 2001 Great Dust Storms - Tharsis | [image](https://images-assets.nasa.gov/image/PIA03171/PIA03171~medium.jpg) |

### Smithsonian

| Title | Image URL |
| --- | --- |
| Oil Pumping Station, near Athens, for the Lehigh Valley Railroad | [image](https://ids.si.edu/ids/deliveryService?id=SAAM-1994.91.150_1&max=1200) |
| Untitled (French Landscape) | [image](https://ids.si.edu/ids/deliveryService?id=NMAAHC-2010_51_1a_002-000001&max=1200) |
| Irish Cottage | [image](https://ids.si.edu/ids/deliveryService?id=SAAM-1966.31.8_2&max=1200) |
| Roses | [image](https://ids.si.edu/ids/deliveryService?id=SAAM-1909.7.60_2&max=1200) |
| A Tree Study | [image](https://ids.si.edu/ids/deliveryService?id=SAAM-1930.12.55_1&max=1200) |

Successful image downloads through `CachedClient`:

```text
IMAGE NASA Martian Aurora 25 Times Brighter Than Prior Brightest 101016 JPEG (1032, 1280)
IMAGE NASA Earth Observation taken during the Expedition 37 mission 138092 JPEG (1280, 851)
IMAGE SI Oil Pumping Station, near Athens, for the Lehigh Valley Railroad 183072 JPEG (1200, 1010)
IMAGE SI Untitled (French Landscape) 132494 JPEG (1200, 963)
```

Smithsonian IDS received `User-Agent: mise-catalog/0.1`. All four responses decoded as JPEGs.
NASA attribution includes the statement that mise generates its feelings and descriptions;
these are not NASA content. See the [NASA media guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/).
No NASA logos were added.

## Size and time

- Vectors: 15,993 x 384 x 2 = 12,282,624 bytes (11.71 MiB), up 3.66 MiB.
- Item JSON: about 8.62 MiB, scaled from the current 6,213,906 bytes for 10,993 works.
- Model and tokenizer: about 23.98 MiB, independent of catalog size. The 24 MiB export cap
  applies only to the ONNX model.
- Total bundle with images: about 908 MiB, up from 589 MiB. This uses current average WebP
  sizes by category; NASA and Smithsonian images can change the actual result.
- Added item labeling: about 30 minutes for 5,000 net new works at the latest bulk rate
  (3,518 requests in 21m 14s; profile step 21m 28s in `20260928-185144-all.log`). The older
  1,744-request run took 26m 15s, which would imply about 75 minutes.
- The current plan marks the old item signatures stale too. A full 15,993-item pass would take
  about 97 minutes at that bulk rate. Changed membership and repaired facts can also change item prompts.
  The 30-minute estimate is added item work, not a promise for the whole overnight run.
- The 30,000 label-query and 40,000 distillation targets remain fixed. Their current plans show
  regeneration from prior signature changes. Judge membership must be ranked after training.

Export sizes vectors and manifest counts from the actual catalog. Browser search validates
vectors against `items.length * dims`. The judge takes 20 results per category per system from
actual rankings and records catalog-dependent provenance. Eval and resolve use current rows
and group targets. There is no fixed 11,000-item cap in these paths.

## Verification

- Fixed: duplicate artist IDs could exceed the name cap. Both limits now apply; no candidate
  artist name supplies more than three songs.
- Fixed: non-Latin titles could collapse to an empty key. Title deduplication keeps their letters.
- Fixed: 357 recordings shared 51 slugs. Recording-ID suffixes keep those works separate.
- Fixed: saved resolved rows could retain old era and priority fields. Resolve refreshes those
  fields while it keeps repaired facts and images for the same work.
- Fixed: NASA nature queries could return technical photos. The metadata filter excludes them.
- Fixed: the CLI preflight treated the optional key as required. The missing-key preflight passes.
- Fixed: museum quota rounding could change the fallback total. The last Met class takes the
  remainder; the missing-key smoke call preserves all 56 slots.

The final source measurements, image calls, missing-key preflight, and checks below cover these
changes. No failed check remains.

Passed: `uv run ruff check`, `uv run ruff format --check`, imports of every module,
`bun run check`, `bunx eslint src`, and `bunx prettier --check src`.
Svelte autofixer returned no issues or suggestions.

Passed: `uv run download --plan`, `uv run label --plan`, and `uv run train --plan`.
Plans report pending work without running a model or changing pipeline outputs.

No tests, real download/label/train/publish command, push, merge, or deployment ran.
Scratch programs ran outside the repository. API calls filled only the HTTP cache.
