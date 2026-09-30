# mise

Live at [mise.art](https://mise.art).

Type how you feel in one sentence, like "a snowy december and i just made warm hot chocolate". mise
answers with a small wall of things that match it: an artwork, a film, a song, a poem, and a book.
The whole page takes on the feeling too. Its colors, its light, and its typeface change, and a line
at the bottom names a scent.

## Who it is for

mise is for anyone who knows how they feel but not what to watch, read, or listen to next.

## The problem

Recommendation apps ask what you liked before. Search boxes want a title, a genre, or a name.
Neither one helps when all you have is a mood, like "the last warm night before school starts".
You can describe that feeling in a sentence, and mise gives you a place to start from it.

## How it works for the user

1. Open the site. A sample feeling rotates in the text box. Press Tab to use it, or type your own.
2. Press Enter. A soft pulse shows while the model loads. The page fades to the new palette, five
   tiles rise in (an artwork, a film, a song, a poem, and a book), and a scent writes itself out
   below them.
3. Click a tile to travel into that work's world. The page takes on the work's own palette, light,
   and typeface, and the works nearest to it fill the wall. A song plays a 30-second preview, and
   each work links to its source, such as IMDb, Hardcover, a music service, or the museum page.
4. Keep going: click any nearby work to travel again. A trail at the top shows your path. Click a
   step, press Escape, or use the browser's back button to retrace it.
5. You can also type the name of a work or an artist, like "blade runner" or "pride and prejudice by
   jane austen". Then a line reads "in the key of" that work, and the picks lean toward it.
6. Copy the URL to share the feeling. Each feeling gets a short link like `mise.art/k3x9Q2a`.
   Each path through its worlds gets its own short link too, and that link opens the same world.
7. Edit the sentence to try again, or click the mise logo to start over.

The page also covers the other states. The first visit shows a breathing background while the
model downloads. A failed load or a failed search shows a short note under the text box. A shared
link that no longer exists says "that feeling has faded". A song without a preview hides the play
button.

## Tech stack

| Part             | What I used                                                                                     |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| Front end        | SvelteKit 2 and Svelte 5 (runes), TypeScript, Tailwind CSS 4                                    |
| In-browser model | ONNX Runtime Web and Hugging Face tokenizers, in a web worker                                   |
| Hosting          | Cloudflare Workers, with Workers KV for share links and R2 for the model and images             |
| Model training   | Python with uv, PyTorch, Hugging Face Transformers, and ONNX (see [ml/README.md](ml/README.md)) |
| Tooling          | Bun, Vite, ESLint, Prettier, svelte-check                                                       |

The course default is React and Next.js. The instructor approved Svelte and a free choice of host.

The code is small and split by job:

| Path                  | What it holds                                                                    |
| --------------------- | -------------------------------------------------------------------------------- |
| `src/routes/`         | the pages (the main page and attribution), the API routes, and the preview image |
| `src/lib/components/` | the UI: the text box, the wall, a tile, a work's world, the room light           |
| `src/lib/mood/`       | the search engine: the web worker, the model call, and the catalog search        |
| `src/lib/server/`     | the preview image: its layout, its font, and the palette check                   |
| `src/lib/color/`      | OKLab color math and the palette fade                                            |
| `scripts/`            | a small sample bundle for local work, and a command line search                  |
| `ml/`                 | the pipeline that builds the catalog and trains the model                        |

## APIs

At run time the app calls three services:

- Deezer. Its preview links expire after about 15 minutes, so the catalog can't store
  them. When you open a song, the route `/api/preview/deezer/<id>` asks Deezer for a fresh link and
  redirects the audio player to it.
- Google Fonts. Each mood picks a typeface. The page downloads only the letters it needs.
- Cloudflare Workers KV. It stores the sentence behind each share link, the item ids of a shared
  path, and the palette of the mood.

Each share link also has its own preview image for chat apps and social sites. The route
`/og/<code>.png` draws a 1200 by 630 PNG of the feeling in quotes, in the five colors of its mood,
with the mise wordmark. The palette comes from your browser: after the page finds the mood, it sends
the palette once to `/api/feeling/<code>/palette`. Until that palette arrives, the image uses the
brand colors. The Worker draws the image with Satori and resvg in WebAssembly and keeps it in the
Cloudflare cache. The feeling is set in EB Garamond Italic, under the SIL Open Font License.

The catalog itself comes from APIs too. I built it ahead of time, because a live query to a dozen
services for each feeling would be slow and would hit rate limits. The pipeline reads TMDB (films),
Hardcover and Open Library (books), ListenBrainz, MusicBrainz, Last.fm and Deezer (songs),
PoetryDB (poems), and open-access collections from the Met, the Art Institute of Chicago, and the
Cleveland Museum of Art (artworks). Their data is everything the wall shows: the titles, the
posters and covers, the song tags, and the links.

The search runs in your browser. A small language model (23 MB) turns your sentence into a list of
numbers and compares it with the numbers for about 11,000 works. The same model also picks the
palette, the light, the typeface, and the scent. After the first load, a search takes a fraction of
a second. The server only stores the sentence and its palette for the share link.

## Run it locally

You need [Bun](https://bun.sh).

```sh
bun install
bun run dev
```

Open http://localhost:5173. The local site loads the published model from `cdn.mise.art`, so you
don't need to train anything. Share links work locally too: Wrangler gives the dev server an
empty local copy of the KV store.

To work offline with a small hand-made sample instead of the real catalog:

```powershell
bun run stub
$env:PUBLIC_BUNDLE_URL = '/bundle/'
bun run dev
```

You can also search from the terminal: `bun run query "a dark and stormy night"`.

The app has no secrets. The API keys are only for the training pipeline and live in `ml/.env`,
which git ignores. [ml/.env.example](ml/.env.example) lists them.

To deploy, run `bun run deploy`. It builds the site and publishes it to Cloudflare Workers.

## Viewing context

I designed for desktop and mobile equally. The layout, the tile sizes, and the type scale adapt
to the screen width, and the small controls keep a touch target of at least 44 px.

## Known limitations

- The first visit is heavy. The model, the vectors, and the catalog are about 40 MB before
  compression. On a slow phone connection the first search can take a while. Later visits use the
  browser cache.
- The catalog is fixed at about 11,000 works, chosen in September 2026. Newer releases aren't in
  it, and plenty of good older ones are missing too.
- An AI model labeled the catalog. A local language model wrote the mood descriptions and palettes
  that the small model learned from, so the app's taste is partly that model's taste.
- The poems and the art are public domain only. Modern poetry and art aren't in the catalog.
- Names must be exact. An anchor needs the full title or artist name, and the name must be the
  whole sentence (words like "movie" or "song" are fine). "blade runner" anchors, but "a rainy night like blade runner" does not.
- It only understands English.
- Share links are public. Anyone with a link can read its sentence.

## What I would improve next

- Apply what I learn in the design half of the course: the layout of the wall, the empty state,
  and how the anchor line reads.
- Make the first load feel shorter, so the wait for the model reads as part of the experience.
- Make it clearer that you can name a film, song, or book, and show which names mise knows.

## AI assistance

I used Claude Code and OpenAI Codex as coding partners. They wrote most of the code from my
direction. Every change went through a branch and a pull request. I decided what the product is,
who it's for, and how it should look and feel. I rejected ideas that made it more complex than it needed to
be: a server-side search, a request rate limit, and a private catalog.

## Credit

[Wave](https://github.com/SophiaYifei/wave-recsys) sparked the idea of turning a described feeling
into picks across media. mise shares no code or design with it. The [attribution page](https://mise.art/attribution)
credits every data source.
