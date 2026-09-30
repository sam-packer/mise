// Build the stub bundle from scripts/stub/ and install it in static/bundle/.
// Run: bun scripts/build-stub-bundle.ts
import sharp from 'sharp';
import { Tokenizer } from '@huggingface/tokenizers';
import * as ortNode from 'onnxruntime-node';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { copyFile, cp, mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { createEncoder, type EncoderIO } from '../src/lib/mood/engine';
import { buildNameData } from '../src/lib/mood/name-data';
import {
	CATEGORIES,
	LIGHTS,
	type Anchor,
	type Category,
	type Item,
	type Manifest,
	type OKLab,
	type Palette,
	type Vocab
} from '../src/lib/mood/types';

const ROOT = path.resolve(import.meta.dirname, '..');
const STUB = path.join(ROOT, 'scripts', 'stub');
const CACHE = path.join(STUB, '.cache');
const OUT = path.join(ROOT, 'ml', 'out', 'stub', 'bundle');
const MODEL_REPO = 'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main';
const MODEL_FILES = {
	'model.onnx': 'onnx/model_quantized.onnx',
	'tokenizer.json': 'tokenizer.json',
	'tokenizer_config.json': 'tokenizer_config.json'
};
const MAX_MODEL_BYTES = 24 * 1024 * 1024;
const IMAGE_EDGE = 800;
const USER_AGENT = 'moodboard-stub-build/0.1 (design class project)';

type Source = {
	title: string;
	creator: string;
	year: number;
	hint: string;
	vibe: string;
	mood: string;
	match?: string;
	titleKey?: string;
	artistKey?: string;
	albumKey?: string;
};
type Resolved = Pick<Item, 'links' | 'text' | 'preview' | 'album'> & { imageUrl?: string };
type AnchorSources = {
	lights: Record<string, string[]>;
	typefaces: Record<string, string[]>;
	scents: Record<string, string[]>;
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const sha1 = (s: string) => createHash('sha1').update(s).digest('hex');
const readJson = async <T>(file: string) => JSON.parse(await readFile(file, 'utf8')) as T;

function norm(s: string): string {
	return s
		.normalize('NFD')
		.replace(/[\u0300-\u036f]/g, '')
		.toLowerCase()
		.replace(/&/g, 'and')
		.replace(/[^a-z0-9]/g, '');
}

function words(s: string): Set<string> {
	return new Set(
		s
			.normalize('NFD')
			.replace(/[\u0300-\u036f]/g, '')
			.toLowerCase()
			.split(/[^a-z0-9]+/)
			.filter((w) => w.length > 2)
	);
}

function overlap(a: string, b: string): number {
	const wa = words(a);
	const wb = words(b);
	let shared = 0;
	for (const w of wa) if (wb.has(w)) shared++;
	return wa.size ? shared / wa.size : 0;
}

function slug(s: string): string {
	return s
		.normalize('NFD')
		.replace(/[\u0300-\u036f]/g, '')
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '-')
		.replace(/^-|-$/g, '');
}

function hexToOklab(hex: string): OKLab {
	const n = parseInt(hex.slice(1), 16);
	return srgbToOklab((n >> 16) & 255, (n >> 8) & 255, n & 255);
}

function srgbToOklab(r8: number, g8: number, b8: number): OKLab {
	const lin = (c: number) => {
		c /= 255;
		return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
	};
	const r = lin(r8);
	const g = lin(g8);
	const b = lin(b8);
	const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
	const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
	const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
	return [
		round(0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s),
		round(1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s),
		round(0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s)
	];
}

const round = (x: number) => Math.round(x * 10000) / 10000;

// Space requests by host and cache responses for repeat builds.

const HOST_GAP: Record<string, number> = {
	'itunes.apple.com': 3200,
	'v3.sg.media-imdb.com': 400,
	'openlibrary.org': 1000,
	'covers.openlibrary.org': 300,
	'collectionapi.metmuseum.org': 400,
	'images.metmuseum.org': 150,
	'poetrydb.org': 300,
	'fonts.googleapis.com': 100
};
const lastHit = new Map<string, number>();

async function throttle(url: string) {
	const host = new URL(url).host;
	const gap = HOST_GAP[host] ?? 200;
	const wait = (lastHit.get(host) ?? 0) + gap - Date.now();
	lastHit.set(host, Math.max(Date.now(), (lastHit.get(host) ?? 0) + gap));
	if (wait > 0) await sleep(wait);
}

async function fetchRetry(url: string): Promise<Response | null> {
	for (let attempt = 0; attempt < 5; attempt++) {
		await throttle(url);
		const res = await fetch(url, { headers: { 'user-agent': USER_AGENT } });
		if (res.ok) return res;
		if (res.status === 404) return null;
		if (res.status === 403 || res.status === 429 || res.status >= 500) {
			await sleep(5000 * (attempt + 1));
			continue;
		}
		return null;
	}
	throw new Error(`gave up on ${url}`);
}

async function getJson<T>(url: string): Promise<T | null> {
	const file = path.join(CACHE, 'http', `${sha1(url)}.json`);
	if (existsSync(file)) return readJson<T>(file);
	// The Met can return an HTML bot check with status 200. Pause before retrying a non-JSON response.
	for (let attempt = 0; attempt < 5; attempt++) {
		const res = await fetchRetry(url);
		if (!res) return null;
		const text = await res.text();
		try {
			const data = JSON.parse(text) as T;
			await writeFile(file, text);
			return data;
		} catch {
			await sleep(15000 * (attempt + 1));
		}
	}
	throw new Error(`no JSON from ${url}`);
}

async function getBytes(url: string, dir = 'bytes'): Promise<Buffer | null> {
	const file = path.join(CACHE, dir, sha1(url));
	if (existsSync(file)) return readFile(file);
	const res = await fetchRetry(url);
	if (!res) return null;
	const bytes = Buffer.from(await res.arrayBuffer());
	await writeFile(file, bytes);
	return bytes;
}

// Resolve each category through its source API before building the shared item format.

async function resolveFilm(s: Source): Promise<Resolved | string> {
	type Hit = { id: string; l: string; y?: number; qid?: string; i?: { imageUrl: string } };
	const data = await getJson<{ d?: Hit[] }>(
		`https://v3.sg.media-imdb.com/suggestion/x/${encodeURIComponent(s.hint)}.json`
	);
	const hits = (data?.d ?? []).filter((h) => h.id.startsWith('tt') && h.i?.imageUrl && h.y);
	const hit =
		hits.find((h) => norm(h.l) === norm(s.title) && Math.abs(h.y! - s.year) <= 1) ??
		hits.find(
			(h) =>
				h.y === s.year && (norm(h.l).includes(norm(s.title)) || norm(s.title).includes(norm(h.l)))
		);
	if (!hit) return `no IMDb match among ${hits.map((h) => `${h.l} (${h.y})`).join(', ')}`;
	return { imageUrl: hit.i!.imageUrl, links: { primary: `https://www.imdb.com/title/${hit.id}/` } };
}

async function resolveSong(s: Source): Promise<Resolved | string> {
	type Track = {
		kind?: string;
		artistName: string;
		trackName: string;
		collectionName?: string;
		trackViewUrl: string;
		previewUrl?: string;
		artworkUrl100?: string;
	};
	const url = `https://itunes.apple.com/search?media=music&entity=song&limit=50&country=us&term=${encodeURIComponent(s.hint)}`;
	const data = await getJson<{ results: Track[] }>(url);
	const titleKey = norm(s.titleKey ?? s.title.replace(/\(.*?\)/g, ''));
	const artistKey = norm(s.artistKey ?? s.creator.split(/ & |, /)[0]);
	const tracks = (data?.results ?? []).filter(
		(t) =>
			t.kind === 'song' &&
			t.previewUrl &&
			t.artworkUrl100 &&
			norm(t.trackName).includes(titleKey) &&
			norm(t.artistName).includes(artistKey) &&
			(!s.albumKey || norm(t.collectionName ?? '').includes(norm(s.albumKey))) &&
			!/karaoke|tribute|made famous|in the style/i.test(`${t.collectionName} ${t.artistName}`)
	);
	// Prefer the original album so the displayed name identifies the release.
	const compilation =
		/best of|greatest|hits|collection|essential|anthology|gold|classics|now that/i;
	const track = tracks.find((t) => !compilation.test(t.collectionName ?? '')) ?? tracks[0];
	if (!track) {
		const seen = (data?.results ?? []).slice(0, 5).map((t) => `${t.artistName} - ${t.trackName}`);
		return `no iTunes match; top results: ${seen.join(' | ')}`;
	}
	const apple = track.trackViewUrl.replace(/[?&]uo=\d+/, '');
	const q = encodeURIComponent(`${s.creator.split(/ & /)[0]} ${s.title.replace(/\(.*?\)/g, '')}`);
	return {
		imageUrl: track.artworkUrl100!.replace('100x100bb', '1000x1000bb'),
		preview: track.previewUrl,
		album: track.collectionName?.replace(/ - (Single|EP)$/, ''),
		links: {
			primary: apple,
			apple,
			spotify: `https://open.spotify.com/search/${q}`,
			youtube: `https://www.youtube.com/results?search_query=${q}`
		}
	};
}

async function resolveBook(s: Source): Promise<Resolved | string> {
	type Doc = {
		key: string;
		title: string;
		author_name?: string[];
		cover_i?: number;
		editions?: { docs: { cover_i?: number; language?: string[] }[] };
	};
	// With lang=en, Open Library returns its top English edition. The work's cover can show a translation.
	const url = `https://openlibrary.org/search.json?limit=10&lang=en&fields=key,title,author_name,cover_i,editions,editions.cover_i,editions.language&q=${encodeURIComponent(s.hint)}`;
	const data = await getJson<{ docs: Doc[] }>(url);
	const surname = norm(s.creator.split(' ').at(-1)!);
	// Match the author first. Some works list the author only in another script, so fall back to an
	// exact title match. A title prefix match would also catch study guides ("... Notes").
	const docs = (data?.docs ?? []).filter((d) => d.cover_i);
	const doc =
		docs.find((d) => (d.author_name ?? []).some((a) => norm(a).includes(surname))) ??
		docs.find((d) => norm(d.title) === norm(s.title)) ??
		// A key:/works/OL…W hint identifies works with a non-Latin title and author.
		(s.hint.startsWith('key:') ? docs[0] : undefined);
	if (!doc) return 'no Open Library work with a cover';
	const english = doc.editions?.docs.find((e) => e.cover_i && e.language?.includes('eng'));
	return {
		imageUrl: `https://covers.openlibrary.org/b/id/${english?.cover_i ?? doc.cover_i}-L.jpg`,
		links: { primary: `https://openlibrary.org${doc.key}` }
	};
}

function excerpt(lines: string[]): string {
	const clean = lines.map((l) => l.replace(/\s+$/, ''));
	while (clean.length && !clean[0].trim()) clean.shift();
	let cut = clean.slice(0, 14);
	if (clean.length > 14) {
		const lastBreak = cut.lastIndexOf('');
		if (lastBreak >= 6) cut = cut.slice(0, lastBreak);
	}
	while (cut.length && !cut.at(-1)!.trim()) cut.pop();
	return cut.join('\n');
}

async function resolvePoem(s: Source): Promise<Resolved | string> {
	type Poem = { title: string; author: string; lines: string[] };
	const data = await getJson<Poem[] | { status: number }>(
		`https://poetrydb.org/title/${encodeURIComponent(s.hint)}`
	);
	if (!Array.isArray(data)) return 'no PoetryDB title match';
	const surname = norm(s.creator.split(' ').at(-1)!);
	const byAuthor = data.filter((p) => norm(p.author).includes(surname));
	const poem = byAuthor.find((p) => norm(p.title) === norm(s.hint)) ?? byAuthor[0];
	if (!poem)
		return `no PoetryDB poem by ${s.creator}; found ${data.map((p) => p.author).join(', ')}`;
	return {
		text: excerpt(poem.lines),
		links: {
			primary: `https://en.wikisource.org/w/index.php?search=${encodeURIComponent(`${s.title} ${s.creator}`)}`
		}
	};
}

async function resolveArt(s: Source): Promise<Resolved | string> {
	type MetObject = {
		title: string;
		artistDisplayName: string;
		isPublicDomain: boolean;
		primaryImage: string;
		primaryImageSmall: string;
		objectURL: string;
	};
	const base = 'https://collectionapi.metmuseum.org/public/collection/v1';
	const search = await getJson<{ objectIDs: number[] | null }>(
		`${base}/search?hasImages=true&q=${encodeURIComponent(s.hint)}`
	);
	let best: { obj: MetObject; score: number } | null = null;
	for (const id of (search?.objectIDs ?? []).slice(0, 20)) {
		const obj = await getJson<MetObject>(`${base}/objects/${id}`);
		if (!obj || !obj.isPublicDomain || !obj.primaryImageSmall) continue;
		if (!norm(obj.artistDisplayName).includes(norm(s.match ?? s.creator))) continue;
		const score = overlap(s.title, obj.title);
		if (!best || score > best.score) best = { obj, score };
		if (score === 1) break;
	}
	if (!best || best.score < 0.5) {
		return `no public-domain Met match (best: ${best ? `${best.obj.title} ${best.score}` : 'none'})`;
	}
	return { imageUrl: best.obj.primaryImageSmall, links: { primary: best.obj.objectURL } };
}

const RESOLVERS: Record<Category, (s: Source) => Promise<Resolved | string>> = {
	film: resolveFilm,
	song: resolveSong,
	book: resolveBook,
	poem: resolvePoem,
	art: resolveArt
};

// Store display images and average their colors in OKLab for the tile background.

async function processImage(url: string, id: string): Promise<NonNullable<Item['image']> | null> {
	const bytes = await getBytes(url, 'img');
	if (!bytes) return null;
	const file = `${slug(id)}.webp`;
	const info = await sharp(bytes)
		.rotate()
		.resize({ width: IMAGE_EDGE, height: IMAGE_EDGE, fit: 'inside', withoutEnlargement: true })
		.webp({ quality: 80 })
		.toFile(path.join(OUT, 'img', file));
	const raw = await sharp(bytes).resize(32, 32, { fit: 'fill' }).removeAlpha().raw().toBuffer();
	const sum = [0, 0, 0];
	const px = raw.length / 3;
	for (let i = 0; i < raw.length; i += 3) {
		const lab = srgbToOklab(raw[i], raw[i + 1], raw[i + 2]);
		for (let c = 0; c < 3; c++) sum[c] += lab[c];
	}
	const tone = sum.map((v) => round(v / px)) as OKLab;
	return { src: `/bundle/img/${file}`, w: info.width, h: info.height, tone };
}

// Assemble the model, resolved items, and vectors into the browser bundle.

async function fetchModel() {
	await mkdir(path.join(OUT, 'model'), { recursive: true });
	for (const [name, remote] of Object.entries(MODEL_FILES)) {
		const cached = path.join(CACHE, 'model', name);
		if (!existsSync(cached)) {
			console.log(`downloading ${remote}`);
			const res = await fetch(`${MODEL_REPO}/${remote}`);
			if (!res.ok) throw new Error(`model download failed: ${remote} ${res.status}`);
			await writeFile(cached, Buffer.from(await res.arrayBuffer()));
		}
		await copyFile(cached, path.join(OUT, 'model', name));
	}
}

async function resolveCategory(category: Category, sources: Source[]) {
	const items: { item: Item; source: Source }[] = [];
	const failures: string[] = [];
	for (const s of sources) {
		const id = `${category}:${slug(s.title)}-${s.year}`;
		const resolved = await RESOLVERS[category](s);
		if (typeof resolved === 'string') {
			failures.push(`${id}: ${resolved}`);
			continue;
		}
		let image: Item['image'] = null;
		if (resolved.imageUrl) {
			image = await processImage(resolved.imageUrl, id);
			if (!image) {
				failures.push(`${id}: image download failed (${resolved.imageUrl})`);
				continue;
			}
		}
		const item: Item = {
			id,
			category,
			title: s.title,
			creator: s.creator,
			year: s.year,
			vibe: s.vibe,
			image,
			links: resolved.links
		};
		if (resolved.text) item.text = resolved.text;
		if (resolved.album) item.album = resolved.album;
		if (resolved.preview) item.preview = resolved.preview;
		items.push({ item, source: s });
	}
	console.log(`${category}: ${items.length}/${sources.length} resolved`);
	return { items, failures };
}

function embeddingText(item: Item, s: Source): string {
	return `${s.vibe}. ${s.mood.replace(/\.$/, '')}. ${item.category}: ${item.title} by ${item.creator}.`;
}

function fileIO(): EncoderIO {
	return {
		ort: ortNode as unknown as EncoderIO['ort'],
		async fetchBytes(p) {
			const b = await readFile(path.join(OUT, p));
			return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
		},
		fetchJson: (p) => readJson(path.join(OUT, p))
	};
}

function toBin(vectors: Float32Array[]): Buffer {
	const flat = new Float32Array(vectors.length * (vectors[0]?.length ?? 0));
	vectors.forEach((v, i) => flat.set(v, i * v.length));
	return Buffer.from(flat.buffer);
}

async function dirSize(dir: string): Promise<number> {
	let total = 0;
	for (const entry of await readdir(dir, { withFileTypes: true })) {
		const p = path.join(dir, entry.name);
		total += entry.isDirectory() ? await dirSize(p) : (await stat(p)).size;
	}
	return total;
}

const mib = (n: number) => `${(n / 1024 / 1024).toFixed(2)} MiB`;

async function main() {
	const started = Date.now();
	for (const d of ['http', 'bytes', 'img', 'model']) {
		await mkdir(path.join(CACHE, d), { recursive: true });
	}
	await rm(OUT, { recursive: true, force: true });
	await mkdir(path.join(OUT, 'img'), { recursive: true });

	const vocab = await readJson<Vocab>(path.join(STUB, 'vocab.json'));
	const anchorSources = await readJson<AnchorSources>(path.join(STUB, 'anchors.json'));
	const palettes = await readJson<{ phrase: string; hex: string[] }[]>(
		path.join(STUB, 'palettes.json')
	);
	const sources = Object.fromEntries(
		await Promise.all(
			CATEGORIES.map(async (c) => [c, await readJson<Source[]>(path.join(STUB, `${c}.json`))])
		)
	) as Record<Category, Source[]>;

	await fetchModel();
	const manifest: Manifest = {
		version: `stub-${new Date().toISOString().slice(0, 10)}`,
		encoder: {
			model: 'model/model.onnx',
			tokenizer: 'model/',
			dims: 384,
			maxTokens: 96,
			pooling: 'mean',
			normalize: true,
			outputs: { hidden: 'last_hidden_state' }
		},
		files: {
			items: { path: 'items.json', format: 'json' },
			vectors: { path: 'vectors.bin', format: 'fp16-le' },
			names: { path: 'search-index.json', format: 'json' },
			vocab: { path: 'vocab.json', format: 'json' },
			anchors: { path: 'anchors.json', format: 'json' },
			anchorVectors: { path: 'anchors.bin', format: 'fp32-le' }
		},
		heads: { kind: 'anchors' },
		counts: { items: 0 }
	};

	const results = await Promise.all(CATEGORIES.map((c) => resolveCategory(c, sources[c])));
	const resolved = results.flatMap((r) => r.items);
	const failures = results.flatMap((r) => r.failures);
	const items = resolved.map((r) => r.item);
	manifest.counts.items = items.length;

	const anchors: Anchor[] = [
		...palettes.map((p): Anchor => ({
			kind: 'palette',
			phrase: p.phrase,
			value: p.hex.map(hexToOklab) as Palette
		})),
		...(['lights', 'typefaces', 'scents'] as const).flatMap((group) =>
			Object.entries(anchorSources[group]).flatMap(([id, phrases]) =>
				phrases.map((phrase): Anchor => ({
					kind: group === 'lights' ? 'light' : group === 'typefaces' ? 'typeface' : 'scent',
					phrase,
					value: id
				}))
			)
		)
	];

	console.log(`embedding ${items.length} items and ${anchors.length} anchors`);
	const encoder = await createEncoder(fileIO(), manifest);
	const itemVectors = await encoder.embed(resolved.map((r) => embeddingText(r.item, r.source)));
	const anchorVectors = await encoder.embed(anchors.map((a) => a.phrase));

	const flat = new Float32Array(items.length * manifest.encoder.dims);
	itemVectors.forEach((vector, row) => flat.set(vector, row * manifest.encoder.dims));
	const half = new DataView(new ArrayBuffer(flat.length * 2)) as DataView & {
		setFloat16(offset: number, value: number, littleEndian: boolean): void;
	};
	flat.forEach((value, i) => half.setFloat16(i * 2, value, true));
	await writeFile(path.join(OUT, 'vectors.bin'), Buffer.from(half.buffer));
	await writeFile(path.join(OUT, 'anchors.bin'), toBin(anchorVectors));
	await writeFile(path.join(OUT, 'items.json'), JSON.stringify(items));
	await writeFile(path.join(OUT, 'anchors.json'), JSON.stringify(anchors));
	const tokenizerJson = await readJson<{ model: { vocab: Record<string, number> } }>(
		path.join(OUT, 'model', 'tokenizer.json')
	);
	const tokenizer = new Tokenizer(
		tokenizerJson,
		await readJson<object>(path.join(OUT, 'model', 'tokenizer_config.json'))
	);
	const candidates = new Set(
		Object.keys(tokenizerJson.model.vocab).filter((word) => /^[a-z0-9]+$/.test(word))
	);
	for (const item of items) {
		const names = [item.title, item.creator, item.album ?? ''].join(' ');
		for (const word of names
			.normalize('NFD')
			.replace(/[\u0300-\u036f]/g, '')
			.toLowerCase()
			.replace(/&/g, ' and ')
			.replace(/['\u2019]/g, '')
			.split(/[^a-z0-9]+/)
			.filter(Boolean))
			candidates.add(word);
	}
	const words = [...candidates]
		.filter((word) => tokenizer.encode(word, { add_special_tokens: false }).ids.length === 1)
		.sort();
	await writeFile(
		path.join(OUT, 'search-index.json'),
		JSON.stringify(buildNameData(items, flat, manifest.encoder.dims, words))
	);
	await writeFile(path.join(OUT, 'vocab.json'), JSON.stringify(vocab));
	await writeFile(path.join(OUT, 'manifest.json'), JSON.stringify(manifest, null, '\t'));

	// Check item metadata and vector dimensions before installing the bundle.
	const errors: string[] = [...failures.map((f) => `unresolved ${f}`)];
	for (const item of items) {
		if (item.category !== 'poem' && !item.image) errors.push(`${item.id}: no image`);
		if (item.category === 'poem' && !item.text) errors.push(`${item.id}: no text`);
		if (item.text && item.text.split('\n').length > 14) errors.push(`${item.id}: excerpt too long`);
		if (!item.links.primary) errors.push(`${item.id}: no primary link`);
		if (item.category === 'song' && !item.links.apple) errors.push(`${item.id}: no apple link`);
		if (item.category === 'song' && !item.album) errors.push(`${item.id}: no album`);
	}
	if (new Set(items.map((i) => i.id)).size !== items.length) errors.push('duplicate item ids');
	const dims = manifest.encoder.dims;
	const vecBytes = (await stat(path.join(OUT, 'vectors.bin'))).size;
	const ancBytes = (await stat(path.join(OUT, 'anchors.bin'))).size;
	if (vecBytes !== items.length * dims * 2) errors.push('vectors.bin row count mismatch');
	if (ancBytes !== anchors.length * dims * 4) errors.push('anchors.bin row count mismatch');
	for (const v of [...itemVectors, ...anchorVectors]) {
		const n = Math.sqrt(v.reduce((a, x) => a + x * x, 0));
		if (v.length !== dims || Math.abs(n - 1) > 1e-3) {
			errors.push('a vector is not unit length');
			break;
		}
	}
	const modelBytes = (await stat(path.join(OUT, manifest.encoder.model))).size;
	if (modelBytes > MAX_MODEL_BYTES) errors.push(`model is ${mib(modelBytes)}, over 24 MiB`);

	const paletteCount = anchors.filter((a) => a.kind === 'palette').length;
	if (paletteCount < 150) errors.push(`only ${paletteCount} palette anchors`);
	if (palettes.some((p) => p.hex.length !== 5 || p.hex.some((h) => !/^#[0-9a-f]{6}$/i.test(h)))) {
		errors.push('a palette anchor does not have five hex colors');
	}
	if (JSON.stringify(vocab.lights) !== JSON.stringify(LIGHTS)) errors.push('vocab lights differ');
	const need: [string, string[], number][] = [
		['light', [...vocab.lights], 4],
		['typeface', vocab.typefaces.map((t) => t.id), 2],
		['scent', vocab.scents.map((s) => s.id), 2]
	];
	for (const [kind, ids, min] of need) {
		const known = new Set(ids);
		for (const a of anchors.filter((x) => x.kind === kind)) {
			if (!known.has(a.value as string)) errors.push(`${kind} anchor for unknown id ${a.value}`);
		}
		for (const id of ids) {
			const n = anchors.filter((a) => a.kind === kind && a.value === id).length;
			if (n < min) errors.push(`${kind} ${id} has ${n} anchor phrases, needs ${min}`);
		}
	}
	for (const t of vocab.typefaces) {
		const url = `https://fonts.googleapis.com/css2?family=${t.family.replace(/ /g, '+')}:${t.axes}&text=Aa`;
		const file = path.join(CACHE, 'http', `${sha1(url)}.css`);
		if (existsSync(file)) continue;
		await throttle(url);
		const res = await fetch(url);
		if (res.ok) await writeFile(file, await res.text());
		else errors.push(`typeface ${t.id}: Google Fonts returned ${res.status} for "${t.axes}"`);
	}

	const counts = CATEGORIES.map((c) => `${c} ${items.filter((i) => i.category === c).length}`);
	console.log(`\nitems: ${items.length} (${counts.join(', ')})`);
	console.log(
		`anchors: ${anchors.length} (palette ${paletteCount}, light ${anchors.filter((a) => a.kind === 'light').length}, typeface ${anchors.filter((a) => a.kind === 'typeface').length}, scent ${anchors.filter((a) => a.kind === 'scent').length})`
	);
	console.log(`vocab: ${vocab.typefaces.length} typefaces, ${vocab.scents.length} scents`);
	console.log(
		`size: model ${mib(await dirSize(path.join(OUT, 'model')))}, img ${mib(await dirSize(path.join(OUT, 'img')))}, vectors ${mib(vecBytes)}, anchors.bin ${mib(ancBytes)}, total ${mib(await dirSize(OUT))}`
	);
	console.log(`time: ${((Date.now() - started) / 1000).toFixed(1)} s`);
	if (errors.length) {
		console.error(`\nvalidation failed (${errors.length}):\n  ${errors.join('\n  ')}`);
		process.exit(1);
	}
	console.log('validation passed');
	const target = path.join(ROOT, 'static', 'bundle');
	await rm(target, { recursive: true, force: true });
	await cp(OUT, target, { recursive: true });
}

await main();
