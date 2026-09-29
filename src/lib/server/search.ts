import {
	CATEGORIES,
	type Anchor,
	type Category,
	type Item,
	type OKLab,
	type Palette,
	type MatchResult
} from '../mood/types';
export type CatalogInfo = {
	version: string;
	dims: number;
	counts: { items: number };
	heads: { kind: 'onnx' | 'anchors' };
	words: string[];
};
/** How much of the anchored item goes into the wall's vector; the rest is the query. */
const ANCHOR_ITEM_WEIGHT = 0.7;
/** Share of the query's content words that a named entity must cover. */
const ANCHOR_COVERAGE = 0.75;
const STOP_WORDS = new Set(
	(
		'the a an of and or by in on at to for with from feat ft i im me my mine you your we our us ' +
		'it its is am are was be been this that these those just so he she him her his they them their'
	).split(' ')
);
/** Words that say what kind of thing is named. A query may add them without breaking a match. */
const FILLER_WORDS = new Set(
	(
		'film movie song track album record ep book novel poem painting art artwork listening ' +
		'watching reading playing vibe vibes mood energy'
	).split(' ')
);
const ANCHOR_CATEGORY_ORDER: Category[] = ['film', 'song', 'book', 'art', 'poem'];

function allWords(text: string): string[] {
	return text
		.normalize('NFD')
		.replace(/[\u0300-\u036f]/g, '')
		.toLowerCase()
		.replace(/&/g, ' and ')
		.replace(/['\u2019]/g, '')
		.split(/[^a-z0-9]+/)
		.filter(Boolean);
}

function contentWords(text: string): string[] {
	return allWords(text).filter((w) => !STOP_WORDS.has(w));
}

/** A name as typed, without filler words or a leading "the". */
function exactName(text: string): string {
	const words = allWords(text).filter((w) => !FILLER_WORDS.has(w));
	if (words[0] === 'the') words.shift();
	return words.join(' ');
}

/** Index of the first place where `phrase` appears as a run of words in `words`, or -1. */
function findPhrase(words: string[], phrase: string[]): number {
	if (!phrase.length) return -1;
	outer: for (let i = 0; i + phrase.length <= words.length; i++) {
		for (let k = 0; k < phrase.length; k++) if (words[i + k] !== phrase[k]) continue outer;
		return i;
	}
	return -1;
}

const PALETTE_TOP_K = 3;
const PALETTE_TEMPERATURE = 0.03;
/** Above this weight the best anchor's palette is returned as it is. */
const PALETTE_DOMINANT = 0.7;

/** Blends palettes slot by slot in OKLCH. Averaging a/b directly cancels distinct hues and leaves
 * grey, so the hue is a circular mean (the shortest arc) and the chroma is the mean chroma. */
function blendPalettes(palettes: Palette[], weights: number[]): Palette {
	return Array.from({ length: 5 }, (_, slot) => {
		let L = 0;
		let C = 0;
		let hx = 0;
		let hy = 0;
		palettes.forEach((p, k) => {
			const [l, a, b] = p[slot];
			const c = Math.hypot(a, b);
			L += weights[k] * l;
			C += weights[k] * c;
			if (c > 1e-6) {
				hx += weights[k] * (a / c);
				hy += weights[k] * (b / c);
			}
		});
		const h = Math.atan2(hy, hx);
		return [L, C * Math.cos(h), C * Math.sin(h)] as OKLab;
	}) as Palette;
}

function l2normalize(v: Float32Array): Float32Array {
	let sum = 0;
	for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
	const inv = sum > 0 ? 1 / Math.sqrt(sum) : 0;
	for (let i = 0; i < v.length; i++) v[i] *= inv;
	return v;
}

function dot(a: Float32Array, b: Float32Array, offset = 0): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[offset + i];
	return s;
}

export function createSearch(
	info: CatalogInfo,
	items: Item[],
	vectors: Float32Array,
	anchors: Anchor[] = [],
	anchorVectors = new Float32Array(0)
) {
	const dims = info.dims;
	const commonWords = new Set(info.words);
	if (items.length !== info.counts.items || vectors.length !== items.length * dims)
		throw new Error('vectors.bin does not match items');
	if (anchorVectors.length !== anchors.length * dims)
		throw new Error('anchors.bin does not match anchors');
	const byCategory = new Map<Category, number[]>(CATEGORIES.map((c) => [c, []]));
	items.forEach((item, i) => byCategory.get(item.category)!.push(i));

	const names = items.map((item) => ({
		exactTitle: exactName(item.title),
		exactAlbum: item.album ? exactName(item.album) : '',
		title: contentWords(item.title),
		creator: contentWords(item.creator),
		album: item.album ? contentWords(item.album) : []
	}));

	/** The item whose vector is closest to the mean vector of `rows`: an album's or artist's most
	 * representative piece. */
	function representative(rows: number[]): number {
		const mean = new Float32Array(dims);
		for (const i of rows) for (let d = 0; d < dims; d++) mean[d] += vectors[i * dims + d];
		let best = rows[0];
		for (const i of rows)
			if (dot(mean, vectors, i * dims) > dot(mean, vectors, best * dims)) best = i;
		return best;
	}

	/**
	 * Finds the catalog item the query names, or -1. A name is a title, an album, or a creator, and
	 * a title or an album may come with its creator ("essex honey by blood orange"). Each name must
	 * appear in the query as a whole run of words, and together the matched names must cover at
	 * least ANCHOR_COVERAGE of the query's content words. A match of one common word ("rain",
	 * "blonde") is too weak and is ignored. Ties go to the larger coverage, then title before album
	 * before creator, then the order film, song, book, art, poem, then catalog order.
	 */
	function findAnchor(query: string): number {
		const words = contentWords(query);
		const exactQuery = exactName(query);
		const counted = words.map((w) => !FILLER_WORDS.has(w));
		const total = counted.filter(Boolean).length;
		if (!total) return -1;

		type Match = { row: number; covered: number; kind: number };
		let best: Match | null = null;
		const better = (a: Match, b: Match) =>
			a.covered !== b.covered
				? a.covered > b.covered
				: a.kind !== b.kind
					? a.kind < b.kind
					: ANCHOR_CATEGORY_ORDER.indexOf(items[a.row].category) <
						ANCHOR_CATEGORY_ORDER.indexOf(items[b.row].category);

		items.forEach((item, row) => {
			const n = names[row];
			const creatorAt = findPhrase(words, n.creator);
			const options: [number, string[][]][] = [
				[0, [n.title]],
				[1, [n.album]],
				[2, []]
			];
			for (const [kind, phrases] of options) {
				const parts = [...phrases, n.creator];
				for (const withCreator of [true, false]) {
					const used = withCreator ? parts : phrases;
					if (!used.length || used.some((p) => !p.length)) continue;
					const hit = new Array(words.length).fill(false);
					let ok = true;
					for (const p of used) {
						const at = p === n.creator ? creatorAt : findPhrase(words, p);
						if (at < 0) ok = false;
						else for (let k = 0; k < p.length; k++) hit[at + k] = true;
					}
					if (!ok) continue;
					const matched = used.flat();
					// A name made only of common words ("lovely day", "rain") reads as a feeling. Without
					// its creator it anchors only when the query is exactly that name. One common word
					// never anchors.
					if (matched.every((w) => commonWords.has(w))) {
						if (matched.length === 1) continue;
						if (!withCreator && exactQuery !== (kind === 0 ? n.exactTitle : n.exactAlbum)) continue;
					}
					const covered = hit.filter((h, i) => h && counted[i]).length;
					if (covered / total < ANCHOR_COVERAGE) continue;
					const match = { row, covered, kind };
					if (!best || better(match, best)) best = match;
				}
			}
		});
		if (!best) return -1;
		const { row, kind } = best as Match;
		const item = items[row];
		if (kind === 1) {
			return representative(
				items.flatMap((x, i) => (x.creator === item.creator && x.album === item.album ? [i] : []))
			);
		}
		if (kind === 2) {
			return representative(items.flatMap((x, i) => (x.creator === item.creator ? [i] : [])));
		}
		return row;
	}

	function pick(q: Float32Array, anchor: Item | null): Record<Category, Item> {
		const picks = {} as Record<Category, Item>;
		for (const [category, rows] of byCategory) {
			let best = -1;
			let bestScore = -Infinity;
			for (const i of rows) {
				if (anchor && (items[i].creator === anchor.creator || items[i].title === anchor.title))
					continue;
				const s = dot(q, vectors, i * dims);
				if (s > bestScore) {
					bestScore = s;
					best = i;
				}
			}
			if (best >= 0) picks[category] = items[best];
		}
		return picks;
	}

	function anchorHeads(q: Float32Array) {
		const scored = anchors.map((anchor, i) => ({ anchor, score: dot(q, anchorVectors, i * dims) }));
		const top = scored
			.filter((s) => s.anchor.kind === 'palette')
			.sort((a, b) => b.score - a.score)
			.slice(0, PALETTE_TOP_K);
		const max = top[0].score;
		const raw = top.map((s) => Math.exp((s.score - max) / PALETTE_TEMPERATURE));
		const total = raw.reduce((a, b) => a + b, 0);
		const weights = raw.map((w) => w / total);
		const palettes = top.map((s) => s.anchor.value as Palette);
		const palette =
			weights[0] >= PALETTE_DOMINANT
				? (palettes[0].map((c) => [...c]) as Palette)
				: blendPalettes(palettes, weights);

		const best = (kind: 'light' | 'typeface' | 'scent') => {
			let id = '';
			let bestScore = -Infinity;
			for (const s of scored) {
				if (s.anchor.kind === kind && s.score > bestScore) {
					bestScore = s.score;
					id = s.anchor.value as string;
				}
			}
			return id;
		};
		return { palette, light: best('light'), typeface: best('typeface'), scent: best('scent') };
	}

	return {
		dims,
		version: info.version,
		match(query: string, embedding: Float32Array): MatchResult {
			const row = findAnchor(query);
			const anchor = row >= 0 ? items[row] : null;
			let q = embedding;
			if (anchor) {
				q = new Float32Array(dims);
				for (let d = 0; d < dims; d++)
					q[d] =
						ANCHOR_ITEM_WEIGHT * vectors[row * dims + d] + (1 - ANCHOR_ITEM_WEIGHT) * embedding[d];
				l2normalize(q);
			}
			return {
				picks: pick(q, anchor),
				anchor,
				...(info.heads.kind === 'anchors' ? { heads: anchorHeads(q) } : {})
			};
		}
	};
}
