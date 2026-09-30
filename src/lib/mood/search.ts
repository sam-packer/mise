import { createAnchorMatcher } from './anchor-search';
import {
	CATEGORIES,
	type Anchor,
	type Category,
	type Item,
	type OKLab,
	type Palette,
	type MatchResult
} from './types';
export type CatalogInfo = {
	dims: number;
	counts: { items: number };
	heads: { kind: 'onnx' | 'anchors' };
	words: string[];
};
/** How much of the anchored item goes into the wall's vector; the rest is the query. */
const ANCHOR_ITEM_WEIGHT = 0.7;
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

export function dot(a: Float32Array, b: Float32Array, offset = 0): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[offset + i];
	return s;
}

export function createSearch(
	info: CatalogInfo,
	items: Item[],
	vectors: Float32Array,
	anchors: Anchor[] = [],
	anchorVectors = new Float32Array(0),
	representatives: Record<string, number> = {}
) {
	const dims = info.dims;
	if (items.length !== info.counts.items || vectors.length !== items.length * dims)
		throw new Error('vectors.bin does not match items');
	if (anchorVectors.length !== anchors.length * dims)
		throw new Error('anchors.bin does not match anchors');
	const byCategory = new Map<Category, number[]>(CATEGORIES.map((c) => [c, []]));
	items.forEach((item, i) => byCategory.get(item.category)!.push(i));

	const findAnchor = createAnchorMatcher(
		items,
		info.words,
		(rows) => representatives[rows.join(',')] ?? representative(vectors, dims, rows)
	);

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

	const anchorHeads = createAnchorHeads(anchors, anchorVectors, dims);

	return {
		match(query: string, embedding: Float32Array): MatchResult {
			const row = findAnchor(query);
			const anchor = row >= 0 ? items[row] : null;
			const q = anchor
				? blendAnchor(embedding, vectors.subarray(row * dims, (row + 1) * dims))
				: embedding;
			return {
				picks: pick(q, anchor),
				anchor,
				...(info.heads.kind === 'anchors' ? { heads: anchorHeads(q) } : {})
			};
		}
	};
}

/** The item whose vector is closest to the mean vector of `rows`: an album's or artist's most
 * representative piece. */
export function representative(vectors: Float32Array, dims: number, rows: number[]): number {
	const mean = new Float32Array(dims);
	for (const i of rows) for (let d = 0; d < dims; d++) mean[d] += vectors[i * dims + d];
	let best = rows[0];
	for (const i of rows)
		if (dot(mean, vectors, i * dims) > dot(mean, vectors, best * dims)) best = i;
	return best;
}

export function createAnchorHeads(anchors: Anchor[], anchorVectors: Float32Array, dims: number) {
	return (q: Float32Array) => {
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
	};
}

export function blendAnchor(embedding: Float32Array, vector: Float32Array) {
	const q = new Float32Array(embedding.length);
	for (let d = 0; d < q.length; d++)
		q[d] = ANCHOR_ITEM_WEIGHT * vector[d] + (1 - ANCHOR_ITEM_WEIGHT) * embedding[d];
	return l2normalize(q);
}
