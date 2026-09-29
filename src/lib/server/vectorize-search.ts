import { CATEGORIES, type Anchor, type Category, type Item, type MatchResult } from '../mood/types';
import { createAnchorMatcher } from './anchor-search';
import { exclusionKeys, expandItems, type CatalogIndex } from './catalog-index';
import { blendAnchor, createAnchorHeads, dot } from './search';

export function createVectorizeSearch(
	index: VectorizeIndex,
	namespace: string,
	catalog: CatalogIndex,
	anchors: Anchor[] = [],
	anchorVectors = new Float32Array(0)
) {
	const { info } = catalog;
	const items = expandItems(catalog);
	if (items.length !== info.counts.items || anchorVectors.length !== anchors.length * info.dims)
		throw new Error('search index does not match catalog');
	const keys = exclusionKeys(items);
	const findAnchor = createAnchorMatcher(items, info.words, (rows) => {
		if (rows.length === 1) return rows[0];
		const row = catalog.representatives[rows.join(',')];
		if (row === undefined) throw new Error('missing anchor representative');
		return row;
	});
	const anchorHeads = createAnchorHeads(anchors, anchorVectors, info.dims);
	const id = (row: number) => namespace + row;
	function record(vector: VectorizeVector): Item {
		if (typeof vector.metadata?.item !== 'string') throw new Error('missing item metadata');
		return JSON.parse(vector.metadata.item) as Item;
	}
	async function get(row: number) {
		const vectors = await index.getByIds([id(row)]);
		const vector = vectors.find((v) => v.id === id(row));
		if (!vector || vector.values.length !== info.dims) throw new Error('missing catalog vector');
		return vector;
	}
	return {
		dims: info.dims,
		version: info.version,
		async match(query: string, embedding: Float32Array): Promise<MatchResult> {
			const row = findAnchor(query);
			const anchored = row >= 0 ? await get(row) : null;
			const anchor = anchored ? record(anchored) : null;
			const q = anchored ? blendAnchor(embedding, Float32Array.from(anchored.values)) : embedding;
			const zero = q.every((value) => value === 0);
			const picks = {} as Record<Category, Item>;
			await Promise.all(
				CATEGORIES.map(async (category) => {
					// A zero vector ties every item. Preserve catalog order without an ANN query.
					if (zero) {
						const first = items.findIndex(
							(item) =>
								item.category === category &&
								(!anchor || (item.creator !== anchor.creator && item.title !== anchor.title))
						);
						if (first >= 0) picks[category] = record(await get(first));
						return;
					}
					const filter: VectorizeVectorMetadataFilter = { category };
					if (anchor) {
						filter.creator = { $ne: keys[row].creator };
						filter.title = { $ne: keys[row].title };
					}
					const result = await index.query(q, {
						namespace,
						filter,
						topK: 50,
						returnValues: true,
						returnMetadata: 'all'
					});
					let best: Item | undefined;
					let bestScore = -Infinity;
					let bestRow = Infinity;
					for (const vector of result.matches) {
						const candidateRow = vector.metadata?.row;
						if (!vector.values || typeof candidateRow !== 'number')
							throw new Error('missing candidate vector or row');
						const item = record(vector as VectorizeVector);
						if (anchor && (item.creator === anchor.creator || item.title === anchor.title))
							continue;
						const score = dot(q, Float32Array.from(vector.values));
						if (score > bestScore || (score === bestScore && candidateRow < bestRow)) {
							best = item;
							bestScore = score;
							bestRow = candidateRow;
						}
					}
					if (best) picks[category] = best;
				})
			);
			return { picks, anchor, ...(info.heads.kind === 'anchors' ? { heads: anchorHeads(q) } : {}) };
		}
	};
}
