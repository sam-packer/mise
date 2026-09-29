import type { Anchor, Item, MatchResult } from '../mood/types';
import { CATALOG_PREFIX } from './catalog';
import type { CatalogIndex } from './catalog-index';
import { createSearch, type CatalogInfo } from './search';
import { createVectorizeSearch } from './vectorize-search';

type Search = {
	dims: number;
	version: string;
	match(query: string, embedding: Float32Array): MatchResult | Promise<MatchResult>;
};
const loaded = new WeakMap<
	R2Bucket,
	Map<VectorizeIndex | undefined, Map<string, Promise<Search>>>
>();

export function loadCatalog(bucket: R2Bucket, prefix = CATALOG_PREFIX, index?: VectorizeIndex) {
	let stores = loaded.get(bucket);
	if (!stores) loaded.set(bucket, (stores = new Map()));
	let catalogs = stores.get(index);
	if (!catalogs) stores.set(index, (catalogs = new Map()));
	let catalog = catalogs.get(prefix);
	if (!catalog) {
		catalog = readCatalog(bucket, prefix, index).catch((cause: unknown) => {
			catalogs.delete(prefix);
			throw cause;
		});
		catalogs.set(prefix, catalog);
	}
	return catalog;
}

async function readCatalog(bucket: R2Bucket, prefix: string, index?: VectorizeIndex) {
	const get = async (name: string) => {
		const object = await bucket.get(prefix + name);
		if (!object) throw new Error(`missing catalog file: ${name}`);
		return object;
	};
	if (index) {
		const catalog = await (await get('search-index.json')).json<CatalogIndex>();
		const [anchors, anchorVectors] =
			catalog.info.heads.kind === 'anchors'
				? await Promise.all([
						get('anchors.json').then((object) => object.json<Anchor[]>()),
						get('anchors.bin').then((object) => object.arrayBuffer())
					])
				: [[], new ArrayBuffer(0)];
		return createVectorizeSearch(index, prefix, catalog, anchors, new Float32Array(anchorVectors));
	}
	const info = await (await get('catalog.json')).json<CatalogInfo>();
	const [items, vectors, anchors, anchorVectors] = await Promise.all([
		get('items.json').then((object) => object.json<Item[]>()),
		get('vectors.bin').then((object) => object.arrayBuffer()),
		info.heads.kind === 'anchors'
			? get('anchors.json').then((object) => object.json<Anchor[]>())
			: [],
		info.heads.kind === 'anchors'
			? get('anchors.bin').then((object) => object.arrayBuffer())
			: new ArrayBuffer(0)
	]);
	return createSearch(
		info,
		items,
		new Float32Array(vectors),
		anchors,
		new Float32Array(anchorVectors)
	);
}
