import type { Anchor, Item } from '../mood/types';
import { CATALOG_PREFIX } from './catalog';
import { createSearch, type CatalogInfo } from './search';

const loaded = new WeakMap<R2Bucket, Map<string, Promise<ReturnType<typeof createSearch>>>>();

export function loadCatalog(bucket: R2Bucket, prefix = CATALOG_PREFIX) {
	let catalogs = loaded.get(bucket);
	if (!catalogs) loaded.set(bucket, (catalogs = new Map()));
	let catalog = catalogs.get(prefix);
	if (!catalog) {
		catalog = readCatalog(bucket, prefix).catch((cause: unknown) => {
			catalogs.delete(prefix);
			throw cause;
		});
		catalogs.set(prefix, catalog);
	}
	return catalog;
}

async function readCatalog(bucket: R2Bucket, prefix: string) {
	const get = async (name: string) => {
		const object = await bucket.get(prefix + name);
		if (!object) throw new Error(`missing catalog file: ${name}`);
		return object;
	};
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
