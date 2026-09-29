import type { Category, Item } from '../mood/types';
import type { AnchorItem } from './anchor-search';
import { representative, type CatalogInfo } from './search';

export type CatalogIndex = {
	info: CatalogInfo;
	items: [Category, string, string, string | null][];
	representatives: Record<string, number>;
};

export function expandItems(index: CatalogIndex): AnchorItem[] {
	return index.items.map(([category, title, creator, album]) => ({
		category,
		title,
		creator,
		...(album === null ? {} : { album })
	}));
}

/** Use the first row as a collision-free identifier for each exact string. */
export function exclusionKeys(items: AnchorItem[]) {
	const creators = new Map<string, number>();
	const titles = new Map<string, number>();
	return items.map((item, row) => {
		if (!creators.has(item.creator)) creators.set(item.creator, row);
		if (!titles.has(item.title)) titles.set(item.title, row);
		return { creator: creators.get(item.creator)!, title: titles.get(item.title)! };
	});
}

export function buildCatalogIndex(
	info: CatalogInfo,
	items: Item[],
	vectors: Float32Array
): CatalogIndex {
	if (items.length !== info.counts.items || vectors.length !== items.length * info.dims)
		throw new Error('vectors.bin does not match items');
	const groups = new Map<string, number[]>();
	items.forEach((item, row) => {
		for (const key of [
			JSON.stringify([item.creator]),
			JSON.stringify([item.creator, item.album])
		]) {
			const rows = groups.get(key) ?? [];
			rows.push(row);
			groups.set(key, rows);
		}
	});
	const representatives: Record<string, number> = {};
	for (const rows of groups.values()) {
		if (rows.length > 1) representatives[rows.join(',')] = representative(vectors, info.dims, rows);
	}
	return {
		info,
		items: items.map((item) => [item.category, item.title, item.creator, item.album ?? null]),
		representatives
	};
}
