import type { Item } from './types';
import { allWords } from './anchor-search';
import { representative } from './search';

export type NameData = { words: string[]; representatives: Record<string, number> };

export function buildNameData(
	items: Item[],
	vectors: Float32Array,
	dims: number,
	words: string[]
): NameData {
	if (vectors.length !== items.length * dims) throw new Error('vectors.bin does not match items');
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
		if (rows.length > 1) representatives[rows.join(',')] = representative(vectors, dims, rows);
	}
	const nameWords = new Set(
		items.flatMap((item) => allWords([item.title, item.creator, item.album ?? ''].join(' ')))
	);
	return {
		words: words.filter((word) => nameWords.has(word)),
		representatives
	};
}
