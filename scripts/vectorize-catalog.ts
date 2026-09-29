import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import type { Item } from '../src/lib/mood/types';
import { buildCatalogIndex, exclusionKeys } from '../src/lib/server/catalog-index';
import type { CatalogInfo } from '../src/lib/server/search';

const [directory, namespace, output] = process.argv.slice(2);
if (!directory || !namespace || !output)
	throw new Error('Use: bun scripts/vectorize-catalog.ts <catalog> <namespace> <output>');
if (Buffer.byteLength(namespace) > 64) throw new Error('Vectorize namespace exceeds 64 bytes');
const info: CatalogInfo = JSON.parse(await readFile(path.join(directory, 'catalog.json'), 'utf8'));
const items: Item[] = JSON.parse(await readFile(path.join(directory, 'items.json'), 'utf8'));
const bytes = await readFile(path.join(directory, 'vectors.bin'));
const vectors = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
if (info.dims !== 384) throw new Error('mise-catalog requires 384 dimensions');
const catalog = buildCatalogIndex(info, items, vectors);
const keys = exclusionKeys(items);
await mkdir(output, { recursive: true });
await writeFile(path.join(output, 'search-index.json'), JSON.stringify(catalog));
let maxMetadataBytes = 0;
const batches: string[] = [];
for (let start = 0; start < items.length; start += 5000) {
	const lines = items.slice(start, start + 5000).map((item, offset) => {
		const row = start + offset;
		const id = namespace + row;
		if (Buffer.byteLength(id) > 64) throw new Error('Vectorize ID exceeds 64 bytes');
		const metadata = { category: item.category, row, ...keys[row], item: JSON.stringify(item) };
		maxMetadataBytes = Math.max(maxMetadataBytes, Buffer.byteLength(JSON.stringify(metadata)));
		if (maxMetadataBytes > 10240) throw new Error(`${item.id}: metadata exceeds 10 KiB`);
		const values = Array.from(vectors.subarray(row * info.dims, (row + 1) * info.dims));
		if (values.some((value) => !Number.isFinite(value)))
			throw new Error(`${item.id}: invalid vector`);
		return JSON.stringify({ id, namespace, values, metadata });
	});
	const name = `vectors-${start}.ndjson`;
	await writeFile(path.join(output, name), lines.join('\n') + '\n');
	batches.push(name);
}
await writeFile(
	path.join(output, 'upload.json'),
	JSON.stringify({ namespace, count: items.length, maxMetadataBytes, batches })
);
console.log(`Prepared ${items.length} vectors; largest metadata record: ${maxMetadataBytes} bytes`);
