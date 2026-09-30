// Build the name index for the exporter from item vectors before fp16 rounding.
import { readFile, writeFile } from 'node:fs/promises';
import { buildNameData } from '../src/lib/mood/name-data';
import type { Item } from '../src/lib/mood/types';

const [itemsPath, vectorsPath, wordsPath, output] = process.argv.slice(2);
if (!output)
	throw new Error('Use: bun scripts/build-name-data.ts <items> <fp32 vectors> <words> <output>');
const items: Item[] = JSON.parse(await readFile(itemsPath, 'utf8'));
const words: string[] = JSON.parse(await readFile(wordsPath, 'utf8'));
const bytes = await readFile(vectorsPath);
const vectors = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
await writeFile(
	output,
	JSON.stringify(buildNameData(items, vectors, vectors.length / items.length, words))
);
