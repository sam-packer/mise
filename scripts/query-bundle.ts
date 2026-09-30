// Run feelings through static/bundle/ with the browser worker's engine and print the selected items.
// Run: bun scripts/query-bundle.ts "driving home at 2am with the windows down"
import * as ortNode from 'onnxruntime-node';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { createMoodEngine, type EngineIO } from '../src/lib/mood/engine';
import { CATEGORIES, type OKLab } from '../src/lib/mood/types';

const BUNDLE = path.resolve(import.meta.dirname, '..', 'static', 'bundle');

function oklabToHex([L, a, b]: OKLab): string {
	const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
	const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
	const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
	const rgb = [
		4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
		-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
		-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s
	];
	return (
		'#' +
		rgb
			.map((c) => {
				const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
				return Math.round(Math.min(1, Math.max(0, v)) * 255)
					.toString(16)
					.padStart(2, '0');
			})
			.join('')
	);
}

const queries = process.argv.slice(2);
if (!queries.length) {
	console.error('usage: bun scripts/query-bundle.ts "<feeling>" ["<feeling>" ...]');
	process.exit(1);
}

const io: EngineIO = {
	ort: ortNode as unknown as EngineIO['ort'],
	async fetchBytes(p) {
		const b = await readFile(path.join(BUNDLE, p));
		return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
	},
	fetchJson: async (p) => JSON.parse(await readFile(path.join(BUNDLE, p), 'utf8'))
};

const engine = await createMoodEngine(io);
await engine.infer('warm up');
for (const query of queries) {
	const mood = await engine.infer(query);
	console.log(`\n"${mood.query}"`);
	const a = mood.anchor;
	console.log(
		`  anchor    ${a ? `${a.title}, ${a.creator}${a.album ? ` [${a.album}]` : ''} (${a.category})` : 'none'}`
	);
	console.log(`  palette   ${mood.palette.map(oklabToHex).join(' ')}`);
	console.log(`  light     ${mood.light}`);
	console.log(`  typeface  ${mood.typeface.family}`);
	console.log(`  scent     ${mood.scent.text}`);
	for (const c of CATEGORIES) {
		const item = mood.picks[c];
		console.log(
			`  ${c.padEnd(9)} ${item ? `${item.title}, ${item.creator} (${item.year})` : 'none'}`
		);
	}
	console.log(`  ms        ${mood.ms.toFixed(1)}`);
}
