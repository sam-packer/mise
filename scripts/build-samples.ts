// Run every sample feeling in src/lib/examples.ts through a bundle with the browser worker's engine,
// and write the rooms into the bundle: samples/<code>.json holds the mood and the world behind each of
// its tiles, and samples/index.json names the file and the palette of each sample. The page shows a
// sample's room from its file while the model loads. The ML export runs this for each new bundle.
// Run: bun scripts/build-samples.ts <bundle folder>
import * as ort from 'onnxruntime-web';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { feelingCode } from '../src/lib/code';
import { EXAMPLES } from '../src/lib/examples';
import { createMoodEngine, type EngineIO } from '../src/lib/mood/engine';
import type { Sample, SampleIndex } from '../src/lib/mood/samples';
import type { OKLab, Palette } from '../src/lib/mood/types';

// The browser's runtime, as the worker sets it up. The int8 kernels of onnxruntime-node round
// differently, and change some picks.
ort.env.wasm.numThreads = 1;

export async function writeSamples(bundle: string) {
	const io: EngineIO = {
		ort,
		async fetchBytes(p) {
			const b = await readFile(path.join(bundle, p));
			return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
		},
		fetchJson: async (p) => JSON.parse(await readFile(path.join(bundle, p), 'utf8'))
	};
	const engine = await createMoodEngine(io);
	const index: SampleIndex = {};
	const files = new Map<string, Sample>();
	for (const text of EXAMPLES) {
		// A fixed time keeps the files the same from run to run.
		const mood = { ...(await engine.infer(text)), ms: 0 };
		const worlds: Sample['worlds'] = {};
		// A tile on the mood's wall opens the world of its item, with the item kept off that world's wall.
		for (const item of [...mood.picks, ...(mood.anchor ? [mood.anchor] : [])]) {
			worlds[item.id] = await engine.world(item.id, [item.id]);
		}
		const file = `${await feelingCode(text)}.json`;
		if (files.has(file)) throw new Error(`two samples share the file ${file}`);
		files.set(file, { mood, worlds });
		// The tagline only shows the palette, so four decimals are enough.
		const round = (c: OKLab) => c.map((v) => Math.round(v * 1e4) / 1e4) as OKLab;
		index[text] = { file, palette: mood.palette.map(round) as Palette };
	}

	const dir = path.join(bundle, 'samples');
	await rm(dir, { recursive: true, force: true });
	await mkdir(dir, { recursive: true });
	let bytes = 0;
	for (const [file, sample] of files) {
		const json = JSON.stringify(sample);
		bytes += Buffer.byteLength(json);
		await writeFile(path.join(dir, file), json);
	}
	await writeFile(path.join(dir, 'index.json'), JSON.stringify(index));
	console.log(`samples: ${files.size} rooms, ${(bytes / 1024).toFixed(0)} KiB -> ${dir}`);
}

if (import.meta.main) {
	const bundle = process.argv[2];
	if (!bundle) {
		console.error('usage: bun scripts/build-samples.ts <bundle folder>');
		process.exit(1);
	}
	await writeSamples(path.resolve(bundle));
}
