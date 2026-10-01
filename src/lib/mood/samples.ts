// Read the rooms of the sample feelings from the bundle. `bun scripts/build-samples.ts` computes them
// with the same engine when the bundle is exported, so a tap on a sample shows its room before the
// model loads. A feeling without a file, or a file that does not load, goes to the model.

import type { Mood, Palette, World } from './types';

/** The room of one sample feeling: its mood, and the world behind each of its tiles, by item id. */
export type Sample = { mood: Mood; worlds: Record<string, World> };

/** Each sample feeling's file in the samples folder and its palette, by the feeling's text. */
export type SampleIndex = Record<string, { file: string; palette: Palette }>;

let setBase: (base: string) => void;
/** The bundle root. The page sets it when it starts the model. */
const base = new Promise<string>((resolve) => (setBase = resolve));

async function getJson<T>(url: string): Promise<T> {
	// A request that hangs must not hold a feeling back from the model.
	const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
	if (!res.ok) throw new Error(`${url}: ${res.status}`);
	return res.json();
}

/** The bundle's sample index. Empty when the bundle has none. */
const index: Promise<SampleIndex> = base
	.then((b) => getJson<SampleIndex>(`${b}samples/index.json`))
	.catch(() => ({}));

const rooms = new Map<string, Promise<Sample | null>>();

/** Start to read the samples of the bundle at this root, which ends in '/'. */
export function loadSamples(bundleBase: string) {
	setBase(bundleBase);
}

/** The palette of each sample feeling, by its text. */
export const samplePalettes: Promise<Record<string, Palette>> = index.then((i) =>
	Object.fromEntries(Object.entries(i).map(([text, entry]) => [text, entry.palette]))
);

/** The precomputed room of a feeling, or null when the feeling has none or its file does not load. */
export function sample(text: string): Promise<Sample | null> {
	let room = rooms.get(text);
	if (!room) {
		room = Promise.all([base, index])
			.then(([b, i]) =>
				Object.hasOwn(i, text) ? getJson<Sample>(`${b}samples/${i[text].file}`) : null
			)
			.catch(() => null);
		rooms.set(text, room);
	}
	return room;
}
