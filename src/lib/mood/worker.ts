// Web Worker: loads the bundle, runs the encoder, and answers infer requests.

import * as ort from 'onnxruntime-web';
import { createMoodEngine, type MoodEngine } from './engine';
import type { WorkerRequest, WorkerResponse } from './client';

const BUNDLE = '/bundle/';
// Version of the installed onnxruntime-web; the WASM is larger than the asset limit, so it comes from the CDN.
const ORT_VERSION = '1.30.0';

ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
ort.env.wasm.numThreads = 1;

const post = (msg: WorkerResponse) => self.postMessage(msg);

async function fetchOk(path: string): Promise<Response> {
	const res = await fetch(new URL(BUNDLE + path, self.location.origin));
	if (!res.ok) throw new Error(`${path}: ${res.status}`);
	return res;
}

const engine: Promise<MoodEngine> = createMoodEngine({
	ort,
	fetchBytes: (path) => fetchOk(path).then((r) => r.arrayBuffer()),
	fetchJson: (path) => fetchOk(path).then((r) => r.json())
});

engine.then(
	() => post({ type: 'ready' }),
	(e: unknown) => post({ type: 'failed', message: e instanceof Error ? e.message : String(e) })
);

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
	const { id, text } = e.data;
	try {
		const mood = await (await engine).infer(text);
		post({ type: 'result', id, mood });
	} catch (err) {
		post({ type: 'error', id, message: err instanceof Error ? err.message : String(err) });
	}
};
