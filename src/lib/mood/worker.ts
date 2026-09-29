// Web Worker: loads the bundle, runs the encoder, and answers infer requests.

import * as ort from 'onnxruntime-web';
import { createMoodEngine, type MoodEngine } from './engine';
import type { WorkerRequest, WorkerResponse } from './client';

// Version of the installed onnxruntime-web; the WASM is larger than the asset limit, so it comes from the CDN.
const ORT_VERSION = '1.30.0';

ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
ort.env.wasm.numThreads = 1;

const post = (msg: WorkerResponse) => self.postMessage(msg);

// The main thread sends the bundle root (a same-origin path or a CDN URL) as the first message.
let engine: Promise<MoodEngine> | null = null;

function load(base: string): Promise<MoodEngine> {
	const fetchOk = async (path: string): Promise<Response> => {
		const res = await fetch(new URL(base + path, self.location.origin));
		if (!res.ok) throw new Error(`${path}: ${res.status}`);
		return res;
	};
	const loading = createMoodEngine({
		ort,
		fetchBytes: (path) => fetchOk(path).then((r) => r.arrayBuffer()),
		fetchJson: (path) => fetchOk(path).then((r) => r.json()),
		async match(query, embedding, version) {
			const res = await fetch(new URL('/api/match', self.location.origin), {
				method: 'POST',
				headers: { 'content-type': 'application/json', accept: 'application/json' },
				body: JSON.stringify({ query, embedding, version })
			});
			if (!res.ok) {
				const body = (await res.json().catch(() => null)) as { message?: unknown } | null;
				throw new Error(typeof body?.message === 'string' ? body.message : `match: ${res.status}`);
			}
			return res.json();
		}
	});
	loading.then(
		() => post({ type: 'ready' }),
		(e: unknown) => post({ type: 'failed', message: e instanceof Error ? e.message : String(e) })
	);
	return loading;
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
	const msg = e.data;
	if (msg.type === 'init') {
		engine ??= load(msg.base);
		return;
	}
	const { id, text } = msg;
	try {
		if (!engine) throw new Error('worker was not initialized');
		const ready = await engine;
		const mood = await ready.infer(text, (local) => post({ type: 'local', id, mood: local }));
		post({ type: 'result', id, mood });
	} catch (err) {
		post({ type: 'error', id, message: err instanceof Error ? err.message : String(err) });
	}
};
