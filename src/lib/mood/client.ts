// Send inference requests to the browser worker and pair responses with their callers.

import type { Mood } from './types';

export type WorkerRequest =
	{ type: 'init'; base: string } | { type: 'infer'; id: number; text: string };
export type WorkerResponse =
	| { type: 'ready' }
	| { type: 'failed'; message: string }
	| { type: 'result'; id: number; mood: Mood }
	| { type: 'error'; id: number; message: string };

type Pending = {
	resolve: (m: Mood) => void;
	reject: (e: Error) => void;
};

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, Pending>();

let resolveReady: () => void;
let rejectReady: (e: Error) => void;

/** Resolves when the bundle and the model are loaded. Rejects when the load fails. */
export const ready: Promise<void> = new Promise((res, rej) => {
	resolveReady = res;
	rejectReady = rej;
});

/** Start loading once. The bundle root can use any origin and must end in '/'. */
export function start(base = '/bundle/') {
	if (worker || typeof Worker === 'undefined') return;
	// Vite requires this `new Worker(new URL(...))` form to find and bundle the worker.
	worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
	worker.postMessage({ type: 'init', base } satisfies WorkerRequest);
	worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
		const msg = e.data;
		if (msg.type === 'ready') resolveReady();
		else if (msg.type === 'failed') rejectReady(new Error(msg.message));
		else {
			const p = pending.get(msg.id);
			if (!p) return;
			pending.delete(msg.id);
			if (msg.type === 'result') p.resolve(msg.mood);
			else p.reject(new Error(msg.message));
		}
	};
	worker.onerror = (e) => {
		rejectReady(new Error(e.message || 'worker failed'));
		for (const p of pending.values()) p.reject(new Error('worker failed'));
		pending.clear();
	};
}

/** Wait for the bundle to load, then request a mood from the worker. */
export async function infer(text: string): Promise<Mood> {
	start();
	await ready;
	const id = nextId++;
	return new Promise((resolve, reject) => {
		pending.set(id, { resolve, reject });
		worker!.postMessage({ type: 'infer', id, text } satisfies WorkerRequest);
	});
}
