// Store feeling text, or a path through the worlds of a feeling, under a code derived here, not
// supplied by the client. Keep the first value stored under each code so shared URLs retain their meaning.
import { error } from '@sveltejs/kit';
import { env } from 'cloudflare:workers';
import {
	MAX_FEELING,
	MAX_TRAIL,
	feelingCode,
	isBundle,
	isItemId,
	normalize,
	pathCode
} from '#lib/code.js';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
	const body = (await request.json().catch(() => null)) as {
		text?: unknown;
		trail?: unknown;
		bundle?: unknown;
	} | null;
	const text = typeof body?.text === 'string' ? normalize(body.text) : '';
	if (!text || text.length > MAX_FEELING) error(400, 'bad feeling');
	const trail = body?.trail;
	if (
		trail !== undefined &&
		(!Array.isArray(trail) ||
			trail.length < 1 ||
			trail.length > MAX_TRAIL ||
			!trail.every(isItemId))
	)
		error(400, 'bad path');
	// The page names its own bundle, so a page from before a deploy still gets the code it computed.
	const bundle = body?.bundle;
	if (trail !== undefined && !isBundle(bundle)) error(400, 'bad bundle');

	const store = trail === undefined ? env.MOODS : env.PATHS;

	if (trail === undefined) {
		const code = await feelingCode(text);
		if ((await store.get(code)) === null) await store.put(code, text);
		return Response.json({ code });
	}
	const code = await pathCode(text, trail, bundle as string);
	if ((await store.get(code)) === null)
		await store.put(code, JSON.stringify({ text, trail, bundle }));
	return Response.json({ code });
};
