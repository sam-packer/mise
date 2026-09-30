// Store feeling text, or a path through the worlds of a feeling, under a code derived here, not
// supplied by the client. Keep the first value stored under each code so shared URLs retain their meaning.
import { error, json } from '@sveltejs/kit';
import { MAX_FEELING, MAX_TRAIL, feelingCode, isItemId, normalize, pathCode } from '$lib/code';
import { pathKey } from '$lib/server/path';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request, platform }) => {
	const body = (await request.json().catch(() => null)) as {
		text?: unknown;
		trail?: unknown;
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

	const moods = platform?.env.MOODS;
	if (!moods) error(503, 'no feeling store');

	if (trail === undefined) {
		const code = await feelingCode(text);
		if ((await moods.get(code)) === null) await moods.put(code, text);
		return json({ code });
	}
	const code = await pathCode(text, trail);
	const key = pathKey(code);
	if ((await moods.get(key)) === null) await moods.put(key, JSON.stringify({ text, trail }));
	return json({ code });
};
