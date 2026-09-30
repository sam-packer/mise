// Store feeling text under a code derived here, not supplied by the client.
// Keep the first text stored under each code so shared URLs retain their meaning.
import { error, json } from '@sveltejs/kit';
import { MAX_FEELING, feelingCode, normalize } from '$lib/code';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request, platform }) => {
	const body = (await request.json().catch(() => null)) as { text?: unknown } | null;
	const text = typeof body?.text === 'string' ? normalize(body.text) : '';
	if (!text || text.length > MAX_FEELING) error(400, 'bad feeling');

	const moods = platform?.env.MOODS;
	if (!moods) error(503, 'no feeling store');

	const code = await feelingCode(text);
	if ((await moods.get(code)) === null) await moods.put(code, text);
	return json({ code });
};
