import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, platform, setHeaders }) => {
	const moods = platform?.env.MOODS;
	if (!moods) error(503, 'no feeling store');

	const text = await moods.get(params.code);
	if (text === null) error(404, 'no such feeling');

	// A code never changes its text once stored.
	setHeaders({ 'cache-control': 'public, max-age=31536000, immutable' });
	return json({ text });
};
