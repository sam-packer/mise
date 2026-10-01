// Retrieve the stored feeling, or the stored path, behind a code for shared URLs and browser navigation.
import { error, json } from '@sveltejs/kit';
import type { Shared } from '$lib/code';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, platform, setHeaders }) => {
	const moods = platform?.env.MOODS;
	const paths = platform?.env.PATHS;
	if (!moods || !paths) error(503, 'no feeling store');

	let body: Shared;
	const text = await moods.get(params.code);
	if (text !== null) body = { text, trail: [] };
	else {
		const path = await paths.get(params.code);
		if (path === null) error(404, 'no such feeling');
		body = JSON.parse(path) as Shared;
	}

	// A code never changes its value once stored.
	setHeaders({ 'cache-control': 'public, max-age=31536000, immutable' });
	return json(body);
};
