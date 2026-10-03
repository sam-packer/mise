// Retrieve the stored feeling, or the stored path, behind a code for shared URLs and browser navigation.
import { error } from '@sveltejs/kit';
import { env } from 'cloudflare:workers';
import type { Shared } from '#lib/code.js';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, setHeaders }) => {
	let body: Shared;
	const text = await env.MOODS.get(params.code);
	if (text !== null) body = { text, trail: [] };
	else {
		const path = await env.PATHS.get(params.code);
		if (path === null) error(404, 'no such feeling');
		body = JSON.parse(path) as Shared;
	}

	// A code never changes its value once stored.
	setHeaders({ 'cache-control': 'public, max-age=31536000, immutable' });
	return Response.json(body);
};
