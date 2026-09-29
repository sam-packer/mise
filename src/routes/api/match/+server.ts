import { error, json } from '@sveltejs/kit';
import { MAX_FEELING, normalize } from '$lib/code';
import { loadCatalog } from '$lib/server/load-catalog';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request, platform, getClientAddress }) => {
	// The app has no accounts, so the client IP is the only key. The limit leaves room for shared IPs.
	const limiter = platform?.env.MATCH_LIMIT;
	if (limiter && !(await limiter.limit({ key: getClientAddress() })).success) {
		error(429, 'Too many searches. Wait a minute and try again.');
	}
	const body = (await request.json().catch(() => null)) as {
		query?: unknown;
		embedding?: unknown;
		version?: unknown;
	} | null;
	const query = typeof body?.query === 'string' ? normalize(body.query) : '';
	const embedding = body?.embedding;
	if (!query || query.length > MAX_FEELING) error(400, 'bad feeling');
	if (
		!Array.isArray(embedding) ||
		!embedding.every((n) => typeof n === 'number' && Number.isFinite(n))
	) {
		error(400, 'bad embedding');
	}

	const bucket = platform?.env.CATALOG;
	if (!bucket) error(503, 'no catalog store');
	const catalog = await loadCatalog(bucket, platform?.env.CATALOG_PREFIX || undefined);
	if (typeof body?.version !== 'string' || body.version !== catalog.version) {
		error(
			409,
			'The model and catalog versions do not match. Reload the page or install a matching bundle and catalog.'
		);
	}
	if (embedding.length !== catalog.dims) error(400, 'bad embedding');
	// Scale first to keep finite JSON numbers within the Float32 range.
	const scale = embedding.reduce((max, n) => Math.max(max, Math.abs(n)), 0);
	const scaled = embedding.map((n) => (scale ? n / scale : 0));
	const norm = Math.sqrt(scaled.reduce((sum, n) => sum + n * n, 0));
	const vector = Float32Array.from(scaled, (n) => (norm ? n / norm : 0));
	return json(catalog.match(query, vector));
};
