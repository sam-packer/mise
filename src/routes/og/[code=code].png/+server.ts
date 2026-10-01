// Serve the link preview image of a shared feeling, and keep it in the Workers cache.
import { error } from '@sveltejs/kit';
import { cache as assets } from '@cf-wasm/og';
import { BUNDLE } from '$lib/code';
import { card } from '$lib/server/og';
import { paletteKey, parsePalette } from '$lib/server/palette';
import type { RequestHandler } from './$types';

/**
 * A stored palette never changes in its bundle. The page links the image with its bundle in the
 * query, so a new bundle gets a new URL.
 */
const FINAL = 'public, max-age=31536000, immutable';
/** The brand colors stand in until the palette arrives, so check again soon. */
const INTERIM = 'public, max-age=60';

type Store = {
	match(key: Request): Promise<Response | undefined>;
	put(key: Request, res: Response): Promise<void>;
};

export const GET: RequestHandler = async ({ params, platform, url }) => {
	const moods = platform?.env.MOODS;
	const palettes = platform?.env.PALETTES;
	if (!moods || !palettes) error(503, 'no feeling store');

	// Key the cache on this build's bundle, not on the query, so a query string cannot make a second
	// copy. The local dev cache stores nothing. The platform types use the Workers Request and
	// Response, which differ from the global ones.
	const store = platform?.caches?.default as unknown as Store | undefined;
	const key = new Request(`${url.origin}${url.pathname}?b=${BUNDLE}`);
	const hit = await store?.match(key);
	if (hit) return hit;

	const [text, stored] = await Promise.all([
		moods.get(params.code),
		palettes.get(paletteKey(BUNDLE, params.code))
	]);
	if (text === null) error(404, 'no such feeling');
	const palette = stored === null ? null : parsePalette(JSON.parse(stored));

	// Let the renderer cache the fallback fonts that it fetches for emoji and non-Latin scripts.
	if (platform?.ctx) assets.setExecutionContext(platform.ctx);
	const png = await card(text, palette);
	const res = new Response(png, {
		headers: { 'content-type': 'image/png', 'cache-control': palette ? FINAL : INTERIM }
	});
	if (store && platform?.ctx) platform.ctx.waitUntil(store.put(key, res.clone()));
	return res;
};
