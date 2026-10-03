// Serve the link preview image of a shared feeling, and keep it in the Workers cache.
import { error } from '@sveltejs/kit';
import { cache as assets } from '@cf-wasm/og';
import { env, waitUntil } from 'cloudflare:workers';
import { BUNDLE } from '#lib/code.js';
import { card } from '#lib/server/og.js';
import { paletteKey, parsePalette } from '#lib/server/palette.js';
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

export const GET: RequestHandler = async ({ params, url }) => {
	// Key the cache on this build's bundle, not on the query, so a query string cannot make a second
	// copy. The local dev cache stores nothing. The page types give `caches` the DOM type, which
	// lacks the Workers default cache and uses other Request and Response types.
	const store = (caches as unknown as { default: Store }).default;
	const key = new Request(`${url.origin}${url.pathname}?b=${BUNDLE}`);
	const hit = await store.match(key);
	if (hit) return hit;

	const [text, stored] = await Promise.all([
		env.MOODS.get(params.code),
		env.PALETTES.get(paletteKey(BUNDLE, params.code))
	]);
	if (text === null) error(404, 'no such feeling');
	const palette = stored === null ? null : parsePalette(JSON.parse(stored));

	// Let the renderer cache the fallback fonts that it fetches for emoji and non-Latin scripts.
	assets.setExecutionContext({ waitUntil });
	const png = await card(text, palette);
	const res = new Response(png, {
		headers: { 'content-type': 'image/png', 'cache-control': palette ? FINAL : INTERIM }
	});
	waitUntil(store.put(key, res.clone()));
	return res;
};
