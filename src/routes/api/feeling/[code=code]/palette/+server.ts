// Store the palette of a feeling's mood once, so the feeling's link preview image shows its colors.
// The browser computes the mood, so the server only checks the shape of the palette.
import { error } from '@sveltejs/kit';
import { MAX_FEELING, feelingCode, isBundle, normalize } from '$lib/code';
import { paletteKey, parsePalette } from '$lib/server/palette';
import type { RequestHandler } from './$types';

export const PUT: RequestHandler = async ({ params, request, platform }) => {
	const body = (await request.json().catch(() => null)) as {
		text?: unknown;
		palette?: unknown;
		bundle?: unknown;
	} | null;
	const text = typeof body?.text === 'string' ? normalize(body.text) : '';
	if (!text || text.length > MAX_FEELING) error(400, 'bad feeling');
	const palette = parsePalette(body?.palette);
	if (!palette) error(400, 'bad palette');
	const bundle = body?.bundle;
	if (!isBundle(bundle)) error(400, 'bad bundle');
	// Accept a palette only for the code that the text produces.
	if ((await feelingCode(text)) !== params.code) error(400, 'the code does not match the feeling');

	const moods = platform?.env.MOODS;
	const palettes = platform?.env.PALETTES;
	if (!moods || !palettes) error(503, 'no feeling store');
	if ((await moods.get(params.code)) !== text) error(404, 'no such feeling');

	// Keep the first palette of each bundle, so the preview image of a code changes only with the model.
	const key = paletteKey(bundle, params.code);
	if ((await palettes.get(key)) === null) await palettes.put(key, JSON.stringify(palette));
	return new Response(null, { status: 204 });
};
