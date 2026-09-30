// Redirect audio playback to a fresh Deezer preview URL.
// Preview URLs expire after about 15 minutes, so the bundle stores this route.
import { error, redirect } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, fetch, setHeaders }) => {
	if (!/^\d+$/.test(params.id)) error(400, 'bad track id');

	const res = await fetch(`https://api.deezer.com/track/${params.id}`);
	if (!res.ok) error(502, 'deezer is unavailable');
	const track: { preview?: string; error?: unknown } = await res.json();
	if (track.error || !track.preview) error(404, 'no preview for this track');

	// Shorter than the link's lifetime, so a cached redirect never points at an expired link.
	setHeaders({ 'cache-control': 'public, max-age=300' });
	redirect(302, track.preview);
};
