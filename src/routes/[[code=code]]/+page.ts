// Resolve share codes, of a feeling or of a path through its worlds, from the browser session or the
// feeling API before rendering the page.
import { browser } from '$app/env';
import { BUNDLE, feelingCode, shared, type Shared } from '#lib/code.js';
import type { PageLoad } from './$types';

type Data = Shared & {
	/** The code of the feeling, for its link preview image. */
	feeling?: string;
	note?: string;
};

export const load: PageLoad = async ({ params, fetch }): Promise<Data> => {
	const code = params.code;
	if (!code) return { text: '', trail: [] };

	let found = browser ? shared.get(code) : undefined;
	if (!found) {
		let res: Response;
		try {
			res = await fetch(`/api/feeling/${code}`);
		} catch {
			return { text: '', trail: [], note: 'the moods are out — try again soon' };
		}
		if (res.status === 404) return { text: '', trail: [], note: 'that feeling has faded' };
		if (!res.ok) return { text: '', trail: [], note: 'the moods are out — try again soon' };
		found = (await res.json()) as Shared;
		if (browser) shared.set(code, found);
	}
	if (!found.trail.length) return { ...found, feeling: code };
	const feeling = await feelingCode(found.text);
	// The ids of a path name works in the catalog of its own bundle. Another bundle may not have them,
	// or may set other works around them, so show the feeling itself.
	if (found.bundle !== BUNDLE)
		return {
			text: found.text,
			trail: [],
			feeling,
			note: 'this path has changed since it was shared'
		};
	return { ...found, feeling };
};
