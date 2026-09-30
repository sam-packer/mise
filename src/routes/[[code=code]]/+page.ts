// Resolve share codes, of a feeling or of a path through its worlds, from the browser session or the
// feeling API before rendering the page.
import { browser } from '$app/environment';
import { feelingCode, shared, type Shared } from '$lib/code';
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
	return { ...found, feeling: found.trail.length ? await feelingCode(found.text) : code };
};
