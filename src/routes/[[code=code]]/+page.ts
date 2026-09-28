import { browser } from '$app/environment';
import { feelings } from '$lib/code';
import type { PageLoad } from './$types';

/** The feeling a code names: from this session when the browser knows it, else from the store. */
export const load: PageLoad = async ({ params, fetch }) => {
	const code = params.code;
	if (!code) return { text: '' };

	const known = browser ? feelings.get(code) : undefined;
	if (known !== undefined) return { text: known };

	let res: Response;
	try {
		res = await fetch(`/api/feeling/${code}`);
	} catch {
		return { text: '', note: 'the moods are out — try again soon' };
	}
	if (res.status === 404) return { text: '', note: 'that feeling has faded' };
	if (!res.ok) return { text: '', note: 'the moods are out — try again soon' };

	const { text }: { text: string } = await res.json();
	if (browser) feelings.set(code, text);
	return { text };
};
