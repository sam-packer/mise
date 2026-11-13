// Tell the page which hemisphere the visitor is in, from the location that Cloudflare gives the
// request. A request without a location counts as the north.
import type { LayoutServerLoad } from './$types';

export const load: LayoutServerLoad = ({ request }) => {
	const latitude = Number(request.cf?.latitude);
	return { south: latitude < 0 };
};
