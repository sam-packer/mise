// Define Cloudflare bindings and navigation state for SvelteKit routes.
import type { Category } from '$lib/mood/types';

declare global {
	namespace App {
		interface Platform {
			env: Env;
			ctx: ExecutionContext;
			caches: CacheStorage;
			cf?: IncomingRequestCfProperties;
		}

		interface PageState {
			/** The category whose full view is open, or the anchor the feeling names. */
			open?: Category | 'anchor';
		}
	}
}

export {};
