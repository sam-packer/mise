// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces
import type { Category } from '$lib/mood/types';

declare global {
	namespace App {
		interface Platform {
			env: Env & { CATALOG_PREFIX?: string };
			ctx: ExecutionContext;
			caches: CacheStorage;
			cf?: IncomingRequestCfProperties;
		}

		interface PageState {
			/** The category whose full view is open, or the anchor the feeling names. */
			open?: Category | 'anchor';
		}

		// interface Error {}
		// interface Locals {}
		// interface PageData {}
	}
}

export {};
