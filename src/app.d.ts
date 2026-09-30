// Define Cloudflare bindings and navigation state for SvelteKit routes.
declare global {
	namespace App {
		interface Platform {
			env: Env;
			ctx: ExecutionContext;
			caches: CacheStorage;
			cf?: IncomingRequestCfProperties;
		}

		interface PageState {
			/** The item ids of the worlds on the path from the feeling, in order. The last one is on screen. */
			trail?: string[];
		}
	}
}

export {};
