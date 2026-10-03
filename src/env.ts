import { defineEnvVars } from '@sveltejs/kit/env';

// Point the app at a local bundle in development. An empty value selects the published bundle.
export const variables = defineEnvVars({
	PUBLIC_BUNDLE_URL: { public: true, schema: (input) => input ?? '' }
});
