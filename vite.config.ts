// Configure the Svelte app build for Cloudflare and load the inference runtime from its CDN.
import tailwindcss from '@tailwindcss/vite';
import adapter from '@sveltejs/adapter-cloudflare';
import { sveltekit } from '@sveltejs/kit/vite';
import { defaultClientConditions, defineConfig } from 'vite';

export default defineConfig({
	// The worker loads ORT WASM from the CDN to exclude files larger than 25 MiB from the app build.
	// Keep Vite's default conditions so Svelte resolves to its browser build and runs onMount.
	resolve: { conditions: [...defaultClientConditions, 'onnxruntime-web-use-extern-wasm'] },
	plugins: [
		tailwindcss(),
		sveltekit({
			compilerOptions: {
				// Use runes in app components and preserve each library's compiler mode.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			adapter: adapter()
		})
	]
});
