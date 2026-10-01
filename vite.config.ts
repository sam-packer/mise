// Configure the Svelte app build for Cloudflare and load the inference runtime from its CDN.
import tailwindcss from '@tailwindcss/vite';
import adapter from '@sveltejs/adapter-cloudflare';
import { sveltekit } from '@sveltejs/kit/vite';
import { defaultClientConditions, defineConfig } from 'vite';

export default defineConfig({
	// The worker loads ORT WASM from the CDN to exclude files larger than 25 MiB from the app build.
	// Keep Vite's default conditions so Svelte resolves to its browser build and runs onMount.
	resolve: { conditions: [...defaultClientConditions, 'onnxruntime-web-use-extern-wasm'] },
	// Leave the preview image renderer to the runtime: the dev server loads its Node build, and
	// Wrangler bundles its workerd build, with the WebAssembly modules, into the worker.
	ssr: { external: ['@cf-wasm/og', '@cf-wasm/resvg', '@cf-wasm/satori'] },
	// The ML project holds over 100,000 files, and a training run writes logs all the time.
	// The app never imports from it, so the dev server does not watch it.
	server: { watch: { ignored: ['**/ml/**'] } },
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
