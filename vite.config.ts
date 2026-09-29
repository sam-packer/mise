import tailwindcss from '@tailwindcss/vite';
import adapter from '@sveltejs/adapter-cloudflare';
import { sveltekit } from '@sveltejs/kit/vite';
import { defaultClientConditions, defineConfig } from 'vite';

export default defineConfig({
	// The ORT WASM loads from the CDN (see worker.ts). The extra condition keeps Vite from emitting the
	// >25 MiB .wasm files. Setting `conditions` replaces Vite's defaults, so keep them: without
	// `browser`, `svelte` resolves to its server build and `onMount` never runs.
	resolve: { conditions: [...defaultClientConditions, 'onnxruntime-web-use-extern-wasm'] },
	plugins: [
		tailwindcss(),
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			adapter: adapter()
		})
	]
});
