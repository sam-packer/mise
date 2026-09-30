<script lang="ts">
	// Show song links and remember the user's preferred music service when storage is available.
	import type { Item } from '$lib/mood/types';

	type Platform = 'spotify' | 'apple' | 'deezer' | 'youtube';
	const PLATFORMS: Platform[] = ['spotify', 'apple', 'deezer', 'youtube'];
	const KEY = 'moodboard:platform';
	const LABELS: Record<Platform, string> = {
		spotify: 'Spotify',
		apple: 'Apple Music',
		deezer: 'Deezer',
		youtube: 'YouTube'
	};

	let { links }: { links: Item['links'] } = $props();

	function stored(): Platform {
		try {
			const v = localStorage.getItem(KEY);
			if (PLATFORMS.includes(v as Platform)) return v as Platform;
		} catch {
			// Storage can be unavailable during server rendering or blocked by the browser.
		}
		return 'spotify';
	}

	let platform = $state<Platform>(stored());
	const available = $derived(PLATFORMS.filter((p) => links[p]));
	const href = $derived(links[platform] ?? links[available[0]] ?? links.primary);
	const name = $derived(
		links[platform] ? LABELS[platform] : available[0] ? LABELS[available[0]] : 'the song'
	);

	function choose(p: Platform) {
		platform = p;
		try {
			localStorage.setItem(KEY, p);
		} catch {
			// Keep the selection for this view when the browser blocks storage.
		}
	}
</script>

<div class="links">
	<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
	<a {href} target="_blank" rel="noopener">open in {name} →</a>
	{#if available.length > 1}
		<div class="switch" role="group" aria-label="music service">
			{#each available as p (p)}
				<button
					type="button"
					class:on={p === platform}
					aria-pressed={p === platform}
					onclick={() => choose(p)}
				>
					{LABELS[p]}
				</button>
			{/each}
		</div>
	{/if}
</div>

<style>
	.links {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
		align-items: flex-start;
	}

	a {
		color: var(--ink);
		text-decoration: none;
		border-bottom: 1px solid color-mix(in oklab, var(--ink) 40%, transparent);
		transition: border-color 300ms var(--ease);
	}

	a:hover,
	a:focus-visible {
		border-color: var(--ink);
	}

	.switch {
		display: flex;
		gap: 1rem;
		font-size: 0.85rem;
	}

	button {
		padding: 0;
		border: 0;
		background: none;
		color: var(--ink-soft);
		font: inherit;
		text-decoration: underline 1px transparent;
		text-underline-offset: 0.3em;
		cursor: pointer;
		transition:
			color 300ms var(--ease),
			text-decoration-color 300ms var(--ease);
	}

	button.on,
	button:hover,
	button:focus-visible {
		color: var(--ink);
	}

	button.on {
		text-decoration-color: currentColor;
	}

	button:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: 4px;
	}
</style>
