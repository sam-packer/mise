<script lang="ts">
	// Link a song to each music service that has it. Each name opens that service directly.
	import type { Item } from '$lib/mood/types';
	import Chevron from './Chevron.svelte';

	type Platform = 'spotify' | 'apple' | 'deezer' | 'youtube';
	const PLATFORMS: Platform[] = ['spotify', 'apple', 'deezer', 'youtube'];
	const LABELS: Record<Platform, string> = {
		spotify: 'Spotify',
		apple: 'Apple Music',
		deezer: 'Deezer',
		youtube: 'YouTube'
	};

	let { links }: { links: Item['links'] } = $props();

	const available = $derived(PLATFORMS.filter((p) => links[p]));
</script>

<div class="links">
	<p class="label">listen on</p>
	{#if available.length}
		<ul>
			{#each available as p (p)}
				<li>
					<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
					<a href={links[p]} target="_blank" rel="noopener">{LABELS[p]}</a>
				</li>
			{/each}
		</ul>
	{:else}
		<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
		<a href={links.primary} target="_blank" rel="noopener">the song<Chevron /></a>
	{/if}
</div>

<style>
	.links {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		align-items: flex-start;
	}

	.label {
		margin: 0;
		color: var(--ink-soft);
		font-size: 0.85rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
	}

	ul {
		display: flex;
		flex-wrap: wrap;
		gap: 0 1.1rem;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	/* Vertical padding makes a 44px touch target without moving the text. */
	a {
		display: inline-block;
		padding: 0.7rem 0;
		color: var(--ink);
		text-decoration: underline 1px color-mix(in oklab, var(--ink) 40%, transparent);
		text-underline-offset: 0.3em;
		transition: text-decoration-color 300ms var(--ease);
	}

	a:hover,
	a:focus-visible {
		text-decoration-color: var(--ink);
	}

	a:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: 2px;
	}
</style>
