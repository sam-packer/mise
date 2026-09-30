<script lang="ts">
	// Arrange category picks as tiles and coordinate their exit before the next mood appears.
	import { CATEGORIES, type Category, type Item } from '$lib/mood/types';
	import Tile from './Tile.svelte';

	let {
		picks,
		leaving,
		active,
		onopen
	}: {
		picks: Record<Category, Item>;
		/** Fade the wall out before the next mood arrives. */
		leaving: boolean;
		active: Category | null;
		onopen: (item: Item, el: HTMLButtonElement) => void;
	} = $props();
</script>

<section class="wall" class:leaving aria-label="picks">
	{#each CATEGORIES as category, i (category)}
		{#if picks[category]}
			<Tile item={picks[category]} index={i} active={active === category} {onopen} />
		{/if}
	{/each}
	<div class="spacer" aria-hidden="true"></div>
</section>

<style>
	.wall {
		--space: max(2.5vw, 12px);
		--min-height: clamp(160px, 18vw, 360px);
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		gap: var(--space);
		width: 100%;
		max-width: 1440px;
		margin: 0 auto;
		padding: var(--space);
		box-sizing: border-box;
		transition: opacity 200ms ease-out;
	}

	.wall.leaving {
		opacity: 0;
		pointer-events: none;
	}

	.spacer {
		flex-grow: 1000000000;
	}
</style>
