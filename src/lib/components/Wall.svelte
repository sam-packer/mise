<script lang="ts">
	// Arrange items as tiles and coordinate their exit before the next mood or world appears.
	import type { Item } from '#lib/mood/types.js';
	import Tile from './Tile.svelte';

	let {
		items,
		label,
		leaving,
		active,
		returned,
		onopen
	}: {
		items: Item[];
		label: string;
		/** Fade the wall out before the next mood arrives. */
		leaving: boolean;
		/** The id of the tile that a world grows from or returns to. */
		active: string | null;
		/** The id of the tile that the path went on through, after a step back over several worlds. */
		returned: string | null;
		onopen: (item: Item, el: HTMLButtonElement) => void;
	} = $props();
</script>

<section class="wall" class:leaving aria-label={label}>
	{#each items as item, i (item.id)}
		<Tile {item} index={i} active={active === item.id} returned={returned === item.id} {onopen} />
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
