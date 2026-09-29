<script lang="ts">
	import { CATEGORIES, type Category, type Item } from '$lib/mood/types';
	import Tile from './Tile.svelte';

	let {
		picks,
		leaving,
		active,
		onopen
	}: {
		/** Null while the server search runs: the wall holds a quiet place for each pick. */
		picks: Record<Category, Item> | null;
		/** Fade the wall out before the next mood arrives. */
		leaving: boolean;
		active: Category | null;
		onopen: (item: Item, el: HTMLButtonElement) => void;
	} = $props();

	// The usual shape of each kind of pick, so the real tiles land close to where the placeholders sat.
	const SHAPES: Record<Category, [w: number, h: number]> = {
		art: [4, 5],
		film: [16, 9],
		song: [1, 1],
		poem: [3, 2],
		book: [2, 3]
	};
</script>

<section class="wall" class:leaving aria-label="picks" aria-busy={picks === null}>
	{#each CATEGORIES as category, i (category)}
		{#if picks}
			<Tile item={picks[category]} index={i} active={active === category} {onopen} />
		{:else}
			<div
				class="placeholder"
				style:--w={SHAPES[category][0]}
				style:--h={SHAPES[category][1]}
				style:--i={i}
				aria-hidden="true"
			>
				<div class="face"></div>
				<span class="kind">{category}</span>
				<span class="bar"></span>
			</div>
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

	/* Same flex sizing as a tile, so the wall keeps its layout when the picks arrive. */
	.placeholder {
		display: flex;
		flex-direction: column;
		gap: 0.7rem;
		flex-grow: calc(var(--w) * (100000 / var(--h)));
		flex-basis: calc(var(--min-height) * (var(--w) / var(--h)));
		min-width: 0;
		animation: settle 600ms var(--ease) backwards;
		animation-delay: calc(300ms + var(--i) * 60ms);
	}

	.placeholder .face {
		width: 100%;
		aspect-ratio: var(--w) / var(--h);
		background: color-mix(in oklab, var(--mid-1) 55%, transparent);
		animation: breathe 2.4s ease-in-out infinite;
		animation-delay: calc(var(--i) * 180ms);
	}

	.kind {
		color: var(--ink);
		font-size: 0.8rem;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		line-height: 1.35;
		opacity: 0.4;
	}

	.bar {
		width: 60%;
		height: 0.55rem;
		margin-top: -0.3rem;
		background: color-mix(in oklab, var(--ink) 10%, transparent);
	}

	@keyframes settle {
		from {
			opacity: 0;
		}
		to {
			opacity: 1;
		}
	}

	@keyframes breathe {
		0%,
		100% {
			opacity: 0.55;
		}
		50% {
			opacity: 1;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.placeholder .face {
			animation: none;
		}
	}
</style>
