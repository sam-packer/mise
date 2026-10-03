<script lang="ts">
	// Render one wall item and name its image for the transition into its world.
	import type { Item } from '#lib/mood/types.js';

	let {
		item,
		index,
		active,
		returned,
		onopen
	}: {
		item: Item;
		index: number;
		/** The tile that a world grows from or returns to. */
		active: boolean;
		/** The tile that the path went on through, after a step back over several worlds. */
		returned: boolean;
		onopen: (item: Item, el: HTMLButtonElement) => void;
	} = $props();

	let loaded = $state(false);
	const w = $derived(item.image?.w ?? 3);
	const h = $derived(item.image?.h ?? 2);
	const tone = $derived(
		item.image
			? `oklab(${item.image.tone[0]} ${item.image.tone[1]} ${item.image.tone[2]})`
			: 'var(--mid-1)'
	);
</script>

<button
	type="button"
	class="tile"
	style:--w={w}
	style:--h={h}
	style:--i={index}
	data-item={item.id}
	aria-label="{item.category}: {item.title}, {item.creator}"
	onclick={(e) => onopen(item, e.currentTarget)}
>
	<div
		class="face"
		class:returned
		style:background={tone}
		style:view-transition-name={active ? 'mood-tile' : null}
	>
		{#if item.image}
			<img
				src={item.image.src}
				alt=""
				width={item.image.w}
				height={item.image.h}
				decoding="async"
				class:loaded
				onload={() => (loaded = true)}
				{@attach (img) => {
					if (img.complete && img.naturalWidth > 0) loaded = true;
				}}
			/>
		{:else}
			<p class="poem">{item.text}</p>
		{/if}
	</div>
	<span class="caption" aria-hidden="true">
		<span class="kind">{item.category}</span>
		<span class="name"><span class="title">{item.title}</span>, {item.creator}</span>
	</span>
</button>

<style>
	.tile {
		position: relative;
		display: flex;
		flex-direction: column;
		gap: 0.7rem;
		flex-grow: calc(var(--w) * (100000 / var(--h)));
		flex-basis: calc(var(--min-height) * (var(--w) / var(--h)));
		min-width: 0;
		margin: 0;
		padding: 0;
		border: 0;
		background: transparent;
		color: inherit;
		font: inherit;
		text-align: left;
		cursor: pointer;
		animation: rise 600ms var(--ease) backwards;
		animation-delay: calc(600ms + var(--i) * 60ms);
	}

	.tile:focus-visible {
		outline: none;
	}

	.face {
		position: relative;
		width: 100%;
		aspect-ratio: var(--w) / var(--h);
		overflow: hidden;
		/* A poem sizes its type to the tile, so a large tile shows larger type. */
		container-type: inline-size;
		transition:
			transform 500ms var(--ease),
			box-shadow 500ms var(--ease);
	}

	/* A brief ring shows where the user came back to. Focus, which comes after, keeps its ring. */
	.face.returned {
		outline: 2px solid transparent;
		outline-offset: 4px;
		animation: mood-returned 1200ms var(--ease) 120ms;
	}

	.tile:hover .face,
	.tile:focus-visible .face {
		z-index: 2;
		transform: scale(1.02);
		box-shadow: 0 18px 50px -12px oklch(0 0 0 / 0.35);
	}

	.tile:focus-visible .face {
		outline: 2px solid var(--ink);
		outline-offset: 4px;
	}

	img {
		display: block;
		width: 100%;
		height: 100%;
		object-fit: cover;
		opacity: 0;
		transition: opacity 700ms var(--ease);
	}

	img.loaded {
		opacity: 1;
	}

	/* The block is as wide as its longest line and sits in the middle of the tile. The lines stay
	   aligned left, as the poet set them. */
	.poem {
		box-sizing: border-box;
		width: fit-content;
		max-width: 100%;
		height: 100%;
		margin: 0 auto;
		padding: clamp(14px, 6cqi, 56px) clamp(16px, 7cqi, 64px);
		overflow: hidden;
		color: var(--paper-ink);
		font-family: var(--mood-font, var(--serif));
		font-style: var(--mood-style, italic);
		/* About 10 lines fill a 3:2 tile at any width. The fade at the bottom hides the rest. */
		font-size: clamp(0.8rem, 3.8cqi, 1.9rem);
		line-height: 1.5;
		white-space: pre-line;
		mask-image: linear-gradient(to bottom, black 55%, transparent 96%);
	}

	.caption {
		display: flex;
		flex-direction: column;
		gap: 0.1rem;
		color: var(--ink);
		line-height: 1.35;
		transition: opacity 400ms var(--ease);
	}

	.kind {
		font-size: 0.8rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		color: var(--ink-soft);
	}

	.name {
		display: -webkit-box;
		overflow: hidden;
		-webkit-box-orient: vertical;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		font-size: 0.9rem;
		color: var(--ink-soft);
		transition: color 400ms var(--ease);
	}

	.title {
		color: var(--ink);
	}

	.tile:hover .name,
	.tile:focus-visible .name {
		color: var(--ink);
	}

	@keyframes rise {
		from {
			opacity: 0;
			transform: translateY(12px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.tile {
			animation-name: appear;
		}

		.face {
			transition: box-shadow 500ms var(--ease);
		}

		.tile:hover .face,
		.tile:focus-visible .face {
			transform: none;
		}

		@keyframes appear {
			from {
				opacity: 0;
			}
			to {
				opacity: 1;
			}
		}
	}
</style>
