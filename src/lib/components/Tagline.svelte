<script lang="ts">
	// Show the brand line under the mark. Each word sits on one color of a palette: the palette of
	// the example feeling in the field, from the bundle's sample rooms.
	import { onMount } from 'svelte';
	import { fromHex, hex, textOn } from '$lib/color/oklab';
	import { samplePalettes } from '$lib/mood/samples';
	import type { Palette } from '$lib/mood/types';

	let {
		example = '',
		away = false,
		waiting = false
	}: {
		/** The example feeling that the field shows. */
		example?: string;
		/** A mood is on the wall. */
		away?: boolean;
		/** The page waits for the model before it can show the mood. */
		waiting?: boolean;
	} = $props();

	const WORDS = ['every', 'feeling', 'has', 'a', 'world.'];
	/** The colors of the logo files, for the first paint and for an example without a palette. */
	const BRAND = '#4f6e73 #c58e6e #e3c7a8 #b98a93 #3f3340'.split(' ').map(fromHex);

	let palettes = $state<Record<string, Palette>>({});

	onMount(() => {
		samplePalettes.then((p) => (palettes = p));
	});

	// Light to dark, left to right, as on the brand swatches. The text on each block comes from the
	// same palette and meets WCAG AA.
	const blocks = $derived.by(() => {
		const palette = [...(palettes[example] ?? BRAND)].sort((a, b) => b[0] - a[0]);
		return palette.map((color) => ({ fill: hex(color), text: hex(textOn(color, palette)) }));
	});
</script>

<p class="tagline" class:away class:waiting aria-hidden={away}>
	<span class="sr-only">{WORDS.join(' ')}</span>
	<span class="blocks" aria-hidden="true">
		{#each WORDS as word, i (i)}
			<span
				class="word"
				style:--i={i}
				style:background-color={blocks[i].fill}
				style:color={blocks[i].text}>{word}</span
			>
		{/each}
	</span>
</p>

<style>
	.tagline {
		--fade: 1200ms;
		margin: 0;
		animation: arrive 1100ms var(--ease) 150ms backwards;
		transition:
			opacity 1100ms var(--ease) 350ms,
			visibility 0s linear 0s;
	}

	.tagline.away {
		opacity: 0;
		visibility: hidden;
		transition:
			opacity 250ms ease-out,
			visibility 0s linear 250ms;
	}

	/* A faint edge keeps a block visible where its color is close to the page. */
	.blocks {
		display: flex;
		width: max-content;
		box-shadow: 0 0 0 1px color-mix(in oklab, var(--ink) 10%, transparent);
		font-size: clamp(0.92rem, 0.25vw + 0.85rem, 1.05rem);
		line-height: 1;
		letter-spacing: 0.01em;
	}

	/* A new palette moves across the words from left to right. */
	.word {
		padding: 0.24em 0.38em 0.3em;
		transition:
			background-color var(--fade) var(--ease) calc(var(--i) * 90ms),
			color var(--fade) var(--ease) calc(var(--i) * 90ms);
	}

	/* While the page waits for the model, a slow wave lifts the words one after another. */
	.waiting .word {
		animation: lift 2.6s ease-in-out calc(var(--i) * 160ms) infinite;
	}

	@keyframes arrive {
		from {
			opacity: 0;
		}
	}

	@keyframes lift {
		50% {
			transform: translateY(-0.14em);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.tagline {
			--fade: 400ms;
			animation: none;
		}

		.word {
			transition-delay: 0s;
		}

		.waiting .word {
			animation: none;
		}
	}

	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		margin: -1px;
		padding: 0;
		overflow: hidden;
		clip: rect(0 0 0 0);
		white-space: nowrap;
		border: 0;
	}
</style>
