<script lang="ts" module>
	import { ready } from '$lib/mood/client';

	/** The model loaded before this component mounted, for example after a return from another page. */
	let loaded = false;
	ready.then(
		() => (loaded = true),
		() => (loaded = true)
	);
</script>

<script lang="ts">
	// Show the brand line above the feeling line. The words breathe in a slow wave while the model
	// loads, then settle one by one. The line gives way when a mood is on the wall.
	import { onMount } from 'svelte';

	let {
		away = false
	}: {
		/** A mood is on the wall. */
		away?: boolean;
	} = $props();

	const WORDS = ['every', 'feeling', 'has', 'a', 'world.'];

	/** The words that finished their last breath. Each word stops only at the end of a cycle. */
	let resting = $state(WORDS.map(() => loaded));
	let done = false;

	onMount(() => {
		const settle = () => (done = true);
		ready.then(settle, settle);
	});

	function onanimationiteration(e: AnimationEvent) {
		if (!done) return;
		const i = Number((e.target as HTMLElement).dataset.i);
		if (!Number.isNaN(i)) resting[i] = true;
	}
</script>

<div class="tagline" class:away aria-hidden={away}>
	<p {onanimationiteration}>
		{#each WORDS as word, i (i)}<span class="word" class:rest={resting[i]} data-i={i} style:--i={i}
				>{word}</span
			>{i < WORDS.length - 1 ? ' ' : ''}{/each}
	</p>
</div>

<style>
	/* The box has no height, so the line above the field never moves the layout. */
	.tagline {
		position: relative;
		width: 100%;
		height: 0;
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

	p {
		position: absolute;
		left: 0;
		right: 0;
		bottom: 1.35rem;
		margin: 0;
		color: var(--ink-soft);
		font-size: clamp(1.02rem, 0.45vw + 0.9rem, 1.28rem);
		line-height: 1.3;
		letter-spacing: 0.01em;
		text-align: center;
		pointer-events: none;
	}

	/* A slow wave across the words: each one rises a little and softens, then comes back to rest. */
	.word {
		display: inline-block;
		animation: breathe 3.6s ease-in-out infinite;
		animation-delay: calc(var(--i) * 220ms);
	}

	.word.rest {
		animation: none;
	}

	@keyframes breathe {
		0%,
		100% {
			opacity: 1;
			transform: none;
		}
		50% {
			opacity: 0.5;
			transform: translateY(-0.08em);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.word {
			animation: none;
		}
	}
</style>
