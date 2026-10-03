<script lang="ts">
	// Show one work as its own world: the work in the middle, its path above, its neighbors below.
	import type { Item, World } from '#lib/mood/types.js';
	import type { LoadedFace } from './typeface';
	import Chevron from './Chevron.svelte';
	import SongLinks from './SongLinks.svelte';
	import Wall from './Wall.svelte';
	import Trail from './Trail.svelte';

	let {
		world,
		feeling,
		path,
		live,
		focus,
		still,
		returned,
		leaving,
		face,
		note,
		heading = $bindable(),
		onstep,
		onclose,
		ontravel
	}: {
		world: World;
		/** The feeling at the start of the path. */
		feeling: string;
		/** The works between the feeling and this one, in order. */
		path: Item[];
		/** This world is the one on screen. A world under it stays in the page, out of view. */
		live: boolean;
		/** The id of the neighbor that the next or the last world grows from. */
		focus: string | null;
		/** A move over several worlds: nothing morphs into the work on screen. */
		still: boolean;
		/** The id of the neighbor that the path went on through, after a step back over several worlds. */
		returned: string | null;
		/** Fade the world out before the page starts over. */
		leaving: boolean;
		face: LoadedFace | null;
		note: string | null;
		heading?: HTMLHeadingElement | null;
		/** Go back to a step of the path. Step 0 is the feeling. */
		onstep: (step: number) => void;
		/** Leave every world and go back to the feeling. */
		onclose: () => void;
		ontravel: (item: Item, el: HTMLButtonElement) => void;
	} = $props();

	const id = $props.id();
	const item = $derived(world.item);
	// The work on screen carries the transition name unless the focus tile is on its own wall, or the
	// move spans several worlds and no tile morphs into it.
	const named = $derived(live && !still && !world.neighbors.some((n) => n.id === focus));
	const canPlay = $derived(item.category === 'song' && !!item.preview);
	const tone = $derived(
		item.image
			? `oklab(${item.image.tone[0]} ${item.image.tone[1]} ${item.image.tone[2]})`
			: 'var(--mid-1)'
	);

	let audio = $state<HTMLAudioElement | null>(null);
	let currentTime = $state(0);
	let duration = $state(0);
	let paused = $state(true);
	let previewFailed = $state(false);
	const progress = $derived(duration > 0 ? currentTime / duration : 0);

	const autoplay = (el: HTMLAudioElement) => {
		el.play().catch(() => {});
	};

	// A world out of view is silent.
	$effect(() => {
		if (!live) audio?.pause();
	});

	function toggle() {
		if (!audio) return;
		if (audio.paused) audio.play().catch(() => {});
		else audio.pause();
	}
</script>

<article
	class="world"
	class:leaving
	style:--mood-font={face?.family ?? null}
	style:--mood-style={face?.style ?? null}
	aria-labelledby="{id}-title"
>
	<div class="top">
		<Trail {feeling} {path} current={item.title} {onstep} />
	</div>

	<!-- Fixed, so the way out stays in view over the wall below. -->
	<button type="button" class="close" aria-label="close, back to your feeling" onclick={onclose}>
		<span class="x" aria-hidden="true"></span>close<kbd aria-hidden="true">esc</kbd>
	</button>

	{#if note}
		<p class="note">{note}</p>
	{/if}

	<section class="work">
		<figure
			class="media"
			style:background={tone}
			style:view-transition-name={named ? 'mood-tile' : null}
		>
			{#if item.image}
				<img
					src={item.image.src}
					alt=""
					width={item.image.w}
					height={item.image.h}
					decoding="async"
				/>
			{:else}
				<p class="text">{item.text}</p>
			{/if}
		</figure>

		<div class="meta">
			<p class="kicker">{item.year ? `${item.category} · ${item.year}` : item.category}</p>
			<h1 id="{id}-title" tabindex="-1" title={item.title} bind:this={heading}>{item.title}</h1>
			<p class="by">
				{item.creator}{#if item.category === 'song' && item.album && item.album !== item.title}
					<span class="from">, from <span class="album">{item.album}</span></span>
				{/if}
			</p>

			<p class="vibe">{item.vibe}</p>

			{#if canPlay}
				<audio
					src={item.preview}
					preload="auto"
					bind:this={audio}
					{@attach autoplay}
					bind:currentTime
					bind:duration
					bind:paused
					onerror={() => (previewFailed = true)}
				></audio>
				{#if !previewFailed}
					<div class="player">
						<button type="button" class="play" onclick={toggle}>
							<span class="ring" aria-hidden="true"><i class="icon" class:paused></i></span>
							{paused ? 'play preview' : 'pause preview'}
						</button>
						<div class="progress" aria-hidden="true">
							<div class="bar" style:transform="scaleX({progress})"></div>
						</div>
					</div>
				{/if}
			{/if}

			<div class="links">
				{#if item.category === 'song'}
					<SongLinks links={item.links} />
				{:else}
					<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
					<a href={item.links.primary} target="_blank" rel="noopener">
						{item.category === 'art'
							? 'see it at the museum'
							: item.category === 'poem'
								? 'read more'
								: 'more about it'}<Chevron />
					</a>
				{/if}
			</div>
		</div>
	</section>

	<section class="near" aria-labelledby="{id}-near">
		<h2 id="{id}-near">the same feeling, elsewhere.</h2>
		<Wall
			items={world.neighbors}
			label="works that share the feeling of {item.title}"
			leaving={false}
			active={focus}
			{returned}
			onopen={ontravel}
		/>
	</section>
</article>

<style>
	.world {
		/* The height of the row that holds the mark. */
		--top: calc(44px + 2 * max(1.3vw, 10px));
		--gutter: max(2.5vw, 12px);
		display: flex;
		flex-direction: column;
		padding-bottom: 4rem;
		transition: opacity 200ms ease-out;
	}

	.world.leaving {
		opacity: 0;
		pointer-events: none;
	}

	/* The path shares the top row with the mark on the left and the close control on the right. */
	/* Above the work, so the whole path can open over it. */
	.top {
		position: relative;
		z-index: 4;
		display: flex;
		align-items: center;
		justify-content: center;
		min-height: var(--top);
		padding: 0 calc(var(--gutter) + 9rem);
		box-sizing: border-box;
		animation: appear 500ms var(--ease) both;
		/* The row spans the page above the mark. Only the trail takes clicks, so the mark stays usable. */
		pointer-events: none;
	}

	.top > :global(*) {
		pointer-events: auto;
	}

	/*
	 * Solid ink with ground-colored text: a clear control, readable over the wall that scrolls under it.
	 * A pill, so it matches the shape of the path beside it.
	 */
	.close {
		position: fixed;
		top: max(1.3vw, 10px);
		right: var(--gutter);
		z-index: 5;
		display: flex;
		align-items: center;
		gap: 0.55rem;
		min-width: 44px;
		min-height: 44px;
		padding: 0 1rem 0 1.1rem;
		border: 0;
		border-radius: 999px;
		background: var(--ink);
		color: var(--ground);
		font: inherit;
		font-size: 1rem;
		cursor: pointer;
		animation: appear 500ms var(--ease) both;
		transition: background-color 300ms var(--ease);
	}

	.close:hover,
	.close:focus-visible {
		background: var(--ink-soft);
	}

	.close:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: 3px;
	}

	kbd {
		margin-left: 0.2rem;
		font: inherit;
		font-size: 0.8rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
	}

	/* The key hint is for a keyboard. A touch screen has none. */
	@media not (hover: hover) {
		kbd {
			display: none;
		}
	}

	/* A drawn cross: two hairlines, so it takes the text color and stays crisp at any size. */
	.x {
		position: relative;
		width: 0.75rem;
		height: 0.75rem;
	}

	.x::before,
	.x::after {
		content: '';
		position: absolute;
		top: 50%;
		left: -10%;
		width: 120%;
		height: 1.25px;
		background: currentColor;
		transform: rotate(45deg);
	}

	.x::after {
		transform: rotate(-45deg);
	}

	.note {
		margin: 0 0 0.75rem;
		color: var(--ink-soft);
		font-size: 0.95rem;
		text-align: center;
	}

	/*
	 * The work and its details fit in the first screen. --room is the height that the top row and a
	 * margin leave. The image keeps its shape inside it at any aspect ratio. The details scale their
	 * type and spacing with the screen height and clamp the long names, so they fit in it too.
	 */
	.work {
		--room: calc(100svh - var(--top) - 3.5rem);
		display: grid;
		grid-template-columns: minmax(0, 7fr) minmax(0, 5fr);
		gap: clamp(2rem, 5vw, 5rem);
		align-items: center;
		width: min(100% - 2 * var(--gutter), 1200px);
		margin: 0.75rem auto 0;
	}

	.media {
		position: relative;
		justify-self: end;
		max-width: 100%;
		margin: 0;
		box-shadow: 0 30px 80px -24px oklch(0 0 0 / 0.45);
	}

	.media img {
		display: block;
		max-width: 100%;
		max-height: max(14rem, var(--room));
		width: auto;
		height: auto;
		object-fit: contain;
	}

	/* A drawn play triangle, or two pause bars, so the icon takes the ink color. */
	.icon {
		width: 0.7rem;
		height: 0.8rem;
		margin-left: 0.15rem;
		background: currentColor;
		clip-path: polygon(0 0, 100% 50%, 0 100%);
	}

	.icon:not(.paused) {
		margin-left: 0;
		background: linear-gradient(
			to right,
			currentColor 0 35%,
			transparent 35% 65%,
			currentColor 65% 100%
		);
		clip-path: none;
	}

	.text {
		box-sizing: border-box;
		width: min(100%, 34rem);
		margin: 0;
		padding: 2em 2.25em;
		color: var(--paper-ink);
		font-family: var(--mood-font, var(--serif));
		font-style: var(--mood-style, italic);
		/* The poem's 14 lines and padding take about 26em. Size the em so the poem fits in --room. */
		font-size: clamp(0.78rem, min(1vw + 0.6rem, var(--room) / 27), 1.3rem);
		line-height: 1.55;
		white-space: pre-line;
	}

	.meta {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		/* The space between the groups of details: less on a short screen. */
		--beat: clamp(0.8rem, 2.6svh, 1.5rem);
		max-width: 28rem;
		color: var(--ink);
		font-size: 1rem;
		line-height: 1.45;
		animation: rise 700ms var(--ease) 150ms both;
	}

	.meta p,
	.meta h1 {
		margin: 0;
	}

	.kicker {
		color: var(--ink-soft);
		font-size: 0.85rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
	}

	.meta h1 {
		margin-top: 0.35rem;
		font-family: var(--mood-font, var(--serif));
		font-style: var(--mood-style, italic);
		font-size: clamp(1.6rem, min(2.6vw + 0.8rem, 4.2svh + 0.4rem), 3.2rem);
		font-weight: 400;
		/* The clamp hides overflow, so the line box must hold the deepest descenders ("g", "y"). */
		line-height: 1.18;
		letter-spacing: -0.015em;
		overflow-wrap: anywhere;
		text-wrap: balance;
		/* A catalog title can run to 300 characters. The title attribute and the trail keep all of it. */
		display: -webkit-box;
		overflow: hidden;
		-webkit-box-orient: vertical;
		-webkit-line-clamp: 3;
		line-clamp: 3;
	}

	h1:focus {
		outline: none;
	}

	.meta .by {
		display: -webkit-box;
		margin-top: 0.6rem;
		overflow: hidden;
		-webkit-box-orient: vertical;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		color: var(--ink-soft);
		font-size: 1.05rem;
	}

	.album {
		color: var(--ink);
	}

	/* A short rule sets the vibe line apart as the sentence the world grows from. */
	.meta .vibe {
		margin-top: var(--beat);
		padding-top: var(--beat);
		border-top: 1px solid color-mix(in oklab, var(--ink) 25%, transparent);
		font-size: 1.2rem;
		line-height: 1.45;
		text-wrap: pretty;
	}

	.player {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		align-self: stretch;
		margin-top: var(--beat);
	}

	.play {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		align-self: flex-start;
		min-height: 44px;
		padding: 0 0.75rem 0 0;
		border: 0;
		background: none;
		color: var(--ink);
		font: inherit;
		cursor: pointer;
	}

	.play:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: 2px;
	}

	.ring {
		display: grid;
		place-items: center;
		width: 2.5rem;
		height: 2.5rem;
		border: 1px solid color-mix(in oklab, var(--ink) 40%, transparent);
		transition: border-color 300ms var(--ease);
	}

	.play:hover .ring,
	.play:focus-visible .ring {
		border-color: var(--ink);
	}

	.progress {
		height: 1px;
		background: color-mix(in oklab, var(--ink) 18%, transparent);
	}

	.bar {
		height: 100%;
		background: var(--ink);
		transform-origin: left;
		transition: transform 250ms linear;
	}

	.links {
		margin-top: var(--beat);
	}

	.links > a {
		color: var(--ink);
		text-decoration: none;
		border-bottom: 1px solid color-mix(in oklab, var(--ink) 40%, transparent);
		transition: border-color 300ms var(--ease);
	}

	.links > a:hover,
	.links > a:focus-visible {
		border-color: var(--ink);
	}

	.links > a:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: 4px;
	}

	.near {
		width: min(100%, 1440px);
		margin: clamp(4.5rem, 14vh, 9rem) auto 0;
	}

	/*
	 * One line in the world's own typeface, set as a moment between the work and its neighbors.
	 * A short rule above it gives the pause before it.
	 */
	.near h2 {
		margin: 0 var(--gutter) clamp(0.5rem, 2vh, 1.25rem);
		color: var(--ink);
		font-family: var(--mood-font, var(--serif));
		font-style: var(--mood-style, italic);
		font-size: clamp(1.75rem, 2.6vw + 0.9rem, 3.1rem);
		font-weight: 400;
		line-height: 1.1;
		letter-spacing: -0.015em;
		text-align: center;
		text-wrap: balance;
	}

	.near h2::before {
		content: '';
		display: block;
		width: 2.5rem;
		height: 1px;
		margin: 0 auto clamp(1.25rem, 3vh, 2rem);
		background: color-mix(in oklab, var(--ink) 35%, transparent);
	}

	@media (max-width: 720px) {
		/* The mark and the close control take the top row. The path takes the row below it. */
		.top {
			justify-content: flex-start;
			min-height: 0;
			padding: var(--top) var(--gutter) 0;
		}

		.work {
			--room: calc(100svh - var(--top) - 46px - 3.5rem);
			gap: clamp(1.25rem, 4vw, 2rem);
			margin-top: 0.5rem;
		}
	}

	/* A phone stacks the work over its details. A half screen window keeps them side by side. */
	@media (max-width: 600px) {
		.work {
			/* Leave room for the path's row, the category, and a title of three lines. */
			--room: calc(100svh - var(--top) - 46px - 17rem);
			grid-template-columns: minmax(0, 1fr);
			gap: 1.5rem;
		}

		.media {
			justify-self: center;
		}
	}

	@keyframes appear {
		from {
			opacity: 0;
		}
		to {
			opacity: 1;
		}
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
		.meta {
			animation-name: appear;
		}
	}
</style>
