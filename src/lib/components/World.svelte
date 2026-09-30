<script lang="ts">
	// Show one work as its own world: the work in the middle, its path above, its neighbors below.
	import type { Item, World } from '$lib/mood/types';
	import type { LoadedFace } from './typeface';
	import SongLinks from './SongLinks.svelte';
	import Wall from './Wall.svelte';

	let {
		world,
		feeling,
		path,
		focus,
		leaving,
		face,
		note,
		heading = $bindable(null),
		onstep,
		onclose,
		ontravel
	}: {
		world: World;
		/** The feeling at the start of the path. */
		feeling: string;
		/** The works between the feeling and this one, in order. */
		path: Item[];
		/** The id of the neighbor that the next or the last world grows from. */
		focus: string | null;
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
	// The work carries the transition name unless a neighbor on this wall carries it.
	const named = $derived(!world.neighbors.some((n) => n.id === focus));
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
	// Show the newest step when the path is wider than the screen.
	const toEnd = (el: HTMLElement) => {
		el.scrollLeft = el.scrollWidth;
	};

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
	<nav class="trail" aria-label="your path">
		<ol {@attach toEnd}>
			<li>
				<button
					type="button"
					class="step"
					aria-label="back to your feeling, {feeling}"
					onclick={() => onstep(0)}>“{feeling}”</button
				>
			</li>
			{#each path as step, i (step.id)}
				<li>
					<span class="arrow" aria-hidden="true">→</span>
					<button type="button" class="step" onclick={() => onstep(i + 1)}>{step.title}</button>
				</li>
			{/each}
			<li aria-current="step">
				<span class="arrow" aria-hidden="true">→</span>
				<span class="step here">{item.title}</span>
			</li>
		</ol>
	</nav>

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
				{#if canPlay && !previewFailed}
					<button
						type="button"
						class="art"
						onclick={toggle}
						aria-label={paused ? 'play preview' : 'pause preview'}
					>
						<span class="badge" aria-hidden="true"><i class="icon" class:paused></i></span>
					</button>
				{/if}
			{:else}
				<p class="text">{item.text}</p>
			{/if}
		</figure>

		<div class="meta">
			<p class="kicker">{item.year ? `${item.category} · ${item.year}` : item.category}</p>
			<h1 id="{id}-title" tabindex="-1" bind:this={heading}>{item.title}</h1>
			<p class="by">
				{item.creator}{#if item.category === 'song' && item.album && item.album !== item.title}
					<span class="from">, from <span class="album">{item.album}</span></span>
				{/if}
			</p>

			<p class="vibe">{item.vibe}</p>
			<p class="scent"><span class="kicker">scent</span> {world.scent.text}</p>

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
							? 'see it at the museum →'
							: item.category === 'poem'
								? 'read more →'
								: 'more about it →'}
					</a>
				{/if}
			</div>
		</div>
	</section>

	<section class="near" aria-labelledby="{id}-near">
		<div class="head">
			<h2 id="{id}-near">nearby</h2>
			<p>choose one to travel on</p>
		</div>
		<Wall
			items={world.neighbors}
			label="works near {item.title}"
			leaving={false}
			active={focus}
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
	.trail {
		display: flex;
		justify-content: center;
		min-height: var(--top);
		padding: 0 calc(var(--gutter) + 9rem);
		box-sizing: border-box;
		animation: appear 500ms var(--ease) both;
	}

	.trail ol {
		display: flex;
		align-items: center;
		min-width: 0;
		margin: 0;
		padding: 0;
		overflow-x: auto;
		scrollbar-width: none;
		list-style: none;
		white-space: nowrap;
	}

	.trail ol::-webkit-scrollbar {
		display: none;
	}

	.trail li {
		display: flex;
		flex: none;
		align-items: center;
	}

	.arrow {
		color: var(--ink-soft);
		font-style: normal;
	}

	.step {
		display: block;
		max-width: 14em;
		min-height: 44px;
		padding: 0 0.5rem;
		border: 0;
		background: none;
		color: var(--ink-soft);
		font: inherit;
		font-size: 0.95rem;
		line-height: 44px;
		overflow: hidden;
		text-overflow: ellipsis;
		cursor: pointer;
		transition: color 300ms var(--ease);
	}

	button.step {
		text-decoration: underline 1px color-mix(in oklab, var(--ink) 30%, transparent);
		text-underline-offset: 0.25em;
	}

	button.step:hover,
	button.step:focus-visible {
		color: var(--ink);
		text-decoration-color: var(--ink);
	}

	button.step:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: -6px;
	}

	.here {
		color: var(--ink);
		cursor: default;
	}

	/* Solid ink with ground-colored text: a clear control, readable over the wall that scrolls under it. */
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
		padding: 0 0.85rem 0 0.95rem;
		border: 0;
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
	 * The work and its title fit in the first screen at any aspect ratio: the image keeps its
	 * shape inside the height that the top row and a margin leave.
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

	/* The whole cover plays the preview. The badge in its corner shows that it can. */
	.art {
		position: absolute;
		inset: 0;
		display: flex;
		align-items: flex-end;
		justify-content: flex-end;
		padding: clamp(0.75rem, 2vw, 1.25rem);
		border: 0;
		background: none;
		cursor: pointer;
	}

	.art:focus-visible {
		outline: 2px solid var(--ink);
		outline-offset: -6px;
	}

	/* Solid ink like the close control, so the icon keeps its contrast over any cover. */
	.badge {
		display: grid;
		place-items: center;
		width: 3rem;
		height: 3rem;
		background: var(--ink);
		color: var(--ground);
		transition: transform 400ms var(--ease);
	}

	.art:hover .badge,
	.art:focus-visible .badge {
		transform: scale(1.06);
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
		padding: clamp(1.5rem, 4vw, 3rem) clamp(1.5rem, 4.5vw, 3.5rem);
		color: var(--paper-ink);
		font-family: var(--mood-font, var(--serif));
		font-style: var(--mood-style, italic);
		font-size: clamp(1rem, 1vw + 0.6rem, 1.3rem);
		line-height: 1.55;
		white-space: pre-line;
	}

	.meta {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
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
		font-size: clamp(1.9rem, 2.6vw + 0.8rem, 3.2rem);
		font-weight: 400;
		line-height: 1.05;
		letter-spacing: -0.015em;
		overflow-wrap: anywhere;
		text-wrap: balance;
	}

	h1:focus {
		outline: none;
	}

	.meta .by {
		margin-top: 0.6rem;
		color: var(--ink-soft);
		font-size: 1.05rem;
	}

	.album {
		color: var(--ink);
	}

	/* A short rule sets the vibe line apart as the sentence the world grows from. */
	.meta .vibe {
		margin-top: 1.5rem;
		padding-top: 1.5rem;
		border-top: 1px solid color-mix(in oklab, var(--ink) 25%, transparent);
		font-size: 1.2rem;
		line-height: 1.45;
		text-wrap: pretty;
	}

	.meta .scent {
		margin-top: 0.9rem;
		color: var(--ink-soft);
		font-size: 0.95rem;
	}

	.scent .kicker {
		margin-right: 0.35rem;
	}

	.player {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		align-self: stretch;
		margin-top: 1.5rem;
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
		margin-top: 1.5rem;
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
		margin: clamp(3.5rem, 10vh, 6rem) auto 0;
	}

	.head {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: 0.25rem 1rem;
		margin: 0 var(--gutter);
		padding-top: 1rem;
		border-top: 1px solid color-mix(in oklab, var(--ink) 25%, transparent);
	}

	.head h2,
	.head p {
		margin: 0;
		font-size: 0.95rem;
		font-weight: 400;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
	}

	.head h2 {
		color: var(--ink);
	}

	.head p {
		color: var(--ink-soft);
	}

	@media (max-width: 720px) {
		/* The mark and the close control take the top row. The path takes the row below it. */
		.trail {
			justify-content: flex-start;
			min-height: 0;
			padding: var(--top) var(--gutter) 0;
		}

		.trail ol {
			margin: 0 -0.5rem;
		}

		.work {
			--room: calc(100svh - var(--top) - 44px - 19rem);
			grid-template-columns: minmax(0, 1fr);
			gap: 1.5rem;
			margin-top: 0.5rem;
		}

		.media {
			justify-self: center;
		}

		.meta .vibe {
			margin-top: 1.1rem;
			padding-top: 1.1rem;
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

		.badge {
			transition: none;
		}

		.art:hover .badge,
		.art:focus-visible .badge {
			transform: none;
		}
	}
</style>
