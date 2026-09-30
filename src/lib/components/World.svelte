<script lang="ts">
	// Show one work as its own world: the work in the middle, its path above, its neighbors around it.
	import type { Item, World } from '$lib/mood/types';
	import type { LoadedFace } from './typeface';
	import SongLinks from './SongLinks.svelte';
	import Wall from './Wall.svelte';
	import Scent from './Scent.svelte';

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
	<nav class="trail veil" aria-label="your path">
		<ol {@attach toEnd}>
			<li>
				<button type="button" class="step" title={feeling} onclick={() => onstep(0)}
					>{feeling}</button
				>
			</li>
			{#each path as step, i (step.id)}
				<li>
					<span class="arrow" aria-hidden="true">→</span>
					<button type="button" class="step" title={step.title} onclick={() => onstep(i + 1)}
						>{step.title}</button
					>
				</li>
			{/each}
			<li aria-current="step">
				<span class="arrow" aria-hidden="true">→</span>
				<span class="step here">{item.title}</span>
			</li>
		</ol>
	</nav>

	{#if note}
		<p class="note">{note}</p>
	{/if}

	<section class="work" class:poem={!item.image}>
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
						<span class="badge veil" aria-hidden="true"><i class="icon" class:paused></i></span>
					</button>
				{/if}
			{:else}
				<p class="text">{item.text}</p>
			{/if}
		</figure>

		<div class="meta veil">
			<p class="kicker">{item.year ? `${item.category} · ${item.year}` : item.category}</p>
			<h1 id="{id}-title" tabindex="-1" bind:this={heading}>{item.title}</h1>
			<p class="by">{item.creator}</p>
			{#if item.category === 'song' && item.album && item.album !== item.title}
				<p class="by">from <span class="album">{item.album}</span></p>
			{/if}
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
		<h2 id="{id}-near" class="label">nearby</h2>
		<Wall
			items={world.neighbors}
			label="works near {item.title}"
			leaving={false}
			active={focus}
			onopen={ontravel}
		/>
	</section>

	<Scent text={world.scent.text} delay={1500} />
</article>

<style>
	.world {
		--gutter: max(2.5vw, 12px);
		display: flex;
		flex-direction: column;
		/* Clear the mark, which sits in the top left corner of the page. */
		padding-top: calc(44px + 2 * max(1.3vw, 10px));
		transition: opacity 200ms ease-out;
	}

	.world.leaving {
		opacity: 0;
		pointer-events: none;
	}

	/* The veil keeps both inks readable over any image: oklab.ts derives --veil for 80% opacity. */
	.veil {
		background: color-mix(in srgb, var(--veil) 80%, transparent);
		backdrop-filter: blur(24px) saturate(1.3);
		-webkit-backdrop-filter: blur(24px) saturate(1.3);
		border: 1px solid color-mix(in oklab, var(--ink) 12%, transparent);
	}

	.trail {
		position: sticky;
		top: max(1vw, 8px);
		z-index: 3;
		align-self: center;
		max-width: calc(100% - 2 * var(--gutter));
		box-sizing: border-box;
		border-radius: 999px;
		animation: appear 500ms var(--ease) both;
	}

	.trail ol {
		display: flex;
		align-items: center;
		margin: 0;
		padding: 0 0.45rem;
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
		max-width: 15em;
		min-height: 44px;
		padding: 0 0.6rem;
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

	button.step:hover,
	button.step:focus-visible {
		color: var(--ink);
	}

	button.step:focus-visible {
		border-radius: 999px;
		outline: 1px solid var(--ink);
		outline-offset: -6px;
	}

	.here {
		color: var(--ink);
		cursor: default;
	}

	.note {
		margin: 0.75rem 0 0;
		color: var(--ink-soft);
		font-size: 0.95rem;
		text-align: center;
	}

	.work {
		display: grid;
		grid-template-columns: minmax(0, 7fr) minmax(0, 5fr);
		gap: clamp(1.5rem, 4vw, 4rem);
		align-items: center;
		width: min(100% - 2 * var(--gutter), 1200px);
		margin: clamp(1.5rem, 5vh, 3.5rem) auto 0;
	}

	.media {
		position: relative;
		justify-self: end;
		max-width: 100%;
		margin: 0;
		overflow: hidden;
		box-shadow: 0 30px 80px -24px oklch(0 0 0 / 0.45);
	}

	.media img {
		display: block;
		max-width: 100%;
		max-height: min(72dvh, 760px);
		width: auto;
		height: auto;
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

	.badge {
		display: grid;
		place-items: center;
		width: 3rem;
		height: 3rem;
		border-radius: 999px;
		color: var(--ink);
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
		font-size: clamp(1rem, 1vw + 0.6rem, 1.35rem);
		line-height: 1.55;
		white-space: pre-line;
	}

	.meta {
		display: flex;
		flex-direction: column;
		gap: 0.45rem;
		padding: clamp(1.25rem, 2.6vw, 2.25rem);
		border-radius: 1rem;
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

	h1 {
		font-family: var(--mood-font, var(--serif));
		font-style: var(--mood-style, italic);
		font-size: clamp(1.7rem, 2.4vw + 0.7rem, 2.9rem);
		font-weight: 400;
		line-height: 1.1;
		letter-spacing: -0.01em;
		overflow-wrap: anywhere;
	}

	h1:focus {
		outline: none;
	}

	.by {
		color: var(--ink-soft);
	}

	.album {
		color: var(--ink);
	}

	.meta .vibe {
		max-width: 30rem;
		margin-top: 0.6rem;
		font-size: 1.1rem;
		line-height: 1.5;
	}

	.player {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		margin-top: 0.9rem;
	}

	.play {
		display: flex;
		align-items: center;
		gap: 0.7rem;
		align-self: flex-start;
		min-height: 44px;
		padding: 0 0.9rem 0 0;
		border: 0;
		background: none;
		color: var(--ink);
		font: inherit;
		cursor: pointer;
	}

	.play:focus-visible {
		border-radius: 999px;
		outline: 1px solid var(--ink);
		outline-offset: 2px;
	}

	.ring {
		display: grid;
		place-items: center;
		width: 2.4rem;
		height: 2.4rem;
		border: 1px solid color-mix(in oklab, var(--ink) 40%, transparent);
		border-radius: 999px;
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
		margin-top: 1rem;
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
		margin-top: clamp(3rem, 9vh, 6rem);
	}

	.label {
		width: min(100%, 1440px);
		box-sizing: border-box;
		margin: 0 auto -0.5rem;
		padding: 0 max(2.5vw, 12px);
		color: var(--ink-soft);
		font-size: 0.95rem;
		font-weight: 400;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
	}

	@media (max-width: 720px) {
		.work {
			grid-template-columns: minmax(0, 1fr);
			gap: 1.25rem;
		}

		.media {
			justify-self: center;
		}

		.media img {
			max-height: 56dvh;
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
