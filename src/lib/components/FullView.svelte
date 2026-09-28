<script lang="ts">
	import type { Item } from '$lib/mood/types';
	import SongLinks from './SongLinks.svelte';

	let { item, kicker, onclose }: { item: Item; kicker?: string; onclose: () => void } = $props();

	const id = $props.id();
	let dialog: HTMLElement;

	let audio = $state<HTMLAudioElement | null>(null);
	let currentTime = $state(0);
	let duration = $state(0);
	let paused = $state(true);
	let previewFailed = $state(false);
	const progress = $derived(duration > 0 ? currentTime / duration : 0);

	const focusOnMount = (el: HTMLElement) => el.focus({ preventScroll: true });
	const autoplay = (el: HTMLAudioElement) => {
		el.play().catch(() => {});
	};

	function onkeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') {
			e.preventDefault();
			onclose();
			return;
		}
		if (e.key !== 'Tab') return;
		const focusable = dialog.querySelectorAll<HTMLElement>('a[href], button:not([disabled])');
		if (focusable.length === 0) return;
		const first = focusable[0];
		const last = focusable[focusable.length - 1];
		if (e.shiftKey && document.activeElement === first) {
			e.preventDefault();
			last.focus();
		} else if (!e.shiftKey && document.activeElement === last) {
			e.preventDefault();
			first.focus();
		}
	}

	function toggle() {
		if (!audio) return;
		if (audio.paused) audio.play().catch(() => {});
		else audio.pause();
	}

	const tone = $derived(
		item.image
			? `oklab(${item.image.tone[0]} ${item.image.tone[1]} ${item.image.tone[2]})`
			: 'var(--mid-1)'
	);
	const by = $derived(item.year ? `${item.creator}, ${item.year}` : item.creator);
</script>

<svelte:window {onkeydown} />

<div class="scrim" onclick={(e) => e.target === e.currentTarget && onclose()} role="presentation">
	<div
		class="view"
		class:poem={item.category === 'poem'}
		role="dialog"
		aria-modal="true"
		aria-labelledby="{id}-title"
		bind:this={dialog}
	>
		<figure class="media" style:background={tone}>
			{#if item.image}
				{#if item.category === 'song' && item.preview}
					<button
						type="button"
						class="art"
						onclick={toggle}
						aria-label={paused ? 'play preview' : 'pause preview'}
					>
						<img
							src={item.image.src}
							alt=""
							width={item.image.w}
							height={item.image.h}
							decoding="async"
						/>
					</button>
				{:else}
					<img
						src={item.image.src}
						alt=""
						width={item.image.w}
						height={item.image.h}
						decoding="async"
					/>
				{/if}
			{:else}
				<p class="text">{item.text}</p>
			{/if}
		</figure>

		{#if item.category === 'song' && item.preview}
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
				<div class="progress" aria-hidden="true">
					<div class="bar" style:transform="scaleX({progress})"></div>
				</div>
			{/if}
		{/if}

		<div class="meta">
			<p class="kicker">{kicker ?? item.category}</p>
			<h2 id="{id}-title">{item.title}</h2>
			<p class="by">{by}</p>
			{#if item.category === 'song' && item.album}
				<p class="album">from <span>{item.album}</span></p>
			{/if}
			{#if item.category !== 'poem'}
				<p class="vibe">{item.vibe}</p>
			{/if}
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

		<button type="button" class="close" onclick={onclose} {@attach focusOnMount}>
			<span class="x" aria-hidden="true"></span>close
		</button>
	</div>
</div>

<style>
	.scrim {
		position: fixed;
		inset: 0;
		z-index: 20;
		display: grid;
		place-items: center;
		padding: max(2.5vw, 12px);
		overflow-y: auto;
		background: color-mix(in oklab, var(--ground) 86%, transparent);
		backdrop-filter: blur(18px);
		-webkit-backdrop-filter: blur(18px);
		animation: appear 300ms var(--ease) both;
	}

	.view {
		position: relative;
		display: grid;
		grid-template-columns: minmax(0, 7fr) minmax(0, 5fr);
		gap: clamp(1.5rem, 4vw, 4rem);
		align-items: center;
		width: min(100%, 1100px);
		box-sizing: border-box;
		padding: clamp(1rem, 3vw, 2.5rem);
	}

	.media {
		margin: 0;
		justify-self: end;
		max-width: 100%;
		max-height: min(72dvh, 760px);
		aspect-ratio: auto;
		overflow: hidden;
		box-shadow: 0 30px 80px -24px oklch(0 0 0 / 0.45);
		view-transition-name: mood-tile;
	}

	.media img {
		display: block;
		max-width: 100%;
		max-height: min(72dvh, 760px);
		width: auto;
		height: auto;
	}

	.art {
		display: block;
		margin: 0;
		padding: 0;
		border: 0;
		background: none;
		cursor: pointer;
	}

	.art:focus-visible {
		outline: 2px solid var(--ink);
		outline-offset: 4px;
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

	.progress {
		position: absolute;
		left: clamp(1rem, 3vw, 2.5rem);
		right: clamp(1rem, 3vw, 2.5rem);
		bottom: calc(clamp(1rem, 3vw, 2.5rem) - 10px);
		height: 1px;
		background: color-mix(in oklab, var(--ink) 18%, transparent);
	}

	.bar {
		height: 100%;
		background: var(--ink);
		transform-origin: left;
		transition: transform 250ms linear;
	}

	.meta {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		color: var(--ink);
		font-size: 1rem;
		line-height: 1.45;
	}

	.meta p,
	.meta h2 {
		margin: 0;
	}

	.kicker {
		font-size: 0.85rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		opacity: 0.55;
	}

	h2 {
		font-size: clamp(1.5rem, 2vw + 0.6rem, 2.4rem);
		font-weight: 400;
		line-height: 1.15;
		letter-spacing: -0.01em;
	}

	.by,
	.album {
		opacity: 0.7;
	}

	.album span {
		font-style: normal;
	}

	.vibe {
		margin-top: 0.5rem;
		max-width: 30rem;
	}

	.meta a {
		margin-top: 1rem;
		align-self: flex-start;
		color: var(--ink);
		text-decoration: none;
		border-bottom: 1px solid color-mix(in oklab, var(--ink) 40%, transparent);
		transition: border-color 300ms var(--ease);
	}

	.meta a:hover,
	.meta a:focus-visible {
		border-color: var(--ink);
	}

	.meta :global(.links) {
		margin-top: 1rem;
	}

	.close {
		position: absolute;
		top: 0;
		right: clamp(0.25rem, 2vw, 1.5rem);
		display: flex;
		align-items: center;
		gap: 0.55rem;
		min-width: 44px;
		min-height: 44px;
		padding: 0 1.1rem 0 0.95rem;
		border: 1px solid color-mix(in oklab, var(--ink) 22%, transparent);
		border-radius: 999px;
		background: color-mix(in oklab, var(--ground) 60%, transparent);
		color: var(--ink);
		font: inherit;
		font-size: 1rem;
		opacity: 0.8;
		cursor: pointer;
		transition:
			opacity 300ms var(--ease),
			border-color 300ms var(--ease),
			background-color 300ms var(--ease);
	}

	.close:hover,
	.close:focus-visible {
		opacity: 1;
		border-color: color-mix(in oklab, var(--ink) 55%, transparent);
		background: color-mix(in oklab, var(--ink) 6%, var(--ground));
	}

	.close:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: 3px;
	}

	/* A drawn cross: two hairlines, so it takes the ink color and stays crisp at any size. */
	.x {
		position: relative;
		width: 0.8rem;
		height: 0.8rem;
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

	@media (max-width: 720px) {
		.scrim {
			place-items: start center;
		}

		.view {
			grid-template-columns: minmax(0, 1fr);
			gap: 1.5rem;
			padding-top: 4.25rem;
		}

		.media {
			justify-self: center;
			max-height: 56dvh;
		}

		.media img {
			max-height: 56dvh;
		}

		.progress {
			position: static;
			margin-top: -0.5rem;
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
</style>
