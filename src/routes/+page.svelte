<script lang="ts">
	import { onMount, tick, untrack } from 'svelte';
	import { fade } from 'svelte/transition';
	import { page } from '$app/state';
	import { goto, pushState } from '$app/navigation';
	import { resolve } from '$app/paths';
	import type { Item, Mood, OKLab } from '$lib/mood/types';
	import { infer, ready, start } from '$lib/mood/client';
	import { neutralTokens, paletteFavicon, paletteToTokens } from '$lib/color/oklab';
	import { applyTokens, tweenTokens } from '$lib/color/tween';
	import { loadTypeface, type LoadedFace } from '$lib/components/typeface';
	import MoodLine from '$lib/components/MoodLine.svelte';
	import Wall from '$lib/components/Wall.svelte';
	import FullView from '$lib/components/FullView.svelte';
	import RoomLight from '$lib/components/RoomLight.svelte';

	const EXAMPLES = [
		'a snowy december and i just made warm hot chocolate',
		'rain on the window and nowhere to be',
		'driving home at 2am with the windows down',
		'the last day of summer, sunburnt and a little sad',
		'a library so quiet you can hear the pages turn',
		'neon puddles outside a ramen bar after midnight',
		'golden light through the kitchen window while the bread rises',
		'a night train through the mountains and i am the only one awake',
		'the first cold morning and the smell of somebody else’s fireplace',
		'dancing alone in the kitchen to a song i had forgotten',
		'an empty beach in february, grey water and a long coat',
		'the city from a rooftop just before the storm breaks',
		'a slow sunday with the record player and nothing planned',
		'walking home from a party, still humming, streetlights buzzing'
	];

	const NEUTRAL: OKLab[] = [
		[0.97, 0, 0],
		[0.86, 0, 0],
		[0.72, 0, 0],
		[0.55, 0, 0],
		[0.35, 0, 0]
	];

	type Open = NonNullable<App.PageState['open']>;

	let text = $state(page.url.searchParams.get('m') ?? '');
	let mood = $state<Mood | null>(null);
	let waiting = $state(false);
	let leaving = $state(false);
	let error = $state<string | null>(null);
	let face = $state<LoadedFace | null>(null);
	let lastOpened = $state<Open | null>(null);
	let lineRef = $state<HTMLTextAreaElement | null>(null);

	let modelReady = false;
	let opener: HTMLElement | null = null;
	let seq = 0;
	let reduced = false;

	const urlMood = $derived((page.url.searchParams.get('m') ?? '').trim());
	const open = $derived(mood && page.state.open ? page.state.open : null);
	const openItem = $derived(
		!mood || !open ? null : open === 'anchor' ? (mood.anchor ?? null) : mood.picks[open]
	);
	const favicon = $derived(paletteFavicon(mood ? mood.palette : NEUTRAL));
	const settled = $derived(mood !== null && text.trim() === mood.query);
	const scentDelay = 1300;

	const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
	/** For a song, the album carries the feeling better than the single track. */
	const work = (item: Item) => (item.category === 'song' ? (item.album ?? item.title) : item.title);

	onMount(() => {
		reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
		applyTokens(neutralTokens(matchMedia('(prefers-color-scheme: dark)').matches));
		start();
		ready.then(
			() => (modelReady = true),
			() => {}
		);
	});

	// Type anywhere: a key pressed while nothing has focus goes to the line.
	function typeAnywhere(e: KeyboardEvent) {
		if (open || e.key.length !== 1 || e.ctrlKey || e.metaKey || e.altKey) return;
		if (document.activeElement && document.activeElement !== document.body) return;
		lineRef?.focus({ preventScroll: true });
	}

	// The URL is the source of truth: Enter pushes it, back and forward move it, and this reacts.
	$effect(() => {
		const q = urlMood;
		untrack(() => {
			if (q !== (mood?.query ?? '')) void show(q);
		});
	});

	async function show(q: string) {
		const my = ++seq;
		error = null;
		if (!q) {
			if (mood) {
				leaving = true;
				await sleep(200);
				if (my !== seq) return;
			}
			mood = null;
			leaving = false;
			waiting = false;
			face = null;
			text = '';
			void tweenTokens(neutralTokens(matchMedia('(prefers-color-scheme: dark)').matches));
			return;
		}

		text = q;
		if (!modelReady) waiting = true;
		let m: Mood;
		try {
			m = await infer(q);
		} catch {
			if (my !== seq) return;
			waiting = false;
			error = 'the moods are out — try again soon';
			return;
		}
		if (my !== seq) return;
		waiting = false;

		if (mood) {
			leaving = true;
			await sleep(200);
			if (my !== seq) return;
		}

		void tweenTokens(paletteToTokens(m.palette), 900);
		lastOpened = null;
		leaving = false;
		mood = m;

		loadTypeface(m.typeface, m.picks.poem.text ?? '').then(
			(f) => {
				if (my === seq) face = f;
			},
			() => {
				if (my === seq) face = null;
			}
		);
	}

	function submit(raw: string) {
		const q = raw.trim();
		if (!q) return;
		if (q === urlMood) {
			if (!mood || mood.query !== q) void show(q);
			return;
		}
		// A real navigation, not pushState: shallow routing leaves page.url unchanged, so the effect would not run.
		// The path is the resolved root; only the query changes.
		// eslint-disable-next-line svelte/no-navigation-without-resolve
		void goto(`${resolve('/')}?m=${encodeURIComponent(q)}`, { keepFocus: true, noScroll: true });
	}

	function viewTransition(update: () => void | Promise<void>): Promise<void> {
		if (reduced || !document.startViewTransition) {
			return Promise.resolve(update()).then(() => {});
		}
		const vt = document.startViewTransition(async () => {
			await update();
			await tick();
		});
		return vt.finished.catch(() => {});
	}

	async function openView(key: Open, el: HTMLButtonElement) {
		opener = el;
		lastOpened = key;
		await tick();
		await viewTransition(() => pushState('', { open: key }));
	}

	async function closeView() {
		if (!page.state.open) return;
		await viewTransition(
			() =>
				new Promise<void>((resolve) => {
					addEventListener('popstate', () => resolve(), { once: true });
					history.back();
				})
		);
	}

	// Focus returns to the tile that opened the view, on Esc, backdrop click, or the back button alike.
	$effect(() => {
		if (open || !opener) return;
		opener.focus({ preventScroll: true });
		opener = null;
	});
</script>

<svelte:head>
	<title>{mood ? mood.query : 'mise'}</title>
	<link rel="icon" href={favicon} />
</svelte:head>

<main
	class="stage"
	class:shown={mood !== null}
	style:--mood-font={face?.family ?? null}
	style:--mood-style={face?.style ?? null}
>
	<header class="line">
		<MoodLine
			bind:value={text}
			bind:ref={lineRef}
			examples={EXAMPLES}
			{settled}
			{waiting}
			onsubmit={submit}
		/>

		{#if mood?.anchor}
			{#key mood.query}
				{@const anchor = mood.anchor}
				<p class="anchor" class:leaving class:editing={!settled}>
					<span class="reveal">
						in the key of
						<button
							type="button"
							class="work"
							style:view-transition-name={!open && lastOpened === 'anchor' ? 'mood-tile' : null}
							onclick={(e) => openView('anchor', e.currentTarget)}
							>{work(anchor)} <span class="dash">—</span> {anchor.creator}</button
						>
					</span>
				</p>
			{/key}
		{/if}

		{#if error}
			<p class="note" in:fade={{ duration: 400 }}>{error}</p>
		{/if}
	</header>

	{#if mood}
		{#key mood.query}
			<Wall
				{mood}
				{leaving}
				active={open || lastOpened === 'anchor' ? null : lastOpened}
				onopen={(item, el) => openView(item.category, el)}
			/>
			<div class="scent" class:leaving style:--delay="{scentDelay}ms">
				<p class="label">scent</p>
				<p class="note-text" aria-label={mood.scent.text}>
					{#each mood.scent.text.split('') as ch, i (i)}
						<span style:--i={i} aria-hidden="true">{ch}</span>
					{/each}
				</p>
			</div>
		{/key}
	{/if}

	<!-- After the line in DOM order, so the first Tab lands on the line. -->
	<a class="mark" href={resolve('/')} aria-label="mise, start over">
		<span class="swatch" aria-hidden="true">
			<i style:background="var(--ground)"></i>
			<i style:background="var(--mid-1)"></i>
			<i style:background="var(--mid-2)"></i>
			<i style:background="var(--mid-3)"></i>
			<i style:background="var(--ink)"></i>
		</span>
		mise
	</a>
</main>

<svelte:window onkeydown={typeAnywhere} />

{#if waiting}
	<div class="breath" aria-hidden="true" transition:fade={{ duration: 600 }}></div>
{/if}

<RoomLight light={mood?.light ?? null} />

{#if openItem}
	<FullView
		item={openItem}
		kicker={open === 'anchor' ? 'in the key of' : undefined}
		onclose={closeView}
	/>
{/if}

<style>
	.stage {
		--line-size: clamp(1.55rem, 2.6vw + 0.7rem, 2.7rem);
		position: relative;
		z-index: 1;
		display: flex;
		flex-direction: column;
		min-height: 100dvh;
		box-sizing: border-box;
	}

	.line {
		display: flex;
		flex-direction: column;
		align-items: center;
		width: min(100% - 2 * max(2.5vw, 12px), 44rem);
		margin: 0 auto;
		padding-top: 34vh;
		padding-bottom: 1.5rem;
		box-sizing: border-box;
		transition: padding-top 900ms var(--ease);
	}

	.shown .line {
		padding-top: clamp(5rem, 12vh, 8rem);
	}

	.mark {
		position: absolute;
		top: 0;
		left: 0;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		min-height: 44px;
		padding: max(1.5vw, 10px) max(2.5vw, 12px);
		color: var(--ink);
		font-size: 1.15rem;
		letter-spacing: 0.03em;
		text-decoration: none;
		opacity: 0.75;
		transition: opacity 400ms var(--ease);
	}

	.mark:hover,
	.mark:focus-visible {
		opacity: 1;
	}

	.mark:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: -6px;
	}

	.swatch {
		display: flex;
		height: 0.8rem;
		outline: 1px solid color-mix(in oklab, var(--ink) 25%, transparent);
		outline-offset: 1px;
		transition: gap 500ms var(--ease);
	}

	.swatch i {
		width: 0.3rem;
	}

	.mark:hover .swatch {
		gap: 2px;
	}

	.anchor {
		margin: 0.9rem 0 0;
		font-size: 0.95rem;
		text-align: center;
		transition: opacity 200ms ease-out;
	}

	.anchor .reveal {
		display: inline-block;
		animation: letter 900ms var(--ease) 700ms both;
	}

	.anchor .work {
		position: relative;
		margin: -0.7rem 0;
		padding: 0.7rem 0;
		border: 0;
		background: none;
		color: var(--ink);
		font: inherit;
		cursor: pointer;
	}

	.anchor .work::after {
		content: '';
		position: absolute;
		left: 0;
		right: 0;
		bottom: 0.55rem;
		height: 1px;
		background: color-mix(in oklab, var(--ink) 45%, transparent);
		transform-origin: left;
		animation: draw 900ms var(--ease) 1200ms both;
		transition: background-color 300ms var(--ease);
	}

	.anchor .work:hover::after,
	.anchor .work:focus-visible::after {
		background: var(--ink);
	}

	.anchor .work:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: 2px;
	}

	.dash {
		opacity: 0.5;
	}

	/* The anchor belongs to the feeling on the wall; while the line is edited, the hint takes its place. */
	.anchor.leaving,
	.anchor.editing {
		opacity: 0;
		pointer-events: none;
	}

	.note {
		margin: 0.5rem 0 0;
		font-size: 0.95rem;
		opacity: 0.6;
	}

	.scent {
		margin: 0.75rem auto 0;
		padding: 0 max(2.5vw, 12px) 4rem;
		text-align: center;
		transition: opacity 200ms ease-out;
	}

	.scent p {
		margin: 0;
	}

	.scent .label {
		font-size: 0.95rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		opacity: 0;
		animation: dim 700ms var(--ease) both;
		animation-delay: calc(var(--delay) - 300ms);
	}

	.note-text {
		font-size: 1.05rem;
		white-space: pre-wrap;
	}

	.scent.leaving {
		opacity: 0;
	}

	.note-text span {
		opacity: 0;
		animation: letter 500ms var(--ease) both;
		animation-delay: calc(var(--delay) + var(--i) * 30ms);
	}

	.breath {
		position: fixed;
		inset: 0;
		z-index: 0;
		background: var(--ink);
		opacity: 0.05;
		animation: breathe 2.4s ease-in-out infinite;
		pointer-events: none;
	}

	@keyframes letter {
		from {
			opacity: 0;
		}
		to {
			opacity: 1;
		}
	}

	@keyframes dim {
		from {
			opacity: 0;
		}
		to {
			opacity: 0.55;
		}
	}

	@keyframes draw {
		from {
			transform: scaleX(0);
		}
		to {
			transform: scaleX(1);
		}
	}

	@keyframes breathe {
		0%,
		100% {
			opacity: 0.015;
		}
		50% {
			opacity: 0.07;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.line {
			transition: none;
		}

		.note-text span {
			animation-delay: var(--delay);
		}

		.anchor .work::after {
			animation: none;
		}

		.mark .swatch {
			transition: none;
		}

		.breath {
			animation: none;
			opacity: 0.04;
		}
	}
</style>
