<script lang="ts">
	import { onMount, tick, untrack } from 'svelte';
	import { fade } from 'svelte/transition';
	import { env } from '$env/dynamic/public';
	import { page } from '$app/state';
	import { goto, pushState } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { BUNDLE_URL } from '$lib/bundle';
	import { feelingCode, feelings, normalize } from '$lib/code';
	import type { Item, Mood, OKLab } from '$lib/mood/types';
	import { infer, ready, start } from '$lib/mood/client';
	import { neutralTokens, paletteFavicon, paletteToTokens } from '$lib/color/oklab';
	import { applyTokens, tweenTokens } from '$lib/color/tween';
	import { loadTypeface, type LoadedFace } from '$lib/components/typeface';
	import { EXAMPLES } from '$lib/examples';
	import MoodLine from '$lib/components/MoodLine.svelte';
	import Wall from '$lib/components/Wall.svelte';
	import FullView from '$lib/components/FullView.svelte';
	import RoomLight from '$lib/components/RoomLight.svelte';

	const NEUTRAL: OKLab[] = [
		[0.97, 0, 0],
		[0.86, 0, 0],
		[0.72, 0, 0],
		[0.55, 0, 0],
		[0.35, 0, 0]
	];

	type Open = NonNullable<App.PageState['open']>;

	let { data } = $props();

	let text = $state(untrack(() => data.text));
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

	const urlMood = $derived(data.text);
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

	// The bundle root: BUNDLE_URL, unless PUBLIC_BUNDLE_URL is set (a local stub for testing).
	const bundleOverride = env.PUBLIC_BUNDLE_URL?.trim();
	const bundleUrl = bundleOverride || BUNDLE_URL;
	const bundleBase = bundleUrl.endsWith('/') ? bundleUrl : `${bundleUrl}/`;

	onMount(() => {
		reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
		applyTokens(neutralTokens(matchMedia('(prefers-color-scheme: dark)').matches));
		start(bundleBase);
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
	// A code the store does not know shows the landing with a note.
	$effect(() => {
		const q = urlMood;
		const note = data.note;
		untrack(() => {
			if (q !== (mood?.query ?? '')) void show(q);
			if (note) error = note;
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
			// The line kept focus through goto's keepFocus, which paused the ghost's rotation. Let it resume.
			lineRef?.blur();
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

	async function submit(raw: string) {
		const q = normalize(raw);
		if (!q) return;
		if (q === urlMood) {
			if (!mood || mood.query !== q) void show(q);
			return;
		}
		// The code comes from the text, so the page moves at once and the store catches up on its own.
		const code = await feelingCode(q);
		feelings.set(code, q);
		fetch('/api/feeling', {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({ text: q })
		}).catch(() => {});
		// A real navigation, not pushState: shallow routing leaves page.url unchanged, so the effect would not run.
		void goto(resolve('/[[code=code]]', { code }), { keepFocus: true, noScroll: true });
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
			boxed={mood !== null}
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
			<i style:--k={0} style:background="var(--ground)"></i>
			<i style:--k={1} style:background="var(--mid-1)"></i>
			<i style:--k={2} style:background="var(--mid-2)"></i>
			<i style:--k={3} style:background="var(--mid-3)"></i>
			<i style:--k={4} style:background="var(--ink)"></i>
		</span>
		<span class="word">mise</span>
	</a>
</main>

{#if !openItem}
	<a class="attribution" href={resolve('/attribution')}>attribution</a>
{/if}

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
		--band: 0.56rem;
		position: absolute;
		top: 0;
		left: 0;
		display: flex;
		align-items: center;
		gap: 0.7rem;
		min-height: 44px;
		padding: max(1.3vw, 10px) max(2.5vw, 12px);
		color: var(--ink);
		text-decoration: none;
	}

	.mark:focus-visible {
		outline: none;
	}

	.word {
		font-size: 1.9rem;
		line-height: 1;
		letter-spacing: 0.015em;
		opacity: 0.85;
		transition: opacity 400ms var(--ease);
	}

	.mark:hover .word,
	.mark:focus-visible .word {
		opacity: 1;
	}

	.mark:focus-visible .word {
		text-decoration: underline 1px;
		text-underline-offset: 0.2em;
	}

	/* The swatch box always holds five bands, so the word never moves. On the landing every band rests
	   behind the ink band beside the word; at reveal they fan out to the left on the compositor. */
	.swatch {
		position: relative;
		flex: none;
		width: calc(5 * var(--band));
		height: 1.5rem;
	}

	.swatch i {
		position: absolute;
		top: 0;
		bottom: 0;
		left: calc(var(--k) * var(--band));
		width: var(--band);
		transform: translateX(calc((4 - var(--k)) * var(--band)));
		transition: transform 650ms cubic-bezier(0.23, 1, 0.32, 1);
		transition-delay: calc((3 - var(--k)) * 25ms);
	}

	.swatch i:first-child {
		box-shadow: inset 0 0 0 1px color-mix(in oklab, var(--ink) 25%, transparent);
	}

	.shown .swatch i {
		transform: none;
		transition-duration: 900ms;
		transition-timing-function: cubic-bezier(0.16, 1, 0.3, 1);
		transition-delay: calc(var(--k) * 35ms);
	}

	/* A hover or focus on the mark spreads the bands evenly around the middle one, transform only
	   so it never touches layout or moves the word. No stagger: every band moves together. */
	.mark:hover .swatch i,
	.mark:focus-visible .swatch i {
		transform: translateX(calc((4 - var(--k)) * var(--band) + (var(--k) - 2) * 2px));
		transition: transform 400ms cubic-bezier(0.16, 1, 0.3, 1);
	}

	.shown .mark:hover .swatch i,
	.shown .mark:focus-visible .swatch i {
		transform: translateX(calc((var(--k) - 2) * 2px));
		transition: transform 400ms cubic-bezier(0.16, 1, 0.3, 1);
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

	.attribution {
		position: fixed;
		right: 0;
		bottom: 0;
		z-index: 2;
		display: flex;
		align-items: center;
		justify-content: center;
		min-width: 44px;
		min-height: 44px;
		padding: max(1vw, 8px) calc(max(2.5vw, 12px) + env(safe-area-inset-right, 0px))
			calc(max(1vw, 8px) + env(safe-area-inset-bottom, 0px)) max(1vw, 8px);
		box-sizing: border-box;
		color: color-mix(in oklab, var(--ink) 42%, transparent);
		font-size: 0.8rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		text-decoration: none;
		transition: color 300ms var(--ease);
	}

	.attribution:hover,
	.attribution:focus-visible {
		color: var(--ink);
	}

	.attribution:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: -3px;
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

		.swatch i {
			transform: none;
			transition: none;
		}

		.mark:hover .swatch i,
		.mark:focus-visible .swatch i,
		.shown .mark:hover .swatch i,
		.shown .mark:focus-visible .swatch i {
			transform: none;
			transition: none;
		}

		.breath {
			animation: none;
			opacity: 0.04;
		}
	}
</style>
