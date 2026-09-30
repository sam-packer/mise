<script lang="ts">
	// Turn shared feeling text into a mood, travel into the worlds of its works, and coordinate
	// the walls, colors, fonts, and navigation.
	import { onMount, tick, untrack } from 'svelte';
	import { fade } from 'svelte/transition';
	import { SvelteMap } from 'svelte/reactivity';
	import { env } from '$env/dynamic/public';
	import { page } from '$app/state';
	import { afterNavigate, goto, pushState } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { BUNDLE_URL } from '$lib/bundle';
	import { feelingCode, feelings, normalize } from '$lib/code';
	import { CATEGORIES, type Item, type Mood, type OKLab, type World } from '$lib/mood/types';
	import { infer, ready, start, world as requestWorld } from '$lib/mood/client';
	import { lightStrength, neutralTokens, paletteFavicon, paletteToTokens } from '$lib/color/oklab';
	import { applyTokens, tweenTokens } from '$lib/color/tween';
	import { loadTypeface, type LoadedFace } from '$lib/components/typeface';
	import { EXAMPLES } from '$lib/examples';
	import MoodLine from '$lib/components/MoodLine.svelte';
	import Wall from '$lib/components/Wall.svelte';
	import Scent from '$lib/components/Scent.svelte';
	import WorldView from '$lib/components/World.svelte';
	import RoomLight from '$lib/components/RoomLight.svelte';

	const NEUTRAL: OKLab[] = [
		[0.97, 0, 0],
		[0.86, 0, 0],
		[0.72, 0, 0],
		[0.55, 0, 0],
		[0.35, 0, 0]
	];

	let { data } = $props();

	let text = $state(untrack(() => data.text));
	let mood = $state<Mood | null>(null);
	let waiting = $state(false);
	let traveling = $state(false);
	let leaving = $state(false);
	let error = $state<string | null>(null);
	let face = $state<LoadedFace | null>(null);
	let worldFace = $state<{ id: string; face: LoadedFace } | null>(null);
	let lineRef = $state<HTMLTextAreaElement | null>(null);
	let heading = $state<HTMLHeadingElement | null>(null);
	/** The item whose tile carries the transition name while its world opens. */
	let pending = $state<string | null>(null);
	/** The item whose world the user left last, so that its tile takes the transition name back. */
	let left = $state<string | null>(null);
	/** The world that stays on screen while it fades out, after the mark starts over. */
	let held = $state<{ world: World; path: Item[]; feeling: string } | null>(null);

	/** Worlds by path: the item ids from the feeling to the world, joined by newlines. */
	const worlds = new SvelteMap<string, World>();
	/** The paths whose worlds the worker builds now. No markup reads it, so it is not reactive. */
	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- only the load effect's guard reads it
	const loading = new Set<string>();
	/** The last scroll position of each path. The empty path is the mood wall. */
	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- written on each scroll, read on navigation
	const scrolls = new Map<string, number>();
	let modelReady = false;
	let seq = 0;
	let reduced = false;
	/** The world that the color tokens show now. */
	let tinted: World | null = null;
	/** The path on screen, read when the browser moves through history. */
	let shownTrail: string[] = [];

	const pathKey = (ids: string[]) => ids.join('\n');
	const urlMood = $derived(data.text);
	const trail = $derived(mood ? (page.state.trail ?? []) : []);
	const key = $derived(pathKey(trail));
	const steps = $derived(trail.map((_, i) => worlds.get(pathKey(trail.slice(0, i + 1))) ?? null));
	const world = $derived(
		steps.length && steps.every((s) => s !== null) ? (steps[steps.length - 1] as World) : null
	);
	const shown = $derived(
		world ? { world, path: steps.slice(0, -1).map((s) => s!.item), feeling: urlMood } : held
	);
	const scene = $derived(shown?.world ?? mood);
	const focus = $derived(pending ?? left);
	const picks = $derived.by(() => {
		const m = mood;
		return m ? CATEGORIES.flatMap((c) => (m.picks[c] ? [m.picks[c]] : [])) : [];
	});
	const favicon = $derived(paletteFavicon(scene ? scene.palette : NEUTRAL));
	const strength = $derived(
		scene ? lightStrength(paletteToTokens(scene.palette, scene.light), scene.light) : 1
	);
	const settled = $derived(mood !== null && text.trim() === mood.query);
	const scentDelay = 1300;

	const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
	const message = (cause: unknown) =>
		cause instanceof Error ? cause.message : 'the moods are out — try again soon';
	/** For a song, the album carries the feeling better than the single track. */
	const work = (item: Item) => (item.category === 'song' ? (item.album ?? item.title) : item.title);

	// Allow a local bundle through PUBLIC_BUNDLE_URL.
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

	// Focus the field when the user types with no other control selected.
	function typeAnywhere(e: KeyboardEvent) {
		if (e.key === 'Escape' && world) {
			e.preventDefault();
			void back(1);
			return;
		}
		if (shown || e.key.length !== 1 || e.ctrlKey || e.metaKey || e.altKey) return;
		if (document.activeElement && document.activeElement !== document.body) return;
		lineRef?.focus({ preventScroll: true });
	}

	// Run inference after full navigation, including the initial page load.
	afterNavigate(() => {
		void show(data.text);
		if (data.note) error = data.note;
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
			held = null;
			leaving = false;
			waiting = false;
			face = null;
			text = '';
			// Release the focus retained by goto so example rotation can resume.
			lineRef?.blur();
			void tweenTokens(neutralTokens(matchMedia('(prefers-color-scheme: dark)').matches));
			return;
		}

		text = q;
		if (!modelReady) waiting = true;
		let m: Mood;
		try {
			m = await infer(q);
		} catch (cause) {
			if (my !== seq) return;
			waiting = false;
			error = message(cause);
			return;
		}
		if (my !== seq) return;
		waiting = false;
		if (mood) {
			leaving = true;
			await sleep(200);
			if (my !== seq) return;
		}
		// History can return to a world of this feeling. Then the world keeps its own colors.
		if (!world) void tweenTokens(paletteToTokens(m.palette, m.light), 900);
		pending = null;
		left = null;
		leaving = false;
		mood = m;
		loadFace(my, m, m.picks.poem?.text ?? '');
	}

	function loadFace(my: number, m: Mood, poem: string) {
		loadTypeface(m.typeface, poem).then(
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
		// Derive the code locally so navigation does not wait for storage.
		const code = await feelingCode(q);
		feelings.set(code, q);
		fetch('/api/feeling', {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({ text: q })
		}).catch(() => {});
		// Navigate to the feeling so afterNavigate starts inference.
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

	// Load every world on the path that is not in memory, for example after the browser returns to
	// a page whose history holds a path.
	$effect(() => {
		trail.forEach((_, i) => {
			const ids = trail.slice(0, i + 1);
			const k = pathKey(ids);
			if (worlds.has(k) || loading.has(k)) return;
			loading.add(k);
			requestWorld(ids[i], ids)
				.then(
					(w) => worlds.set(k, w),
					(cause) => (error = message(cause))
				)
				.finally(() => loading.delete(k));
		});
	});

	// Retint the page when a world opens or closes. show() tints the page for a new mood.
	$effect(() => {
		const w = world;
		if (w === tinted) return;
		tinted = w;
		if (w) {
			void tweenTokens(paletteToTokens(w.palette, w.light), 900);
			const id = w.item.id;
			loadTypeface(w.typeface, w.item.title + (w.item.text ?? '')).then(
				(f) => {
					if (world?.item.id === id) worldFace = { id, face: f };
				},
				() => {}
			);
		} else if (mood && !held) {
			void tweenTokens(paletteToTokens(mood.palette, mood.light), 900);
		}
	});

	/** Travel from the wall on screen into the world of one of its works. */
	async function travel(item: Item) {
		if (traveling) return;
		const from = key;
		const ids = [...trail, item.id];
		const k = pathKey(ids);
		error = null;
		let w = worlds.get(k);
		if (!w) {
			traveling = true;
			try {
				w = await requestWorld(item.id, ids);
			} catch (cause) {
				error = message(cause);
				return;
			} finally {
				traveling = false;
			}
			// The user left this wall while the world loaded.
			if (key !== from) return;
			worlds.set(k, w);
		}
		left = null;
		pending = item.id;
		await tick();
		await viewTransition(() => {
			pushState('', { trail: ids });
			pending = null;
			scrollTo(0, 0);
		});
		heading?.focus({ preventScroll: true });
	}

	/** Go back the given number of steps on the path. The popstate listener restores focus. */
	async function back(count: number) {
		if (!world || count <= 0) return;
		const y = scrolls.get(pathKey(trail.slice(0, trail.length - count))) ?? 0;
		left = world.item.id;
		pending = null;
		await viewTransition(() =>
			new Promise<void>((resolve) => {
				addEventListener('popstate', () => resolve(), { once: true });
				history.go(-count);
			}).then(async () => {
				await tick();
				scrollTo(0, y);
			})
		);
	}

	function startOver(e: MouseEvent) {
		if (!world || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
		held = shown;
	}

	$effect(() => {
		shownTrail = trail;
	});

	// Keep the scroll position of each path, and restore scroll and focus when the browser moves
	// through the path's history.
	onMount(() => {
		const onscroll = () => scrolls.set(key, scrollY);
		const onpopstate = async () => {
			// The router's popstate listener runs first. For a step on the path it sets page.state at
			// once; for another page it navigates later, and afterNavigate takes over.
			if (location.pathname !== page.url.pathname) return;
			const before = shownTrail;
			const after = mood ? (page.state.trail ?? []) : [];
			const y = scrolls.get(pathKey(after)) ?? 0;
			const shorter = after.length < before.length;
			if (shorter) left = before[before.length - 1];
			else if (after.length > before.length) left = null;
			await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
			await tick();
			scrollTo(0, y);
			const tile = shorter
				? [...document.querySelectorAll<HTMLElement>(`[data-item="${CSS.escape(left!)}"]`)].find(
						(el) => !el.closest('[inert]')
					)
				: null;
			if (tile) tile.focus({ preventScroll: true });
			else if (world) heading?.focus({ preventScroll: true });
		};
		addEventListener('scroll', onscroll, { passive: true });
		addEventListener('popstate', onpopstate);
		return () => {
			seq++;
			removeEventListener('scroll', onscroll);
			removeEventListener('popstate', onpopstate);
		};
	});
</script>

<svelte:head>
	<title>{shown ? shown.world.item.title : mood ? mood.query : 'mise'}</title>
	<link rel="icon" href={favicon} />
</svelte:head>

<main
	class="stage"
	class:shown={mood !== null}
	style:--mood-font={face?.family ?? null}
	style:--mood-style={face?.style ?? null}
>
	<!-- The mood stays in the page under a world, so a return shows it as the user left it. -->
	<div class="scene" class:away={shown !== null} inert={shown !== null}>
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
								data-item={anchor.id}
								style:view-transition-name={!shown && focus === anchor.id ? 'mood-tile' : null}
								onclick={() => travel(anchor)}
								>{work(anchor)} <span class="dash">—</span> {anchor.creator}</button
							>
						</span>
					</p>
				{/key}
			{/if}

			{#if error && !shown}
				<p class="note" in:fade={{ duration: 400 }}>{error}</p>
			{/if}
		</header>

		{#if mood}
			{#key mood.query}
				<Wall items={picks} label="picks" {leaving} active={shown ? null : focus} onopen={travel} />
			{/key}
			{#key `${mood.query}\n${mood.scent.id}`}
				<Scent text={mood.scent.text} {leaving} delay={scentDelay} />
			{/key}
		{/if}
	</div>

	{#if shown}
		{#key pathKey([...shown.path.map((p) => p.id), shown.world.item.id])}
			<WorldView
				world={shown.world}
				feeling={shown.feeling}
				path={shown.path}
				{focus}
				leaving={leaving && held !== null}
				face={worldFace?.id === shown.world.item.id ? worldFace.face : null}
				note={error}
				bind:heading
				onstep={(step) => back(trail.length - step)}
				ontravel={travel}
			/>
		{/key}
	{/if}

	<!-- Place the mark after the field so users reach the field first when they press Tab. -->
	<a class="mark" href={resolve('/')} aria-label="mise, start over" onclick={startOver}>
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

<a class="attribution" href={resolve('/attribution')}>attribution</a>

<svelte:window onkeydown={typeAnywhere} />

{#if waiting || traveling}
	<div class="breath" aria-hidden="true" transition:fade={{ duration: 600 }}></div>
{/if}

<RoomLight light={scene?.light ?? null} {strength} />

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

	.scene {
		display: flex;
		flex: 1;
		flex-direction: column;
	}

	/* Out of view and out of the flow, but still rendered, so its animations never start again. */
	.scene.away {
		position: absolute;
		inset: 0 0 auto;
		height: 0;
		overflow: hidden;
		visibility: hidden;
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
		color: var(--ink-soft);
		transition: color 400ms var(--ease);
	}

	.mark:hover .word,
	.mark:focus-visible .word {
		color: var(--ink);
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
		transition:
			opacity 200ms ease-out,
			transform 300ms var(--ease);
	}

	/* The line's hint sits out of the flow in the same place. While it shows, the anchor steps down
	   below it; a transform, so the wall under the header never moves. */
	.line:has(:global(.hint)) .anchor {
		transform: translateY(1.4rem);
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
		color: var(--ink-soft);
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
		color: var(--ink-soft);
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
		border-top-left-radius: 0.75rem;
		/* A veil, so the label stays readable over the tiles that scroll under it. */
		background: color-mix(in srgb, var(--veil) 80%, transparent);
		backdrop-filter: blur(16px);
		-webkit-backdrop-filter: blur(16px);
		color: var(--ink-soft);
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
