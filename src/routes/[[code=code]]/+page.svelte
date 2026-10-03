<script lang="ts">
	// Turn shared feeling text into a mood, travel into the worlds of its works, and coordinate
	// the walls, colors, fonts, and navigation.
	import { onMount, tick, untrack } from 'svelte';
	import { fade } from 'svelte/transition';
	import { SvelteMap } from 'svelte/reactivity';
	import { PUBLIC_BUNDLE_URL } from '$app/env/public';
	import { page } from '$app/state';
	import { afterNavigate, beforeNavigate, goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { BUNDLE_URL } from '#lib/bundle.js';
	import { BUNDLE, feelingCode, normalize, pathCode, shared } from '#lib/code.js';
	import type { Item, Mood, OKLab, Palette, World } from '#lib/mood/types.js';
	import { infer, ready, start, world as requestWorld } from '#lib/mood/client.js';
	import { loadSamples, sample } from '#lib/mood/samples.js';
	import {
		lightStrength,
		neutralTokens,
		paletteFavicon,
		paletteToTokens
	} from '#lib/color/oklab.js';
	import { applyTokens, tweenTokens } from '#lib/color/tween.js';
	import { loadTypeface, type LoadedFace } from '#lib/components/typeface.js';
	import { EXAMPLES } from '#lib/examples.js';
	import MoodLine from '#lib/components/MoodLine.svelte';
	import Wall from '#lib/components/Wall.svelte';
	import WorldView from '#lib/components/World.svelte';
	import RoomLight from '#lib/components/RoomLight.svelte';
	import Tagline from '#lib/components/Tagline.svelte';

	const NEUTRAL: OKLab[] = [
		[0.97, 0, 0],
		[0.86, 0, 0],
		[0.72, 0, 0],
		[0.55, 0, 0],
		[0.35, 0, 0]
	];

	/** One world on the path: its path key, the world, and the works before it. */
	type Layer = { key: string; world: World; path: Item[] };

	let { data } = $props();

	let text = $state(untrack(() => data.text));
	let mood = $state<Mood | null>(null);
	let waiting = $state(false);
	/** The example feeling that the empty line shows. */
	let example = $state('');
	let traveling = $state(false);
	let leaving = $state(false);
	let error = $state<string | null>(null);
	let face = $state<LoadedFace | null>(null);
	let lineRef = $state<HTMLTextAreaElement | null>(null);
	/** The title of each world on the path, by its place on the path. */
	let headings = $state<(HTMLHeadingElement | null)[]>([]);
	/** The item whose tile carries the transition name while its world opens. */
	let pending = $state<string | null>(null);
	/** The item whose world the user left last, so that its tile takes the transition name back. */
	let left = $state<string | null>(null);
	/** The tile that the path went on through, which a step back over several worlds shows briefly. */
	let returned = $state<string | null>(null);
	/** A move over several worlds: the new view names no work, so the old one recedes. */
	let still = $state(false);
	/**
	 * The path to keep on screen while the browser's own Back or Forward starts a view transition. The
	 * router sets the new state before the transition captures the old view, so the page holds it.
	 */
	let hold = $state<string[] | null>(null);
	/** The worlds that stay on screen while they fade out, after the mark starts over. */
	let held = $state<{ layers: Layer[]; feeling: string } | null>(null);

	/** Worlds by path: the item ids from the feeling to the world, joined by newlines. */
	const worlds = new SvelteMap<string, World>();
	/** The loaded typeface of each world, by item id. */
	const faces = new SvelteMap<string, LoadedFace>();
	/** The paths whose worlds the worker builds now. No markup reads it, so it is not reactive. */
	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- only the load effect's guard reads it
	const loading = new Set<string>();
	/** The last scroll position of each path. The empty path is the mood wall. */
	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- written on each scroll, read on navigation
	const scrolls = new Map<string, number>();
	/** The feeling saves that this session started, by code. */
	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- only savePalette reads it
	const saving = new Map<string, Promise<unknown>>();
	/** The codes whose palettes this session sent. */
	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- only savePalette reads it
	const paletteSent = new Set<string>();
	let modelReady = false;
	let seq = 0;
	let reduced = false;
	/** The world that the color tokens show now. */
	let tinted: World | null = null;
	/** back() moves through history inside its own view transition. */
	let ownMove = false;
	/** The path before the last history move. Navigation, travel, and the popstate listener set it. */
	let shownTrail: string[] = [];

	const pathKey = (ids: string[]) => ids.join('\n');
	const urlMood = $derived(data.text);
	/** The path that the URL names. A history entry that the page pushed keeps it in its state. */
	const target = $derived(hold ?? page.state.trail ?? data.trail);
	/** The shortest path that Back in history reaches from this entry. */
	const floor = $derived(page.state.floor ?? data.trail.length);
	const trail = $derived(mood ? target : []);
	const key = $derived(pathKey(trail));
	const steps = $derived(trail.map((_, i) => worlds.get(pathKey(trail.slice(0, i + 1))) ?? null));
	const world = $derived(
		steps.length && steps.every((s) => s !== null) ? (steps[steps.length - 1] as World) : null
	);
	/**
	 * Every world on the path, first to last. All of them stay in the page and only the last one shows,
	 * so a step back shows the world under it as the user left it.
	 */
	const layers: Layer[] = $derived(
		world
			? steps.map((s, i) => ({
					key: pathKey(trail.slice(0, i + 1)),
					world: s!,
					path: steps.slice(0, i).map((p) => p!.item)
				}))
			: (held?.layers ?? [])
	);
	const shown = $derived(
		layers.length
			? { ...layers[layers.length - 1], feeling: world ? urlMood : (held?.feeling ?? urlMood) }
			: null
	);
	const heading = $derived(headings[layers.length - 1] ?? null);
	/** A link to a path opens: the page waits for its worlds, and keeps the mood wall out of view. */
	const arriving = $derived(target.length > 0 && !world && !error);
	const scene = $derived(shown?.world ?? mood);
	const focus = $derived(pending ?? left);
	const favicon = $derived(paletteFavicon(scene ? scene.palette : NEUTRAL));
	const strength = $derived(
		scene ? lightStrength(paletteToTokens(scene.palette, scene.light), scene.light) : 1
	);
	const settled = $derived(mood !== null && text.trim() === mood.query);

	/** How long a travel waits for the next world's typeface, in milliseconds. */
	const FACE_WAIT = 400;

	const sleep = (ms: number) => new Promise<null>((r) => setTimeout(() => r(null), ms));
	/** The text that a world sets in its own typeface: the title, and a poem. */
	const glyphs = (w: World) => w.item.title + (w.item.text ?? '');
	const message = (cause: unknown) =>
		cause instanceof Error ? cause.message : 'the moods are out — try again soon';
	/** The worker's message for an item id that the catalog does not have, for example in a local bundle. */
	const MISSING = 'that work is not in the catalog';
	/** For a song, the album carries the feeling better than the single track. */
	const work = (item: Item) => (item.category === 'song' ? (item.album ?? item.title) : item.title);

	// Allow a local bundle through PUBLIC_BUNDLE_URL.
	const bundleOverride = PUBLIC_BUNDLE_URL.trim();
	const bundleUrl = bundleOverride || BUNDLE_URL;
	const bundleBase = bundleUrl.endsWith('/') ? bundleUrl : `${bundleUrl}/`;

	onMount(() => {
		reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
		applyTokens(neutralTokens(matchMedia('(prefers-color-scheme: dark)').matches));
		start(bundleBase);
		loadSamples(bundleBase);
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

	/** Counts history moves and other navigations, so a history move that a newer one overtakes stops. */
	let pops = 0;
	// The router drops a history move when another navigation starts before it resolves the route.
	beforeNavigate((nav) => {
		if (nav.type !== 'popstate') pops++;
	});

	// Run inference after full navigation, including the initial page load.
	afterNavigate((nav) => {
		// A shallow navigation changes only the history entry, so the feeling stays.
		if (nav.shallow) return;
		// The router does not apply a history entry's state on the first page load, so the page shows
		// the path of the URL. Clear the state left in the entry, which may name another floor.
		// The router accepts shallow navigation only after it starts, just after this callback.
		if (nav.type === 'enter') queueMicrotask(() => void goto('', { shallow: true, replace: true }));
		shownTrail = page.state.trail ?? data.trail;
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
			// Release the focus retained by goto so the samples show again.
			lineRef?.blur();
			void tweenTokens(neutralTokens(matchMedia('(prefers-color-scheme: dark)').matches));
			return;
		}

		text = q;
		// A sample feeling has its room in the bundle, so it does not wait for the model.
		const room = await sample(q);
		if (my !== seq) return;
		let m: Mood;
		if (room) m = room.mood;
		else {
			if (!modelReady) waiting = true;
			try {
				m = await infer(q);
			} catch (cause) {
				if (my !== seq) return;
				waiting = false;
				error = message(cause);
				return;
			}
			if (my !== seq) return;
		}
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
		still = false;
		leaving = false;
		mood = m;
		loadFace(my, m, m.picks.map((p) => p.text ?? '').join(''));
		void savePalette(q, m.palette);
	}

	/** Send the palette of a feeling's mood to the server once, for the link preview image. */
	async function savePalette(q: string, palette: Palette) {
		const code = await feelingCode(q);
		if (paletteSent.has(code)) return;
		paletteSent.add(code);
		// The server keeps a palette only for a stored feeling, so wait for this session's save.
		await saving.get(code);
		fetch(`/api/feeling/${code}/palette`, {
			method: 'PUT',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({ text: q, palette, bundle: BUNDLE })
		}).catch(() => {});
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
		shared.set(code, { text: q, trail: [] });
		saving.set(
			code,
			fetch('/api/feeling', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ text: q })
			}).catch(() => {})
		);
		// Navigate to the feeling so afterNavigate starts inference.
		void goto(resolve('/[[code=code]]', { code }), { reset: false });
	}

	/** Run a page change as a view transition. The type, into a world or out of one, sets the shadow (layout.css). */
	function viewTransition(
		type: 'enter' | 'leave',
		update: () => void | Promise<void>
	): Promise<void> {
		if (reduced || !document.startViewTransition) {
			return Promise.resolve(update()).then(() => {});
		}
		const vt = document.startViewTransition({
			update: async () => {
				await update();
				await tick();
			},
			types: [type]
		});
		// A hidden tab or a newer transition aborts this one. The update still runs.
		vt.ready.catch(() => {});
		return vt.finished.catch(() => {});
	}

	/**
	 * Build the world at the end of a path. A first step from a sample feeling opens the world from
	 * the sample's file, so a tile on a sample's wall does not wait for the model.
	 */
	async function loadWorld(ids: string[]): Promise<World> {
		const id = ids[ids.length - 1];
		if (ids.length === 1) {
			const room = await sample(urlMood);
			if (room && Object.hasOwn(room.worlds, id)) return room.worlds[id];
		}
		return requestWorld(id, ids);
	}

	// Load every world on the path that is not in memory, in order, for example after a link to a path
	// opens or the browser returns to a page whose history holds a path.
	$effect(() => {
		trail.forEach((_, i) => {
			const ids = trail.slice(0, i + 1);
			const k = pathKey(ids);
			if (worlds.has(k) || loading.has(k)) return;
			loading.add(k);
			loadWorld(ids)
				.then(
					(w) => worlds.set(k, w),
					(cause) => {
						if (cause instanceof Error && cause.message === MISSING) void cut(ids.slice(0, i));
						else error = message(cause);
					}
				)
				.finally(() => loading.delete(k));
		});
	});

	/** The share code of a path of this feeling. The empty path is the feeling itself. */
	async function share(ids: string[]): Promise<string> {
		const text = urlMood;
		const code = ids.length ? await pathCode(text, ids, BUNDLE) : await feelingCode(text);
		if (!shared.has(code)) {
			shared.set(code, { text, trail: ids, bundle: BUNDLE });
			// Store the path in the background, as submit does for a feeling. The feeling is stored.
			if (ids.length)
				fetch('/api/feeling', {
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify({ text, trail: ids, bundle: BUNDLE })
				}).catch(() => {});
		}
		return code;
	}

	/** Stop the path of the URL before a work that the catalog no longer has. */
	async function cut(ids: string[]) {
		const before = target;
		const code = await share(ids);
		// Another cut or a history move changed the path meanwhile.
		if (target !== before || !ids.every((id, i) => before[i] === id)) return;
		await goto(resolve('/[[code=code]]', { code }), {
			shallow: true,
			replace: true,
			state: { trail: ids, floor: Math.min(floor, ids.length) }
		});
		shownTrail = ids;
	}

	// Retint the page when a world opens or closes. show() tints the page for a new mood.
	$effect(() => {
		const w = world;
		if (w === tinted) return;
		tinted = w;
		if (w) {
			void tweenTokens(paletteToTokens(w.palette, w.light), 900);
			const id = w.item.id;
			loadTypeface(w.typeface, glyphs(w)).then(
				(f) => {
					faces.set(id, f);
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
		traveling = true;
		const base = floor;
		let code: string;
		let w: World;
		let f: LoadedFace | null;
		try {
			[code, w] = await Promise.all([share(ids), worlds.get(k) ?? loadWorld(ids)]);
			// Give the typeface a moment, so the title does not change font during the transition.
			f = await Promise.race([
				loadTypeface(w.typeface, glyphs(w)).catch(() => null),
				sleep(FACE_WAIT)
			]);
		} catch (cause) {
			error = message(cause);
			return;
		} finally {
			traveling = false;
		}
		// The user left this wall while the world loaded.
		if (key !== from) return;
		worlds.set(k, w);
		if (f) faces.set(item.id, f);
		scrolls.set(from, scrollY);
		left = null;
		returned = null;
		still = false;
		pending = item.id;
		await tick();
		await viewTransition('enter', async () => {
			// Wait for page.state, so the scroll below records under the new path, not this wall.
			await goto(resolve('/[[code=code]]', { code }), {
				shallow: true,
				state: { trail: ids, floor: base }
			});
			shownTrail = ids;
			pending = null;
			scrollTo(0, 0);
		});
		heading?.focus({ preventScroll: true });
	}

	/**
	 * Go back the given number of steps on the path. The popstate listener restores focus. History
	 * holds no entry for a path shorter than the floor, for example after a link to a path opens, so
	 * then the page pushes a new entry.
	 */
	async function back(count: number) {
		if (!world || count <= 0) return;
		const before = trail;
		const ids = trail.slice(0, trail.length - count);
		scrolls.set(key, scrollY);
		const y = scrolls.get(pathKey(ids)) ?? 0;
		// The work on screen keeps the transition name. A step back over several worlds clears the
		// focus in the new view, so the work has no partner there and recedes (layout.css) instead of
		// morphing into a tile of another work.
		left = world.item.id;
		pending = null;
		returned = null;
		still = false;
		if (ids.length < floor) {
			const code = await share(ids);
			if (trail !== before) return;
			await viewTransition('leave', async () => {
				// Wait for page.state, as in travel().
				await goto(resolve('/[[code=code]]', { code }), {
					shallow: true,
					state: { trail: ids, floor: ids.length }
				});
				shownTrail = ids;
				if (count > 1) {
					left = null;
					still = true;
				}
				await tick();
				scrollTo(0, y);
			});
			refocus(before, ids);
			return;
		}
		await viewTransition('leave', async () => {
			const routed = nextState();
			ownMove = true;
			addEventListener('popstate', () => (ownMove = false), { once: true });
			history.go(-count);
			await routed;
			await tick();
			scrollTo(0, y);
		});
	}

	/**
	 * Resolve when the router next sets page.state. After a history move the router sets it only
	 * after it resolves the route, some time after the popstate event.
	 */
	function nextState(): Promise<void> {
		const from = page.state;
		return new Promise((resolve) => {
			const stop = $effect.root(() => {
				$effect(() => {
					if (page.state === from) return;
					resolve();
					queueMicrotask(stop);
				});
			});
		});
	}

	/**
	 * After a step back on the path, focus the work on this wall that the path went on through. After
	 * a step back over several worlds no morph lands on it, so it shows briefly.
	 */
	function refocus(before: string[], after: string[]) {
		const next = after.length < before.length ? before[after.length] : null;
		if (before.length - after.length > 1) returned = next;
		const tile = next
			? [...document.querySelectorAll<HTMLElement>(`[data-item="${CSS.escape(next)}"]`)].find(
					(el) => !el.closest('[inert]')
				)
			: null;
		if (tile) tile.focus({ preventScroll: true });
		else if (world) heading?.focus({ preventScroll: true });
	}

	function startOver(e: MouseEvent) {
		if (!world || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
		held = { layers, feeling: urlMood };
	}

	// Keep the scroll position of each path, and restore scroll and focus when the browser moves
	// through the path's history.
	onMount(() => {
		const onscroll = () => scrolls.set(key, scrollY);
		const onpopstate = async () => {
			// For another feeling the router navigates later, and afterNavigate takes over. Every path
			// of this feeling in history has its code in the session cache, with the path it names.
			const entry = shared.get(location.pathname.slice(1));
			if (entry?.text !== urlMood) return;
			const my = ++pops;
			const routed = nextState();
			// This listener runs before the router's, which sets page.state and scrolls. Copy the scroll
			// positions first: the scroll event that follows would record the router's position.
			const saved = new Map(scrolls);
			const own = ownMove;
			const before = shownTrail;
			const next = entry.trail;
			returned = null;
			still = false;
			const settle = async () => {
				hold = null;
				const after = trail;
				shownTrail = after;
				// Only a move of one world has a tile to morph with. See back().
				still = Math.abs(after.length - before.length) > 1;
				if (after.length < before.length) left = still ? null : before[after.length];
				pending = null;
				await tick();
				scrollTo(0, saved.get(pathKey(after)) ?? 0);
				return after;
			};
			let after: string[];
			if (own || reduced || !document.startViewTransition) {
				await routed;
				if (my !== pops) return;
				if (!own && next.length > before.length) left = null;
				after = await settle();
			} else {
				// The browser's own Back or Forward: hold the path on screen until the transition captures
				// it, and name the tile that the morph joins: the one to return to, or the one to enter.
				const y = scrollY;
				hold = before;
				if (next.length < before.length) left = before[before.length - 1];
				else if (next.length > before.length) {
					left = null;
					// Over several worlds, the tile and the work at the end differ, so nothing morphs.
					pending = next.length === before.length + 1 ? next[before.length] : null;
				}
				// The router scrolls at once. Show the old view again in the frame that captures it.
				requestAnimationFrame(() => {
					if (hold) scrollTo(0, y);
				});
				let done: string[] = [];
				const vt = document.startViewTransition({
					update: async () => {
						await routed;
						if (my === pops) done = await settle();
					},
					types: [next.length > before.length ? 'enter' : 'leave']
				});
				vt.ready.catch(() => {});
				await vt.updateCallbackDone.catch(() => {});
				if (my !== pops) return;
				after = done;
			}
			refocus(before, after);
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
	<title>{shown ? shown.world.item.title : mood ? mood.query : data.text || 'mise'}</title>
	<link rel="icon" href={favicon} />
</svelte:head>

<main
	class="stage"
	class:shown={mood !== null}
	style:--mood-font={face?.family ?? null}
	style:--mood-style={face?.style ?? null}
>
	<!-- The mood stays in the page under a world, so a return shows it as the user left it. -->
	<div class="scene" class:away={shown !== null || arriving} inert={shown !== null || arriving}>
		<header class="line">
			<MoodLine
				bind:value={text}
				bind:ref={lineRef}
				examples={EXAMPLES}
				boxed={mood !== null}
				{settled}
				{waiting}
				onsubmit={submit}
				onexample={(current) => (example = current)}
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
								class:returned={!shown && returned === anchor.id}
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
				<Wall
					items={mood.picks}
					label="picks"
					{leaving}
					active={shown ? null : focus}
					returned={shown ? null : returned}
					onopen={travel}
				/>
			{/key}
		{/if}
	</div>

	{#each layers as layer, i (layer.key)}
		{@const top = i === layers.length - 1}
		<!-- A world under the one on screen keeps its state, so a step back shows it as the user left it. -->
		<div class="layer" class:away={!top} inert={!top}>
			<WorldView
				world={layer.world}
				feeling={shown?.feeling ?? urlMood}
				path={layer.path}
				live={top}
				focus={top ? focus : null}
				still={top && still}
				returned={top ? returned : null}
				leaving={top && leaving && held !== null}
				face={faces.get(layer.world.item.id) ?? null}
				note={top ? error : null}
				bind:heading={headings[i]}
				onstep={(step) => back(trail.length - step)}
				onclose={() => back(trail.length)}
				ontravel={travel}
			/>
		</div>
	{/each}

	<!-- Place the mark after the field so users reach the field first when they press Tab. -->
	<a
		class="mark"
		href={resolve('/[[code=code]]', {})}
		aria-label="mise, start over"
		onclick={startOver}
	>
		<span class="swatch" aria-hidden="true">
			<i style:--k={0} style:background="var(--ground)"></i>
			<i style:--k={1} style:background="var(--mid-1)"></i>
			<i style:--k={2} style:background="var(--mid-2)"></i>
			<i style:--k={3} style:background="var(--mid-3)"></i>
			<i style:--k={4} style:background="var(--ink)"></i>
		</span>
		<span class="word">mise</span>
	</a>

	<!-- The brand line sits under the word of the mark, as in the brand lockup. -->
	<div class="motto">
		<Tagline {example} {waiting} away={mood !== null} />
	</div>

	<nav class="foot" aria-label="about mise">
		<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
		<a href="https://sampacker.com" target="_blank" rel="noopener">who made this</a>
		<a href={resolve('/attribution')}>attribution</a>
	</nav>
</main>

<svelte:window onkeydown={typeAnywhere} />

<!-- While a world loads, the page breathes. -->
{#if traveling || arriving}
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

	/* Keep room under the wall for the links at the foot of the page. */
	.shown .scene {
		padding-bottom: 4rem;
	}

	/* Out of view and out of the flow, but still rendered, so its animations never start again. */
	.scene.away,
	.layer.away {
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

	/* Under the word of the mark: past its padding, its five bands, and its gap. */
	.motto {
		position: absolute;
		top: calc(max(1.3vw, 10px) + 1.9rem + 0.7rem);
		left: calc(max(2.5vw, 12px) + 5 * 0.56rem + 0.7rem);
		pointer-events: none;
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

	/* The same brief ring as a tile's, after a step back over several worlds. */
	.anchor .work.returned {
		outline: 1px solid transparent;
		outline-offset: 2px;
		animation: mood-returned 1200ms var(--ease) 120ms;
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

	/* At the end of the page, under the last wall, so it never sits on an image. */
	.foot {
		position: absolute;
		right: 0;
		bottom: 0;
		display: flex;
		padding: 0 calc(max(2.5vw, 12px) - max(1vw, 8px) + env(safe-area-inset-right, 0px))
			env(safe-area-inset-bottom, 0px) 0;
	}

	.foot a {
		display: flex;
		align-items: center;
		justify-content: center;
		min-width: 44px;
		min-height: 44px;
		padding: max(1vw, 8px);
		box-sizing: border-box;
		color: var(--ink-soft);
		font-size: 0.8rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		text-decoration: none;
		transition: color 300ms var(--ease);
	}

	.foot a:hover,
	.foot a:focus-visible {
		color: var(--ink);
	}

	.foot a:focus-visible {
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
