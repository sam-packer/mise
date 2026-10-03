<script lang="ts">
	// Offer a few sample feelings under the empty line: one on a phone, as many as fit on one line on a
	// wider screen, and a button for other ones.
	import { onMount } from 'svelte';
	import { fade } from 'svelte/transition';
	import { sample } from '#lib/mood/samples.js';

	let {
		examples,
		onpick,
		onexample
	}: {
		examples: string[];
		onpick: (text: string) => void;
		/** The sample that the tagline shows: the one under the pointer or the focus, or the first. */
		onexample?: (text: string) => void;
	} = $props();

	/** The most samples on one line. */
	const MOST = 3;

	let order = $state<string[]>([]);
	let start = $state(0);
	/** How many of the candidates fit on one line. */
	let fit = $state(1);
	let near = $state<string | null>(null);
	let measure = $state<HTMLDivElement | null>(null);

	const candidates = $derived(
		order.length
			? Array.from(
					{ length: Math.min(MOST, order.length) },
					(_, i) => order[(start + i) % order.length]
				)
			: []
	);
	const shown = $derived(candidates.slice(0, fit));

	// Shuffle after mounting to avoid a mismatch between server HTML and browser state.
	onMount(() => {
		const shuffled = [...examples];
		for (let i = shuffled.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
		}
		order = shuffled;
	});

	// Count the candidates that share the first line with the refresh button in the hidden copy.
	$effect(() => {
		const el = measure;
		if (!el || !candidates.length) return;
		const phone = matchMedia('(max-width: 600px)');
		const count = () => {
			if (phone.matches) {
				fit = 1;
				return;
			}
			const [button, ...chips] = [...el.children] as HTMLElement[];
			const n = chips.filter((c) => c.offsetTop === button.offsetTop).length;
			fit = Math.max(1, n);
		};
		count();
		const observer = new ResizeObserver(count);
		observer.observe(el);
		phone.addEventListener('change', count);
		return () => {
			observer.disconnect();
			phone.removeEventListener('change', count);
		};
	});

	$effect(() => onexample?.(near ?? shown[0] ?? ''));

	// Load the rooms of the samples on screen while the page is idle, so a tap shows its room at once.
	$effect(() => {
		const texts = shown;
		const load = () => texts.forEach((text) => void sample(text));
		// Safari has no requestIdleCallback.
		if ('requestIdleCallback' in window) {
			const id = requestIdleCallback(load, { timeout: 2000 });
			return () => cancelIdleCallback(id);
		}
		const id = setTimeout(load, 500);
		return () => clearTimeout(id);
	});

	function refresh() {
		start = (start + shown.length) % order.length;
		near = null;
	}
</script>

{#if shown.length}
	<div class="examples">
		<p class="label">or start from one of these</p>
		<div class="row">
			{#key start}
				<ul class="chips" in:fade={{ duration: 400 }}>
					{#each shown as text (text)}
						<li>
							<button
								type="button"
								class="chip"
								class:single={shown.length === 1}
								onclick={() => onpick(text)}
								onpointerenter={() => (near = text)}
								onpointerleave={() => (near = null)}
								onfocus={() => (near = text)}
								onblur={() => (near = null)}>{text}</button
							>
						</li>
					{/each}
				</ul>
			{/key}
			<button type="button" class="refresh" aria-label="show other samples" onclick={refresh}>
				<svg viewBox="0 0 24 24" aria-hidden="true">
					<path d="M20 12a8 8 0 1 1-2.34-5.66" />
					<path d="M20 4v4.5h-4.5" />
				</svg>
			</button>
		</div>

		<!-- A hidden copy of every candidate on one wrapping line, to count the ones that fit. -->
		<div class="row measure" bind:this={measure} aria-hidden="true">
			<span class="refresh"></span>
			{#each candidates as text (text)}
				<span class="chip">{text}</span>
			{/each}
		</div>
	</div>
{/if}

<style>
	.examples {
		position: relative;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.6rem;
		width: 100%;
	}

	.label {
		margin: 0;
		color: var(--ink-soft);
		font-size: 0.85rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
	}

	.row {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.6rem;
		max-width: 100%;
	}

	.chips {
		display: flex;
		justify-content: center;
		gap: 0.6rem;
		min-width: 0;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.chips li {
		min-width: 0;
	}

	.chip {
		box-sizing: border-box;
		min-height: 44px;
		margin: 0;
		padding: 0.5rem 0.9rem;
		border: 1px solid color-mix(in oklab, var(--ink) 22%, transparent);
		background: none;
		color: var(--ink-soft);
		font: inherit;
		font-family: var(--serif);
		font-style: italic;
		font-size: 1rem;
		line-height: 1.35;
		white-space: nowrap;
		cursor: pointer;
		transition:
			color 300ms var(--ease),
			border-color 300ms var(--ease);
	}

	/* One sample alone may be longer than a phone is wide, so it wraps. */
	.chip.single {
		white-space: normal;
		text-wrap: balance;
	}

	.chip:hover,
	.chip:focus-visible {
		border-color: var(--ink);
		color: var(--ink);
	}

	.chip:focus-visible,
	.refresh:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: 2px;
	}

	.refresh {
		display: grid;
		flex: none;
		place-items: center;
		width: 44px;
		height: 44px;
		margin: 0;
		padding: 0;
		border: 0;
		background: none;
		color: var(--ink-soft);
		cursor: pointer;
		transition: color 300ms var(--ease);
	}

	.refresh:hover,
	.refresh:focus-visible {
		color: var(--ink);
	}

	.refresh svg {
		width: 1.15rem;
		height: 1.15rem;
		fill: none;
		stroke: currentColor;
		stroke-width: 1.5;
		stroke-linecap: round;
		stroke-linejoin: round;
		transition: transform 500ms var(--ease);
	}

	.refresh:hover svg {
		transform: rotate(60deg);
	}

	.measure {
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		flex-wrap: wrap;
		justify-content: flex-start;
		visibility: hidden;
		pointer-events: none;
	}

	@media (prefers-reduced-motion: reduce) {
		.refresh svg {
			transition: none;
		}

		.refresh:hover svg {
			transform: none;
		}
	}
</style>
