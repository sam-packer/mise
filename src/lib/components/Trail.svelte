<script lang="ts">
	// Show the path from the feeling to the world on screen. A long path folds its middle steps.
	import type { Item } from '$lib/mood/types';

	let {
		feeling,
		path,
		current,
		onstep
	}: {
		feeling: string;
		/** The works between the feeling and the world on screen, in order. */
		path: Item[];
		/** The title of the world on screen. */
		current: string;
		/** Go back to a step of the path. Step 0 is the feeling. */
		onstep: (step: number) => void;
	} = $props();

	/** The steps before the world on screen that stay in view when the path folds: fewer on a phone. */
	const TAIL = 2;
	const NARROW_TAIL = 1;

	const id = $props.id();
	let narrow = $state(false);
	const keep = $derived(narrow ? NARROW_TAIL : TAIL);
	const folded = $derived(path.length > keep + 1);
	const tail = $derived(folded ? path.slice(-keep) : path);
	const hidden = $derived(path.length - tail.length);

	let open = $state(false);
	let nav = $state<HTMLElement | null>(null);
	let more = $state<HTMLButtonElement | null>(null);

	// Show the newest step when the path is still wider than the screen.
	const toEnd = (el: HTMLElement) => {
		el.scrollLeft = el.scrollWidth;
	};

	// Follow the width of the screen, so a phone keeps fewer steps in view.
	const watch = () => {
		const mq = matchMedia('(max-width: 600px)');
		const update = () => (narrow = mq.matches);
		update();
		mq.addEventListener('change', update);
		return () => mq.removeEventListener('change', update);
	};

	function go(step: number) {
		open = false;
		onstep(step);
	}

	// Escape closes the list first. This listener runs before the page's, which goes back a step,
	// and stops the key there.
	function onkeydown(e: KeyboardEvent) {
		if (e.key !== 'Escape' || !open) return;
		e.preventDefault();
		e.stopPropagation();
		open = false;
		more?.focus();
	}

	function onfocusout(e: FocusEvent) {
		if (!nav?.contains(e.relatedTarget as Node | null)) open = false;
	}
</script>

<svelte:window
	onkeydowncapture={onkeydown}
	onclick={(e) => {
		if (open && !nav?.contains(e.target as Node)) open = false;
	}}
/>

<nav class="trail" aria-label="your path" bind:this={nav} {onfocusout} {@attach watch}>
	<ol class="pill" {@attach toEnd}>
		<li class="first">
			<button
				type="button"
				class="step"
				title={feeling}
				aria-label="back to your feeling, {feeling}"
				onclick={() => go(0)}>“{feeling}”</button
			>
		</li>
		{#if folded}
			<li>
				<span class="arrow" aria-hidden="true">→</span>
				<button
					type="button"
					class="step more"
					aria-expanded={open}
					aria-controls="{id}-all"
					aria-label="{hidden} more steps, show the whole path"
					bind:this={more}
					onclick={() => (open = !open)}><span class="chip">+{hidden}</span></button
				>
			</li>
		{/if}
		{#each tail as step, i (step.id)}
			<li>
				<span class="arrow" aria-hidden="true">→</span>
				<button type="button" class="step" title={step.title} onclick={() => go(hidden + i + 1)}
					>{step.title}</button
				>
			</li>
		{/each}
		<li aria-current="step">
			<span class="arrow" aria-hidden="true">→</span>
			<span class="step here" title={current}>{current}</span>
		</li>
	</ol>

	{#if open}
		<ol class="all" id="{id}-all" aria-label="the whole path">
			<li>
				<button type="button" class="row" onclick={() => go(0)}>
					<span class="n" aria-hidden="true">1</span><span class="name">“{feeling}”</span>
				</button>
			</li>
			{#each path as step, i (step.id)}
				<li>
					<button type="button" class="row" onclick={() => go(i + 1)}>
						<span class="n" aria-hidden="true">{i + 2}</span><span class="name">{step.title}</span>
					</button>
				</li>
			{/each}
			<li aria-current="step">
				<span class="row here">
					<span class="n" aria-hidden="true">{path.length + 2}</span><span class="name"
						>{current}</span
					>
				</span>
			</li>
		</ol>
	{/if}
</nav>

<style>
	.trail {
		position: relative;
		display: flex;
		justify-content: center;
		min-width: 0;
		max-width: 100%;
	}

	/*
	 * The one rounded shape on the page: a pill of blurred ground, so the path reads as one object
	 * that floats over the room. It sits on the ground and the light, never on an image.
	 */
	.pill {
		display: flex;
		align-items: center;
		min-width: 0;
		max-width: 100%;
		margin: 0;
		padding: 0 0.55rem;
		box-sizing: border-box;
		border: 1px solid color-mix(in oklab, var(--ink) 14%, transparent);
		border-radius: 999px;
		background: color-mix(in oklab, var(--ground) 72%, transparent);
		backdrop-filter: blur(18px) saturate(1.2);
		-webkit-backdrop-filter: blur(18px) saturate(1.2);
		overflow-x: auto;
		scrollbar-width: none;
		list-style: none;
		white-space: nowrap;
	}

	.pill::-webkit-scrollbar {
		display: none;
	}

	.pill li {
		display: flex;
		flex: none;
		align-items: center;
		min-width: 0;
	}

	/* The feeling gives way first, so it never pushes the other steps out. */
	.pill li.first {
		flex: 0 1 auto;
		min-width: 5.5em;
		max-width: 16em;
	}

	.arrow {
		color: var(--ink-soft);
		font-style: normal;
	}

	.step {
		display: block;
		max-width: 11em;
		min-height: 44px;
		padding: 0 0.5rem;
		border: 0;
		background: none;
		color: var(--ink-soft);
		font: inherit;
		font-size: 0.92rem;
		line-height: 44px;
		overflow: hidden;
		text-overflow: ellipsis;
		cursor: pointer;
		transition: color 300ms var(--ease);
	}

	.first .step {
		max-width: 100%;
	}

	/* An underline marks a step that goes back. */
	button.step:not(.more) {
		text-decoration: underline 1px color-mix(in oklab, var(--ink) 30%, transparent);
		text-underline-offset: 0.25em;
	}

	button.step:hover,
	button.step:focus-visible,
	.more[aria-expanded='true'] {
		color: var(--ink);
	}

	button.step:focus-visible {
		border-radius: 999px;
		outline: 1px solid var(--ink);
		outline-offset: -7px;
	}

	.more {
		font-style: normal;
		font-variant-numeric: tabular-nums;
		letter-spacing: 0.02em;
	}

	/* A chip, so the folded steps read as something to open. */
	.chip {
		padding: 0.15em 0.55em;
		border: 1px solid color-mix(in oklab, var(--ink) 30%, transparent);
		border-radius: 999px;
		transition: border-color 300ms var(--ease);
	}

	.more:hover .chip,
	.more:focus-visible .chip,
	.more[aria-expanded='true'] .chip {
		border-color: var(--ink);
	}

	.here {
		color: var(--ink);
		cursor: default;
	}

	/* On the solid ground, because the list opens over the work. */
	.all {
		position: absolute;
		top: calc(100% + 0.5rem);
		left: 50%;
		z-index: 6;
		width: min(24rem, calc(100vw - 2 * max(2.5vw, 12px)));
		max-height: min(60vh, 28rem);
		margin: 0;
		padding: 0.4rem;
		box-sizing: border-box;
		overflow-y: auto;
		border: 1px solid color-mix(in oklab, var(--ink) 14%, transparent);
		border-radius: 1.1rem;
		background: var(--ground);
		box-shadow: 0 24px 60px -20px oklch(0 0 0 / 0.4);
		list-style: none;
		transform: translateX(-50%);
		animation: drop 220ms var(--ease) both;
	}

	.row {
		display: flex;
		align-items: baseline;
		gap: 0.75rem;
		width: 100%;
		min-height: 44px;
		padding: 0.6rem 0.75rem;
		box-sizing: border-box;
		border: 0;
		border-radius: 0.7rem;
		background: none;
		color: var(--ink-soft);
		font: inherit;
		font-size: 0.95rem;
		line-height: 1.35;
		text-align: left;
		cursor: pointer;
		transition: color 200ms var(--ease);
	}

	button.row:hover,
	button.row:focus-visible {
		color: var(--ink);
	}

	button.row:hover .name {
		text-decoration: underline 1px;
		text-underline-offset: 0.25em;
	}

	button.row:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: -1px;
	}

	.row.here {
		color: var(--ink);
		cursor: default;
	}

	.n {
		flex: none;
		min-width: 1.4em;
		font-size: 0.8rem;
		font-style: normal;
		font-variant-numeric: tabular-nums;
		text-align: right;
	}

	.name {
		min-width: 0;
		overflow-wrap: anywhere;
	}

	@media (max-width: 720px) {
		.trail {
			justify-content: flex-start;
		}

		.step {
			max-width: 6.5em;
		}

		.pill {
			padding: 0 0.3rem;
		}

		.pill li.first {
			min-width: 3.5em;
			max-width: 11em;
		}

		.all {
			left: 0;
			transform: none;
		}
	}

	@keyframes drop {
		from {
			opacity: 0;
			translate: 0 -6px;
		}
		to {
			opacity: 1;
			translate: 0 0;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.all {
			animation: none;
		}
	}
</style>
