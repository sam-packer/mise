<script lang="ts">
	import { onMount } from 'svelte';
	import { fade } from 'svelte/transition';

	let {
		value = $bindable(''),
		examples,
		settled = false,
		waiting = false,
		onsubmit,
		ref = $bindable(null)
	}: {
		value?: string;
		examples: string[];
		/** The line shows the feeling on the wall: the field edge folds into a short rule. */
		settled?: boolean;
		/** The model is still loading. */
		waiting?: boolean;
		onsubmit: (text: string) => void;
		ref?: HTMLTextAreaElement | null;
	} = $props();

	const id = $props.id();
	let index = $state(0);
	let focused = $state(false);
	let coarse = $state(false);
	const example = $derived(examples[index % examples.length]);
	const empty = $derived(value === '');

	// Rotate the ghost only while nobody is at the line, so Tab never takes a sentence mid-fade.
	$effect(() => {
		if (!empty || focused) return;
		const timer = setInterval(() => (index += 1), 4000);
		return () => clearInterval(timer);
	});

	onMount(() => {
		coarse = matchMedia('(pointer: coarse)').matches;
	});

	const hint = $derived.by(() => {
		if (waiting) return 'getting the room ready';
		if (coarse) return empty && focused ? 'tap to use this' : '';
		if (empty) return focused ? 'tab to use this · enter to see it' : '';
		return settled ? '' : 'enter to see it';
	});

	function use() {
		value = example;
		ref?.focus({ preventScroll: true });
	}

	function onkeydown(e: KeyboardEvent) {
		if (e.key === 'Tab' && !e.shiftKey && empty) {
			e.preventDefault();
			use();
			return;
		}
		if (e.key !== 'Enter') return;
		e.preventDefault();
		if (e.shiftKey) return;
		if (empty) value = example;
		onsubmit(value);
	}
</script>

<div class="line" class:empty class:settled>
	<label class="sr-only" for={id}>describe a feeling</label>
	<div class="stack">
		<div class="mirror" aria-hidden="true">{value + ' '}</div>
		<textarea
			{id}
			bind:this={ref}
			bind:value
			rows="1"
			autocomplete="off"
			autocapitalize="off"
			spellcheck="false"
			enterkeyhint="go"
			placeholder={example}
			onfocus={() => (focused = true)}
			onblur={() => (focused = false)}
			{onkeydown}></textarea>
		{#if empty}
			<!-- A tap on the ghost keeps the keyboard up (no blur) and fills the line. -->
			<div
				class="ghost"
				class:tappable={coarse && focused}
				aria-hidden="true"
				onmousedown={(e) => e.preventDefault()}
				onclick={use}
			>
				{#key example}
					<span in:fade={{ duration: 700 }} out:fade={{ duration: 500 }}
						><i class="caret"></i>{example}</span
					>
				{/key}
			</div>
		{/if}
	</div>
	<div class="edge" aria-hidden="true"></div>
	{#if hint}
		{#key hint}
			<p
				class="hint"
				aria-live="polite"
				in:fade={{ duration: 400, delay: 200 }}
				out:fade={{ duration: 150 }}
			>
				{hint}
			</p>
		{/key}
	{/if}
</div>

<style>
	.line {
		position: relative;
		width: 100%;
		cursor: text;
	}

	/* Mirror, textarea, and ghost share one grid cell, so the cell is always as tall as the text it shows. */
	.stack {
		display: grid;
	}

	.mirror,
	textarea,
	.ghost,
	.ghost span {
		grid-area: 1 / 1;
	}

	.mirror,
	textarea,
	.ghost {
		margin: 0;
		padding: 0;
		font-family: var(--mood-font, var(--serif));
		font-style: var(--mood-style, italic);
		font-size: var(--line-size);
		line-height: 1.3;
		text-align: center;
		letter-spacing: -0.005em;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}

	.mirror {
		visibility: hidden;
	}

	textarea {
		display: block;
		width: 100%;
		border: 0;
		border-radius: 0;
		background: transparent;
		color: var(--ink);
		caret-color: var(--ink);
		resize: none;
		overflow: hidden;
		outline: none;
		box-shadow: none;
	}

	/* The ghost is the visible placeholder; the native one stays for assistive tech only. */
	textarea::placeholder {
		color: transparent;
	}

	/* The drawn caret stands in for the real one while the line is empty. */
	.empty textarea {
		caret-color: transparent;
	}

	textarea:focus-visible {
		outline: none;
	}

	.ghost {
		display: grid;
		pointer-events: none;
		color: color-mix(in oklab, var(--ink) 32%, transparent);
		user-select: none;
	}

	.ghost.tappable {
		pointer-events: auto;
		cursor: pointer;
	}

	.caret {
		display: inline-block;
		width: 1.5px;
		height: 1.05em;
		margin-right: 0.14em;
		vertical-align: -0.17em;
		background: var(--ink);
		opacity: 0.55;
		animation: blink 1.1s steps(1) infinite;
		transition: opacity 300ms var(--ease);
	}

	.line:focus-within .caret {
		opacity: 1;
	}

	.edge {
		height: 1px;
		margin-top: 0.45em;
		background: var(--ink);
		opacity: 0.24;
		transform-origin: center;
		transition:
			transform 900ms var(--ease),
			opacity 500ms var(--ease);
	}

	.line:hover .edge {
		opacity: 0.36;
	}

	.line:focus-within .edge {
		opacity: 0.6;
		transform: none;
	}

	.settled .edge {
		transform: scaleX(0.08);
		opacity: 0.4;
	}

	.settled:not(:focus-within):hover .edge {
		transform: scaleX(0.16);
	}

	/* Under the edge, out of the flow: the hint never moves the page. */
	.hint {
		position: absolute;
		top: calc(100% + 0.7rem);
		left: 0;
		right: 0;
		margin: 0;
		color: color-mix(in oklab, var(--ink) 45%, transparent);
		font-size: 0.85rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		text-align: center;
		pointer-events: none;
	}

	@keyframes blink {
		50% {
			opacity: 0;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.caret {
			animation: none;
		}

		.edge {
			transition: opacity 500ms var(--ease);
		}
	}

	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		margin: -1px;
		padding: 0;
		overflow: hidden;
		clip: rect(0 0 0 0);
		white-space: nowrap;
		border: 0;
	}
</style>
