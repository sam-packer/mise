<script lang="ts">
	// Collect feeling text and offer rotating examples before the page requests a mood.
	import { onMount } from 'svelte';
	import { fade } from 'svelte/transition';
	import { MAX_FEELING } from '$lib/code';

	let {
		value = $bindable(''),
		examples,
		boxed = false,
		settled = false,
		waiting = false,
		onsubmit,
		ref = $bindable(null)
	}: {
		value?: string;
		examples: string[];
		/** Show an outline after a mood appears to make the field easier to find. */
		boxed?: boolean;
		/** The line holds the feeling on the wall, unedited. */
		settled?: boolean;
		/** The model is still loading. */
		waiting?: boolean;
		onsubmit: (text: string) => void;
		ref?: HTMLTextAreaElement | null;
	} = $props();

	const id = $props.id();
	let order = $state<string[]>([]);
	let index = $state(0);
	let focused = $state(false);
	let coarse = $state(false);
	const example = $derived(order.length ? order[index % order.length] : '');
	const empty = $derived(value === '');

	// Shuffle after mounting to avoid a mismatch between server HTML and browser state.
	onMount(() => {
		coarse = matchMedia('(pointer: coarse)').matches;
		const shuffled = [...examples];
		for (let i = shuffled.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
		}
		order = shuffled;
	});

	// Pause examples while the field has focus, so users can press Tab without selecting a fading sentence.
	$effect(() => {
		if (!empty || focused || !order.length) return;
		const timer = setInterval(() => (index += 1), 4000);
		return () => clearInterval(timer);
	});

	const hint = $derived.by(() => {
		if (waiting) return 'getting the room ready';
		if (coarse) return empty && focused ? 'tap to use this' : '';
		if (!focused) return empty || settled ? '' : 'enter to see it';
		if (empty) return 'tab to use this · enter to see it';
		return settled ? 'type to change it · esc to clear' : 'enter to see it';
	});

	function use() {
		value = example;
		ref?.focus({ preventScroll: true });
	}

	function clear() {
		value = '';
		index += 1;
		ref?.focus({ preventScroll: true });
	}

	// Select the whole feeling so typing replaces it.
	// Prevent the first click from moving the caret and clearing the selection after focus.
	function onmousedown(e: MouseEvent) {
		if (focused || !settled || !ref) return;
		e.preventDefault();
		ref.focus({ preventScroll: true });
		ref.select();
	}

	function onfocus() {
		if (settled) ref?.select();
	}

	function onkeydown(e: KeyboardEvent) {
		// Keep focus for completion or editing. Press Shift+Tab to leave at any time.
		if (e.key === 'Tab' && !e.shiftKey) {
			if (empty && example) {
				e.preventDefault();
				use();
				return;
			}
			if (!empty && !settled) {
				e.preventDefault();
				return;
			}
		}
		if (e.key === 'Escape' && settled) {
			e.preventDefault();
			clear();
			return;
		}
		if (e.key !== 'Enter') return;
		e.preventDefault();
		if (e.shiftKey) return;
		if (empty) value = example;
		onsubmit(value);
	}
</script>

<div
	class="line"
	class:empty
	class:boxed
	onfocusin={() => (focused = true)}
	onfocusout={() => (focused = false)}
>
	<label class="sr-only" for={id}>describe a feeling</label>
	<div class="box" aria-hidden="true"></div>
	<div class="stack">
		<div class="mirror" aria-hidden="true">{value + ' '}</div>
		<textarea
			{id}
			bind:this={ref}
			bind:value
			rows="1"
			maxlength={MAX_FEELING}
			autocomplete="off"
			autocapitalize="off"
			spellcheck="false"
			enterkeyhint="go"
			placeholder={example}
			{onmousedown}
			{onfocus}
			{onkeydown}></textarea>
		{#if empty && example}
			<!-- Prevent blur so users can select an example without closing the touch keyboard. -->
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

	{#if focused && !empty}
		<button
			type="button"
			class="tag"
			onmousedown={(e) => e.preventDefault()}
			onclick={clear}
			transition:fade={{ duration: 250 }}>clear</button
		>
	{:else if boxed && settled && !focused}
		<label class="tag" for={id} aria-hidden="true" transition:fade={{ duration: 250 }}>edit</label>
	{/if}

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

	textarea::selection {
		background: color-mix(in oklab, var(--ink) 22%, transparent);
		color: var(--ink);
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
		transition: opacity 500ms var(--ease);
	}

	.line:hover .edge {
		opacity: 0.36;
	}

	.line:focus-within .edge {
		opacity: 0.6;
	}

	/* After a reveal the underline gives way to a faint outline: a field you can see without hovering. */
	.box {
		position: absolute;
		inset: -0.55rem -0.6rem 0;
		border: 1px solid transparent;
		border-radius: 0.6rem;
		pointer-events: none;
		transition: border-color 900ms var(--ease);
	}

	.boxed .box {
		border-color: color-mix(in oklab, var(--ink) 20%, transparent);
	}

	.boxed:hover .box {
		border-color: color-mix(in oklab, var(--ink) 38%, transparent);
		transition-duration: 300ms;
	}

	.boxed:focus-within .box {
		border-color: color-mix(in oklab, var(--ink) 62%, transparent);
		transition-duration: 300ms;
	}

	.line.boxed .edge {
		opacity: 0;
	}

	/* "edit" at rest, "clear" while editing: one quiet corner label on the field's top edge. */
	.tag {
		position: absolute;
		right: -0.6rem;
		bottom: calc(100% + 0.55rem);
		display: flex;
		align-items: flex-end;
		min-width: 44px;
		min-height: 44px;
		justify-content: flex-end;
		box-sizing: border-box;
		margin: 0;
		padding: 0 0.15rem 0.3rem 0.6rem;
		border: 0;
		background: none;
		color: color-mix(in oklab, var(--ink) 50%, transparent);
		font: inherit;
		font-size: 0.85rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		cursor: pointer;
		transition: color 300ms var(--ease);
	}

	.tag:hover,
	.tag:focus-visible,
	.boxed:hover .tag {
		color: var(--ink);
	}

	.tag:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: -2px;
	}

	/* Under the field, out of the flow: the hint never moves the page. */
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
