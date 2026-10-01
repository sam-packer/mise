<script lang="ts">
	// Collect feeling text, and offer sample feelings under the line before the page shows a mood.
	import { onMount } from 'svelte';
	import { fade } from 'svelte/transition';
	import { MAX_FEELING } from '$lib/code';
	import Examples from './Examples.svelte';

	let {
		value = $bindable(''),
		examples,
		boxed = false,
		settled = false,
		waiting = false,
		onsubmit,
		onexample,
		ref = $bindable(null)
	}: {
		value?: string;
		examples: string[];
		/** Outline the field once a mood is on the wall, to show that the feeling can change. */
		boxed?: boolean;
		/** The line holds the feeling on the wall, unedited. */
		settled?: boolean;
		/** The model is still loading. */
		waiting?: boolean;
		onsubmit: (text: string) => void;
		/** The sample feeling that the tagline shows changed. */
		onexample?: (text: string) => void;
		ref?: HTMLTextAreaElement | null;
	} = $props();

	/** The empty line says what to do and what comes back. */
	const PLACEHOLDER = 'describe a moment. get its art, songs, and stories.';

	const id = $props.id();
	let focused = $state(false);
	let coarse = $state(false);
	const empty = $derived(value === '');

	onMount(() => {
		coarse = matchMedia('(pointer: coarse)').matches;
	});

	/** A short answer to a command, shown in place of the hint. */
	let notice = $state('');

	$effect(() => {
		if (!notice) return;
		const timer = setTimeout(() => (notice = ''), 3000);
		return () => clearTimeout(timer);
	});

	const hint = $derived.by(() => {
		if (notice) return notice;
		if (waiting) return 'getting the room ready';
		if (coarse || empty) return '';
		if (!focused) return settled ? '' : 'enter to see it';
		return settled ? 'type to change it · esc to clear' : 'enter to see it';
	});

	function pick(text: string) {
		value = text;
		onsubmit(text);
	}

	function clear() {
		value = '';
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
		// Keep Tab in the field while an edit is not submitted. Shift+Tab always leaves.
		if (e.key === 'Tab' && !e.shiftKey && !empty && !settled) {
			e.preventDefault();
			return;
		}
		if (e.key === 'Escape' && settled) {
			e.preventDefault();
			clear();
			return;
		}
		if (e.key !== 'Enter') return;
		e.preventDefault();
		if (e.shiftKey) return;
		// Opt this browser out of Plausible, or back in, with the exact command. It is not a feeling.
		if (
			value === 'localStorage.plausible_ignore=true' ||
			value === 'localStorage.plausible_ignore=false'
		) {
			const off = value.endsWith('true');
			try {
				if (off) localStorage.setItem('plausible_ignore', 'true');
				else localStorage.removeItem('plausible_ignore');
				notice = off ? 'analytics off in this browser' : 'analytics on in this browser';
			} catch {
				notice = 'this browser blocks storage';
			}
			value = '';
			return;
		}
		if (!empty) onsubmit(value);
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
			placeholder={PLACEHOLDER}
			{onmousedown}
			{onfocus}
			{onkeydown}
			oninput={() => (notice = '')}></textarea>
		{#if empty}
			<div class="ghost" aria-hidden="true"><span><i class="caret"></i>{PLACEHOLDER}</span></div>
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
	{:else if empty && !boxed}
		<div class="samples" transition:fade={{ duration: 250 }}>
			<Examples {examples} onpick={pick} {onexample} />
		</div>
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
		color: var(--ink-soft);
		user-select: none;
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
		color: var(--ink-soft);
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
		color: var(--ink-soft);
		font-size: 0.85rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		text-align: center;
		pointer-events: none;
	}

	/* The samples take the hint's place under the field, wider than the field and out of the flow. */
	.samples {
		position: absolute;
		top: calc(100% + 1.6rem);
		left: 50%;
		width: min(100vw - 2 * max(2.5vw, 12px), 64rem);
		transform: translateX(-50%);
		cursor: auto;
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
