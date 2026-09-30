<script lang="ts">
	// Write the scent note one letter at a time after the wall has settled.
	let {
		text,
		leaving = false,
		delay
	}: {
		text: string;
		leaving?: boolean;
		/** When the letters start, in milliseconds. */
		delay: number;
	} = $props();
</script>

<div class="scent" class:leaving style:--delay="{delay}ms">
	<p class="label">scent</p>
	<p class="note-text" aria-label={text}>
		{#each text.split('') as ch, i (i)}
			<span style:--i={i} aria-hidden="true">{ch}</span>
		{/each}
	</p>
</div>

<style>
	.scent {
		margin: 0.75rem auto 0;
		padding: 0 max(2.5vw, 12px) 4rem;
		text-align: center;
		transition: opacity 200ms ease-out;
	}

	.scent p {
		margin: 0;
	}

	.label {
		color: var(--ink-soft);
		font-size: 0.95rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		animation: letter 700ms var(--ease) both;
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

	@keyframes letter {
		from {
			opacity: 0;
		}
		to {
			opacity: 1;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.note-text span {
			animation-delay: var(--delay);
		}
	}
</style>
