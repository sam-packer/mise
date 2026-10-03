<script lang="ts">
	// Show a missing page or a failed page in the voice of mise, with one way back to the start.
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import Chevron from '#lib/components/Chevron.svelte';

	const missing = $derived(page.status === 404);
	const line = $derived(
		missing ? 'this world doesn’t exist yet.' : 'something went wrong in this world.'
	);
	// SvelteKit gives an unexpected error the plain message "Internal Error", so no detail leaks here.
	const label = $derived([page.status, page.error?.message].filter(Boolean).join(' · '));
</script>

<svelte:head>
	<title>{missing ? 'not found' : 'something went wrong'} — mise</title>
</svelte:head>

<main class="stage">
	<div class="center">
		<p class="kicker">{label}</p>
		<h1>{line}</h1>
		<a class="home" href={resolve('/[[code=code]]', {})}>start over<Chevron /></a>
	</div>

	<!-- Place the mark after the message so users reach the way home first when they press Tab. -->
	<a class="mark" href={resolve('/[[code=code]]', {})} aria-label="mise, start over">
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

<style>
	.stage {
		position: relative;
		display: flex;
		min-height: 100dvh;
		box-sizing: border-box;
		/* Keep the message clear of the mark on a short screen. */
		padding: 5rem max(2.5vw, 12px);
	}

	.center {
		display: flex;
		flex-direction: column;
		align-items: center;
		width: min(100%, 40rem);
		margin: auto;
		text-align: center;
		animation: rise 900ms var(--ease) 100ms both;
	}

	.kicker {
		margin: 0 0 1rem;
		max-width: 100%;
		color: var(--ink-soft);
		font-size: 0.85rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		overflow-wrap: anywhere;
	}

	h1 {
		margin: 0;
		color: var(--ink);
		font-size: clamp(1.75rem, 2.6vw + 0.9rem, 3.1rem);
		font-weight: 400;
		line-height: 1.2;
		text-wrap: balance;
	}

	/* A short hairline gives a pause between the line and the way home. */
	.home::before {
		content: '';
		position: absolute;
		top: 0;
		left: 50%;
		width: 2.5rem;
		height: 1px;
		background: color-mix(in oklab, var(--ink) 18%, transparent);
		transform: translateX(-50%);
	}

	.home {
		position: relative;
		display: inline-flex;
		align-items: center;
		min-height: 44px;
		margin-top: 2rem;
		padding: 1.25rem 0.25rem 0.5rem;
		color: var(--ink-soft);
		font-size: 0.95rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		text-decoration: none;
		transition: color 300ms var(--ease);
	}

	.home:hover,
	.home:focus-visible {
		color: var(--ink);
	}

	.home:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: 2px;
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

	/* The five bands show fanned out, as in the brand lockup. A hover spreads them a little. */
	.swatch {
		display: flex;
		flex: none;
		height: 1.5rem;
	}

	.swatch i {
		width: var(--band);
		transition: transform 400ms cubic-bezier(0.16, 1, 0.3, 1);
	}

	.swatch i:first-child {
		box-shadow: inset 0 0 0 1px color-mix(in oklab, var(--ink) 25%, transparent);
	}

	.mark:hover .swatch i,
	.mark:focus-visible .swatch i {
		transform: translateX(calc((var(--k) - 2) * 2px));
	}

	@keyframes rise {
		from {
			opacity: 0;
			transform: translateY(12px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	@keyframes appear {
		from {
			opacity: 0;
		}
		to {
			opacity: 1;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.center {
			animation-name: appear;
		}

		.swatch i,
		.mark:hover .swatch i,
		.mark:focus-visible .swatch i {
			transform: none;
			transition: none;
		}
	}
</style>
