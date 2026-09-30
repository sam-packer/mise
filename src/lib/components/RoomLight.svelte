<script lang="ts">
	// Layer the selected light and grain behind the mood wall.
	import { fade } from 'svelte/transition';
	import type { Light } from '$lib/mood/types';

	let { light }: { light: Light | null } = $props();
</script>

<div class="room" aria-hidden="true">
	{#key light}
		{#if light}
			<div class="light {light}" transition:fade={{ duration: 900 }}></div>
		{/if}
	{/key}
	<div class="grain"></div>
</div>

<style>
	.room,
	.light,
	.grain {
		position: fixed;
		inset: 0;
		pointer-events: none;
	}

	.room {
		z-index: 0;
	}

	.grain {
		opacity: 0.045;
		mix-blend-mode: overlay;
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 0.5 0 0 0 0 0.5 0 0 0 0 0.5 0 0 0 1 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
		background-size: 240px 240px;
	}

	.dawn {
		background:
			linear-gradient(to top, oklch(0.86 0.09 40 / 0.3), transparent 55%),
			linear-gradient(to bottom, oklch(0.76 0.06 250 / 0.2), transparent 50%);
	}

	.golden-hour {
		background: radial-gradient(90% 75% at 12% 100%, oklch(0.82 0.15 70 / 0.42), transparent 72%);
	}

	.overcast {
		background: linear-gradient(oklch(0.82 0.02 240 / 0.24), oklch(0.72 0.025 240 / 0.24));
	}

	.neon {
		background:
			radial-gradient(45% 95% at 0% 50%, oklch(0.66 0.26 340 / 0.36), transparent 66%),
			radial-gradient(45% 95% at 100% 50%, oklch(0.82 0.14 200 / 0.36), transparent 66%);
	}

	.candle {
		background: radial-gradient(40% 38% at 50% 88%, oklch(0.82 0.15 62 / 0.45), transparent 100%);
		animation: flicker 3.6s ease-in-out infinite;
	}

	.moonlight {
		background: radial-gradient(75% 60% at 50% -12%, oklch(0.86 0.045 240 / 0.32), transparent 72%);
	}

	.desk-lamp {
		background:
			radial-gradient(48% 52% at 93% 3%, oklch(0.9 0.11 76 / 0.5), transparent 72%),
			radial-gradient(130% 130% at 93% 3%, transparent 42%, oklch(0 0 0 / 0.14));
	}

	.fluorescent {
		background:
			repeating-linear-gradient(to bottom, oklch(1 0 0 / 0.025) 0 2px, transparent 2px 5px),
			linear-gradient(
				to bottom,
				oklch(0.96 0.035 150 / 0.4),
				oklch(0.9 0.02 150 / 0.14) 28%,
				transparent 62%
			);
	}

	@keyframes flicker {
		0%,
		100% {
			opacity: 1;
		}
		37% {
			opacity: 0.82;
		}
		61% {
			opacity: 0.94;
		}
		78% {
			opacity: 0.86;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.candle {
			animation: none;
		}
	}
</style>
