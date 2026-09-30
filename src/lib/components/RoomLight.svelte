<script lang="ts">
	// Layer the selected light and grain behind the mood wall.
	import { fade } from 'svelte/transition';
	import { cssLight, LIGHTS } from '$lib/color/oklab';
	import type { Light } from '$lib/mood/types';

	let {
		light,
		strength = 1
	}: {
		light: Light | null;
		/** From lightStrength: the share of the light that the text colors can hold. */
		strength?: number;
	} = $props();
</script>

<div class="room" aria-hidden="true">
	{#key light}
		{#if light}
			<div class="glow" style:opacity={strength} transition:fade={{ duration: 900 }}>
				<div
					class="light {light}"
					style:--l0={cssLight(LIGHTS[light][0])}
					style:--l1={LIGHTS[light][1] ? cssLight(LIGHTS[light][1]) : null}
					style:--l2={LIGHTS[light][2] ? cssLight(LIGHTS[light][2]) : null}
				></div>
			</div>
		{/if}
	{/key}
	<div class="grain"></div>
</div>

<style>
	.room,
	.glow,
	.light,
	.grain {
		position: fixed;
		inset: 0;
		pointer-events: none;
	}

	.room {
		z-index: 0;
	}

	.glow {
		transition: opacity 900ms var(--ease);
	}

	.grain {
		opacity: 0.045;
		mix-blend-mode: overlay;
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 0.5 0 0 0 0 0.5 0 0 0 0 0.5 0 0 0 1 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
		background-size: 240px 240px;
	}

	.dawn {
		background:
			linear-gradient(to top, var(--l0), transparent 55%),
			linear-gradient(to bottom, var(--l1), transparent 50%);
	}

	.golden-hour {
		background: radial-gradient(90% 75% at 12% 100%, var(--l0), transparent 72%);
	}

	.overcast {
		background: linear-gradient(var(--l0), var(--l1));
	}

	.neon {
		background:
			radial-gradient(45% 95% at 0% 50%, var(--l0), transparent 66%),
			radial-gradient(45% 95% at 100% 50%, var(--l1), transparent 66%);
	}

	.candle {
		background: radial-gradient(40% 38% at 50% 88%, var(--l0), transparent 100%);
		animation: flicker 3.6s ease-in-out infinite;
	}

	.moonlight {
		background: radial-gradient(75% 60% at 50% -12%, var(--l0), transparent 72%);
	}

	.desk-lamp {
		background:
			radial-gradient(48% 52% at 93% 3%, var(--l0), transparent 72%),
			radial-gradient(130% 130% at 93% 3%, transparent 42%, var(--l1));
	}

	.fluorescent {
		background:
			repeating-linear-gradient(to bottom, var(--l0) 0 2px, transparent 2px 5px),
			linear-gradient(to bottom, var(--l1), var(--l2) 28%, transparent 62%);
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
