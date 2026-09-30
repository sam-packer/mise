// Tween the palette tokens in OKLCH and write them as CSS variables on <html>.

import type { OKLab } from '$lib/mood/types';
import { cssLch, labToLch, toGamut, type OKLCH, type Tokens } from './oklab';

const KEYS = [
	'--ground',
	'--ink',
	'--ink-soft',
	'--mid-1',
	'--mid-2',
	'--mid-3',
	'--paper-ink'
] as const;

function flat(t: Tokens): OKLab[] {
	return [t.ground, t.ink, t.inkSoft, ...t.mids, t.paperInk];
}

function mixLch(a: OKLCH, b: OKLCH, t: number): OKLCH {
	const L = a[0] + (b[0] - a[0]) * t;
	const C = a[1] + (b[1] - a[1]) * t;
	// A near-grey has no hue of its own: borrow the other end's hue.
	let ha = a[1] < 0.005 ? b[2] : a[2];
	const hb = b[1] < 0.005 ? a[2] : b[2];
	let d = hb - ha;
	if (d > 180) ha += 360;
	else if (d < -180) ha -= 360;
	d = hb - ha;
	return [L, C, (ha + d * t + 360) % 360];
}

const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

let frame = 0;
let current: Tokens | null = null;

export function applyTokens(t: Tokens) {
	cancelAnimationFrame(frame);
	current = t;
	const root = document.documentElement.style;
	flat(t).forEach((c, i) => root.setProperty(KEYS[i], cssLch(toGamut(labToLch(c)))));
}

/** Animate from the last target palette. The duration is in milliseconds. */
export function tweenTokens(to: Tokens, duration = 900): Promise<void> {
	if (!current || duration === 0) {
		applyTokens(to);
		return Promise.resolve();
	}
	cancelAnimationFrame(frame);
	const from = flat(current).map(labToLch);
	const target = flat(to).map(labToLch);
	current = to;
	const root = document.documentElement.style;
	const start = performance.now();
	return new Promise((resolve) => {
		const step = (now: number) => {
			const t = easeOut(Math.min(1, (now - start) / duration));
			from.forEach((f, i) => root.setProperty(KEYS[i], cssLch(toGamut(mixLch(f, target[i], t)))));
			if (t < 1) frame = requestAnimationFrame(step);
			else resolve();
		};
		frame = requestAnimationFrame(step);
	});
}
