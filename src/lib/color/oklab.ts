// Convert model palettes to display colors, readable text colors, and the tab icon.

import type { OKLab, Palette } from '$lib/mood/types';

export type OKLCH = [L: number, C: number, H: number];
type RGB = [r: number, g: number, b: number];

export function labToLch([L, a, b]: OKLab): OKLCH {
	const C = Math.hypot(a, b);
	let H = (Math.atan2(b, a) * 180) / Math.PI;
	if (H < 0) H += 360;
	return [L, C, C < 1e-4 ? 0 : H];
}

function lchToLab([L, C, H]: OKLCH): OKLab {
	const h = (H * Math.PI) / 180;
	return [L, C * Math.cos(h), C * Math.sin(h)];
}

// OKLab → linear sRGB (Björn Ottosson's reference matrices).
function labToLinear([L, a, b]: OKLab): RGB {
	const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
	const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
	const s_ = L - 0.0894841775 * a - 1.291485548 * b;
	const l = l_ * l_ * l_;
	const m = m_ * m_ * m_;
	const s = s_ * s_ * s_;
	return [
		4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
		-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
		-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s
	];
}

function linearToLab([r, g, b]: RGB): OKLab {
	const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
	const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
	const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
	return [
		0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
		1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
		0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s
	];
}

function inGamut(rgb: RGB, eps = 1e-4): boolean {
	return rgb.every((c) => c >= -eps && c <= 1 + eps);
}

// Reduce chroma until the color fits in sRGB. Lightness and hue stay fixed.
export function toGamut(lch: OKLCH): OKLCH {
	const [L, C, H] = lch;
	if (L <= 0) return [0, 0, H];
	if (L >= 1) return [1, 0, H];
	if (inGamut(labToLinear(lchToLab(lch)))) return lch;
	let lo = 0;
	let hi = C;
	for (let i = 0; i < 20; i++) {
		const mid = (lo + hi) / 2;
		if (inGamut(labToLinear(lchToLab([L, mid, H])))) lo = mid;
		else hi = mid;
	}
	return [L, lo, H];
}

function luminance(lab: OKLab): number {
	const [r, g, b] = labToLinear(lchToLab(toGamut(labToLch(lab)))).map((c) =>
		Math.min(1, Math.max(0, c))
	);
	return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: OKLab, b: OKLab): number {
	const la = luminance(a) + 0.05;
	const lb = luminance(b) + 0.05;
	return la > lb ? la / lb : lb / la;
}

// Move the lightness of `ink` away from `ground` until contrast ≥ target.
function clampContrast(ink: OKLab, ground: OKLab, target = 4.5): OKLab {
	if (contrast(ink, ground) >= target) return ink;
	const [, C, H] = labToLch(ink);
	const gL = labToLch(ground)[0];
	const light: OKLab = lchToLab(toGamut([1, C, H]));
	const dark: OKLab = lchToLab(toGamut([0, C, H]));
	const goLight =
		contrast(light, ground) >= target ? gL < 0.5 : contrast(light, ground) > contrast(dark, ground);
	// Binary search the smallest move that reaches the target.
	let lo = labToLch(ink)[0];
	let hi = goLight ? 1 : 0;
	for (let i = 0; i < 24; i++) {
		const mid = (lo + hi) / 2;
		const c = lchToLab(toGamut([mid, C, H]));
		if (contrast(c, ground) >= target) hi = mid;
		else lo = mid;
	}
	return lchToLab(toGamut([hi, C, H]));
}

export function cssLch([L, C, H]: OKLCH): string {
	return `oklch(${L.toFixed(4)} ${C.toFixed(4)} ${H.toFixed(2)})`;
}

function hex(lab: OKLab): string {
	const rgb = labToLinear(lchToLab(toGamut(labToLch(lab)))).map((c) => {
		const v = Math.min(1, Math.max(0, c));
		const s = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
		return Math.round(s * 255)
			.toString(16)
			.padStart(2, '0');
	});
	return `#${rgb.join('')}`;
}

export type Tokens = {
	ground: OKLab;
	ink: OKLab;
	mids: [OKLab, OKLab, OKLab];
	/** Ink for text set on `mids[0]` paper. */
	paperInk: OKLab;
};

const NEUTRAL_LIGHT: OKLab = [1, 0, 0];
const NEUTRAL_DARK: OKLab = linearToLab([0.0027, 0.0027, 0.0036]); // #09090b

export function neutralTokens(dark: boolean): Tokens {
	const ground = dark ? NEUTRAL_DARK : NEUTRAL_LIGHT;
	const ink: OKLab = dark ? [0.86, 0, 0] : [0.32, 0, 0];
	const mid: OKLab = dark ? [0.25, 0, 0] : [0.94, 0, 0];
	return { ground, ink, mids: [mid, mid, mid], paperInk: ink };
}

/** Use the dominant color as the background and adjust text colors to meet the contrast target. */
export function paletteToTokens(palette: Palette): Tokens {
	const ground = palette[0];
	const rest = palette.slice(1) as OKLab[];
	let best = 0;
	for (let i = 1; i < rest.length; i++) {
		if (contrast(rest[i], ground) > contrast(rest[best], ground)) best = i;
	}
	const ink = clampContrast(rest[best], ground);
	const mids = rest.filter((_, i) => i !== best) as [OKLab, OKLab, OKLab];
	const paper = mids[0];
	const candidate = contrast(ink, paper) >= contrast(ground, paper) ? ink : ground;
	const paperInk = clampContrast(candidate, paper);
	return { ground, ink, mids, paperInk };
}

// A five-band swatch as an SVG data URL, for the tab icon.
export function paletteFavicon(palette: readonly OKLab[]): string {
	const bands = palette
		.map((c, i) => `<rect x="${i * 12.8}" y="0" width="12.8" height="64" fill="${hex(c)}"/>`)
		.join('');
	const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><clipPath id="r"><rect width="64" height="64" rx="12"/></clipPath><g clip-path="url(#r)">${bands}</g></svg>`;
	return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}
