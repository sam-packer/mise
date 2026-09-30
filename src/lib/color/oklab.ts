// Convert model palettes to display colors, readable text colors, and the tab icon.

import type { Light, OKLab, Palette } from '$lib/mood/types';

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

/** The lowest contrast of `ink` against any of the grounds. */
function worst(ink: OKLab, grounds: OKLab[]): number {
	return Math.min(...grounds.map((g) => contrast(ink, g)));
}

// Move the lightness of `ink` away from the grounds until its contrast with each is ≥ target.
// The first ground decides the side when both sides reach the target.
function clampContrast(ink: OKLab, grounds: OKLab[], target = 4.5): OKLab {
	if (worst(ink, grounds) >= target) return ink;
	const [, C, H] = labToLch(ink);
	const gL = labToLch(grounds[0])[0];
	const light = worst(lchToLab(toGamut([1, C, H])), grounds);
	const dark = worst(lchToLab(toGamut([0, C, H])), grounds);
	// Go to the side that can reach the target. On a mid grey only one side can.
	const goLight =
		light >= target && dark >= target
			? gL < 0.5
			: light >= target || (dark < target && light > dark);
	// Binary search the smallest move that reaches the target.
	let lo = labToLch(ink)[0];
	let hi = goLight ? 1 : 0;
	for (let i = 0; i < 24; i++) {
		const mid = (lo + hi) / 2;
		const c = lchToLab(toGamut([mid, C, H]));
		if (worst(c, grounds) >= target) hi = mid;
		else lo = mid;
	}
	return lchToLab(toGamut([hi, C, H]));
}

function toSrgb(lab: OKLab): RGB {
	return labToLinear(lchToLab(toGamut(labToLch(lab)))).map((c) => {
		const v = Math.min(1, Math.max(0, c));
		return v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
	}) as RGB;
}

function fromSrgb(rgb: RGB): OKLab {
	return linearToLab(
		rgb.map((s) => (s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4))) as RGB
	);
}

/** One layer of a room light: its color and its alpha where it is strongest. */
export type LightLayer = [L: number, C: number, H: number, alpha: number];

/**
 * The colors of each room light, in the order RoomLight.svelte uses them as --l0, --l1, --l2.
 * The token math reads the same colors, so text keeps its contrast where a light is strongest.
 */
export const LIGHTS: Record<Light, LightLayer[]> = {
	dawn: [
		[0.86, 0.09, 40, 0.3],
		[0.76, 0.06, 250, 0.2]
	],
	'golden-hour': [[0.82, 0.15, 70, 0.42]],
	overcast: [
		[0.82, 0.02, 240, 0.24],
		[0.72, 0.025, 240, 0.24]
	],
	neon: [
		[0.66, 0.26, 340, 0.36],
		[0.82, 0.14, 200, 0.36]
	],
	candle: [[0.82, 0.15, 62, 0.45]],
	moonlight: [[0.86, 0.045, 240, 0.32]],
	'desk-lamp': [
		[0.9, 0.11, 76, 0.5],
		[0, 0, 0, 0.14]
	],
	fluorescent: [
		[1, 0, 0, 0.025],
		[0.96, 0.035, 150, 0.4],
		[0.9, 0.02, 150, 0.14]
	]
};

export function cssLight([L, C, H, alpha]: LightLayer, strength = 1): string {
	return `oklch(${L} ${C} ${H} / ${+(alpha * strength).toFixed(4)})`;
}

/** The ground under each layer of the light at its strongest point, as the browser blends it (in sRGB). */
function litGrounds(ground: OKLab, light: Light | null, strength = 1): OKLab[] {
	if (!light) return [ground];
	const under = toSrgb(ground);
	return [
		ground,
		...LIGHTS[light].map(([L, C, H, alpha]) => {
			const top = toSrgb(lchToLab([L, C, H]));
			const a = alpha * strength;
			return fromSrgb(under.map((u, i) => top[i] * a + u * (1 - a)) as RGB);
		})
	];
}

export function cssLch([L, C, H]: OKLCH): string {
	return `oklch(${L.toFixed(4)} ${C.toFixed(4)} ${H.toFixed(2)})`;
}

function hex(lab: OKLab): string {
	return `#${toSrgb(lab)
		.map((s) =>
			Math.round(s * 255)
				.toString(16)
				.padStart(2, '0')
		)
		.join('')}`;
}

export type Tokens = {
	ground: OKLab;
	ink: OKLab;
	/** A quieter ink for secondary text. It meets the same contrast floor as `ink`. */
	inkSoft: OKLab;
	mids: [OKLab, OKLab, OKLab];
	/** Ink for text set on `mids[0]` paper. */
	paperInk: OKLab;
	/** The color of a veil panel. At VEIL_ALPHA both inks keep TEXT_TARGET on it over any backdrop. */
	veil: OKLab;
};

/** Ink contrast on the plain ground (WCAG AAA) where the ground allows it. */
const INK_TARGET = 7;
/** The floor for every text color on the ground under any room light: WCAG AA, plus a margin for the grain. */
const TEXT_TARGET = 4.6;
/** How far the soft ink moves from the ink toward the ground before the floor pulls it back. */
const SOFT_MIX = 0.4;
/** The opacity of a veil panel. The CSS uses the same value: 80% of --veil. */
const VEIL_ALPHA = 0.8;

/**
 * Find the veil color nearest to the ground that keeps both inks at TEXT_TARGET when the veil
 * lets VEIL_ALPHA of it through over the worst backdrop: white under light ink, black under dark ink.
 */
function veilFor(ground: OKLab, ink: OKLab, inkSoft: OKLab): { veil: OKLab; seen: OKLab } {
	const lightInk = luminance(ink) > luminance(ground);
	const backdrop = toSrgb(lightInk ? [1, 0, 0] : [0, 0, 0]);
	const seen = (veil: OKLab) =>
		fromSrgb(toSrgb(veil).map((v, i) => v * VEIL_ALPHA + backdrop[i] * (1 - VEIL_ALPHA)) as RGB);
	const holds = (veil: OKLab) => worst(seen(veil), [ink, inkSoft]) >= TEXT_TARGET;
	if (holds(ground)) return { veil: ground, seen: seen(ground) };
	const [L, C, H] = labToLch(ground);
	// Move away from the ink: darker under light ink, lighter under dark ink.
	let lo = L;
	let hi = lightInk ? 0 : 1;
	for (let i = 0; i < 24; i++) {
		const mid = (lo + hi) / 2;
		if (holds(lchToLab(toGamut([mid, C, H])))) hi = mid;
		else lo = mid;
	}
	const veil = lchToLab(toGamut([hi, C, H]));
	return { veil, seen: seen(veil) };
}

const NEUTRAL_LIGHT: OKLab = [1, 0, 0];
const NEUTRAL_DARK: OKLab = linearToLab([0.0027, 0.0027, 0.0036]); // #09090b

export function neutralTokens(dark: boolean): Tokens {
	const ground = dark ? NEUTRAL_DARK : NEUTRAL_LIGHT;
	const ink: OKLab = dark ? [0.86, 0, 0] : [0.32, 0, 0];
	const inkSoft: OKLab = dark ? [0.7, 0, 0] : [0.48, 0, 0];
	const mid: OKLab = dark ? [0.25, 0, 0] : [0.94, 0, 0];
	const { veil } = veilFor(ground, ink, inkSoft);
	return { ground, ink, inkSoft, mids: [mid, mid, mid], paperInk: ink, veil };
}

/**
 * Use the dominant color as the background and adjust text colors to meet the contrast targets.
 * With a light, the text colors also keep TEXT_TARGET where each layer of that light is strongest.
 */
export function paletteToTokens(palette: Palette, light: Light | null = null): Tokens {
	const ground = palette[0];
	const grounds = litGrounds(ground, light);
	const rest = palette.slice(1) as OKLab[];
	let best = 0;
	for (let i = 1; i < rest.length; i++) {
		if (contrast(rest[i], ground) > contrast(rest[best], ground)) best = i;
	}
	// Hold the floor under the light where some ink can. Where none can, take the most contrast the
	// plain ground allows, and lightStrength dims the light to what that ink can hold.
	const floor = (c: OKLab) => {
		const lit = clampContrast(c, grounds, TEXT_TARGET);
		return worst(lit, grounds) >= TEXT_TARGET ? lit : clampContrast(c, [ground], INK_TARGET);
	};
	let ink = floor(clampContrast(rest[best], [ground], INK_TARGET));
	let inkSoft = floor(ink.map((v, i) => v + (ground[i] - v) * SOFT_MIX) as OKLab);
	// Where no veil color is enough on its own (a near-white or near-black ground), move the inks.
	const { veil, seen } = veilFor(ground, ink, inkSoft);
	ink = clampContrast(ink, [seen], TEXT_TARGET);
	inkSoft = clampContrast(inkSoft, [seen], TEXT_TARGET);
	const mids = rest.filter((_, i) => i !== best) as [OKLab, OKLab, OKLab];
	const paper = mids[0];
	const candidate = contrast(ink, paper) >= contrast(ground, paper) ? ink : ground;
	const paperInk = clampContrast(candidate, [paper]);
	return { ground, ink, inkSoft, mids, paperInk, veil };
}

/**
 * The strength (0 to 1) at which to show a light, so that both inks keep TEXT_TARGET.
 * It is 1 unless the ground is so near mid grey that no ink can hold the target under the full light.
 */
export function lightStrength(tokens: Tokens, light: Light | null): number {
	if (!light) return 1;
	const holds = (k: number) =>
		Math.min(
			worst(tokens.ink, litGrounds(tokens.ground, light, k)),
			worst(tokens.inkSoft, litGrounds(tokens.ground, light, k))
		) >= TEXT_TARGET;
	if (holds(1)) return 1;
	let lo = 0;
	let hi = 1;
	for (let i = 0; i < 16; i++) {
		const mid = (lo + hi) / 2;
		if (holds(mid)) lo = mid;
		else hi = mid;
	}
	return lo;
}
// A five-band swatch as an SVG data URL, for the tab icon.
export function paletteFavicon(palette: readonly OKLab[]): string {
	const bands = palette
		.map((c, i) => `<rect x="${i * 12.8}" y="0" width="12.8" height="64" fill="${hex(c)}"/>`)
		.join('');
	const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><clipPath id="r"><rect width="64" height="64" rx="12"/></clipPath><g clip-path="url(#r)">${bands}</g></svg>`;
	return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}
