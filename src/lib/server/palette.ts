// Check and store the palette of a feeling's mood for its link preview image.
import { fromHex } from '#lib/color/oklab.js';
import type { OKLab, Palette } from '#lib/mood/types.js';

/** The PALETTES key of a feeling's palette under one bundle. A new model gives the feeling new colors. */
export const paletteKey = (bundle: string, code: string) => `${bundle}:${code}`;

const HEX = /^#[0-9a-f]{6}$/i;

const inRange = (v: unknown, lo: number, hi: number): v is number =>
	typeof v === 'number' && Number.isFinite(v) && v >= lo && v <= hi;

function color(value: unknown): OKLab | null {
	if (typeof value === 'string') return HEX.test(value) ? fromHex(value) : null;
	if (!Array.isArray(value) || value.length !== 3) return null;
	const [L, a, b] = value;
	return inRange(L, 0, 1) && inRange(a, -0.5, 0.5) && inRange(b, -0.5, 0.5) ? [L, a, b] : null;
}

/**
 * Accept exactly five colors, each an OKLab triple in range or a #rrggbb string. Return them as
 * OKLab rounded to four places, or null.
 */
export function parsePalette(value: unknown): Palette | null {
	if (!Array.isArray(value) || value.length !== 5) return null;
	const colors = value.map(color);
	if (colors.some((c) => c === null)) return null;
	return (colors as OKLab[]).map((c) => c.map((v) => Math.round(v * 1e4) / 1e4)) as Palette;
}
