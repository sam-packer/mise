// Render the link preview image of a feeling: the feeling in quotes, set in the colors of its mood.
// Satori lays the card out as SVG, and resvg draws it as a PNG. Both run in WebAssembly.
import { render } from '@cf-wasm/og';
import { hex, paletteToTokens } from '$lib/color/oklab';
import type { Palette } from '$lib/mood/types';
import garamond from './fonts/EBGaramond-Italic.ttf?inline';
import brandCard from '../../../static/brand/og.svg?raw';

type Node = { type: string; props: Record<string, unknown> };

const h = (type: string, props: Record<string, unknown>, ...children: (Node | string)[]): Node => ({
	type,
	props: { ...props, children: children.length === 1 ? children[0] : children }
});

const WIDTH = 1200;
const HEIGHT = 630;
/** The left and right margin, the same as the static card. */
const SIDE = 96;
const TOP = 72;
/** The box that holds the quote, between the lockup and the tagline. */
const BOX_W = WIDTH - 2 * SIDE;
const BOX_H = 330;
const SIZES = [104, 96, 88, 80, 72, 64, 58, 52, 46, 40];
const LEADING = 1.2;
/** A little more than the mean advance of EB Garamond Italic in running text, in em. */
const ADVANCE = 0.4;

/** The brand colors of the static card, for a feeling whose palette is not stored yet. */
const BRAND = {
	ground: '#f4eee6',
	ink: '#221d1a',
	inkSoft: '#5e524a',
	swatch: ['#4f6e73', '#c58e6e', '#e3c7a8', '#b98a93', '#3f3340']
};

// The static card holds the outlined wordmark and tagline. Palatino is not open-licensed, so the
// image reuses those outlines and sets only the feeling in a font.
const outline = (fill: string) => {
	const d = brandCard.match(new RegExp(`<path fill="${fill}" d="([^"]+)"`))?.[1];
	if (!d) throw new Error(`static/brand/og.svg has no ${fill} path`);
	return d;
};
const WORD = outline('#221d1a');
const TAGLINE = outline('#5e524a');

const font = Uint8Array.from(atob(garamond.slice(garamond.indexOf(',') + 1)), (c) =>
	c.charCodeAt(0)
).buffer;

/** Count the lines that `text` takes at `size`, with a greedy wrap on estimated widths. */
function lines(text: string, size: number): number {
	const perLine = BOX_W / (size * ADVANCE);
	let count = 1;
	let used = 0;
	for (const word of text.split(/\s+/)) {
		const need = used ? used + 1 + word.length : word.length;
		if (need <= perLine) {
			used = need;
			continue;
		}
		if (used) count++;
		// A word longer than a line breaks across lines.
		count += Math.ceil(word.length / perLine) - 1;
		used = word.length % perLine || perLine;
	}
	return count;
}

/** Pick the largest size at which the quote fits. At the smallest size, cut the feeling to fit. */
function fit(feeling: string): { quote: string; size: number; max: number } {
	for (const size of SIZES) {
		const max = Math.floor(BOX_H / (size * LEADING));
		const quote = `“${feeling}”`;
		if (lines(quote, size) <= max) return { quote, size, max };
	}
	const size = SIZES[SIZES.length - 1];
	const max = Math.floor(BOX_H / (size * LEADING));
	let cut = feeling;
	while (cut.length > 1 && lines(`“${cut}…”`, size) > max) cut = cut.slice(0, -1);
	// End on a whole word where the cut leaves one.
	const words = cut.replace(/\s+\S*$/, '');
	cut = (words || cut).replace(/[\s,;:.!?-]+$/, '');
	return { quote: `“${cut}…”`, size, max };
}

function colors(palette: Palette | null) {
	if (!palette) return BRAND;
	const t = paletteToTokens(palette);
	return {
		ground: hex(t.ground),
		ink: hex(t.ink),
		inkSoft: hex(t.inkSoft),
		swatch: palette.map(hex)
	};
}

/** The swatch and the wordmark, in the geometry of the static card. */
function lockup(swatch: string[], ground: string, ink: string): Node {
	const height = 44;
	const bands = swatch.map((fill, i) =>
		h('rect', {
			x: 96 + i * 34.78,
			y: 132,
			width: i === 4 ? 34.78 : 52.17,
			height: 93.16,
			fill,
			// The first band can be the ground itself. A faint edge keeps it in view.
			...(i === 0 && fill === ground ? { stroke: ink, strokeOpacity: 0.25, strokeWidth: 2 } : {})
		})
	);
	return h(
		'svg',
		{ viewBox: '95 131 439 99', width: (439 * height) / 99, height },
		...bands,
		h('path', { d: WORD, fill: ink })
	);
}

function tagline(fill: string): Node {
	const height = 27;
	return h(
		'svg',
		{ viewBox: '315 257 479 50', width: (479 * height) / 50, height },
		h('path', { d: TAGLINE, fill })
	);
}

/** Render the card as a 1200 by 630 PNG. Without a palette, the card uses the brand colors. */
export async function card(
	feeling: string,
	palette: Palette | null
): Promise<Uint8Array<ArrayBuffer>> {
	const c = colors(palette);
	const { quote, size, max } = fit(feeling);
	const element = h(
		'div',
		{
			style: {
				display: 'flex',
				flexDirection: 'column',
				width: WIDTH,
				height: HEIGHT,
				padding: `${TOP}px ${SIDE}px`,
				backgroundColor: c.ground
			}
		},
		lockup(c.swatch, c.ground, c.ink),
		h(
			'div',
			{ style: { display: 'flex', flex: 1, flexDirection: 'column', justifyContent: 'center' } },
			h(
				'div',
				{
					style: {
						display: 'block',
						fontFamily: 'EB Garamond',
						fontStyle: 'italic',
						fontSize: size,
						lineHeight: LEADING,
						color: c.ink,
						wordBreak: 'break-word',
						// A safety net for an estimate that runs short. fit() cuts the text first.
						lineClamp: max
					}
				},
				quote
			)
		),
		tagline(c.inkSoft)
	);
	const { image } = await render(element, {
		width: WIDTH,
		height: HEIGHT,
		fonts: [{ name: 'EB Garamond', data: font, weight: 400, style: 'italic' }]
	}).asPng();
	return image;
}
