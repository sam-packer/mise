// The mood bundle contract. See docs/superpowers/specs/2026-09-27-moodboard-design.md §4.

export type OKLab = [L: number, a: number, b: number];

export const CATEGORIES = ['art', 'film', 'song', 'poem', 'book'] as const;
export type Category = (typeof CATEGORIES)[number];

export type Item = {
	id: string;
	category: Category;
	title: string;
	creator: string;
	year: number | null;
	vibe: string;
	image: { src: string; w: number; h: number; tone: OKLab } | null;
	text?: string;
	/** Song only: the album the track is on. */
	album?: string;
	preview?: string;
	links: { primary: string; spotify?: string; apple?: string; deezer?: string; youtube?: string };
};

export type Manifest = {
	version: string;
	encoder: {
		model: string;
		tokenizer: string;
		dims: number;
		maxTokens: number;
		pooling: 'mean' | 'none';
		normalize: boolean;
		outputs: {
			hidden?: string;
			embedding?: string;
			palette?: string;
			light?: string;
			typeface?: string;
			scent?: string;
		};
	};
	files: {
		items: { path: string; format: 'json' };
		vectors: { path: string; format: 'fp16-le' };
		names: { path: string; format: 'json' };
		vocab: { path: string; format: 'json' };
		anchors?: { path: string; format: 'json' };
		anchorVectors?: { path: string; format: 'fp32-le' };
	};
	assets: Record<string, string>;
	heads: { kind: 'anchors' | 'onnx' };
	counts: { items: number };
};

export const LIGHTS = [
	'dawn',
	'golden-hour',
	'overcast',
	'neon',
	'candle',
	'moonlight',
	'desk-lamp',
	'fluorescent'
] as const;
export type Light = (typeof LIGHTS)[number];

export type Typeface = { id: string; family: string; axes: string };
export type Scent = { id: string; text: string };

export type Vocab = {
	lights: Light[];
	typefaces: Typeface[];
	scents: Scent[];
};

export type Anchor =
	| { kind: 'palette'; phrase: string; value: [OKLab, OKLab, OKLab, OKLab, OKLab] }
	| { kind: 'light' | 'typeface' | 'scent'; phrase: string; value: string };

export type Palette = [OKLab, OKLab, OKLab, OKLab, OKLab];

export type Mood = {
	query: string;
	/** Five colors in dominance order. */
	palette: Palette;
	light: Light;
	typeface: Typeface;
	scent: Scent;
	/** Inference time in milliseconds. */
	ms: number;

	picks: Record<Category, Item>;
	/**
	 * The catalog item the feeling names ("blood orange essex honey"), or null. When set, the wall takes that
	 * item's vibe, and the item itself is not one of the picks.
	 */
	anchor: Item | null;
};

export type MatchResult = {
	picks: Record<Category, Item>;
	anchor: Item | null;
	heads?: { palette: Palette; light: string; typeface: string; scent: string };
};
