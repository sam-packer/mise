// Define the bundle format shared by export scripts, inference, and the interface.

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
	heads: {
		kind: 'anchors' | 'onnx';
		corrections?: Record<'light' | 'typeface', { prior: number[]; tau: number }>;
	};
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

export type Vocab = {
	lights: Light[];
	typefaces: Typeface[];
};

export type Anchor =
	| { kind: 'palette'; phrase: string; value: [OKLab, OKLab, OKLab, OKLab, OKLab] }
	| { kind: 'light' | 'typeface'; phrase: string; value: string };

export type Palette = [OKLab, OKLab, OKLab, OKLab, OKLab];

export type Mood = {
	query: string;
	/** Five colors in dominance order. */
	palette: Palette;
	light: Light;
	typeface: Typeface;
	/** Inference time in milliseconds. */
	ms: number;

	/** The works closest to the feeling, with the categories mixed. */
	picks: Item[];
	/**
	 * The catalog item named in the query, or null. Its vector guides the picks, which exclude the item itself.
	 */
	anchor: Item | null;
};

/** One work as a place of its own: the mood of its vibe line, and the works closest to it. */
export type World = Pick<Mood, 'palette' | 'light' | 'typeface'> & {
	item: Item;
	/** Nearest works, with the categories mixed. They exclude the work, its creator, and the path so far. */
	neighbors: Item[];
};

export type MatchResult = {
	picks: Item[];
	anchor: Item | null;
	heads?: { palette: Palette; light: string; typeface: string };
};
