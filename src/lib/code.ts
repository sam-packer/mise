// Derive share codes from feeling text and paths in both the browser and the Cloudflare Worker.
// Encode the first 48 hash bits as seven base62 characters.

const ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
const LENGTH = 7;

/** The longest feeling the line takes and the API stores. */
export const MAX_FEELING = 500;

/**
 * Trim only the outer whitespace. Preserve the same text for the model, the tab title, and the share code.
 */
export const normalize = (text: string) => text.trim();

export const isCode = (value: string) => /^[0-9A-Za-z]{7}$/.test(value);

/** The most worlds that one path link holds. */
export const MAX_TRAIL = 50;

/** A catalog item id: a category, a colon, and a slug, like "song:the-marias-sienna". */
export const isItemId = (value: unknown): value is string =>
	typeof value === 'string' && /^[a-z]+:[a-z0-9-]{1,160}$/.test(value);

export const feelingCode = (text: string) => hashCode(normalize(text));

/**
 * Derive the code of a path: a feeling and the item ids of the worlds on it, in order.
 * A feeling never starts with a newline, so a path code never uses the input of a feeling code.
 */
export const pathCode = (text: string, trail: string[]) =>
	hashCode('\n' + JSON.stringify({ text: normalize(text), trail }));

async function hashCode(input: string): Promise<string> {
	const bytes = new TextEncoder().encode(input);
	const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
	// A double holds 48 bits exactly. Seven base62 digits reduce the value modulo 62^7.
	let n = 0;
	for (let i = 0; i < 6; i++) n = n * 256 + digest[i];
	let code = '';
	for (let i = 0; i < LENGTH; i++) {
		code = ALPHABET[n % 62] + code;
		n = Math.floor(n / 62);
	}
	return code;
}

/** A share link: a feeling, and for a path, the item ids of the worlds on it. */
export type Shared = { text: string; trail: string[] };

/** Cache share links in the browser session to avoid a fetch after submission or history navigation. */
export const shared = new Map<string, Shared>();
