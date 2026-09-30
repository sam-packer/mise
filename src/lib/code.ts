// Derive share codes from feeling text in both the browser and the Cloudflare Worker.
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

export async function feelingCode(text: string): Promise<string> {
	const bytes = new TextEncoder().encode(normalize(text));
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

/** Cache feelings in the browser session to avoid a fetch after submission or history navigation. */
export const feelings = new Map<string, string>();
