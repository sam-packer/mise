// A feeling's short code: the first 48 bits of the SHA-256 of its text, as 7 base62 characters.
// The browser and the Worker both run this, so the same feeling always gets the same link.

const ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
const LENGTH = 7;

/** The longest feeling the line takes and the API stores. */
export const MAX_FEELING = 500;

/**
 * The feeling as the app submits it: trimmed, and nothing else. The text goes to the model and the
 * tab title as typed, so a code must name exactly one text.
 */
export const normalize = (text: string) => text.trim();

export const isCode = (value: string) => /^[0-9A-Za-z]{7}$/.test(value);

export async function feelingCode(text: string): Promise<string> {
	const bytes = new TextEncoder().encode(normalize(text));
	const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
	// 48 bits fit a double exactly, and 62^7 < 2^48, so 7 base62 digits take the value mod 62^7.
	let n = 0;
	for (let i = 0; i < 6; i++) n = n * 256 + digest[i];
	let code = '';
	for (let i = 0; i < LENGTH; i++) {
		code = ALPHABET[n % 62] + code;
		n = Math.floor(n / 62);
	}
	return code;
}

/** Browser only: the feelings of this session by code, so Enter and back/forward skip the network. */
export const feelings = new Map<string, string>();
