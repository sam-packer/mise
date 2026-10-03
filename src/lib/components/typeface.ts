// Load a Google Font with only the glyphs the page needs (CSS2 `&text=`).

import type { Typeface } from '#lib/mood/types.js';

export type LoadedFace = { family: string; style: 'italic' | 'normal' };

// Include common input characters so edits can use the loaded font immediately.
const BASE = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?\'"’“”‘—–-()&';

const cache = new Map<string, Promise<LoadedFace>>();

/** Load and cache a font subset for this text, with a timeout in milliseconds. */
export function loadTypeface(face: Typeface, text: string, timeoutMs = 5000): Promise<LoadedFace> {
	const glyphs = Array.from(new Set(BASE + text))
		.sort()
		.join('');
	const key = `${face.id}|${glyphs}`;
	const hit = cache.get(key);
	if (hit) return hit;

	const style: LoadedFace['style'] = face.axes.includes('1,') ? 'italic' : 'normal';
	const family = face.family.replace(/ /g, '+');
	const spec = face.axes ? `${family}:${face.axes}` : family;
	const url = `https://fonts.googleapis.com/css2?family=${spec}&text=${encodeURIComponent(glyphs)}&display=swap`;

	const p = new Promise<LoadedFace>((resolve, reject) => {
		const timer = setTimeout(() => reject(new Error('font timeout')), timeoutMs);
		const link = document.createElement('link');
		link.rel = 'stylesheet';
		link.href = url;
		link.onerror = () => {
			clearTimeout(timer);
			link.remove();
			reject(new Error('font css failed'));
		};
		link.onload = () => {
			document.fonts.load(`${style} 400 1em "${face.family}"`, glyphs).then(
				(faces) => {
					clearTimeout(timer);
					if (faces.length) resolve({ family: `"${face.family}"`, style });
					else reject(new Error('font not available'));
				},
				(e) => {
					clearTimeout(timer);
					reject(e);
				}
			);
		};
		document.head.appendChild(link);
	});
	cache.set(key, p);
	p.catch(() => cache.delete(key));
	return p;
}
