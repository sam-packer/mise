// Match names in a feeling to catalog items before vector search.
import type { Category, Item } from './types';

type AnchorItem = Pick<Item, 'category' | 'title' | 'creator' | 'album'>;

/** Share of the query's content words that a named entity must cover. */
const ANCHOR_COVERAGE = 0.75;
const STOP_WORDS = new Set(
	(
		'the a an of and or by in on at to for with from feat ft i im me my mine you your we our us ' +
		'it its is am are was be been this that these those just so he she him her his they them their'
	).split(' ')
);
/** Words that say what kind of thing is named. A query may add them without breaking a match. */
const FILLER_WORDS = new Set(
	(
		'film movie song track album record ep book novel poem painting art artwork listening ' +
		'watching reading playing vibe vibes mood energy'
	).split(' ')
);
const ANCHOR_CATEGORY_ORDER: Category[] = ['film', 'song', 'book', 'art', 'poem'];

/** Split lowercase text into words after removing accents and apostrophes. */
export function allWords(text: string): string[] {
	return text
		.normalize('NFD')
		.replace(/[\u0300-\u036f]/g, '')
		.toLowerCase()
		.replace(/&/g, ' and ')
		.replace(/['\u2019]/g, '')
		.split(/[^a-z0-9]+/)
		.filter(Boolean);
}

function contentWords(text: string): string[] {
	return allWords(text).filter((w) => !STOP_WORDS.has(w));
}

/** A name as typed, without filler words or a leading "the". */
function exactName(text: string): string {
	const words = allWords(text).filter((w) => !FILLER_WORDS.has(w));
	if (words[0] === 'the') words.shift();
	return words.join(' ');
}

/** Index of the first place where `phrase` appears as a run of words in `words`, or -1. */
function findPhrase(words: string[], phrase: string[]): number {
	if (!phrase.length) return -1;
	outer: for (let i = 0; i + phrase.length <= words.length; i++) {
		for (let k = 0; k < phrase.length; k++) if (words[i + k] !== phrase[k]) continue outer;
		return i;
	}
	return -1;
}

/** Build a matcher that returns an item row, or -1 when the query names no item. */
export function createAnchorMatcher(
	items: AnchorItem[],
	words: string[],
	representative: (rows: number[]) => number
) {
	const commonWords = new Set(words);
	const names = items.map((item) => ({
		exactTitle: exactName(item.title),
		exactAlbum: item.album ? exactName(item.album) : '',
		title: contentWords(item.title),
		creator: contentWords(item.creator),
		album: item.album ? contentWords(item.album) : []
	}));

	/**
	 * Match titles, albums, and creators as complete word sequences.
	 * Together, the names must cover at least ANCHOR_COVERAGE of the query's content words.
	 * Reject a match that contains only one common word, such as "rain".
	 * Prefer greater coverage, then title, album, and creator matches, in that order.
	 * Break remaining ties by category order, then catalog order.
	 */
	function findAnchor(query: string): number {
		const words = contentWords(query);
		const exactQuery = exactName(query);
		const counted = words.map((w) => !FILLER_WORDS.has(w));
		const total = counted.filter(Boolean).length;
		if (!total) return -1;

		type Match = { row: number; covered: number; kind: number };
		let best: Match | null = null;
		const better = (a: Match, b: Match) =>
			a.covered !== b.covered
				? a.covered > b.covered
				: a.kind !== b.kind
					? a.kind < b.kind
					: ANCHOR_CATEGORY_ORDER.indexOf(items[a.row].category) <
						ANCHOR_CATEGORY_ORDER.indexOf(items[b.row].category);

		items.forEach((item, row) => {
			const n = names[row];
			const creatorAt = findPhrase(words, n.creator);
			const options: [number, string[][]][] = [
				[0, [n.title]],
				[1, [n.album]],
				[2, []]
			];
			for (const [kind, phrases] of options) {
				const parts = [...phrases, n.creator];
				for (const withCreator of [true, false]) {
					const used = withCreator ? parts : phrases;
					if (!used.length || used.some((p) => !p.length)) continue;
					const hit = new Array(words.length).fill(false);
					let ok = true;
					for (const p of used) {
						const at = p === n.creator ? creatorAt : findPhrase(words, p);
						if (at < 0) ok = false;
						else for (let k = 0; k < p.length; k++) hit[at + k] = true;
					}
					if (!ok) continue;
					const matched = used.flat();
					// Common words can describe a feeling. Require an exact name match when the creator is absent.
					// Never anchor on one common word.
					if (matched.every((w) => commonWords.has(w))) {
						if (matched.length === 1) continue;
						if (!withCreator && exactQuery !== (kind === 0 ? n.exactTitle : n.exactAlbum)) continue;
					}
					const covered = hit.filter((h, i) => h && counted[i]).length;
					if (covered / total < ANCHOR_COVERAGE) continue;
					const match = { row, covered, kind };
					if (!best || better(match, best)) best = match;
				}
			}
		});
		if (!best) return -1;
		const { row, kind } = best as Match;
		const item = items[row];
		if (kind === 1) {
			return representative(
				items.flatMap((x, i) => (x.creator === item.creator && x.album === item.album ? [i] : []))
			);
		}
		if (kind === 2) {
			return representative(items.flatMap((x, i) => (x.creator === item.creator ? [i] : [])));
		}
		return row;
	}

	return findAnchor;
}
