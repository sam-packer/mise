// Define navigation state for SvelteKit routes.
declare global {
	namespace App {
		interface PageState {
			/** The item ids of the worlds on the path from the feeling, in order. The last one is on screen. */
			trail?: string[];
			/**
			 * The length of the path at the first entry of this run of history entries. Back in history
			 * reaches no shorter path, so the page pushes a new entry for a shorter one.
			 */
			floor?: number;
		}
	}
}

export {};
