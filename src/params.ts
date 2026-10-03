// Restrict feeling routes to the seven-character share code format.
import { defineParams } from '@sveltejs/kit/params';
import { isCode } from '#lib/code.ts';

export const params = defineParams({
	code: (param) => (isCode(param) ? param : undefined)
});
