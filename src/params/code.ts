import type { ParamMatcher } from '@sveltejs/kit';
import { isCode } from '$lib/code';

export const match: ParamMatcher = isCode;
