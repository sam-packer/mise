// Check the endpoint with real local R2 storage. No model or remote service is required.
import assert from 'node:assert/strict';
import path from 'node:path';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { createServer } from 'vite';
import { getPlatformProxy } from 'wrangler';
import { CATALOG_PREFIX } from '../src/lib/server/catalog';
import { installLocalCatalog, selectLocalCatalog } from './local-catalog';
import {
	CATEGORIES,
	type Anchor,
	type Item,
	type MatchResult,
	type Palette
} from '../src/lib/mood/types';

const root = path.resolve(import.meta.dirname, '..');
const items: Item[] = CATEGORIES.flatMap((category) =>
	['Warm', 'Cool', 'Mild'].map((name, i) => ({
		id: `${category}-${i}`,
		category,
		title: category === 'film' && i === 1 ? 'Distant Signal' : `${name} ${category}`,
		creator: `${category} creator ${i}`,
		year: 2000,
		vibe: 'a quiet evening',
		image: null,
		links: { primary: 'https://example.com/' }
	}))
);
const vectors = new Float32Array(
	CATEGORIES.flatMap(() => [1, 0, 0, 1, Math.SQRT1_2, Math.SQRT1_2])
);
const palette = Array.from({ length: 5 }, () => [0.5, 0.1, 0.1]) as Palette;

// Exercise the same local upload path that install, publish, and stub use.
const delivery = await mkdtemp(path.join(tmpdir(), 'mise-delivery-'));
try {
	const directory = path.join(delivery, 'catalog');
	await mkdir(directory);
	await writeFile(
		path.join(delivery, 'wrangler.jsonc'),
		JSON.stringify({
			name: 'mise-delivery-check',
			compatibility_date: '2026-09-27',
			r2_buckets: [{ binding: 'CATALOG', bucket_name: 'mise-catalog' }]
		})
	);
	const bodies = {
		'catalog.json': Buffer.from(
			JSON.stringify({
				version: 'delivery-v1',
				dims: 2,
				counts: { items: items.length },
				heads: { kind: 'onnx' },
				words: []
			})
		),
		'items.json': Buffer.from(
			JSON.stringify(
				items.map((item) => ({
					...item,
					image: { src: 'https://example.com/published.webp', width: 1, height: 1 }
				}))
			)
		),
		'vectors.bin': Buffer.from(vectors.buffer)
	};
	for (const [name, body] of Object.entries(bodies))
		await writeFile(path.join(directory, name), body);
	const tracked = await readFile(path.join(root, 'src/lib/server/catalog.ts'));
	const local = await installLocalCatalog(directory, undefined, delivery);
	assert.match(local, /^catalog\/\d{4}-\d{2}-\d{2}-[a-f0-9]{8}\/$/);
	const published = 'catalog/published-check/';
	await installLocalCatalog(directory, published, delivery);
	assert.deepEqual(await readFile(path.join(root, 'src/lib/server/catalog.ts')), tracked);
	const proxy = await getPlatformProxy<App.Platform['env']>({
		configPath: path.join(delivery, 'wrangler.jsonc'),
		persist: { path: path.join(delivery, '.wrangler', 'state', 'v3') }
	});
	try {
		assert.equal(proxy.env.CATALOG_PREFIX, published);
		for (const prefix of [local, published]) {
			for (const [name, bytes] of Object.entries(bodies)) {
				const object = await proxy.env.CATALOG.get(prefix + name);
				assert.ok(object);
				assert.deepEqual(Buffer.from(await object.arrayBuffer()), bytes);
			}
		}
	} finally {
		await proxy.dispose();
	}
	console.log(
		'delivery: exact bytes, generated and published prefixes, override, and unchanged tracked prefix passed'
	);
} finally {
	await rm(delivery, { recursive: true, force: true });
}

for (const kind of ['onnx', 'anchors'] as const) {
	const fixture = await mkdtemp(path.join(tmpdir(), 'mise-match-'));
	const prefix = `catalog/check-${kind}/`;
	await writeFile(
		path.join(fixture, 'wrangler.jsonc'),
		JSON.stringify({
			name: `mise-check-${kind}`,
			compatibility_date: '2026-09-27',
			r2_buckets: [{ binding: 'CATALOG', bucket_name: 'mise-catalog' }]
		})
	);
	const vars = path.join(fixture, '.dev.vars');
	await writeFile(vars, '# Keep this comment.\nOTHER_SETTING=keep-me\nCATALOG_PREFIX=old\n');
	await selectLocalCatalog(prefix, fixture);
	assert.equal(
		await readFile(vars, 'utf8'),
		`# Keep this comment.\nOTHER_SETTING=keep-me\nCATALOG_PREFIX=${prefix}\n`
	);
	const platform = await getPlatformProxy<App.Platform['env']>({
		configPath: path.join(fixture, 'wrangler.jsonc'),
		persist: false
	});
	assert.equal(platform.env.CATALOG_PREFIX, prefix);
	const server = await createServer({
		configFile: false,
		envFile: false,
		root,
		resolve: { alias: { $lib: path.join(root, 'src/lib') } },
		server: { middlewareMode: true, watch: null },
		logLevel: 'error'
	});
	try {
		const bucket = platform.env.CATALOG;
		await bucket.put(
			prefix + 'catalog.json',
			JSON.stringify({
				version: 'check-v1',
				dims: 2,
				counts: { items: items.length },
				heads: { kind },
				words: ['warm', 'cool', 'mild', 'distant', 'signal', ...CATEGORIES]
			})
		);
		await bucket.put(prefix + 'items.json', JSON.stringify(items));
		await bucket.put(prefix + 'vectors.bin', vectors.buffer);
		if (kind === 'anchors') {
			const anchors: Anchor[] = [
				{ kind: 'palette', phrase: 'quiet', value: palette },
				{ kind: 'light', phrase: 'quiet', value: 'dawn' },
				{ kind: 'typeface', phrase: 'quiet', value: 'serif' },
				{ kind: 'scent', phrase: 'quiet', value: 'pine' }
			];
			await bucket.put(prefix + 'anchors.json', JSON.stringify(anchors));
			await bucket.put(prefix + 'anchors.bin', new Float32Array([1, 0, 1, 0, 1, 0, 1, 0]).buffer);
		}
		const { POST } = await server.ssrLoadModule('/src/routes/api/match/+server.ts');
		const request = (body: string, env: Partial<App.Platform['env']> = platform.env) =>
			POST({
				request: new Request('http://localhost/api/match', { method: 'POST', body }),
				platform: { env }
			}) as Promise<Response>;
		const match = async (query: string) =>
			(
				await request(JSON.stringify({ query, embedding: [1, 0], version: 'check-v1' }))
			).json() as Promise<MatchResult>;
		await assert.rejects(
			request(JSON.stringify({ query: 'quiet', embedding: [1, 0, 0], version: 'wrong-version' })),
			(error: { status?: number; body?: { message?: string } }) =>
				error.status === 409 && Boolean(error.body?.message?.includes('versions do not match'))
		);
		const plain = await match('a quiet evening');
		assert.deepEqual(
			await (
				await request(
					JSON.stringify({ query: 'a quiet evening', embedding: [1e100, 0], version: 'check-v1' })
				)
			).json(),
			plain
		);
		assert.equal(plain.anchor, null);
		assert.deepEqual(
			Object.keys(plain).sort(),
			kind === 'anchors' ? ['anchor', 'heads', 'picks'] : ['anchor', 'picks']
		);
		for (const category of CATEGORIES) assert.equal(plain.picks[category].id, `${category}-0`);
		if (kind === 'anchors')
			assert.deepEqual(plain.heads, { palette, light: 'dawn', typeface: 'serif', scent: 'pine' });
		const named = await match('  Distant Signal  ');
		assert.equal(named.anchor?.id, 'film-1');
		for (const category of CATEGORIES) assert.equal(named.picks[category].id, `${category}-2`);
		assert.equal((await match('I remember distant signal tonight')).anchor, null);
		for (const body of [
			'{',
			'null',
			'{}',
			JSON.stringify({ query: ' ', embedding: [1, 0] }),
			JSON.stringify({ query: 'x'.repeat(501), embedding: [1, 0] }),
			JSON.stringify({ query: 'x', embedding: [1], version: 'check-v1' }),
			'{"query":"x","embedding":[1e400,0]}',
			JSON.stringify({ query: 'x', embedding: ['1', 0] })
		]) {
			await assert.rejects(request(body), (error: { status?: number }) => error.status === 400);
		}
		await assert.rejects(
			request(JSON.stringify({ query: 'x', embedding: [1, 0] }), {}),
			(error: { status?: number }) => error.status === 503
		);
		// Without an override, the endpoint must load the tracked prefix.
		await bucket.put(
			CATALOG_PREFIX + 'catalog.json',
			JSON.stringify({
				version: 'published-v2',
				dims: 2,
				counts: { items: items.length },
				heads: { kind: 'onnx' },
				words: []
			})
		);
		await bucket.put(CATALOG_PREFIX + 'items.json', JSON.stringify(items));
		await bucket.put(CATALOG_PREFIX + 'vectors.bin', vectors.buffer);
		const fallback = await request(
			JSON.stringify({ query: 'quiet', embedding: [1, 0], version: 'published-v2' }),
			{ CATALOG: bucket }
		);
		assert.equal(fallback.status, 200);
		assert.deepEqual(await match('a quiet evening'), plain);
		// The second request must use the cached catalog after the R2 files are removed.
		await bucket.delete((await bucket.list({ prefix })).objects.map((object) => object.key));
		assert.deepEqual(await match('a quiet evening'), plain);
		console.log(
			`${kind}: local R2, override, fallback, version conflict, search, anchor blend, validation, and cache passed`
		);
	} finally {
		await server.close();
		await platform.dispose();
		await rm(fixture, { recursive: true, force: true });
	}
}
