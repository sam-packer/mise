// Put a private catalog in local R2 and select it for development.
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { unstable_readConfig } from 'wrangler';

const ROOT = path.resolve(import.meta.dirname, '..');

export async function selectLocalCatalog(prefix: string, root = ROOT) {
	const file = path.join(root, '.dev.vars');
	const content = await readFile(file, 'utf8').catch((error: NodeJS.ErrnoException) => {
		if (error.code !== 'ENOENT') throw error;
		return '';
	});
	const newline = content.includes('\r\n') ? '\r\n' : '\n';
	const setting = `CATALOG_PREFIX=${prefix}`;
	const pattern = /^[\t ]*(?:export[\t ]+)?CATALOG_PREFIX[\t ]*=[^\r\n]*/gm;
	let found = false;
	const merged = content.replace(pattern, () => {
		const value = found ? '' : setting;
		found = true;
		return value;
	});
	await writeFile(
		file,
		found
			? merged
			: `${content}${content && !content.endsWith('\n') ? newline : ''}${setting}${newline}`
	);
}

export async function installLocalCatalog(directory: string, prefix?: string, root = ROOT) {
	const config = unstable_readConfig({ config: path.join(root, 'wrangler.jsonc') });
	const bucket = config.r2_buckets.find((binding) => binding.binding === 'CATALOG')?.bucket_name;
	if (!bucket) throw new Error('Set the CATALOG bucket in wrangler.jsonc.');
	const meta = JSON.parse(await readFile(path.join(directory, 'catalog.json'), 'utf8'));
	const names = ['catalog.json', 'items.json', 'vectors.bin'];
	if (meta.heads.kind === 'anchors') names.push('anchors.json', 'anchors.bin');
	const digest = createHash('sha256');
	for (const name of names.sort()) {
		const bytes = await readFile(path.join(directory, name));
		digest.update(`${name}\0${createHash('sha256').update(bytes).digest('hex')}\n`);
	}
	prefix ??= `catalog/${new Date().toISOString().slice(0, 10)}-${digest.digest('hex').slice(0, 8)}/`;
	for (const name of names.sort(
		(a, b) => Number(a === 'catalog.json') - Number(b === 'catalog.json') || a.localeCompare(b)
	)) {
		const child = Bun.spawn(
			[
				'bunx',
				'wrangler',
				'r2',
				'object',
				'put',
				`${bucket}/${prefix}${name}`,
				'--local',
				'--config',
				path.join(root, 'wrangler.jsonc'),
				'--persist-to',
				path.join(root, '.wrangler', 'state'),
				'--file',
				path.resolve(directory, name)
			],
			{ cwd: ROOT, stdout: 'inherit', stderr: 'inherit' }
		);
		if ((await child.exited) !== 0) throw new Error(`Could not install ${name} in local R2.`);
	}
	await selectLocalCatalog(prefix, root);
	return prefix;
}

if (import.meta.main) {
	const [directory, prefix] = process.argv.slice(2);
	if (!directory) throw new Error('Use: bun scripts/local-catalog.ts <directory> [prefix]');
	await installLocalCatalog(directory, prefix);
}
