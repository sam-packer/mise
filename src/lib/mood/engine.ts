// Load the encoder and search the bundle for the browser worker and Bun scripts.
import type * as ORT from 'onnxruntime-web';
import { Tokenizer } from '@huggingface/tokenizers';
import { createSearch } from './search';
import { decodeFp16 } from './fp16';
import type { NameData } from './name-data';
import {
	type Light,
	type Manifest,
	type Mood,
	type OKLab,
	type Palette,
	type Vocab,
	type Anchor,
	type Item,
	type MatchResult,
	type World
} from './types';

export type EncoderIO = {
	ort: typeof ORT;
	fetchBytes(path: string): Promise<ArrayBuffer>;
	fetchJson<T>(path: string): Promise<T>;
	sessionOptions?: ORT.InferenceSession.SessionOptions;
};

export type EngineIO = EncoderIO;

export type MoodEngine = {
	infer(query: string): Promise<Mood>;
	/** Build the world of one catalog item. `exclude` holds item ids to keep off its wall. */
	world(id: string, exclude: string[]): Promise<World>;
};

type Encoder = {
	/** Runs the model on one text and returns every output tensor plus the pooled embedding. */
	run(text: string): Promise<{ embedding: Float32Array; outputs: ORT.InferenceSession.ReturnType }>;
	embed(texts: string[]): Promise<Float32Array[]>;
};

function joinPath(dir: string, file: string): string {
	return dir.endsWith('/') ? dir + file : `${dir}/${file}`;
}

function l2normalize(v: Float32Array): Float32Array {
	let sum = 0;
	for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
	const inv = sum > 0 ? 1 / Math.sqrt(sum) : 0;
	for (let i = 0; i < v.length; i++) v[i] *= inv;
	return v;
}

function argmax(values: ArrayLike<number>): number {
	let best = 0;
	for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
	return best;
}

/** Load the tokenizer and ONNX session. Encode texts separately because batch padding changes dynamic quantization ranges. */
export async function createEncoder(io: EncoderIO, manifest: Manifest): Promise<Encoder> {
	const { encoder } = manifest;
	const [tokenizerJson, tokenizerConfig, modelBytes] = await Promise.all([
		io.fetchJson<object>(joinPath(encoder.tokenizer, 'tokenizer.json')),
		io.fetchJson<object>(joinPath(encoder.tokenizer, 'tokenizer_config.json')),
		io.fetchBytes(encoder.model)
	]);
	const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
	const session = await io.ort.InferenceSession.create(
		new Uint8Array(modelBytes),
		io.sessionOptions ?? {}
	);

	async function run(text: string) {
		let ids = tokenizer.encode(text).ids;
		// Truncate like the Python tokenizers: keep the closing special token ([SEP]).
		if (ids.length > encoder.maxTokens) {
			ids = [...ids.slice(0, encoder.maxTokens - 1), ids[ids.length - 1]];
		}
		const n = ids.length;
		const feeds: Record<string, ORT.Tensor> = {};
		const toTensor = (values: number[]) =>
			new io.ort.Tensor(
				'int64',
				BigInt64Array.from(values, (x) => BigInt(x)),
				[1, n]
			);
		for (const name of session.inputNames) {
			if (name === 'input_ids') feeds[name] = toTensor(ids);
			else if (name === 'attention_mask') feeds[name] = toTensor(new Array(n).fill(1));
			else if (name === 'token_type_ids') feeds[name] = toTensor(new Array(n).fill(0));
			else throw new Error(`unknown model input: ${name}`);
		}
		const outputs = await session.run(feeds);

		let embedding: Float32Array;
		if (encoder.pooling === 'mean') {
			const hidden = outputs[encoder.outputs.hidden ?? 'last_hidden_state'];
			const data = hidden.data as Float32Array;
			const dims = encoder.dims;
			embedding = new Float32Array(dims);
			for (let t = 0; t < n; t++) {
				for (let d = 0; d < dims; d++) embedding[d] += data[t * dims + d];
			}
			for (let d = 0; d < dims; d++) embedding[d] /= n;
		} else {
			const out = outputs[encoder.outputs.embedding ?? 'embedding'];
			embedding = Float32Array.from(out.data as Float32Array);
		}
		if (encoder.normalize) l2normalize(embedding);
		return { embedding, outputs };
	}

	return {
		run,
		async embed(texts) {
			const result: Float32Array[] = [];
			for (const text of texts) result.push((await run(text)).embedding);
			return result;
		}
	};
}

/** Load bundle assets and combine encoder outputs with catalog matches. */
export async function createMoodEngine(io: EngineIO): Promise<MoodEngine> {
	const manifest = await io.fetchJson<Manifest>('manifest.json');
	const { files } = manifest;
	const [encoder, vocab, items, bytes, penalty, names, anchors, anchorBytes] = await Promise.all([
		createEncoder(io, manifest),
		io.fetchJson<Vocab>(files.vocab.path),
		io.fetchJson<Item[]>(files.items.path),
		io.fetchBytes(files.vectors.path),
		// Older bundles have no penalty file; search then uses a penalty of zero.
		files.penalty ? io.fetchBytes(files.penalty.path).then(decodeFp16) : undefined,
		io.fetchJson<NameData>(files.names.path),
		files.anchors ? io.fetchJson<Anchor[]>(files.anchors.path) : [],
		files.anchorVectors ? io.fetchBytes(files.anchorVectors.path) : new ArrayBuffer(0)
	]);
	const search = createSearch(
		{ ...manifest, dims: manifest.encoder.dims, words: names.words },
		items,
		decodeFp16(bytes),
		penalty,
		anchors,
		new Float32Array(anchorBytes),
		names.representatives
	);
	const typefaces = new Map(vocab.typefaces.map((t) => [t.id, t]));

	function onnxHeads(outputs: ORT.InferenceSession.ReturnType) {
		const names = manifest.encoder.outputs;
		const read = (name: string | undefined) => {
			if (!name || !outputs[name]) throw new Error(`missing model output: ${name}`);
			return outputs[name].data as Float32Array;
		};
		const choice = (name: 'light' | 'typeface') => {
			const logits = read(names[name]);
			const correction = manifest.heads.corrections?.[name];
			if (!correction) return argmax(logits);
			return argmax(
				Array.from(logits, (score, i) => score - correction.tau * Math.log(correction.prior[i]))
			);
		};
		const flat = read(names.palette);
		const palette = Array.from(
			{ length: 5 },
			(_, slot) => [flat[slot * 3], flat[slot * 3 + 1], flat[slot * 3 + 2]] as OKLab
		) as Palette;
		return {
			palette,
			light: vocab.lights[choice('light')],
			typeface: vocab.typefaces[choice('typeface')].id
		};
	}

	function toRoom(
		heads: NonNullable<MatchResult['heads']>
	): Pick<World, 'palette' | 'light' | 'typeface'> {
		return {
			palette: heads.palette,
			light: heads.light as Light,
			typeface: typefaces.get(heads.typeface)!
		};
	}

	return {
		async infer(query) {
			const start = performance.now();
			const { embedding, outputs } = await encoder.run(query);
			const { picks, anchor, heads } = search.match(query, embedding);
			const selectedHeads =
				manifest.heads.kind === 'anchors'
					? heads!
					: onnxHeads(anchor ? (await encoder.run(anchor.vibe)).outputs : outputs);
			return { query, ...toRoom(selectedHeads), ms: performance.now() - start, picks, anchor };
		},

		async world(id, exclude) {
			const { item, neighbors } = search.world(id, exclude);
			// The vibe line carries the work's feeling, as it does for a named anchor in infer.
			const { embedding, outputs } = await encoder.run(item.vibe);
			const heads =
				manifest.heads.kind === 'anchors' ? search.heads(embedding) : onnxHeads(outputs);
			return { ...toRoom(heads), item, neighbors };
		}
	};
}
