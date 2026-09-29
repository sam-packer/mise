// Environment-agnostic inference core. The browser worker and the Bun scripts both use it.
import type * as ORT from 'onnxruntime-web';
import { Tokenizer } from '@huggingface/tokenizers';
import {
	type Light,
	type Manifest,
	type Mood,
	type OKLab,
	type Palette,
	type Vocab,
	type MatchResult
} from './types';

export type EncoderIO = {
	ort: typeof ORT;
	fetchBytes(path: string): Promise<ArrayBuffer>;
	fetchJson<T>(path: string): Promise<T>;
	sessionOptions?: ORT.InferenceSession.SessionOptions;
};

export type EngineIO = EncoderIO & {
	match(query: string, embedding: number[], version: string): Promise<MatchResult>;
};

export type MoodEngine = {
	infer(query: string): Promise<Mood>;
};

export type Encoder = {
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

/** Loads the tokenizer and the ONNX session. Texts are encoded one at a time, so a text always gets
 * the same vector: batch padding would change the dynamic quantization ranges. */
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

export async function createMoodEngine(io: EngineIO): Promise<MoodEngine> {
	const manifest = await io.fetchJson<Manifest>('manifest.json');
	const [encoder, vocab] = await Promise.all([
		createEncoder(io, manifest),
		io.fetchJson<Vocab>('vocab.json')
	]);
	const typefaces = new Map(vocab.typefaces.map((t) => [t.id, t]));
	const scents = new Map(vocab.scents.map((s) => [s.id, s]));

	function onnxHeads(outputs: ORT.InferenceSession.ReturnType) {
		const names = manifest.encoder.outputs;
		const read = (name: string | undefined) => {
			if (!name || !outputs[name]) throw new Error(`missing model output: ${name}`);
			return outputs[name].data as Float32Array;
		};
		const flat = read(names.palette);
		const palette = Array.from(
			{ length: 5 },
			(_, slot) => [flat[slot * 3], flat[slot * 3 + 1], flat[slot * 3 + 2]] as OKLab
		) as Palette;
		return {
			palette,
			light: vocab.lights[argmax(read(names.light))],
			typeface: vocab.typefaces[argmax(read(names.typeface))].id,
			scent: vocab.scents[argmax(read(names.scent))].id
		};
	}

	return {
		async infer(query) {
			const start = performance.now();
			const { embedding, outputs } = await encoder.run(query);
			const {
				picks,
				anchor,
				heads: serverHeads
			} = await io.match(query, Array.from(embedding), manifest.version);
			const heads =
				manifest.heads.kind === 'anchors'
					? serverHeads
					: onnxHeads(anchor ? (await encoder.run(anchor.vibe)).outputs : outputs);
			if (!heads) throw new Error('missing anchor heads');

			return {
				query,
				palette: heads.palette,
				light: heads.light as Light,
				typeface: typefaces.get(heads.typeface)!,
				scent: scents.get(heads.scent)!,
				picks,
				anchor,
				ms: performance.now() - start
			};
		}
	};
}
