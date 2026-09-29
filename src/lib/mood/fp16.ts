/** Decode IEEE 754 binary16, little-endian, without changing vector norms. */
export function decodeFp16(bytes: ArrayBuffer): Float32Array {
	if (bytes.byteLength % 2) throw new Error('Invalid fp16 byte length');
	const view = new DataView(bytes);
	const values = new Float32Array(bytes.byteLength / 2);
	for (let i = 0; i < values.length; i++) {
		const bits = view.getUint16(i * 2, true);
		const exponent = (bits >>> 10) & 31;
		const fraction = bits & 1023;
		const magnitude =
			exponent === 0 ? fraction * 2 ** -24 : (1024 + fraction) * 2 ** (exponent - 25);
		if (exponent === 31) throw new Error('Non-finite catalog vector');
		values[i] = bits & 32768 ? -magnitude : magnitude;
	}
	return values;
}
