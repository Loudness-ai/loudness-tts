// SPDX-License-Identifier: GPL-3.0-or-later
// Loudness measurement and normalization (the same method as the Python edition, without dependencies).
// Integrated loudness follows ITU-R BS.1770-4 / EBU R 128 (K-weighting designed for any sample rate, 400 ms blocks,
// absolute and relative gates); true peak uses 4x oversampling. normalize() sets the integrated loudness and keeps the
// true peak at or below the ceiling with a look-ahead limiter, repeating while limiting lowers the loudness.

type Biquad = [number, number, number, number, number]; // b0 b1 b2 a1 a2

function kWeighting(rate: number): [Biquad, Biquad] {
  // Shelf (libebur128 / BS.1770 design at any rate).
  let f0 = 1681.974450955533;
  const gainDb = 3.999843853973347;
  let q = 0.7071752369554196;
  let k = Math.tan(Math.PI * f0 / rate);
  const vh = 10 ** (gainDb / 20);
  const vb = vh ** 0.4996667741545416;
  let a0 = 1 + k / q + k * k;
  const shelf: Biquad = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0,
    2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0];
  // RLB high-pass.
  f0 = 38.13547087602444;
  q = 0.5003270373238773;
  k = Math.tan(Math.PI * f0 / rate);
  a0 = 1 + k / q + k * k;
  const highPass: Biquad = [1, -2, 1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0];
  return [shelf, highPass];
}

function filter(x: Float64Array, [b0, b1, b2, a1, a2]: Biquad): Float64Array {
  const y = new Float64Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1;
    x1 = x[i];
    y2 = y1;
    y1 = v;
    y[i] = v;
  }
  return y;
}

function blockPowers(y: Float64Array, rate: number, seconds: number, hop: number): number[] {
  const size = Math.round(seconds * rate);
  const step = Math.round(hop * rate);
  if (y.length < size) return [];
  const c = new Float64Array(y.length + 1);
  for (let i = 0; i < y.length; i++) c[i + 1] = c[i] + y[i] * y[i];
  const out: number[] = [];
  for (let s = 0; s + size <= y.length; s += step) out.push((c[s + size] - c[s]) / size);
  return out;
}

const lufs = (z: number): number => -0.691 + 10 * Math.log10(z);
const mean = (z: number[]): number => z.reduce((a, b) => a + b, 0) / z.length;

/** Integrated loudness in LUFS, or null for silence or audio shorter than one 400 ms block. */
export function integratedLoudness(samples: Float32Array, rate: number): number | null {
  const [shelf, highPass] = kWeighting(rate);
  let z = blockPowers(filter(filter(Float64Array.from(samples), shelf), highPass), rate, 0.4, 0.1);
  z = z.filter((p) => lufs(p) > -70);
  if (!z.length) return null;
  const gate = lufs(mean(z)) - 10;
  z = z.filter((p) => lufs(p) > gate);
  return z.length ? lufs(mean(z)) : null;
}

// 4x interpolator, 12 taps per phase (48 taps), windowed sinc as in BS.1770-4 Annex 2.
const PHASES = 4;
const TAPS = 12;
const KERNEL: Float64Array[] = (() => {
  const n = PHASES * TAPS;
  const proto = Array.from({ length: n }, (_, i) => {
    const x = (i - (n - 1) / 2) / PHASES;
    const w = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)) + 0.08 * Math.cos(4 * Math.PI * i / (n - 1));
    return (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)) * w;
  });
  return Array.from({ length: PHASES }, (_, p) => {
    const row = proto.filter((_, i) => i % PHASES === p);
    const sum = row.reduce((a, b) => a + b, 0);
    return Float64Array.from(row, (v) => v / sum);
  });
})();

/** Highest absolute value after 4x oversampling (linear). */
export function truePeak(samples: Float32Array): number {
  let peak = 0;
  for (const s of samples) peak = Math.max(peak, Math.abs(s));
  if (samples.length < TAPS) return peak;
  for (const row of KERNEL) {
    for (let i = 0; i + TAPS <= samples.length; i++) {
      let acc = 0;
      for (let t = 0; t < TAPS; t++) acc += samples[i + t] * row[TAPS - 1 - t]; // convolution
      const a = Math.abs(acc);
      if (a > peak) peak = a;
    }
  }
  return peak;
}

export const db = (value: number): number | null => (value > 0 ? 20 * Math.log10(value) : null);

export interface Measurement {
  integratedLufs: number | null;
  truePeakDbtp: number | null;
  rmsDbfs: number | null;
  durationSeconds: number;
}

export function measure(samples: Float32Array, rate: number): Measurement {
  let sum = 0;
  for (const s of samples) sum += s * s;
  return {
    integratedLufs: integratedLoudness(samples, rate),
    truePeakDbtp: db(truePeak(samples)),
    rmsDbfs: db(samples.length ? Math.sqrt(sum / samples.length) : 0),
    durationSeconds: samples.length / rate,
  };
}

/** Look-ahead peak limiter: the gain is a moving average of a two-sided minimum hold of the gain each sample needs,
 * so it reaches full reduction before every peak and recovers within about 10 ms. */
export function limit(samples: Float32Array, rate: number, ceilingDb: number): Float32Array {
  const w = Math.max(1, Math.round(0.005 * rate));
  const c = 10 ** (ceilingDb / 20);
  const n = samples.length;
  const need = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.abs(samples[i]);
    need[i] = a > c ? c / Math.max(a, 1e-12) : 1;
  }
  // Two-sided minimum over need[i-w..i+w] (1 outside), with a monotonic queue.
  const held = new Float64Array(n);
  const queue = new Int32Array(n + 1);
  let head = 0, tail = 0, next = 0;
  for (let i = 0; i < n; i++) {
    while (next < n && next <= i + w) {
      while (tail > head && need[queue[tail - 1]] >= need[next]) tail--;
      queue[tail++] = next++;
    }
    while (queue[head] < i - w) head++;
    held[i] = need[queue[head]]; // need is at most 1, so the padding of ones never wins
  }
  // Centered moving average of held, with the edges repeated.
  const span = 2 * w + 1;
  const at = (j: number): number => held[Math.min(n - 1, Math.max(0, j))];
  const out = new Float32Array(n);
  let acc = 0;
  for (let j = -w; j <= w; j++) acc += at(j);
  for (let i = 0; i < n; i++) {
    out[i] = samples[i] * (acc / span);
    acc += at(i + w + 1) - at(i - w);
  }
  return out;
}

const scaled = (samples: Float32Array, gainDb: number): Float32Array => {
  const g = Math.fround(10 ** (gainDb / 20));
  return Float32Array.from(samples, (s) => s * g);
};

/** Audio at `targetLufs` with true peak at or below `truePeakDb`. */
export function normalize(samples: Float32Array, rate: number, targetLufs = -16, truePeakDb = -1.5):
  { audio: Float32Array; normalized: boolean; limited: boolean } {
  const loud = integratedLoudness(samples, rate);
  if (loud === null) return { audio: samples, normalized: false, limited: false };
  let gainDb = targetLufs - loud;
  let ceiling = truePeakDb - 0.5;
  let limited = false;
  let out = samples;
  for (let round = 0; round < 6; round++) {
    out = scaled(samples, gainDb);
    let tp = db(truePeak(out));
    if (tp !== null && tp > truePeakDb) {
      out = limit(out, rate, ceiling);
      limited = true;
    }
    const now = integratedLoudness(out, rate);
    tp = db(truePeak(out));
    const off = now !== null ? targetLufs - now : 0;
    if (Math.abs(off) <= 0.1 && (tp === null || tp <= truePeakDb + 0.05)) break;
    if (Math.abs(off) <= 0.1) ceiling -= (tp! - truePeakDb) + 0.1;
    gainDb += off;
  }
  return { audio: out, normalized: true, limited };
}
