// SPDX-License-Identifier: GPL-3.0-or-later
// Results: the spoken audio with its timings, WAV/PCM export and SRT/VTT subtitles (the same rules as loudness.ai).
import type { Voice } from "./catalog.js";
import { LocalTTSError } from "./errors.js";
import { platform } from "./platform.js";

/** Text offsets are UTF-16 code units (text.slice(start, end)); times are milliseconds. */
export interface Timings {
  version: 1;
  text: string;
  duration: number;
  /** [start, end, fromMs, toMs] for each spoken segment (sentence, line or clause). */
  segments: Array<[number, number, number, number]>;
  /** [start, end, fromMs, toMs] for each written word. */
  words: Array<[number, number, number, number]>;
}

export function pcm16(samples: Float32Array): Uint8Array {
  const out = new Uint8Array(samples.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < samples.length; i++) {
    view.setInt16(i * 2, Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), true);
  }
  return out;
}

export function wavBytes(samples: Float32Array, rate: number, bitDepth: 16 | 24 = 16): Uint8Array {
  if (bitDepth !== 16 && bitDepth !== 24) throw new LocalTTSError("OPTION_INVALID", "bitDepth must be 16 or 24.");
  const width = bitDepth / 8;
  const size = samples.length * width;
  const out = new Uint8Array(44 + size);
  const view = new DataView(out.buffer);
  const ascii = (offset: number, s: string): void => {
    for (let i = 0; i < s.length; i++) out[offset + i] = s.charCodeAt(i);
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + size, true);
  ascii(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * width, true);
  view.setUint16(32, width, true);
  view.setUint16(34, bitDepth, true);
  ascii(36, "data");
  view.setUint32(40, size, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    if (bitDepth === 16) view.setInt16(44 + i * 2, Math.round(s * 32767), true);
    else {
      const v = Math.round(s * 8388607);
      const o = 44 + i * 3;
      out[o] = v & 255;
      out[o + 1] = (v >> 8) & 255;
      out[o + 2] = (v >> 16) & 255;
    }
  }
  return out;
}

/** Band-limited resampling (windowed sinc), e.g. to the 24 kHz PCM that OpenAI-compatible clients expect. */
export function resample(samples: Float32Array, rate: number, target: number): Float32Array {
  if (rate === target || !samples.length) return samples;
  const n = Math.round(samples.length * target / rate);
  const out = new Float32Array(n);
  const ratio = rate / target;
  const cutoff = Math.min(1, target / rate);
  const half = 16 / cutoff;
  for (let i = 0; i < n; i++) {
    const center = i * ratio;
    const first = Math.max(0, Math.ceil(center - half));
    const last = Math.min(samples.length - 1, Math.floor(center + half));
    let acc = 0;
    for (let j = first; j <= last; j++) {
      const x = j - center;
      const t = x * cutoff;
      const sinc = t === 0 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t);
      const window = 0.42 + 0.5 * Math.cos(Math.PI * x / half) + 0.08 * Math.cos(2 * Math.PI * x / half);
      acc += samples[j] * sinc * window;
    }
    out[i] = acc * cutoff;
  }
  return out;
}

// ---------------------------------------------------------------- subtitles (same rules as loudness.ai)
const pad = (n: number, width: number): string => String(n).padStart(width, "0");
function clock(ms: number, comma: boolean): string {
  const t = Math.max(0, Math.round(ms));
  return `${pad(Math.floor(t / 3600000), 2)}:${pad(Math.floor(t / 60000) % 60, 2)}:${pad(Math.floor(t / 1000) % 60, 2)}`
    + `${comma ? "," : "."}${pad(t % 1000, 3)}`;
}

const squash = (s: string): string => s.replace(/\s+/gu, " ");
const length = (s: string): number => Array.from(s).length;

function twoLines(line: string): string {
  if (length(line) <= 42) return line;
  const chars = Array.from(line);
  const middle = chars.length / 2;
  let best = -1;
  chars.forEach((ch, i) => {
    if (ch === " " && (best < 0 || Math.abs(i - middle) < Math.abs(best - middle))) best = i;
  });
  return best > 0 ? chars.slice(0, best).join("") + "\n" + chars.slice(best + 1).join("") : line;
}

export interface Cue {
  text: string;
  from: number;
  to: number;
}

/** One cue per sentence or line; longer ones split at word gaps to fit two lines of about 42 characters; a cue
 * ending mid-thought (comma, semicolon, colon, ellipsis) joins the next when both fit. */
export function subtitleCues(timings: Timings): Cue[] {
  const text = timings.text;
  const cues: Cue[] = [];
  const add = (start: number, end: number, from: number, to: number): void => {
    const line = squash(text.slice(start, end)).trim();
    if (line) cues.push({ text: twoLines(line), from, to });
  };
  for (const seg of timings.segments) {
    const words = timings.words.filter((w) => w[0] >= seg[0] && w[1] <= seg[1]);
    if (!words.length) {
      add(seg[0], seg[1], seg[2], seg[3]);
      continue;
    }
    let first = 0;
    for (let i = 1; i < words.length; i++) {
      if (length(squash(text.slice(words[first][0], words[i][1]))) > 84) {
        add(words[first][0], words[i - 1][1], words[first][2], words[i - 1][3]);
        first = i;
      }
    }
    add(words[first][0], seg[1], words[first][2], Math.max(words[words.length - 1][3], seg[3]));
  }
  for (let n = cues.length - 2; n >= 0; n--) {
    const joined = cues[n].text.replace(/\n/g, " ") + " " + cues[n + 1].text.replace(/\n/g, " ");
    if (/(,|;|:|\.\.\.|…)$/.test(cues[n].text) && length(joined) <= 84) {
      cues.splice(n, 2, { text: twoLines(joined), from: cues[n].from, to: cues[n + 1].to });
    }
  }
  cues.forEach((cue, i) => {
    cue.to = Math.max(cue.to, cue.from + 700);
    if (i + 1 < cues.length) cue.to = Math.min(cue.to, cues[i + 1].from);
  });
  return cues;
}

export const srt = (timings: Timings): string => subtitleCues(timings)
  .map((c, i) => `${i + 1}\n${clock(c.from, true)} --> ${clock(c.to, true)}\n${c.text}`).join("\n\n") + "\n";

export const vtt = (timings: Timings): string => "WEBVTT\n\n" + subtitleCues(timings)
  .map((c) => `${clock(c.from, false)} --> ${clock(c.to, false)}\n${c.text}`).join("\n\n") + "\n";

async function write(path: string, data: Uint8Array | string): Promise<string> {
  const writeFile = platform().writeFile;
  if (!writeFile) {
    throw new LocalTTSError("NOT_AVAILABLE_HERE", "Saving to a path needs Node, Bun or Deno; in a browser use wav() or blob().");
  }
  await writeFile(path, data);
  return path;
}

export interface Metrics {
  seconds: number;
  audioSeconds: number;
  realTimeFactor: number;
  timings: "exact" | "estimated";
  normalized: boolean;
  limited: boolean;
}

/** Spoken audio (mono float samples in -1..1) with word timings and the voice used. */
export class Speech {
  constructor(
    readonly samples: Float32Array,
    readonly sampleRate: number,
    readonly text: string,
    readonly voice: Voice,
    readonly timings: Timings,
    readonly metrics: Metrics,
    /** The text as the voice read it, after text preparation ([[...]] marks raw phonemes). */
    readonly spoken: string,
  ) {}

  get duration(): number {
    return this.samples.length / this.sampleRate;
  }
  /** A WAV file (16-bit, or 24-bit). */
  wav(bitDepth: 16 | 24 = 16): Uint8Array {
    return wavBytes(this.samples, this.sampleRate, bitDepth);
  }
  /** Raw 16-bit little-endian mono PCM, optionally resampled. */
  pcm(sampleRate?: number): Uint8Array {
    return pcm16(resample(this.samples, this.sampleRate, sampleRate ?? this.sampleRate));
  }
  /** A WAV Blob, for <audio> elements and downloads in a browser. */
  blob(bitDepth: 16 | 24 = 16): Blob {
    return new Blob([this.wav(bitDepth) as BlobPart], { type: "audio/wav" });
  }
  srt(): string {
    return srt(this.timings);
  }
  vtt(): string {
    return vtt(this.timings);
  }
  /** Writes a WAV file (Node, Bun, Deno). */
  save(path: string, bitDepth: 16 | 24 = 16): Promise<string> {
    return write(path, this.wav(bitDepth));
  }
  saveSrt(path: string): Promise<string> {
    return write(path, this.srt());
  }
  saveVtt(path: string): Promise<string> {
    return write(path, this.vtt());
  }
  /** Everything except the samples, for logs and JSON output. */
  toJSON(): Record<string, unknown> {
    return { text: this.text, spoken: this.spoken, sampleRate: this.sampleRate, duration: Math.round(this.duration * 1000) / 1000,
      voice: this.voice, metrics: this.metrics, timings: this.timings };
  }
}

/** One piece of streamed speech: a spoken sentence with the pause after it. */
export class SpeechChunk {
  constructor(
    readonly samples: Float32Array,
    readonly sampleRate: number,
    readonly index: number,
    /** Span of the source text it speaks. */
    readonly start: number,
    readonly end: number,
    readonly final = false,
  ) {}
  pcm(): Uint8Array {
    return pcm16(this.samples);
  }
}
