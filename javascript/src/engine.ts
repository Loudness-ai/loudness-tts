// SPDX-License-Identifier: GPL-3.0-or-later
// The speech engine: Piper voices with Loudness.ai's text preparation, pauses, calmer delivery, click-free joins,
// word timings and loudness normalization. Runs locally (in Node, Bun, Deno or a browser); nothing leaves the computer
// except voice downloads. A port of the Python edition's engine.py, tested against it.
import type * as Ort from "onnxruntime-web";
import { withAlignmentOutput } from "./alignment.js";
import { findVoice, modelInfo, type Voice } from "./catalog.js";
import { CancelledError, LocalTTSError } from "./errors.js";
import { phonemizeForVoice } from "./espeak.js";
import * as loudness from "./loudness.js";
import { onnxWasmUrl, platform, type Progress } from "./platform.js";
import { cpLength, re, strip } from "./pyre.js";
import { Speech, SpeechChunk, type Metrics, type Timings } from "./speech.js";
import { cutPoints, prepare, type Segment } from "./textprep.js";

const PAUSES: Record<string, number> = { short: 0.12, normal: 0.35, long: 0.7 };
// Pause after a segment, as a share of the chosen pause; a paragraph gets more.
const PAUSE_SHARE: Record<string, number> = { clause: 0.4, ellipsis: 1.0, sentence: 1.0, line: 1.0 };
// Calmer than each model's own settings: the random variation small VITS voices add to sound and timing is what makes
// them tremble. Models already at 0.5 or lower keep theirs.
const CALM = 0.7;
const FADE_IN_SECONDS = 0.003; // speech meets silence without a click
const FADE_OUT_SECONDS = 0.012;
const TAIL_SECONDS = 0.12; // players do not clip the last word
const VOICED_DB = -45.0;

/** Piper voice config (the .onnx.json next to each model). */
interface PiperConfig {
  audio: { sample_rate: number; quality?: string };
  espeak: { voice: string };
  phoneme_type?: string;
  phoneme_id_map: Record<string, number[]>;
  num_speakers: number;
  inference?: { noise_scale?: number; length_scale?: number; noise_w?: number };
  hop_length?: number;
  vowel_clusters?: string[][];
  default_speaker_id?: number;
}

interface Loaded {
  session: Ort.InferenceSession;
  config: PiperConfig;
  /** The model reports per-sound durations (exact word timings). */
  aligned: boolean;
  /** American word forms for US voices trained on British phonemes. */
  usWords: boolean;
  sampleRate: number;
  clusters?: Set<string>;
  queue: Promise<unknown>;
}

interface Synthesis {
  speaker: number | null;
  lengthScale: number;
  noiseScale: number;
  noiseW: number;
}

type Sound = [string, number, number];
type Timed = [Segment, number, number, Sound[] | null];

export interface TTSOptions {
  /** Default voice: a voice id, model name, OpenAI voice name, "auto", "best:<language>", or a Piper .onnx path/URL. */
  voice?: string;
  /** Default language for "auto" (en-US). */
  language?: string;
  /** CPU threads for inference (default: up to 4; one in browsers that are not cross-origin isolated). Set before the
   * first voice loads. */
  threads?: number;
  /** Voices kept loaded (default 2). */
  cacheSize?: number;
  /** Never download; use voices already stored. */
  offline?: boolean;
  /** Where voices are kept in Node (default: $LOCAL_TTS_HOME, else the user cache folder, shared with the Python
   * edition). */
  cacheDir?: string;
  /** Exact word timings from the model (default true). */
  timings?: boolean;
  /** Longest text accepted (default 100,000 characters). */
  maxCharacters?: number;
  /** Download progress. */
  progress?: (p: Progress) => void;
}

export interface SpeakOptions {
  voice?: string;
  language?: string;
  /** 0.5 to 2.0. */
  speed?: number;
  /** "short", "normal", "long" or seconds between sentences (0 to 3). */
  pause?: "short" | "normal" | "long" | number;
  /** 0.5 to 1.5: variation in sound. */
  expressiveness?: number;
  /** 0.5 to 1.5: variation in timing. */
  rhythm?: number;
  /** Turn numbers, dates, money and abbreviations into words (default true). */
  prepareText?: boolean;
  /** Loudness normalization (default true): targetLufs (-16) and truePeakDb (-1.5). */
  normalize?: boolean;
  targetLufs?: number;
  truePeakDb?: number;
  /** Stops the work (AbortController, or AbortSignal.timeout(ms) for a time limit). */
  signal?: AbortSignal;
}

function bounded(value: unknown, fallback: number, low: number, high: number, label: string): number {
  if (value === undefined || value === null) return fallback;
  const n = typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : NaN;
  if (Number.isNaN(n)) throw new LocalTTSError("OPTION_INVALID", `${label} is not a number.`);
  if (!(n >= low && n <= high)) throw new LocalTTSError("OPTION_INVALID", `${label} must be between ${low} and ${high}.`);
  return n;
}

const CONTROL = /^[\p{C}]$/u;
/** Normalizes text the way the engine reads it: NFC, no control characters, single spaces. */
export function cleanText(value: unknown, maxCharacters: number): string {
  if (typeof value !== "string") throw new LocalTTSError("TEXT_INVALID", "Text must be a string.");
  let text = Array.from(value.normalize("NFC")).filter((ch) => ch === "\n" || ch === "\t" || !CONTROL.test(ch)).join("");
  text = strip(text.replace(/[ \t]+/g, " "));
  if (!text) throw new LocalTTSError("TEXT_INVALID", "There is no text to speak.");
  if (cpLength(text) > maxCharacters) {
    throw new LocalTTSError("TEXT_TOO_LONG", `Text is longer than ${maxCharacters} characters; raise maxCharacters or split it.`);
  }
  return text;
}

function faded(audio: Float32Array, rate: number): Float32Array {
  const out = Float32Array.from(audio);
  const fadeIn = Math.min(Math.floor(out.length / 4), Math.trunc(rate * FADE_IN_SECONDS));
  const fadeOut = Math.min(Math.floor(out.length / 4), Math.trunc(rate * FADE_OUT_SECONDS));
  for (let i = 0; i < fadeIn; i++) out[i] *= Math.fround(i / fadeIn);
  for (let i = 0; i < fadeOut; i++) out[out.length - fadeOut + i] *= Math.fround(fadeOut > 1 ? 1 - i / (fadeOut - 1) : 1);
  return out;
}

function voiced(audio: Float32Array): [number, number] {
  let max = 0;
  for (const s of audio) max = Math.max(max, Math.abs(s));
  if (!audio.length || max <= 0) return [0, audio.length];
  const threshold = max * 10 ** (VOICED_DB / 20);
  let first = -1;
  let last = -1;
  for (let i = 0; i < audio.length; i++) {
    if (Math.abs(audio[i]) > threshold) {
      if (first < 0) first = i;
      last = i;
    }
  }
  return first < 0 ? [0, audio.length] : [first, last + 1];
}

/** Lets timers, other requests and the page run between sentences (inference itself blocks the event loop). */
function breathe(): Promise<void> {
  const g = globalThis as { setImmediate?: (f: () => void) => void; scheduler?: { yield?: () => Promise<void> } };
  if (g.scheduler?.yield) return g.scheduler.yield();
  return new Promise((resolve) => (g.setImmediate ? g.setImmediate(resolve) : setTimeout(resolve, 0)));
}

function stopIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw signal.reason instanceof Error && signal.reason.name === "TimeoutError"
    ? new LocalTTSError("TIMEOUT", "Speech generation took longer than the time limit.") : new CancelledError();
}

const pauseAfter = (kind: string, pause: number): number =>
  kind === "paragraph" ? Math.max(2 * pause, pause + 0.3) : pause * (PAUSE_SHARE[kind] ?? 0);

/** Python's round(): halves go to the even neighbour. */
function pyRound(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r;
}

const concat = (parts: Float32Array[]): Float32Array => {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
};

const now = (): number => (typeof performance !== "undefined" ? performance.now() : Date.now());

let ortModule: Promise<typeof Ort> | null = null;
function onnxRuntime(threads: number): Promise<typeof Ort> {
  if (!ortModule) {
    // Node, Bun and Deno use onnxruntime-web's Node build; browsers its CPU-only build (half the download).
    ortModule = (import("#ort") as Promise<typeof Ort>).then((ort) => {
      ort.env.logLevel = "error";
      const isolated = platform().node || (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
      ort.env.wasm.numThreads = isolated ? threads : 1;
      if (!platform().node && !ort.env.wasm.wasmPaths) ort.env.wasm.wasmPaths = onnxWasmUrl();
      return ort;
    });
    ortModule.catch(() => {
      ortModule = null;
    });
  }
  return ortModule;
}

/** A reusable speech engine. Keep one per process or page: voices stay loaded between calls.
 *
 *     const tts = new TTS();                                   // best English (US) voice, downloaded on first use
 *     const speech = await tts.synthesize("Dr. Smith paid $5.50 on 4/5/2026.");
 *     await speech.save("speech.wav");                         // Node; in a browser: speech.blob()
 */
export class TTS {
  readonly defaultVoice: string;
  readonly defaultLanguage: string | null;
  readonly threads: number;
  readonly cacheSize: number;
  readonly offline: boolean;
  readonly cacheDir?: string;
  readonly exactTimings: boolean;
  readonly maxCharacters: number;
  private readonly progress?: (p: Progress) => void;
  /** @internal Share of each model's random variation that is used (tests set 0 for repeatable output). */
  calm = CALM;
  private readonly cache = new Map<string, Promise<Loaded>>();

  constructor(options: TTSOptions = {}) {
    this.defaultVoice = options.voice ?? "auto";
    this.defaultLanguage = options.language ?? null;
    this.threads = options.threads ?? Math.max(1, Math.min(4, platform().cpus()));
    this.cacheSize = Math.max(1, options.cacheSize ?? 2);
    this.offline = options.offline ?? false;
    this.cacheDir = options.cacheDir;
    this.exactTimings = options.timings ?? true;
    this.maxCharacters = options.maxCharacters ?? 100_000;
    this.progress = options.progress;
  }

  // ------------------------------------------------------------------ voices
  /** The catalogue voice that `voice`/`language` (or the engine defaults) resolve to. */
  voice(voice?: string | null, language?: string | null): Voice {
    return findVoice(voice || this.defaultVoice, language || this.defaultLanguage, !platform().node);
  }

  /** Downloads (if needed) and loads a voice now, so the first request is fast. */
  async preload(voice?: string | null, language?: string | null): Promise<Voice> {
    const info = this.resolve(voice, language);
    await this.load(info);
    return info;
  }

  private resolve(voice?: string | null, language?: string | null): Voice {
    const value = voice || this.defaultVoice;
    if (value.endsWith(".onnx")) return customVoice(value, language || this.defaultLanguage || "en-US");
    return this.voice(voice, language);
  }

  private load(info: Voice): Promise<Loaded> {
    const key = info.model;
    const held = this.cache.get(key);
    if (held) {
      this.cache.delete(key);
      this.cache.set(key, held);
      return held;
    }
    const loading = this.open(info);
    this.cache.set(key, loading);
    loading.catch(() => this.cache.delete(key));
    while (this.cache.size > this.cacheSize) {
      const [oldest, entry] = this.cache.entries().next().value as [string, Promise<Loaded>];
      this.cache.delete(oldest);
      entry.then((l) => l.queue.then(() => l.session.release())).catch(() => undefined);
    }
    return loading;
  }

  private async open(info: Voice): Promise<Loaded> {
    let model: Uint8Array;
    let configBytes: Uint8Array;
    if (info.source === "local") {
      [model, configBytes] = await Promise.all([readLocal(info.model), readLocal(`${info.model}.json`)]);
    } else {
      const files = modelInfo(info.model).files;
      const store = platform().store;
      const options = { offline: this.offline, progress: this.progress, root: this.cacheDir };
      configBytes = await store.file(info.model, `${info.model}.onnx.json`, files.config, options);
      model = await store.file(info.model, `${info.model}.onnx`, files.model, options);
    }
    const config = JSON.parse(new TextDecoder().decode(configBytes)) as PiperConfig;
    const patched = this.exactTimings ? withAlignmentOutput(model) : null;
    const ort = await onnxRuntime(this.threads);
    let session: Ort.InferenceSession;
    try {
      session = await ort.InferenceSession.create(patched ?? model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    } catch (error) {
      throw new LocalTTSError("ENGINE_FAILED", `The voice model could not be loaded: ${String(error)}`);
    }
    const aligned = patched !== null && session.outputNames.length > 1;
    const clusters = config.vowel_clusters?.length ? new Set(config.vowel_clusters.map((c) => c.join("\0"))) : undefined;
    return { session, config, aligned, usWords: info.locale === "en-US" && config.espeak?.voice === "en",
      sampleRate: config.audio.sample_rate, clusters, queue: Promise.resolve() };
  }

  /** Releases every loaded voice. */
  async close(): Promise<void> {
    const loaded = [...this.cache.values()];
    this.cache.clear();
    await Promise.all(loaded.map((p) => p.then((l) => l.session.release()).catch(() => undefined)));
  }

  // ------------------------------------------------------------------ phonemes and audio
  private phonemize(loaded: Loaded, text: string): Promise<string[][]> {
    return phonemizeForVoice(loaded.config, text, loaded.clusters);
  }

  private run(loaded: Loaded, ids: number[], syn: Synthesis): Promise<[Float32Array, number[] | null]> {
    const task = loaded.queue.then(async () => {
      const ort = await onnxRuntime(this.threads);
      const config = loaded.config;
      const feeds: Record<string, Ort.Tensor> = {
        input: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]),
        input_lengths: new ort.Tensor("int64", BigInt64Array.of(BigInt(ids.length)), [1]),
        scales: new ort.Tensor("float32", Float32Array.of(syn.noiseScale, syn.lengthScale, syn.noiseW), [3]),
      };
      if (config.num_speakers > 1) {
        feeds.sid = new ort.Tensor("int64", BigInt64Array.of(BigInt(syn.speaker ?? config.default_speaker_id ?? 0)), [1]);
      }
      const result = await loaded.session.run(feeds);
      const audio = result[loaded.session.outputNames[0]].data as Float32Array;
      let perId: number[] | null = null;
      if (loaded.aligned) {
        const hop = config.hop_length ?? 256;
        perId = Array.from(result[loaded.session.outputNames[1]].data as Float32Array, (d) => Math.trunc(Math.fround(d * hop)));
      }
      return [audio, perId] as [Float32Array, number[] | null];
    });
    loaded.queue = task.catch(() => undefined);
    return task;
  }

  /** Audio for each sentence of one prepared segment (Piper's own loop, with collapsed word gaps restored), with each
   * sound's start and end sample when the model reports durations. */
  private async sentences(loaded: Loaded, text: string, syn: Synthesis, signal?: AbortSignal):
    Promise<Array<[Float32Array, Sound[] | null]>> {
    const idMap = loaded.config.phoneme_id_map;
    const out: Array<[Float32Array, Sound[] | null]> = [];
    for (const sentence of await this.phonemize(loaded, text)) {
      const phonemes = speakablePhonemes(sentence, idMap);
      if (!phonemes.length) continue;
      const ids = phonemesToIds(phonemes, idMap);
      await breathe();
      stopIfAborted(signal);
      const [audio, perId] = await this.run(loaded, ids, syn);
      if (!audio.length) continue;
      let sounds: Sound[] | null = null;
      if (perId && perId.length === ids.length) {
        const pad = idMap._.length;
        let index = idMap["^"].length + pad;
        let position = perId.slice(0, index).reduce((a, b) => a + b, 0);
        sounds = [];
        for (const phoneme of phonemes) {
          const count = idMap[phoneme].length + pad;
          const length = perId.slice(index, index + count).reduce((a, b) => a + b, 0);
          sounds.push([phoneme, position, position + length]);
          position += length;
          index += count;
        }
      }
      out.push([audio, sounds]);
    }
    return out;
  }

  private options(o: SpeakOptions): [number, number, number, number] {
    let pause: number;
    if (typeof o.pause === "string") {
      if (!(o.pause in PAUSES)) throw new LocalTTSError("OPTION_INVALID", "pause must be short, normal, long or a number of seconds.");
      pause = PAUSES[o.pause];
    } else {
      pause = bounded(o.pause, 0.35, 0.0, 3.0, "pause");
    }
    return [bounded(o.speed, 1.0, 0.5, 2.0, "speed"), pause, bounded(o.expressiveness, 1.0, 0.5, 1.5, "expressiveness"),
      bounded(o.rhythm, 1.0, 0.5, 1.5, "rhythm")];
  }

  /** Yields audio pieces and, after each segment's audio, its timing record. */
  private async *generate(text: string, info: Voice, speed: number, pause: number, expressiveness: number, rhythm: number,
    prepareText: boolean, signal?: AbortSignal): AsyncGenerator<Float32Array | [Timed, Loaded]> {
    const loaded = await this.load(info);
    const config = loaded.config;
    const inference = config.inference ?? {};
    const noiseScale = inference.noise_scale ?? 0.667;
    const noiseW = inference.noise_w ?? 0.8;
    const syn: Synthesis = {
      speaker: info.speaker,
      lengthScale: (inference.length_scale ?? 1.0) / speed,
      noiseScale: (noiseScale > 0.5 ? noiseScale * this.calm : noiseScale) * expressiveness,
      noiseW: (noiseW > 0.5 ? noiseW * this.calm : noiseW) * rhythm,
    };
    const rate = loaded.sampleRate;
    const segments = prepareText ? prepare(text, info.locale, loaded.usWords)
      : [{ text, start: 0, end: text.length, pause: "", tokens: [] } as Segment];
    const gap = new Float32Array(Math.trunc(rate * pause));
    let position = 0;
    for (const segment of segments) {
      stopIfAborted(signal);
      const chunks = await this.sentences(loaded, segment.text, syn, signal);
      stopIfAborted(signal);
      const heard: Array<[number, number]> = [];
      let stream: Sound[] = [];
      let aligned = true;
      for (let index = 0; index < chunks.length; index++) {
        const [raw, sounds] = chunks[index];
        if (index) {
          yield gap;
          position += gap.length;
          stream.push([" ", position - gap.length, position]);
        }
        const chunk = faded(raw, rate);
        const spoken = (sounds ?? []).filter(([p]) => strip(p)).map(([, a, b]) => [a, b]);
        const [first, last] = spoken.length ? [spoken[0][0], spoken[spoken.length - 1][1]] : voiced(chunk);
        if (sounds && sounds.length) stream = stream.concat(sounds.map(([p, a, b]) => [p, position + a, position + b] as Sound));
        else aligned = false;
        heard.push([position + first, position + last]);
        yield chunk;
        position += chunk.length;
      }
      if (heard.length) yield [[segment, heard[0][0], heard[heard.length - 1][1], aligned ? stream : null], loaded];
      const after = chunks.length ? pauseAfter(segment.pause, pause) : 0;
      if (after) {
        const silence = new Float32Array(Math.trunc(rate * after));
        yield silence;
        position += silence.length;
      }
    }
    yield new Float32Array(Math.trunc(rate * TAIL_SECONDS));
  }

  /** Speaks `text` and returns a Speech (samples, timings, subtitles, WAV and file helpers). */
  async synthesize(text: string, options: SpeakOptions = {}): Promise<Speech> {
    const started = now();
    text = cleanText(text, this.maxCharacters);
    const [speed, pause, expressiveness, rhythm] = this.options(options);
    const targetLufs = bounded(options.targetLufs, -16.0, -40.0, -5.0, "targetLufs");
    const truePeakDb = bounded(options.truePeakDb, -1.5, -9.0, 0.0, "truePeakDb");
    const info = this.resolve(options.voice, options.language);
    const parts: Float32Array[] = [];
    const timed: Timed[] = [];
    let loaded: Loaded | null = null;
    for await (const item of this.generate(text, info, speed, pause, expressiveness, rhythm, options.prepareText ?? true,
      options.signal)) {
      if (item instanceof Float32Array) parts.push(item);
      else {
        timed.push(item[0]);
        loaded = item[1];
      }
    }
    if (!timed.length || !loaded) throw new LocalTTSError("NOTHING_TO_SPEAK", "There is nothing to speak in this text.");
    const rate = loaded.sampleRate;
    let audio = concat(parts);
    let peak = 0;
    for (const s of audio) peak = Math.max(peak, Math.abs(s));
    if (peak < 1e-6) throw new LocalTTSError("NOTHING_TO_SPEAK", "There is nothing to speak in this text.");
    const scale = Math.fround(0.9);
    audio = audio.map((s) => Math.max(-1, Math.min(1, Math.fround(Math.fround(s / peak) * scale))));
    let report = { normalized: false, limited: false };
    if (options.normalize ?? true) {
      const result = loudness.normalize(audio, rate, targetLufs, truePeakDb);
      audio = result.audio;
      report = { normalized: result.normalized, limited: result.limited };
    }
    const timings = await this.timings(text, timed, loaded, rate, audio.length);
    const spoken = timed.map(([segment]) => segment.text).join(" ");
    const elapsed = (now() - started) / 1000;
    const metrics: Metrics = {
      seconds: Math.round(elapsed * 1000) / 1000, audioSeconds: Math.round(audio.length / rate * 1000) / 1000,
      realTimeFactor: Math.round(elapsed / Math.max(audio.length / rate, 1e-9) * 10000) / 10000,
      timings: loaded.aligned ? "exact" : "estimated", ...report,
    };
    return new Speech(audio, rate, text, info, timings, metrics, spoken);
  }

  /** Speaks `text` piece by piece (each spoken sentence with the pause after it) so playback can start at once.
   * Loudness is set from the first sentence and peaks are limited per piece, so the level is close to, not exactly,
   * targetLufs. */
  async *stream(text: string, options: SpeakOptions = {}): AsyncGenerator<SpeechChunk> {
    text = cleanText(text, this.maxCharacters);
    const [speed, pause, expressiveness, rhythm] = this.options(options);
    const targetLufs = bounded(options.targetLufs, -16.0, -40.0, -5.0, "targetLufs");
    const truePeakDb = bounded(options.truePeakDb, -1.5, -9.0, 0.0, "truePeakDb");
    const normalize = options.normalize ?? true;
    const info = this.resolve(options.voice, options.language);
    let gain: number | null = null;
    let pending: Float32Array[] = [];
    let index = 0;
    let rate = info.sampleRate;
    for await (const item of this.generate(text, info, speed, pause, expressiveness, rhythm, options.prepareText ?? true,
      options.signal)) {
      if (item instanceof Float32Array) {
        pending.push(item);
        continue;
      }
      const [[segment], loaded] = item;
      rate = loaded.sampleRate;
      let piece = concat(pending);
      pending = [];
      if (normalize && piece.length) {
        if (gain === null) {
          const measured = loudness.integratedLoudness(piece, rate);
          gain = measured !== null ? 10 ** (Math.max(-20, Math.min(20, targetLufs - measured)) / 20) : 1.0;
        }
        const g = Math.fround(gain);
        piece = loudness.limit(piece.map((s) => s * g), rate, truePeakDb - 0.5);
      }
      yield new SpeechChunk(piece, rate, index, segment.start, segment.end);
      index += 1;
    }
    if (pending.length) {
      let tail = concat(pending);
      if (normalize && gain !== null) {
        const g = Math.fround(gain);
        tail = tail.map((s) => s * g);
      }
      yield new SpeechChunk(tail, rate, index, text.length, text.length, true);
    }
  }

  // ------------------------------------------------------------------ word timings
  private async soundCount(loaded: Loaded, token: string, cache: Map<string, number>): Promise<number> {
    let count = cache.get(token);
    if (count === undefined) {
      const sounds = token.includes("[[") ? token.replace(/[[\]]/g, "")
        : (await this.phonemize(loaded, token)).map((s) => s.join("")).join("");
      count = Math.max(1, cpLength(sounds.replace(/ /g, "")));
      cache.set(token, count);
    }
    return count;
  }

  private async tokenTimes(loaded: Loaded, segment: Segment, stream: Sound[] | null, start: number, end: number,
    cache: Map<string, number>): Promise<Array<[number, number]>> {
    const counts: number[] = [];
    for (const [token] of segment.tokens) counts.push(await this.soundCount(loaded, token, cache));
    const total = counts.reduce((a, b) => a + b, 0);
    if (!counts.length) return [];
    if (!stream) {
      const out: Array<[number, number]> = [];
      let done = 0;
      for (const count of counts) {
        out.push([start + (end - start) * done / total, start + (end - start) * (done + count) / total]);
        done += count;
      }
      return out;
    }
    const sounds = stream.map(([p], i) => (p !== " " ? i : -1)).filter((i) => i >= 0);
    if (!sounds.length) return counts.map(() => [start, end]);
    const wordStarts = new Set(sounds.map((i, k) => (k === 0 || stream[i - 1][0] === " " ? k : -1)).filter((k) => k >= 0));
    const bounds = cutPoints(counts, sounds.length, wordStarts);
    const out: Array<[number, number]> = [];
    for (let k = 0; k + 1 < bounds.length; k++) {
      const [a, b] = [bounds[k], bounds[k + 1]];
      if (b > a) out.push([stream[sounds[a]][1], stream[sounds[b - 1]][2]]);
      else {
        const edge = stream[sounds[Math.min(a, sounds.length - 1)]][1];
        out.push([edge, edge]);
      }
    }
    return out;
  }

  private async wordTimings(loaded: Loaded, segment: Segment, text: string, stream: Sound[] | null, start: number,
    end: number, cache: Map<string, number>): Promise<Array<[number, number, number, number]>> {
    const words = Array.from(re("\\S+").finditer(text.slice(segment.start, segment.end)),
      (m) => [m.index + segment.start, m.index + m[0].length + segment.start]);
    const groups: Array<[number[], number, number]> = [];
    const times = await this.tokenTimes(loaded, segment, stream, start, end, cache);
    segment.tokens.forEach(([, ts, te], k) => {
      if (k >= times.length) return;
      const [t0, t1] = times[k];
      const hit = words.map(([a, b], i) => (a < te && ts < b ? i : -1)).filter((i) => i >= 0);
      if (!hit.length) return;
      const lastGroup = groups[groups.length - 1];
      if (lastGroup && lastGroup[0].join(",") === hit.join(",")) lastGroup[2] = t1;
      else groups.push([hit, t0, t1]);
    });
    const spans = new Map<number, [number, number]>();
    for (const [hit, t0, t1] of groups) {
      const lengths = hit.map((i) => words[i][1] - words[i][0]);
      const sum = lengths.reduce((a, b) => a + b, 0);
      let done = 0;
      hit.forEach((i, k) => {
        const piece: [number, number] = [t0 + (t1 - t0) * done / sum, t0 + (t1 - t0) * (done + lengths[k]) / sum];
        const held = spans.get(i);
        spans.set(i, held ? [Math.min(held[0], piece[0]), Math.max(held[1], piece[1])] : piece);
        done += lengths[k];
      });
    }
    return [...spans.keys()].sort((a, b) => a - b).map((i) => [words[i][0], words[i][1], spans.get(i)![0], spans.get(i)![1]]);
  }

  private async timings(text: string, timed: Timed[], loaded: Loaded, rate: number, total: number): Promise<Timings> {
    const ms = (samples: number): number => pyRound(samples * 1000 / rate);
    const cache = new Map<string, number>();
    const words: Array<[number, number, number, number]> = [];
    for (const [segment, first, last, stream] of timed) {
      if (!segment.tokens.length) continue;
      for (const [a, b, t0, t1] of await this.wordTimings(loaded, segment, text, stream, first, last, cache)) {
        words.push([a, b, ms(t0), ms(t1)]);
      }
    }
    return { version: 1, text, duration: ms(total), segments: timed.map(([s, a, b]) => [s.start, s.end, ms(a), ms(b)]), words };
  }
}

/** The phonemes of a sentence the model is given: word gaps collapsed (Piper keeps a gap after punctuation), sounds the
 * model does not know left out, no gaps at the ends. */
export function speakablePhonemes(sentence: string[], idMap: Record<string, number[]>): string[] {
  const phonemes = sentence.filter((p, i) => (p !== " " || (i > 0 && sentence[i - 1] !== " ")) && p in idMap);
  while (phonemes.length && phonemes[0] === " ") phonemes.shift();
  while (phonemes.length && phonemes[phonemes.length - 1] === " ") phonemes.pop();
  return phonemes;
}

/** Piper's phonemes_to_ids: BOS, PAD, then each phoneme followed by PAD, then EOS. */
export function phonemesToIds(phonemes: string[], idMap: Record<string, number[]>): number[] {
  const ids = [...idMap["^"], ...idMap._];
  for (const phoneme of phonemes) {
    const mapped = idMap[phoneme];
    if (!mapped) continue;
    ids.push(...mapped, ...idMap._);
  }
  ids.push(...idMap.$);
  return ids;
}

async function readLocal(path: string): Promise<Uint8Array> {
  if (/^https?:\/\//.test(path)) {
    const response = await fetch(path);
    if (!response.ok) throw new LocalTTSError("VOICE_NOT_FOUND", `Could not load ${path} (HTTP ${response.status}).`);
    return new Uint8Array(await response.arrayBuffer());
  }
  const read = platform().readFile;
  if (!read) throw new LocalTTSError("NOT_AVAILABLE_HERE", "Local model paths need Node, Bun or Deno; in a browser give a URL.");
  try {
    return await read(path);
  } catch {
    throw new LocalTTSError("VOICE_NOT_FOUND", `No model file at ${path}.`);
  }
}

/** A Piper voice of your own (a .onnx file or URL with its .onnx.json next to it). */
function customVoice(path: string, locale: string): Voice {
  const name = path.split(/[\\/]/).pop()!.replace(/\.onnx$/, "");
  return { id: path, model: path, speaker: null, name, language: locale, locale, quality: "medium", sex: null, overall: 0,
    natural: 0, clear: 0, rank: 0, of: 0, licence: "see the model card", use: "unclear", dataset: "",
    source: "local", sampleRate: 22050, note: null, credit: null, browser: true };
}
