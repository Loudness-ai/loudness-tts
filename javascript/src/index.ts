// SPDX-License-Identifier: GPL-3.0-or-later
/**
 * local-tts: fast, natural text to speech on your own computer, server or web page, by Loudness.ai.
 *
 *     import { speak, synthesize, voices } from "local-tts";
 *     await speak("Dr. Smith paid $5.50 on 4/5/2026.", "speech.wav");        // Node; in a browser: speak(text) plays it
 *     const speech = await synthesize("Hola, ¿qué tal?", { language: "es-ES" });
 *     for await (const chunk of stream("A long text, read sentence by sentence.")) play(chunk.pcm());
 *
 * Voices are ranked by Loudness.ai's open voice study (https://loudness.ai/research/piper-tts-voices-ranked/) and
 * downloaded once, with checksums, from their original source. Docs: https://github.com/Loudness-ai/local-tts
 */
import { findVoice, modelInfo, type Voice } from "./catalog.js";
import { TTS, type SpeakOptions } from "./engine.js";
import { LocalTTSError } from "./errors.js";
import { platform, type Progress } from "./platform.js";
import type { Speech, SpeechChunk } from "./speech.js";

export { TTS, cleanText, phonemesToIds, type SpeakOptions, type TTSOptions } from "./engine.js";
export { Speech, SpeechChunk, srt, vtt, subtitleCues, wavBytes, pcm16, resample, type Timings, type Cue, type Metrics } from "./speech.js";
export {
  voices, languages, findVoice, modelInfo, openaiVoiceMap, AUTO_USE, USE_CLASSES, DEFAULT_LANGUAGE,
  type Voice, type VoiceFilter, type LanguageEntry, type ModelInfo, type LicenceUse, type Quality,
} from "./catalog.js";
export { LocalTTSError, CancelledError, type ErrorCode } from "./errors.js";
export { prepare, type Segment } from "./textprep.js";
export { configure, setEspeakDataUrl, type BrowserSettings, type Progress } from "./platform.js";
export * as loudness from "./loudness.js";
export { VERSION as version } from "./version.js";

let shared: TTS | null = null;
/** The engine behind synthesize(), stream() and speak(). */
export function defaultEngine(): TTS {
  return (shared ??= new TTS());
}

/** Speaks `text` with a shared engine (see TTS.synthesize for the options). */
export function synthesize(text: string, options?: SpeakOptions): Promise<Speech> {
  return defaultEngine().synthesize(text, options);
}

/** Speaks `text` piece by piece for low latency (see TTS.stream). */
export function stream(text: string, options?: SpeakOptions): AsyncGenerator<SpeechChunk> {
  return defaultEngine().stream(text, options);
}

export interface SpeakFileOptions extends SpeakOptions {
  /** Also write SRT or VTT subtitles to these paths. */
  srt?: string;
  vtt?: string;
}

/** Node, Bun, Deno: speaks `text` into a WAV file (and optional subtitles). Browser: plays it, when no path is given.
 * Returns the Speech. */
export async function speak(text: string, path?: string, options: SpeakFileOptions = {}): Promise<Speech> {
  const { srt: srtPath, vtt: vttPath, ...rest } = options;
  const speech = await synthesize(text, rest);
  if (path) {
    await speech.save(path);
    if (srtPath) await speech.saveSrt(srtPath);
    if (vttPath) await speech.saveVtt(vttPath);
  } else if (!platform().node && typeof AudioContext !== "undefined") {
    await play(speech);
  } else {
    throw new LocalTTSError("OPTION_INVALID", "Give a file path, e.g. speak(text, \"speech.wav\").");
  }
  return speech;
}

/** Plays speech through the page's speakers (browsers). */
export async function play(speech: Speech | SpeechChunk): Promise<void> {
  if (typeof AudioContext === "undefined") throw new LocalTTSError("NOT_AVAILABLE_HERE", "Playback needs a browser.");
  const context = new AudioContext();
  const buffer = context.createBuffer(1, speech.samples.length, speech.sampleRate);
  buffer.copyToChannel(speech.samples as Float32Array<ArrayBuffer>, 0);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  await new Promise<void>((resolve) => {
    source.onended = () => resolve();
    source.start();
  });
  await context.close();
}

/** Downloads and checks a voice now (useful before going offline, in Docker builds and on page load). */
export async function download(voice = "auto", language?: string, progress?: (p: Progress) => void): Promise<Voice> {
  const info = findVoice(voice, language ?? null, !platform().node);
  const files = modelInfo(info.model).files;
  await platform().store.file(info.model, `${info.model}.onnx.json`, files.config, { progress });
  await platform().store.file(info.model, `${info.model}.onnx`, files.model, { progress });
  return info;
}
