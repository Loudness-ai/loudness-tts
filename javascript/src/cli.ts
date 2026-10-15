#!/usr/bin/env node
// SPDX-License-Identifier: GPL-3.0-or-later
// Command line: `loudness-tts "Hello" -o hello.wav`. Run `loudness-tts --help`.
// Commands: say (default), voices, languages, download, serve, cache, doctor.
// Exit codes: 0 success, 1 error (message on stderr; --json prints {"error": {code, message}}), 130 interrupted.
import { writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { parseArgs, type ParseArgsConfig } from "node:util";
import {
  cacheDir, downloadedModels, encode, ffmpegPath, findVoice, languages, TTSError, modelInfo, removeModel, serve,
  TTS, version, voices, type Progress, type SpeakOptions, type Voice,
} from "./node.js";
import { platform } from "./platform.js";

const COMMANDS = new Set(["say", "voices", "languages", "download", "serve", "cache", "doctor"]);

const HELP = `loudness-tts ${version}: natural text to speech on your own computer or server.

Usage:
  loudness-tts "Text to speak" -o out.wav [--voice ID] [--language es-ES] [--speed 1.1] [--srt out.srt] [--vtt out.vtt]
  echo "Text from a pipe" | loudness-tts - > out.wav
  loudness-tts voices [--language fr] [--sex female] [--min-score 80] [--limit 20] [--json]
  loudness-tts languages [--json]
  loudness-tts download [VOICE ...] [--language de-DE]      for offline machines and Docker images
  loudness-tts serve [--port 8880] [--host 0.0.0.0 --api-key SECRET] [--cors ORIGIN] [--workers 1]
  loudness-tts cache path|list|remove [MODEL ...]
  loudness-tts doctor [--synthesize]

Speech options: --pause short|normal|long|SECONDS, --expressiveness 0.5-1.5, --rhythm 0.5-1.5, --raw (no text
preparation), --no-normalize, --target-lufs -16, --bit-depth 16|24, --format wav|pcm|mp3|opus|aac|flac.
Common options: --threads N, --offline, --quiet, --json.
Voices: https://loudness.ai/research/piper-tts-voices-ranked/ · Docs: https://github.com/Loudness-ai/loudness-tts`;

const COMMON: ParseArgsConfig["options"] = {
  threads: { type: "string" }, offline: { type: "boolean" }, quiet: { type: "boolean", short: "q" },
  json: { type: "boolean" }, help: { type: "boolean", short: "h" },
};
const VOICE: ParseArgsConfig["options"] = { voice: { type: "string", short: "v" }, language: { type: "string", short: "l" } };
const OPTIONS: Record<string, ParseArgsConfig["options"]> = {
  say: { ...COMMON, ...VOICE, output: { type: "string", short: "o" }, speed: { type: "string" }, pause: { type: "string" },
    expressiveness: { type: "string" }, rhythm: { type: "string" }, raw: { type: "boolean" },
    "no-normalize": { type: "boolean" }, "target-lufs": { type: "string" }, "bit-depth": { type: "string" },
    format: { type: "string" }, srt: { type: "string" }, vtt: { type: "string" } },
  voices: { ...COMMON, language: { type: "string", short: "l" }, sex: { type: "string" }, use: { type: "string", multiple: true },
    quality: { type: "string", multiple: true }, "min-score": { type: "string" }, sort: { type: "string" },
    limit: { type: "string" } },
  languages: { ...COMMON },
  download: { ...COMMON, language: { type: "string", short: "l" } },
  serve: { ...COMMON, ...VOICE, host: { type: "string" }, port: { type: "string" }, workers: { type: "string" },
    "cache-size": { type: "string" }, "max-characters": { type: "string" }, "api-key": { type: "string" },
    cors: { type: "string" }, "no-preload": { type: "boolean" } },
  cache: { ...COMMON },
  doctor: { ...COMMON, ...VOICE, synthesize: { type: "boolean" } },
};

type Args = Record<string, string | boolean | string[] | undefined>;
const num = (value: unknown): number | undefined => (value === undefined ? undefined : Number(value));
const err = (text: string): void => void process.stderr.write(text);

function progress(p: Progress): void {
  const total = p.total || 1;
  err(`\rDownloading ${p.file}: ${Math.floor((100 * p.bytes) / total)}%`);
  if (p.bytes >= total) err("\n");
}

const engine = (a: Args, extra = {}): TTS => new TTS({ voice: (a.voice as string) || "auto", language: a.language as string,
  threads: num(a.threads), offline: !!a.offline, progress: a.quiet ? undefined : progress, ...extra });

async function readStdin(): Promise<string> {
  const parts: Buffer[] = [];
  for await (const part of process.stdin) parts.push(part as Buffer);
  return Buffer.concat(parts).toString("utf8");
}

async function say(a: Args, positionals: string[]): Promise<void> {
  const given = positionals[0];
  const text = given === undefined || given === "-" ? await readStdin() : given;
  const output = a.output as string | undefined;
  const format = (a.format as string) || (output ? path.extname(output).slice(1).toLowerCase() : "wav") || "wav";
  if (!output && process.stdout.isTTY) throw new TTSError("OPTION_INVALID", "Give an output file with -o speech.wav (or pipe the output).");
  const pause = a.pause === undefined ? undefined : ["short", "normal", "long"].includes(a.pause as string)
    ? (a.pause as SpeakOptions["pause"]) : Number(a.pause);
  const speech = await engine(a).synthesize(text, { speed: num(a.speed), pause, expressiveness: num(a.expressiveness),
    rhythm: num(a.rhythm), prepareText: !a.raw, normalize: !a["no-normalize"], targetLufs: num(a["target-lufs"]) });
  const depth = (num(a["bit-depth"]) ?? 16) as 16 | 24;
  const data = format === "wav" ? speech.wav(depth) : await encode(speech.samples, speech.sampleRate, format);
  if (output) await writeFile(output, data);
  else process.stdout.write(data);
  if (a.srt) await speech.saveSrt(a.srt as string);
  if (a.vtt) await speech.saveVtt(a.vtt as string);
  if (a.json) console.log(JSON.stringify({ ...speech.toJSON(), output: output ?? null }));
  else if (!a.quiet && output) {
    const credit = speech.voice.credit ? `\nCredit: ${speech.voice.credit}` : "";
    err(`${output}: ${speech.duration.toFixed(1)} s, voice ${speech.voice.id}, made in ${speech.metrics.seconds} s${credit}\n`);
  }
}

function listVoices(a: Args): void {
  const found = voices({ language: a.language as string, sex: a.sex as string, use: a.use as Voice["use"][],
    quality: a.quality as Voice["quality"][], minScore: num(a["min-score"]), sort: (a.sort as never) ?? "overall" });
  const limit = num(a.limit) ?? 20;
  const shown = limit ? found.slice(0, limit) : found;
  if (a.json) return console.log(JSON.stringify(shown));
  const row = (cells: Array<[string, number, boolean?]>): string =>
    cells.map(([t, w, right]) => (right ? t.padStart(w) : t.padEnd(w))).join(" ");
  console.log(row([["VOICE", 38], ["LANGUAGE", 26], ["SEX", 4], ["SCORE", 5, true], ["NAT", 5, true], ["CLEAR", 5, true]]) + "  LICENCE");
  for (const v of shown) {
    console.log(row([[v.id, 38], [v.language.slice(0, 26), 26], [v.sex ?? "-", 4], [v.overall.toFixed(1), 5, true],
      [v.natural.toFixed(1), 5, true], [v.clear.toFixed(1), 5, true]]) + "  " + v.licence);
  }
  if (found.length > shown.length) console.log(`... ${found.length - shown.length} more (use --limit 0 for all)`);
}

function listLanguages(a: Args): void {
  const rows = languages();
  if (a.json) return console.log(JSON.stringify(rows.map((r) => ({ ...r, best: r.best.id }))));
  for (const r of rows) {
    console.log(`${r.locale.padEnd(8)} ${r.language.padEnd(34)} ${String(r.voices).padStart(5)} voices   best: ${r.best.id} (${r.best.overall})`);
  }
}

async function download(a: Args, positionals: string[]): Promise<void> {
  for (const name of positionals.length ? positionals : ["auto"]) {
    const voice = findVoice(name, (a.language as string) ?? null);
    const files = modelInfo(voice.model).files;
    const options = { offline: !!a.offline, progress: a.quiet ? undefined : progress };
    await platform().store.file(voice.model, `${voice.model}.onnx.json`, files.config, options);
    await platform().store.file(voice.model, `${voice.model}.onnx`, files.model, options);
    if (!a.quiet) err(`${voice.model}: ready (${voice.licence})\n`);
  }
}

async function runServer(a: Args): Promise<void> {
  const tts = engine(a, { cacheSize: num(a["cache-size"]) ?? 4, maxCharacters: num(a["max-characters"]) ?? 20000 });
  const server = await serve({ host: (a.host as string) ?? "127.0.0.1", port: num(a.port) ?? 8880, tts,
    workers: num(a.workers) ?? 1, apiKey: (a["api-key"] as string) ?? undefined, cors: (a.cors as string) ?? "",
    preload: !a["no-preload"], quiet: !!a.quiet });
  await new Promise<void>((resolve) => {
    const stop = (): void => {
      server.close();
      resolve();
    };
    process.once("SIGINT", stop);
    process.once("SIGTERM", stop);
  });
}

async function cache(a: Args, positionals: string[]): Promise<void> {
  const [action, ...models] = positionals;
  if (action === "path") console.log(cacheDir());
  else if (action === "list") for (const m of await downloadedModels()) console.log(m);
  else if (action === "remove") for (const m of models) console.log(`${m}: ${(await removeModel(m)) ? "removed" : "not in the cache"}`);
  else throw new TTSError("OPTION_INVALID", "Use: loudness-tts cache path|list|remove [MODEL ...]");
}

async function doctor(a: Args): Promise<void> {
  const ort = await import("onnxruntime-web").then((m) => m.env.versions?.web ?? "installed", (e) => `missing: ${String(e)}`);
  const checks: Record<string, unknown> = {
    loudness_tts: version, node: process.versions.node, platform: `${os.platform()} ${os.arch()}`, cache_dir: cacheDir(),
    ffmpeg: ffmpegPath() ?? "not found (wav and pcm still work)", onnxruntime_web: ort, word_timings: "exact",
    downloaded_voices: await downloadedModels(),
  };
  if (a.synthesize) {
    const started = Date.now();
    const speech = await engine(a).synthesize("Local text to speech is ready.");
    checks.synthesis = { voice: speech.voice.id, seconds: Math.round((Date.now() - started) / 10) / 100,
      audio_seconds: Math.round(speech.duration * 100) / 100 };
  }
  if (a.json) console.log(JSON.stringify(checks));
  else for (const [k, v] of Object.entries(checks)) console.log(`${k.padEnd(18)} ${typeof v === "string" ? v : JSON.stringify(v)}`);
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
  argv = [...argv];
  if (argv[0] === "--version" || argv[0] === "-V") {
    console.log(`loudness-tts ${version}`);
    return 0;
  }
  if (!argv.length || argv[0] === "--help" || argv[0] === "-h") {
    console.log(HELP);
    return 0;
  }
  if (!COMMANDS.has(argv[0])) argv.unshift("say");
  const command = argv[0];
  let a: Args = {};
  try {
    const parsed = parseArgs({ args: argv.slice(1), options: OPTIONS[command], allowPositionals: true, strict: true });
    a = parsed.values as Args;
    if (a.help) {
      console.log(HELP);
      return 0;
    }
    const positionals = parsed.positionals;
    if (command === "say") await say(a, positionals);
    else if (command === "voices") listVoices(a);
    else if (command === "languages") listLanguages(a);
    else if (command === "download") await download(a, positionals);
    else if (command === "serve") await runServer(a);
    else if (command === "cache") await cache(a, positionals);
    else if (command === "doctor") await doctor(a);
    return 0;
  } catch (error) {
    const known = error instanceof TTSError ? error
      : new TTSError("OPTION_INVALID", error instanceof Error ? error.message : String(error));
    if (a.json || argv.includes("--json")) console.log(JSON.stringify({ error: known.toJSON() }));
    else err(`loudness-tts: ${known.message}\n`);
    return 1;
  }
}

main().then((code) => {
  process.exitCode = code;
});
