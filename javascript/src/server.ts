// SPDX-License-Identifier: GPL-3.0-or-later
// A local speech server compatible with OpenAI's speech API, so existing clients can switch to local voices by changing
// their base URL. Node's http module only.
//
//   POST /v1/audio/speech   OpenAI-compatible: {input, voice?, model?, response_format?, speed?}. Extra fields: language,
//                           pause, expressiveness, rhythm. response_format "pcm" streams 24 kHz 16-bit PCM as it is made.
//   POST /v1/tts            JSON answer with base64 WAV, word timings, SRT and VTT, voice and credit line.
//   GET  /v1/audio/voices   voices (?language=es&limit=20); GET /v1/models; GET /health
// Text is never logged or stored.
import { timingSafeEqual } from "node:crypto";
import http from "node:http";
import "./node-platform.js";
import { findVoice, voices } from "./catalog.js";
import { TTS } from "./engine.js";
import { TTSError } from "./errors.js";
import { CONTENT_TYPES, PCM_RATE, encode, ffmpegPath } from "./formats.js";
import { pcm16, resample } from "./speech.js";
import { VERSION } from "./version.js";

const MAX_BODY = 1 << 20;
const STATUS: Record<string, number> = { VOICE_NOT_FOUND: 400, TEXT_INVALID: 400, TEXT_TOO_LONG: 413,
  NOTHING_TO_SPEAK: 422, OPTION_INVALID: 400, FORMAT_UNAVAILABLE: 400, OFFLINE: 503, DOWNLOAD_FAILED: 502,
  CHECKSUM_MISMATCH: 502, TIMEOUT: 504, CANCELLED: 499 };

/** HTTP headers carry ASCII: accents are dropped and the middle dot becomes a hyphen. */
const ascii = (text: string): string => text.replace(/·/g, "-").normalize("NFKD").replace(/[^\x00-\x7f]/g, "");

export interface ServeOptions {
  host?: string;
  port?: number;
  tts?: TTS;
  /** Syntheses at the same time (default 1); others wait up to queueSeconds. */
  workers?: number;
  /** Require this bearer token (default: $LOUDNESS_TTS_API_KEY). */
  apiKey?: string;
  /** Allow this browser origin, e.g. "http://localhost:3000" or "*". */
  cors?: string;
  queueSeconds?: number;
  /** Longest synthesis in seconds (default 300). */
  timeout?: number;
  /** Load the default voice before listening (default true). */
  preload?: boolean;
  quiet?: boolean;
}

class Slots {
  private free: number;
  private waiting: Array<() => void> = [];
  constructor(count: number) {
    this.free = Math.max(1, count);
  }
  acquire(seconds: number): Promise<boolean> {
    if (this.free > 0) {
      this.free--;
      return Promise.resolve(true);
    }
    return new Promise((resolve) => {
      const wake = (): void => {
        clearTimeout(timer);
        resolve(true);
      };
      const timer = setTimeout(() => {
        this.waiting = this.waiting.filter((w) => w !== wake);
        resolve(false);
      }, seconds * 1000);
      this.waiting.push(wake);
    });
  }
  release(): void {
    const next = this.waiting.shift();
    if (next) next();
    else this.free++;
  }
}

function readBody(req: http.IncomingMessage): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const parts: Buffer[] = [];
    let size = 0;
    req.on("data", (b: Buffer) => {
      size += b.length;
      if (size > MAX_BODY) {
        reject(new TTSError("TEXT_INVALID", "Send a JSON body of up to 1 MB."));
        req.destroy();
      } else parts.push(b);
    });
    req.on("end", () => {
      if (!size) return reject(new TTSError("TEXT_INVALID", "Send a JSON body of up to 1 MB."));
      try {
        const body = JSON.parse(Buffer.concat(parts).toString("utf8"));
        if (!body || typeof body !== "object" || Array.isArray(body)) reject(new TTSError("TEXT_INVALID", "Send a JSON object."));
        else resolve(body);
      } catch {
        reject(new TTSError("TEXT_INVALID", "Send valid JSON."));
      }
    });
    req.on("error", reject);
  });
}

/** Creates the server (not yet listening). */
export function createServer(options: ServeOptions = {}): http.Server {
  const tts = options.tts ?? new TTS();
  const apiKey = options.apiKey ?? process.env.LOUDNESS_TTS_API_KEY ?? "";
  const cors = options.cors ?? "";
  const slots = new Slots(options.workers ?? 1);
  const queueSeconds = options.queueSeconds ?? 120;
  const timeout = options.timeout ?? 300;

  const headers = (extra: Record<string, string> = {}): Record<string, string> => ({
    "Cache-Control": "no-store", ...(cors ? { "Access-Control-Allow-Origin": cors, "Access-Control-Expose-Headers": "X-Voice-Id, X-Voice-Credit" } : {}),
    ...extra });
  const json = (res: http.ServerResponse, status: number, payload: unknown): void => {
    const body = Buffer.from(JSON.stringify(payload));
    res.writeHead(status, headers({ "Content-Type": "application/json", "Content-Length": String(body.length) }));
    res.end(body);
  };
  const error = (res: http.ServerResponse, status: number, code: string, message: string): void =>
    json(res, status, { error: { message, type: status < 500 ? "invalid_request_error" : "server_error", code } });
  const authorized = (req: http.IncomingMessage): boolean => {
    if (!apiKey) return true;
    const got = Buffer.from((req.headers.authorization ?? "").replace(/^Bearer /, "").trim());
    const want = Buffer.from(apiKey);
    return got.length === want.length && timingSafeEqual(got, want);
  };
  const speakOptions = (body: Record<string, unknown>): Record<string, unknown> => {
    let speed = body.speed as number | undefined;
    if (speed !== undefined && speed !== null) {
      speed = Number(speed);
      if (Number.isNaN(speed)) throw new TTSError("OPTION_INVALID", "speed must be a number.");
      speed = Math.min(2.0, Math.max(0.5, speed)); // OpenAI allows 0.25-4; Piper sounds right within 0.5-2
    }
    const extra = Object.fromEntries(["language", "pause", "expressiveness", "rhythm"]
      .filter((k) => body[k] !== undefined && body[k] !== null).map((k) => [k, body[k]]));
    return { voice: body.voice || undefined, speed: speed ?? 1.0, ...extra };
  };

  return http.createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    try {
      if (req.method === "OPTIONS") {
        res.writeHead(204, headers({ "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Authorization, Content-Type", "Content-Length": "0" }));
        return void res.end();
      }
      if (req.method === "GET") {
        if (url.pathname === "/health") return json(res, 200, { status: "ok", version: VERSION, ffmpeg: !!ffmpegPath() });
        if (!authorized(req)) return error(res, 401, "UNAUTHORIZED", "Missing or wrong API key.");
        if (url.pathname === "/v1/models") {
          return json(res, 200, { object: "list", data: ["loudness-tts", "tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
            .map((id) => ({ id, object: "model", owned_by: "loudness-tts" })) });
        }
        if (url.pathname === "/v1/audio/voices" || url.pathname === "/v1/voices") {
          const found = voices({ language: url.searchParams.get("language"), sex: url.searchParams.get("sex") });
          const limit = Number(url.searchParams.get("limit") ?? 100);
          return json(res, 200, { object: "list", data: found.slice(0, limit), total: found.length });
        }
        return error(res, 404, "NOT_FOUND", "Not found.");
      }
      if (req.method !== "POST" || !["/v1/audio/speech", "/v1/tts"].includes(url.pathname)) return error(res, 404, "NOT_FOUND", "Not found.");
      if (!authorized(req)) return error(res, 401, "UNAUTHORIZED", "Missing or wrong API key.");
      const body = await readBody(req);
      const text = (body.input ?? body.text) as string;
      const opts = speakOptions(body);
      const voice = findVoice((opts.voice as string) || "auto", (opts.language as string) ?? null);
      opts.voice = voice.id;
      if (!(await slots.acquire(queueSeconds))) return error(res, 429, "BUSY", "The speech server is busy. Try again shortly.");
      try {
        const signal = AbortSignal.timeout(timeout * 1000);
        const extra: Record<string, string> = { "X-Voice-Id": voice.id };
        if (voice.credit) extra["X-Voice-Credit"] = ascii(voice.credit);
        if (url.pathname === "/v1/tts") {
          const speech = await tts.synthesize(text, { ...opts, signal });
          return json(res, 200, { audio: Buffer.from(speech.wav()).toString("base64"), format: "wav",
            sample_rate: speech.sampleRate, duration: Math.round(speech.duration * 1000) / 1000, voice,
            timings: speech.timings, srt: speech.srt(), vtt: speech.vtt(), metrics: speech.metrics });
        }
        const format = (body.response_format as string) || (ffmpegPath() ? "mp3" : "wav");
        if (format === "pcm") {
          const chunks = tts.stream(text, { ...opts, signal });
          const first = await chunks.next(); // errors surface before any audio is sent
          res.writeHead(200, headers({ "Content-Type": "audio/pcm", ...extra }));
          let item = first;
          while (!item.done) {
            const data = pcm16(resample(item.value.samples, item.value.sampleRate, PCM_RATE));
            if (data.length) res.write(data);
            item = await chunks.next();
          }
          return void res.end();
        }
        const speech = await tts.synthesize(text, { ...opts, signal });
        const data = await encode(speech.samples, speech.sampleRate, format);
        res.writeHead(200, headers({ "Content-Type": CONTENT_TYPES[format], "Content-Length": String(data.length), ...extra }));
        res.end(data);
      } finally {
        slots.release();
      }
    } catch (e) {
      if (res.headersSent) return void res.destroy();
      if (e instanceof TTSError) error(res, STATUS[e.code] ?? 500, e.code, e.message);
      else error(res, 500, "ENGINE_FAILED", "Speech generation failed."); // never leak internals to clients
    }
  });
}

/** Starts the server and resolves once it is listening. */
export async function serve(options: ServeOptions = {}): Promise<http.Server> {
  const host = options.host ?? "127.0.0.1";
  const port = options.port ?? 8880;
  const tts = options.tts ?? new TTS();
  if (options.preload ?? true) await tts.preload();
  const server = createServer({ ...options, tts });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => resolve());
  });
  if (!options.quiet) {
    const address = server.address();
    const shown = typeof address === "object" && address ? address.port : port;
    console.log(`loudness-tts ${VERSION} serving on http://${host}:${shown}  (OpenAI-compatible: /v1/audio/speech)`);
  }
  return server;
}
