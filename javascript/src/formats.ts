// SPDX-License-Identifier: GPL-3.0-or-later
// Output formats (Node). WAV and raw PCM are built in; MP3, Opus, AAC and FLAC use an FFmpeg found on the PATH.
import { spawn } from "node:child_process";
import { accessSync, constants } from "node:fs";
import path from "node:path";
import { TTSError } from "./errors.js";
import { pcm16, resample, wavBytes } from "./speech.js";

export const CONTENT_TYPES: Record<string, string> = { wav: "audio/wav", pcm: "audio/pcm", mp3: "audio/mpeg",
  opus: "audio/ogg", aac: "audio/aac", flac: "audio/flac" };
const FFMPEG_ARGS: Record<string, string[]> = {
  mp3: ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
  opus: ["-c:a", "libopus", "-b:a", "48k", "-f", "ogg"],
  aac: ["-c:a", "aac", "-b:a", "96k", "-f", "adts"],
  flac: ["-c:a", "flac", "-f", "flac"],
};
/** As OpenAI's speech API: 24 kHz, 16-bit, little-endian, mono. */
export const PCM_RATE = 24000;

let found: string | null | undefined;
export function ffmpegPath(): string | null {
  if (found !== undefined) return found;
  const names = process.platform === "win32" ? ["ffmpeg.exe", "ffmpeg"] : ["ffmpeg"];
  for (const dir of (process.env.PATH ?? "").split(path.delimiter)) {
    for (const name of names) {
      try {
        accessSync(path.join(dir, name), constants.X_OK);
        return (found = path.join(dir, name));
      } catch {
        // keep looking
      }
    }
  }
  return (found = null);
}

export async function encode(samples: Float32Array, rate: number, format: string): Promise<Uint8Array> {
  if (format === "wav") return wavBytes(samples, rate);
  if (format === "pcm") return pcm16(resample(samples, rate, PCM_RATE));
  if (!(format in FFMPEG_ARGS)) {
    throw new TTSError("FORMAT_UNAVAILABLE", `Unknown format "${format}": use wav, pcm, mp3, opus, aac or flac.`);
  }
  const ffmpeg = ffmpegPath();
  if (!ffmpeg) throw new TTSError("FORMAT_UNAVAILABLE", `${format} needs FFmpeg on the PATH; use wav or pcm, or install FFmpeg.`);
  return new Promise((resolve, reject) => {
    const child = spawn(ffmpeg, ["-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", String(rate), "-ac", "1",
      "-i", "pipe:0", ...FFMPEG_ARGS[format], "pipe:1"], { stdio: ["pipe", "pipe", "ignore"] });
    const out: Buffer[] = [];
    child.stdout.on("data", (b: Buffer) => out.push(b));
    child.on("error", () => reject(new TTSError("FORMAT_UNAVAILABLE", `FFmpeg could not encode ${format}.`)));
    child.on("close", (code) => {
      const data = Buffer.concat(out);
      if (code !== 0 || !data.length) reject(new TTSError("FORMAT_UNAVAILABLE", `FFmpeg could not encode ${format}.`));
      else resolve(new Uint8Array(data));
    });
    child.stdin.on("error", () => undefined);
    child.stdin.end(pcm16(samples));
  });
}
