// SPDX-License-Identifier: GPL-3.0-or-later
// Node, Bun and Deno: voices in a cache folder shared with the Python edition, espeak-ng data from this package, and
// file output. Importing this module installs them.
import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, readdir, readFile, rename, rm, stat, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";
import { allModels } from "./catalog.js";
import { TTSError } from "./errors.js";
import { setPlatform, type FetchOptions, type FileSpec, type VoiceStore } from "./platform.js";
import { VERSION } from "./version.js";


/** $LOUDNESS_TTS_HOME, else $XDG_CACHE_HOME/loudness-tts, else ~/.cache/loudness-tts (Windows: %LOCALAPPDATA%\loudness-tts). */
export function cacheDir(override?: string): string {
  if (override) return override.replace(/^~(?=$|[\\/])/, os.homedir());
  const env = process.env;
  if (env.LOUDNESS_TTS_HOME) return env.LOUDNESS_TTS_HOME.replace(/^~(?=$|[\\/])/, os.homedir());
  if (process.platform === "win32" && env.LOCALAPPDATA) return path.join(env.LOCALAPPDATA, "loudness-tts");
  return path.join(env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache"), "loudness-tts");
}

const digest = (spec: FileSpec): [string, string] => (spec.sha256 ? ["sha256", spec.sha256] : ["md5", spec.md5 ?? ""]);

async function verified(file: string, spec: FileSpec): Promise<boolean> {
  const [algo, want] = digest(spec);
  try {
    const [info, receipt] = await Promise.all([stat(file), readFile(file + ".ok", "utf8")]);
    return info.size === spec.size && receipt === `${algo}:${want}`;
  } catch {
    return false;
  }
}

async function fetchTo(spec: FileSpec, target: string, label: string, options: FetchOptions): Promise<void> {
  const [algo, want] = digest(spec);
  await mkdir(path.dirname(target), { recursive: true });
  let lastError: unknown;
  for (let attempt = 0; attempt < 3; attempt++) {
    const tmp = path.join(path.dirname(target), `.${path.basename(target)}.${process.pid}.${attempt}.tmp`);
    try {
      const response = await fetch(spec.url, { headers: { "User-Agent": `loudness-tts/${VERSION}` }, signal: options.signal });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      const hash = createHash(algo);
      const out = createWriteStream(tmp);
      let done = 0;
      try {
        for await (const block of Readable.fromWeb(response.body as never) as AsyncIterable<Buffer>) {
          done += block.length;
          if (done > spec.size) throw new TTSError("CHECKSUM_MISMATCH", `${label} is larger than expected; download refused.`);
          hash.update(block);
          if (!out.write(block)) await new Promise<void>((resolve) => out.once("drain", () => resolve()));
          options.progress?.({ file: label, bytes: done, total: spec.size });
        }
      } finally {
        await new Promise<void>((resolve) => out.end(() => resolve()));
      }
      if (done !== spec.size || hash.digest("hex") !== want) {
        throw new TTSError("CHECKSUM_MISMATCH", `${label} does not match its published checksum; it was not used.`);
      }
      await rename(tmp, target);
      await writeFile(target + ".ok", `${algo}:${want}`);
      return;
    } catch (error) {
      if (error instanceof TTSError || options.signal?.aborted) throw error;
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 1500 * (attempt + 1)));
    } finally {
      await unlink(tmp).catch(() => undefined);
    }
  }
  throw new TTSError("DOWNLOAD_FAILED", `Could not download ${label}: ${String(lastError)}`);
}

/** Voice files in the cache folder, in the same layout as the Python edition (so both share downloads). */
export class NodeStore implements VoiceStore {
  async file(model: string, name: string, spec: FileSpec, options: FetchOptions): Promise<Uint8Array> {
    const file = path.join(cacheDir(options.root), "voices", model, name);
    if (!(await verified(file, spec))) {
      if (options.offline || process.env.LOUDNESS_TTS_OFFLINE === "1") {
        throw new TTSError("OFFLINE", `${model} is not downloaded yet and downloads are turned off.`);
      }
      await fetchTo(spec, file, name, options);
    }
    return new Uint8Array(await readFile(file));
  }
  async list(models: Record<string, { files: Record<string, FileSpec> }> = allModels()): Promise<string[]> {
    const folder = path.join(cacheDir(), "voices");
    const names = await readdir(folder).catch(() => [] as string[]);
    const out: string[] = [];
    for (const model of names.sort()) {
      const info = models[model];
      if (!info) continue;
      const checks = await Promise.all([verified(path.join(folder, model, `${model}.onnx`), info.files.model),
        verified(path.join(folder, model, `${model}.onnx.json`), info.files.config)]);
      if (checks.every(Boolean)) out.push(model);
    }
    return out;
  }
  async remove(model: string): Promise<boolean> {
    const folder = path.join(cacheDir(), "voices", path.basename(model));
    try {
      await stat(folder);
    } catch {
      return false;
    }
    await rm(folder, { recursive: true, force: true });
    return true;
  }
  location(): string {
    return cacheDir();
  }
}

const DATA = fileURLToPath(new URL("../espeak-ng-data/", import.meta.url));

setPlatform({
  node: true,
  espeakFile: async (name) => new Uint8Array(await readFile(path.join(DATA, path.basename(name)))),
  store: new NodeStore(),
  cpus: () => (typeof os.availableParallelism === "function" ? os.availableParallelism() : os.cpus().length),
  writeFile: (file, data) => writeFile(file, data),
  readFile: async (file) => new Uint8Array(await readFile(file)),
});

/** Voices whose files are downloaded and verified. */
export const downloadedModels = (): Promise<string[]> => new NodeStore().list(allModels());
/** Deletes a downloaded voice. */
export const removeModel = (model: string): Promise<boolean> => new NodeStore().remove(model);
