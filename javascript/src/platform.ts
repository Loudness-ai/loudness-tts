// SPDX-License-Identifier: GPL-3.0-or-later
// What differs between Node (and Bun, Deno) and browsers: where voice files and espeak-ng data live, and file output.
// The browser implementation is the default; the Node entry point (node.ts) installs the Node one.
import { LocalTTSError } from "./errors.js";
import { Md5 } from "./md5.js";
import { ORT_VERSION, VERSION } from "./version.js";

export interface FileSpec {
  url: string;
  size: number;
  md5?: string;
  sha256?: string;
}

export interface Progress {
  file: string;
  bytes: number;
  total: number;
}

export interface FetchOptions {
  offline?: boolean;
  /** Where Node keeps voices (default: $LOCAL_TTS_HOME or the user cache folder). */
  root?: string;
  progress?: (p: Progress) => void;
  signal?: AbortSignal;
}

/** Storage for voice files: downloaded once, checked against the catalogue, then used offline. */
export interface VoiceStore {
  /** The verified bytes of one file of a model, downloading it first when needed. */
  file(model: string, name: string, spec: FileSpec, options: FetchOptions): Promise<Uint8Array>;
  /** Models whose files are all stored and verified. */
  list(models: Record<string, { files: Record<string, FileSpec> }>): Promise<string[]>;
  remove(model: string): Promise<boolean>;
  location(): string;
}

export interface Platform {
  node: boolean;
  /** Reads a file of the bundled espeak-ng data (e.g. "phondata", "en_dict", "lang.json"). */
  espeakFile(name: string): Promise<Uint8Array>;
  store: VoiceStore;
  cpus(): number;
  writeFile?(path: string, data: Uint8Array | string): Promise<void>;
  readFile?(path: string): Promise<Uint8Array>;
}

export const sha256Hex = async (data: Uint8Array): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", data as BufferSource)), (b) => b.toString(16).padStart(2, "0")).join("");

export async function checksum(spec: FileSpec, data: Uint8Array): Promise<[string, string, string]> {
  if (spec.sha256) return ["sha256", spec.sha256, await sha256Hex(data)];
  return ["md5", spec.md5 ?? "", new Md5().update(data).hex()];
}

/** Downloads a file into memory, refusing anything larger than the catalogue says, with three attempts. */
export async function download(spec: FileSpec, label: string, options: FetchOptions): Promise<Uint8Array> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetch(spec.url, { signal: options.signal });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      const out = new Uint8Array(spec.size);
      let done = 0;
      const reader = response.body.getReader();
      for (;;) {
        const { done: finished, value } = await reader.read();
        if (finished) break;
        if (done + value.length > spec.size) {
          await reader.cancel();
          throw new LocalTTSError("CHECKSUM_MISMATCH", `${label} is larger than expected; download refused.`);
        }
        out.set(value, done);
        done += value.length;
        options.progress?.({ file: label, bytes: done, total: spec.size });
      }
      const [, want, got] = await checksum(spec, out);
      if (done !== spec.size || got !== want) {
        throw new LocalTTSError("CHECKSUM_MISMATCH", `${label} does not match its published checksum; it was not used.`);
      }
      return out;
    } catch (error) {
      if (error instanceof LocalTTSError || options.signal?.aborted) throw error;
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 1500 * (attempt + 1)));
    }
  }
  throw new LocalTTSError("DOWNLOAD_FAILED", `Could not download ${label}: ${String(lastError)}`);
}

// ---------------------------------------------------------------- browsers
const CACHE = "local-tts-v1";
const MARK = "x-local-tts-checksum";

const openCache = async (): Promise<Cache | null> => {
  try {
    return typeof caches === "undefined" ? null : await caches.open(CACHE);
  } catch {
    return null; // private windows and some embedded views have no Cache Storage
  }
};

/** Voice files kept in the browser's Cache Storage (or only in memory where it is not available). */
export class BrowserStore implements VoiceStore {
  private memory = new Map<string, Uint8Array>();
  async file(model: string, name: string, spec: FileSpec, options: FetchOptions): Promise<Uint8Array> {
    const want = `${spec.sha256 ? "sha256" : "md5"}:${spec.sha256 ?? spec.md5}`;
    const held = this.memory.get(spec.url);
    if (held) return held;
    const cache = await openCache();
    const hit = await cache?.match(spec.url);
    if (hit && hit.headers.get(MARK) === want) {
      const data = new Uint8Array(await hit.arrayBuffer());
      if (data.length === spec.size) return data;
    }
    if (options.offline) throw new LocalTTSError("OFFLINE", `${model} is not downloaded yet and downloads are turned off.`);
    const data = await download(spec, name, options);
    await cache?.put(spec.url, new Response(data as BodyInit, { headers: { [MARK]: want, "content-type": "application/octet-stream" } }))
      .catch(() => undefined); // storage full: still usable for this page
    if (!cache) this.memory.set(spec.url, data);
    return data;
  }
  async list(models: Record<string, { files: Record<string, FileSpec> }>): Promise<string[]> {
    const cache = await openCache();
    if (!cache) return [];
    const stored = new Set((await cache.keys()).map((r) => r.url));
    return Object.keys(models).filter((m) => Object.values(models[m].files).every((f) => stored.has(f.url)));
  }
  async remove(model: string): Promise<boolean> {
    const cache = await openCache();
    if (!cache) return false;
    let removed = false;
    for (const request of await cache.keys()) {
      if (decodeURIComponent(request.url).includes(model)) removed = (await cache.delete(request)) || removed;
    }
    return removed;
  }
  location(): string {
    return `Cache Storage "${CACHE}"`;
  }
}

let espeakBase = `https://cdn.jsdelivr.net/npm/local-tts@${VERSION}/espeak-ng-data/`;
let onnxBase = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const folder = (url: string): string => (url.endsWith("/") ? url : url + "/");

export interface BrowserSettings {
  /** Where browsers load espeak-ng's data (default: this version of the npm package on jsDelivr). Copy the package's
   * espeak-ng-data folder to your site to serve it yourself. */
  espeakDataUrl?: string;
  /** Where browsers load ONNX Runtime's WebAssembly files (default: onnxruntime-web on jsDelivr). Copy
   * node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.{wasm,mjs} to serve them yourself. */
  onnxWasmUrl?: string;
}

/** Browser settings; call before the first synthesis. Node, Bun and Deno use the files in the package. */
export function configure(settings: BrowserSettings): void {
  if (settings.espeakDataUrl) espeakBase = folder(settings.espeakDataUrl);
  if (settings.onnxWasmUrl) onnxBase = folder(settings.onnxWasmUrl);
}
export const onnxWasmUrl = (): string => onnxBase;
/** Same as configure({ espeakDataUrl }). */
export const setEspeakDataUrl = (url: string): void => configure({ espeakDataUrl: url });

async function browserEspeakFile(name: string): Promise<Uint8Array> {
  const url = espeakBase + name;
  const cache = await openCache();
  const hit = await cache?.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const response = await fetch(url);
  if (!response.ok) throw new LocalTTSError("DOWNLOAD_FAILED", `Could not load speech data ${name} (HTTP ${response.status}).`);
  const data = new Uint8Array(await response.arrayBuffer());
  await cache?.put(url, new Response(data as BodyInit)).catch(() => undefined);
  return data;
}

let current: Platform = {
  node: false,
  espeakFile: browserEspeakFile,
  store: new BrowserStore(),
  cpus: () => (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 1,
};

export const platform = (): Platform => current;
export function setPlatform(next: Platform): void {
  current = next;
}
