// SPDX-License-Identifier: GPL-3.0-or-later
// Text to phonemes with espeak-ng, compiled to WebAssembly from the same espeak-ng commit and with the same clause
// handling as Piper (piper1-gpl 1.8: espeakbridge.c and phonemize_espeak.py), so the phonemes are identical to the
// ones Piper and the Python edition produce. espeak-ng is GPL-3.0-or-later.
import createEspeak, { type EspeakModule } from "./wasm/espeak.mjs";
import { TTSError } from "./errors.js";
import { platform } from "./platform.js";
import { strip } from "./pyre.js";

const ROOT = "/espeak-ng-data";
const BASE = ["phontab", "phonindex", "phondata", "intonations"];

interface LangFile {
  path: string;
  name: string;
  dictionary: string;
}

/** The espeak-ng language files: which dictionary each voice name needs (espeak's own rules in voices.c). */
function indexLanguages(files: Record<string, string>): LangFile[] {
  return Object.entries(files).map(([path, text]) => {
    let name = "";
    let dictionary = "";
    let language = "";
    for (const raw of text.split("\n")) {
      const [key, value] = raw.trim().split(/\s+/, 2);
      if (key === "name" && !name) name = raw.trim().slice(5).trim();
      else if (key === "language" && !language && value) language = value.split("-")[0];
      else if (key === "dictionary" && value) dictionary = value;
    }
    return { path, name, dictionary: dictionary || language };
  });
}

export class Espeak {
  private dictionaries = new Set<string>();
  private current = "";

  private constructor(private readonly m: EspeakModule, private readonly langs: LangFile[]) {}

  static async create(): Promise<Espeak> {
    const read = platform().espeakFile;
    const [m, base, langJson, en] = await Promise.all([
      createEspeak(), Promise.all(BASE.map((name) => read(name))), read("lang.json"), read("en_dict")]);
    const files = JSON.parse(new TextDecoder().decode(langJson)) as Record<string, string>;
    m.FS.mkdir(ROOT);
    BASE.forEach((name, i) => m.FS.writeFile(`${ROOT}/${name}`, base[i]));
    for (const [path, text] of Object.entries(files)) {
      let dir = ROOT;
      for (const part of ["lang", ...path.split("/").slice(0, -1)]) {
        dir += "/" + part;
        if (!m.FS.analyzePath(dir).exists) m.FS.mkdir(dir);
      }
      m.FS.writeFile(`${ROOT}/lang/${path}`, text);
    }
    m.FS.writeFile(`${ROOT}/en_dict`, en);
    const espeak = new Espeak(m, indexLanguages(files));
    espeak.dictionaries.add("en");
    if (espeak.call(m._lt_initialize, ROOT) < 0) throw new TTSError("ENGINE_FAILED", "espeak-ng could not start.");
    return espeak;
  }

  private call(fn: (ptr: number) => number, text: string): number {
    const ptr = this.m.stringToNewUTF8(text);
    try {
      return fn(ptr);
    } finally {
      this.m._free(ptr);
    }
  }

  /** Loads what a voice needs (its dictionary) and selects it. */
  async use(voice: string): Promise<void> {
    if (voice === this.current) return;
    const wanted = voice.split("+")[0].toLowerCase();
    const file = this.langs.find((l) => l.name.toLowerCase() === wanted)
      ?? this.langs.find((l) => l.path.split("/").pop()!.toLowerCase() === wanted);
    if (file && !this.dictionaries.has(file.dictionary)) {
      const dict = await platform().espeakFile(`${file.dictionary}_dict`).catch(() => null);
      if (dict) this.m.FS.writeFile(`${ROOT}/${file.dictionary}_dict`, dict);
      this.dictionaries.add(file.dictionary);
    }
    if (this.call(this.m._lt_set_voice, voice) !== 0) {
      this.current = "";
      throw new TTSError("ENGINE_FAILED", `espeak-ng has no voice "${voice}".`);
    }
    this.current = voice;
  }

  /** Piper's EspeakPhonemizer.phonemize: phonemes per sentence (each phoneme one code point, accents separate). The
   * voice must have been selected with use(). */
  phonemize(text: string, vowelClusters?: Set<string>): string[][] {
    const ptr = this.call(this.m._lt_phonemes, text);
    if (!ptr) throw new TTSError("ENGINE_FAILED", "espeak-ng ran out of memory.");
    const raw = this.m.UTF8ToString(ptr);
    this.m._free(ptr);
    const all: string[][] = [];
    let sentence: string[] = [];
    for (const record of raw.split("\x1e")) {
      const [phonemes, terminator, end] = record.split("\x1f");
      // Filter out (lang) switch flags around words from other languages.
      let piece = phonemes.replace(/\([^)]+\)/g, "") + terminator;
      if (terminator === "," || terminator === ":" || terminator === ";") piece += " ";
      sentence.push(...Array.from(piece.normalize("NFD")));
      if (end === "1") {
        all.push(vowelClusters ? mergeClusters(sentence, vowelClusters) : sentence);
        sentence = [];
      }
    }
    if (sentence.length) all.push(vowelClusters ? mergeClusters(sentence, vowelClusters) : sentence);
    return all;
  }
}

/** Piper's _merge_known_vowel_clusters (clusters given as phonemes joined with \u0000). */
function mergeClusters(phones: string[], clusters: Set<string>): string[] {
  const maxLen = Math.max(...[...clusters].map((k) => k.split("\0").length));
  const out: string[] = [];
  let i = 0;
  while (i < phones.length) {
    let match: string[] | null = null;
    for (let n = Math.min(maxLen, phones.length - i); n > 1; n--) {
      const candidate = phones.slice(i, i + n);
      if (clusters.has(candidate.join("\0"))) {
        match = candidate;
        break;
      }
    }
    if (match) {
      out.push(match.join(""));
      i += match.length;
    } else {
      out.push(phones[i]);
      i += 1;
    }
  }
  return out;
}

let shared: Promise<Espeak> | null = null;
/** One espeak-ng instance per process or page (it keeps global state; JavaScript runs it one call at a time). */
export function espeak(): Promise<Espeak> {
  if (!shared) {
    shared = Espeak.create();
    shared.catch(() => {
      shared = null;
    });
  }
  return shared;
}

/** What phonemization needs from a Piper voice config. */
export interface PhonemeConfig {
  espeak: { voice: string };
  phoneme_type?: string;
}

/** Piper's PiperVoice.phonemize: phonemes per sentence, with [[raw phoneme]] blocks passed through. */
export async function phonemizeForVoice(config: PhonemeConfig, text: string, vowelClusters?: Set<string>): Promise<string[][]> {
  if (config.phoneme_type === "text") return [Array.from(text.normalize("NFD"))];
  if (config.phoneme_type && config.phoneme_type !== "espeak") {
    throw new TTSError("ENGINE_FAILED", `Voices with ${config.phoneme_type} phonemes are not supported in JavaScript yet.`);
  }
  const engine = await espeak();
  await engine.use(config.espeak.voice);
  const phonemes: string[][] = [];
  const parts = text.split(/(\[\[.*?\]\])/);
  let previousRaw = false;
  parts.forEach((part, i) => {
    if (part.startsWith("[[")) {
      previousRaw = true;
      if (!phonemes.length) phonemes.push([]);
      const last = phonemes[phonemes.length - 1];
      if (i > 0 && parts[i - 1].endsWith(" ")) last.push(" ");
      last.push(...Array.from(strip(part.slice(2, -2))));
      if (i < parts.length - 1 && parts[i + 1].startsWith(" ")) last.push(" ");
      return;
    }
    let partPhonemes = engine.phonemize(part, vowelClusters);
    if (previousRaw && partPhonemes.length) {
      phonemes[phonemes.length - 1].push(...partPhonemes[0]);
      partPhonemes = partPhonemes.slice(1);
    }
    phonemes.push(...partPhonemes);
    previousRaw = false;
  });
  if (phonemes.length && !phonemes[phonemes.length - 1].length) phonemes.pop();
  return phonemes;
}
