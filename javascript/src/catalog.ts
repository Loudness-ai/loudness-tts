// SPDX-License-Identifier: GPL-3.0-or-later
// The voice catalogue: every commercially usable Piper voice, ranked by naturalness and clarity in Loudness.ai's open
// voice study (LA-RB-2026-003, https://loudness.ai/research/piper-tts-voices-ranked/). Compare scores within one
// language only.
import data from "./voices-data.js";
import { TTSError } from "./errors.js";
import type { FileSpec } from "./platform.js";

export type LicenceUse = "none" | "credit" | "copyleft" | "unclear";
export type Quality = "x_low" | "low" | "medium" | "high";

export interface ModelInfo {
  locale: string;
  quality: Quality;
  sampleRate: number;
  speakers: number;
  licence: string;
  use: LicenceUse;
  dataset: string;
  source: "catalog" | "community";
  card: string;
  note?: string;
  files: { model: FileSpec; config: FileSpec };
}

export interface CatalogData {
  version: number;
  source: string;
  pinnedRevision: string;
  fields: string[];
  voices: unknown[][];
  models: Record<string, ModelInfo>;
}

export interface Voice {
  id: string;
  model: string;
  speaker: number | null;
  name: string;
  /** Language variant as ranked in the study, e.g. "English (United States)". */
  language: string;
  /** BCP 47 tag, e.g. "en-US". */
  locale: string;
  quality: Quality;
  /** "F" or "M" where the training data records the reader, else null. */
  sex: "F" | "M" | null;
  /** 0-100: (natural + clear) / 2, +4 for 22 kHz voices. */
  overall: number;
  /** 0-100: UTMOS22 predicted naturalness. */
  natural: number;
  /** 0-100: 100 minus Whisper large-v3 character error rate. */
  clear: number;
  rank: number;
  of: number;
  licence: string;
  use: LicenceUse;
  dataset: string;
  /** "catalog" (official Piper voices), "community", or "local" for a model file of your own. */
  source: "catalog" | "community" | "local";
  sampleRate: number;
  note: string | null;
  /** A credit line to show where your users can see it, or null when the licence asks for none. */
  credit: string | null;
  /** Its files can be downloaded by a web page (the host allows it). */
  browser: boolean;
}

export const USE_CLASSES: LicenceUse[] = ["none", "credit", "copyleft", "unclear"];
/** Voices picked automatically have a clear licence for commercial use (no conditions, or a credit line). */
export const AUTO_USE: LicenceUse[] = ["none", "credit"];
export const DEFAULT_LANGUAGE = "en-US";
const BROWSER_HOSTS = ["huggingface.co"];

const catalog = data as CatalogData;
let all: Voice[] | null = null;
const BROWSER_MODEL_BYTES = 80e6;

/** The same voices with models under 80 MB first (for web pages), each once. */
function smallFirst(pool: Voice[]): Voice[] {
  const small = pool.filter((v) => catalog.models[v.model].files.model.size <= BROWSER_MODEL_BYTES);
  return small.concat(pool.filter((v) => catalog.models[v.model].files.model.size > BROWSER_MODEL_BYTES));
}

const compare = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

function everyVoice(): Voice[] {
  if (all) return all;
  all = catalog.voices.map((row) => {
    const v = Object.fromEntries(catalog.fields.map((f, i) => [f, row[i]])) as Record<string, never>;
    const m = catalog.models[v.model];
    const credit = m.use === "none" ? null
      : `Voice "${v.name}" (Piper voice ${v.model})${m.dataset ? `, trained on ${m.dataset}` : ""}, ${m.licence}.`;
    return {
      id: v.id, model: v.model, speaker: v.speaker, name: v.name, language: v.language, locale: m.locale,
      quality: m.quality, sex: v.sex, overall: v.overall, natural: v.natural, clear: v.clear, rank: v.rank, of: v.of,
      licence: m.licence, use: m.use, dataset: m.dataset, source: m.source, sampleRate: m.sampleRate,
      note: m.note ?? null, credit,
      browser: Object.values(m.files).every((f) => BROWSER_HOSTS.includes(new URL(f.url).hostname)),
    };
  });
  return all;
}

/** Download files, checksums, licence and card of one model file. */
export function modelInfo(model: string): ModelInfo {
  const info = catalog.models[model];
  if (!info) throw new TTSError("VOICE_NOT_FOUND", `No voice model called "${model}" is in the catalogue.`);
  return info;
}

export const allModels = (): Record<string, ModelInfo> => catalog.models;
export const catalogSource = (): { source: string; pinnedRevision: string } =>
  ({ source: catalog.source, pinnedRevision: catalog.pinnedRevision });

function matchesLanguage(v: Voice, language: string): boolean {
  const want = language.replace(/_/g, "-").toLowerCase();
  const have = v.locale.toLowerCase();
  return have === want || have.split("-")[0] === want || v.language.toLowerCase() === want;
}

export interface VoiceFilter {
  /** A locale ("en-US", "es-MX"), a language code ("es") or a study language name. */
  language?: string | null;
  /** "F"/"M" (or "female"/"male"); voices whose reader is not recorded are left out when given. */
  sex?: string | null;
  /** Licence classes: none, credit, copyleft, unclear. */
  use?: LicenceUse | LicenceUse[] | null;
  quality?: Quality | Quality[] | null;
  source?: "catalog" | "community" | null;
  minScore?: number | null;
  /** Only voices a web page can download. */
  browser?: boolean;
  sort?: "overall" | "natural" | "clear" | "name";
}

/** Voices matching every given filter, best first. */
export function voices(filter: VoiceFilter | string = {}): Voice[] {
  const f: VoiceFilter = typeof filter === "string" ? { language: filter } : filter;
  const uses = f.use ? (Array.isArray(f.use) ? f.use : [f.use]) : null;
  const qualities = f.quality ? (Array.isArray(f.quality) ? f.quality : [f.quality]) : null;
  const sex = f.sex ? f.sex.toUpperCase().slice(0, 1) : null;
  const out = everyVoice().filter((v) => (f.language == null || matchesLanguage(v, f.language))
    && (sex == null || v.sex === sex) && (uses == null || uses.includes(v.use))
    && (qualities == null || qualities.includes(v.quality)) && (f.source == null || v.source === f.source)
    && (f.minScore == null || v.overall >= f.minScore) && (!f.browser || v.browser));
  const sort = f.sort ?? "overall";
  if (sort === "name") out.sort((a, b) => compare(a.name.toLowerCase(), b.name.toLowerCase()));
  else if (sort === "overall" || sort === "natural" || sort === "clear") {
    out.sort((a, b) => b[sort] - a[sort] || compare(a.id, b.id));
  } else throw new TTSError("OPTION_INVALID", "sort must be overall, natural, clear or name.");
  return out;
}

export interface LanguageEntry {
  locale: string;
  language: string;
  voices: number;
  best: Voice;
}

/** Locales with at least one voice, with their best voice and how many voices they have. */
export function languages(): LanguageEntry[] {
  const best = new Map<string, LanguageEntry>();
  for (const v of everyVoice()) {
    let entry = best.get(v.locale);
    if (!entry) best.set(v.locale, (entry = { locale: v.locale, language: v.language, voices: 0, best: v }));
    entry.voices += 1;
    if (v.overall > entry.best.overall) entry.best = v;
  }
  return [...best.values()].sort((a, b) => compare(a.locale, b.locale));
}

/** Voice names used by OpenAI's speech API, mapped to strong English (US) voices so existing clients work unchanged. */
const OPENAI_NAMES = {
  female: ["nova", "shimmer", "coral", "sage", "marin"],
  male: ["echo", "onyx", "ash", "ballad", "verse", "cedar"],
  any: ["alloy", "fable"],
};

export function openaiVoiceMap(browser = false): Record<string, string> {
  let pool = voices({ language: DEFAULT_LANGUAGE, use: AUTO_USE, browser });
  if (browser) pool = smallFirst(pool);
  const picks: Record<string, string> = {};
  for (const [sex, names] of [["F", OPENAI_NAMES.female], ["M", OPENAI_NAMES.male]] as const) {
    const ranked = pool.filter((v) => v.sex === sex);
    names.forEach((name, i) => {
      if (ranked[i]) picks[name] = ranked[i].id;
    });
  }
  OPENAI_NAMES.any.forEach((name, i) => {
    if (pool[i]) picks[name] = pool[i].id;
  });
  return picks;
}

/** Resolves a voice id, a model name (its best speaker), an OpenAI voice name, "auto" or "best:<language>".
 * "auto" picks the best voice with a clear commercial licence for `language` (default en-US). In a browser it picks
 * among voices a web page can download, preferring models under 80 MB (a download a page can afford). */
export function findVoice(voice: string | null = "auto", language: string | null = null, browser = false): Voice {
  if (voice == null || voice === "" || voice === "auto" || voice === "best") return best(language || DEFAULT_LANGUAGE, browser);
  if (voice.startsWith("best:")) return best(voice.slice(5), browser);
  const every = everyVoice();
  const byId = every.find((v) => v.id === voice);
  if (byId) return byId;
  const speakers = every.filter((v) => v.model === voice);
  if (speakers.length) return speakers.reduce((a, b) => (b.rank < a.rank || (b.rank === a.rank && b.id < a.id) ? b : a));
  const mapped = openaiVoiceMap(browser)[voice.toLowerCase()];
  if (mapped) return every.find((v) => v.id === mapped)!;
  throw new TTSError("VOICE_NOT_FOUND", `No voice called "${voice}". Run \`loudness-tts voices\` to see them.`);
}

function best(language: string, browser: boolean): Voice {
  let pool = voices({ language, use: AUTO_USE, browser }).concat(voices({ language, browser }));
  if (browser) pool = smallFirst(pool);
  if (!pool.length) {
    throw new TTSError("VOICE_NOT_FOUND", `No voice for "${language}"${browser ? " that a web page can download" : ""}. `
      + "Run `loudness-tts languages` to see the list.");
  }
  return pool[0];
}
