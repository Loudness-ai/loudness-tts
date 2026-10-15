// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Asad Shehzad / Loudness.ai.
// Python string and regular-expression semantics in JavaScript, so the text preparation rules (written for Python's
// `re`) behave identically here: Unicode \w, \d, \s and \b, Python's `.` and `$`, and str methods such as isupper().

const W = "\\p{L}\\p{N}_";
/** Python's whitespace (str.isspace() and re's \s). */
const S = "\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const WORD = `[${W}]`;
const BOUNDARY = `(?:(?<=${WORD})(?!${WORD})|(?<!${WORD})(?=${WORD}))`;

/** Translates a Python `re` pattern into a JavaScript one (used with the `u` flag). */
export function translate(pattern: string): string {
  pattern = pattern.split("[^\\W\\d_]").join("[\\p{L}\\p{Nl}\\p{No}]"); // a letter
  let out = "";
  let inClass = false;
  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i];
    if (ch === "\\") {
      const next = pattern[i + 1];
      i++;
      if (inClass) {
        if (next === "w") out += W;
        else if (next === "d") out += "\\p{Nd}";
        else if (next === "s") out += S;
        else if (next === "W" || next === "D" || next === "S") throw new Error(`unsupported \\${next} in a class: ${pattern}`);
        else out += "\\" + next;
      } else if (next === "w") out += WORD;
      else if (next === "W") out += `[^${W}]`;
      else if (next === "d") out += "\\p{Nd}";
      else if (next === "D") out += "\\P{Nd}";
      else if (next === "s") out += `[${S}]`;
      else if (next === "S") out += `[^${S}]`;
      else if (next === "b") out += BOUNDARY;
      else out += "\\" + next;
    } else if (inClass) {
      if (ch === "]") inClass = false;
      out += ch;
    } else if (ch === "[") {
      inClass = true;
      out += ch;
      if (pattern[i + 1] === "^") out += pattern[++i];
    } else if (ch === ".") out += "[^\\n]";
    else if (ch === "$") out += "(?=\\n?(?![\\s\\S]))";
    else out += ch;
  }
  return out;
}

/** A compiled Python-style pattern. */
export class Re {
  readonly source: string;
  private readonly flags: string;
  private readonly sticky: RegExp;
  constructor(pattern: string, ignoreCase = false) {
    this.source = translate(pattern);
    this.flags = ignoreCase ? "iu" : "u";
    this.sticky = new RegExp(this.source, "y" + this.flags);
  }
  finditer(text: string): IterableIterator<RegExpExecArray> {
    return text.matchAll(new RegExp(this.source, "g" + this.flags)) as IterableIterator<RegExpExecArray>;
  }
  /** Python's pattern.match(text, pos): a match starting exactly at pos. */
  match(text: string, pos = 0): RegExpExecArray | null {
    this.sticky.lastIndex = pos;
    return this.sticky.exec(text);
  }
  search(text: string): RegExpExecArray | null {
    return new RegExp(this.source, this.flags).exec(text);
  }
  fullmatch(text: string): RegExpExecArray | null {
    return new RegExp(`^(?:${this.source})(?![\\s\\S])`, this.flags).exec(text);
  }
  sub(text: string, replace: string | ((m: RegExpExecArray) => string)): string {
    let out = "";
    let last = 0;
    for (const m of this.finditer(text)) {
      out += text.slice(last, m.index) + (typeof replace === "string" ? replace : replace(m));
      last = m.index + m[0].length;
    }
    return out + text.slice(last);
  }
  findall(text: string): string[] {
    return Array.from(this.finditer(text), (m) => (m.length > 1 ? m[1] : m[0]));
  }
}

const cache = new Map<string, Re>();
/** A cached compiled pattern (rules compile their patterns on every line, as the Python code does). */
export function re(pattern: string, ignoreCase = false): Re {
  const key = (ignoreCase ? "i:" : "s:") + pattern;
  let compiled = cache.get(key);
  if (!compiled) {
    compiled = new Re(pattern, ignoreCase);
    cache.set(key, compiled);
  }
  return compiled;
}

/** Python's re.escape for the characters that matter in a JavaScript pattern. */
export const escape = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export const start = (m: RegExpExecArray): number => m.index;
export const end = (m: RegExpExecArray): number => m.index + m[0].length;

// ---------------------------------------------------------------- str methods
const SPACE = new RegExp(`^[${S}]$`, "u");
const LEADING = new RegExp(`^[${S}]+`, "u");
const TRAILING = new RegExp(`[${S}]+$`, "u");
const ALNUM = /^[\p{L}\p{N}]$/u;
const LOWER = /^[\p{Lowercase}]$/u;
const UPPER = /^[\p{Uppercase}]$/u;
const TITLE = /^\p{Lt}$/u;
const PUNCT = /^\p{P}$/u;

export const isspace = (ch: string): boolean => ch !== "" && Array.from(ch).every((c) => SPACE.test(c));
export const isalnumChar = (ch: string): boolean => ALNUM.test(ch);
export const isPunctuation = (ch: string): boolean => PUNCT.test(ch);
export const strip = (s: string): string => s.replace(LEADING, "").replace(TRAILING, "");
export const lstrip = (s: string): string => s.replace(LEADING, "");
export function rstrip(s: string, chars?: string): string {
  if (chars === undefined) return s.replace(TRAILING, "");
  const set = new Set(Array.from(chars));
  const cps = Array.from(s);
  let n = cps.length;
  while (n && set.has(cps[n - 1])) n--;
  return cps.slice(0, n).join("");
}
/** Python's str.isupper(): at least one cased character and none lowercase or titlecase. */
export function isupper(s: string): boolean {
  let cased = false;
  for (const c of s) {
    if (LOWER.test(c) || TITLE.test(c)) return false;
    if (UPPER.test(c)) cased = true;
  }
  return cased;
}
export function islower(s: string): boolean {
  let cased = false;
  for (const c of s) {
    if (UPPER.test(c) || TITLE.test(c)) return false;
    if (LOWER.test(c)) cased = true;
  }
  return cased;
}
/** Python's str.title() for the words it is used on. */
export function title(s: string): string {
  let out = "";
  let previousCased = false;
  for (const c of s) {
    out += previousCased ? c.toLowerCase() : c.toUpperCase();
    previousCased = LOWER.test(c) || UPPER.test(c) || TITLE.test(c);
  }
  return out;
}
/** First code point (Python's s[:1]) and the code point at k. */
export const first = (s: string): string => (s ? String.fromCodePoint(s.codePointAt(0)!) : "");
export const codePointAt = (s: string, k: number): string => Array.from(s)[k] ?? "";
export const lastChar = (s: string): string => (s ? Array.from(s).pop()! : "");
export const cpLength = (s: string): number => Array.from(s).length;

// Decimal digits of every script, like Python's int(): the zero of each run of ten.
const ZEROS = [0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66, 0xde6, 0xe50,
  0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90, 0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620,
  0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10, 0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0,
  0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50,
  0x16a60, 0x16ac0, 0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0];

export function asciiDigits(s: string): string {
  let out = "";
  for (const c of s) {
    const cp = c.codePointAt(0)!;
    if (cp >= 0x30 && cp <= 0x39) {
      out += c;
      continue;
    }
    let zero = 0;
    for (const z of ZEROS) if (z <= cp) zero = z;
    const value = cp - zero;
    if (!(value >= 0 && value <= 9)) throw new Error(`not a digit: ${c}`);
    out += String(value);
  }
  return out;
}
/** Python's int() on a string of decimal digits. */
export const int = (s: string): number => parseInt(asciiDigits(s), 10);
export const bigint = (s: string): bigint => BigInt(asciiDigits(s));
