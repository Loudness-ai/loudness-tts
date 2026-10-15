// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Asad Shehzad / Loudness.ai.
// A line-by-line port of ttsprep (local_tts/textprep.py in the Python edition), the text preparation layer of the
// Loudness.ai speech engine (https://loudness.ai/research/piper-tts-text-normalization/). This file stays under the
// Apache License 2.0; the rest of local-tts is GPL-3.0-or-later. Its output is tested against the Python edition.
//
// Turns what a person typed into what the voice should say. Written forms that the phonemizer (espeak-ng) reads badly
// become words: abbreviations, money, dates, times, decades, units, ranges, phone numbers, fractions and Roman
// numerals (English and Spanish). Emoji, list markers and markdown go. The text is cut into segments, each followed
// by the right pause, and every segment keeps the span of the text it came from (in UTF-16 code units, so
// text.slice(start, end) works), so speech can be timed against what was written.

import {
  bigint, codePointAt, cpLength, end, escape, first, int, isalnumChar, islower, isPunctuation, isspace, isupper,
  lastChar, lstrip, re, rstrip, start, strip, title,
} from "./pyre.js";

export interface Segment {
  /** What the voice reads; may hold [[raw phonemes]]. */
  text: string;
  /** Span of the source text it came from. */
  start: number;
  end: number;
  /** Pause after it: clause, ellipsis, sentence, line, paragraph or "" (last). */
  pause: string;
  /** [word read, source start, source end] for each word of text. */
  tokens: Array<[string, number, number]>;
}

type Match = RegExpExecArray;
type Out = string | [string, string] | null | undefined;
type Dict<T> = Record<string, T>;

// ---------------------------------------------------------------- word lists

const MONTHS: Dict<string[]> = {
  en: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
    "December"],
  es: ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre",
    "diciembre"],
};
const MONTH_ABBR_EN: Dict<number> = { Jan: 1, Feb: 2, Mar: 3, Apr: 4, Jun: 6, Jul: 7, Aug: 8, Sep: 9, Sept: 9, Oct: 10,
  Nov: 11, Dec: 12 };
// Words a long unpunctuated run may be split before (a comma pause goes there).
const CONJUNCTIONS: Dict<Set<string>> = {
  en: new Set(["and", "but", "or", "so", "because", "which", "while", "when", "where", "although", "though", "if", "then",
    "whereas", "unless", "until", "since"]),
  es: new Set(["y", "pero", "o", "porque", "cuando", "donde", "aunque", "mientras", "si", "pues", "sino", "entonces"]),
  de: new Set(["und", "aber", "oder", "weil", "dass", "wenn", "denn", "sondern", "während", "obwohl"]),
  fr: new Set(["et", "mais", "ou", "parce", "quand", "car", "donc", "puis", "lorsque", "pendant"]),
  it: new Set(["e", "ma", "o", "perché", "quando", "mentre", "se", "quindi", "poi"]),
  pt: new Set(["e", "mas", "ou", "porque", "quando", "enquanto", "se", "então", "pois"]),
  nl: new Set(["en", "maar", "of", "omdat", "wanneer", "terwijl", "als", "dus", "want"]),
};
const RANGE_WORD: Dict<string> = { en: "to", es: "a", de: "bis", fr: "à" };
const PLUS_WORD: Dict<string> = { en: "plus", es: "más", de: "plus", fr: "plus", it: "più", pt: "mais", nl: "plus" };
// A single initial before one of these is more likely the end of a sentence.
const SENTENCE_STARTERS: Dict<Set<string>> = {
  en: new Set(["The", "This", "That", "These", "Those", "Then", "There", "They", "It", "We", "He", "She", "I", "You",
    "But", "And", "So", "A", "An", "In", "On", "At", "If", "When", "After", "Before", "Now", "Today", "Our", "My", "Your",
    "His", "Her", "Their", "What", "Why", "How", "Who", "Where", "Yes", "No", "Next", "Also"]),
  es: new Set(["El", "La", "Los", "Las", "Un", "Una", "Y", "Pero", "Luego", "Entonces", "Es", "Hoy", "Mi", "Su", "Sí",
    "No", "En", "Con", "Por", "Para", "Si", "Cuando", "Después", "Antes", "Ahora", "Qué", "Cómo", "Ella", "Él"]),
};
const REGNAL_NAMES: Dict<Set<string>> = {
  en: new Set(["Henry", "Edward", "George", "William", "Elizabeth", "Charles", "Louis", "James", "Richard", "Mary",
    "John", "Paul", "Pius", "Leo", "Benedict", "Gregory", "Innocent", "Alexander", "Peter", "Catherine", "Philip",
    "Frederick", "Ramesses", "Constantine", "Napoleon", "Edmund", "Harold", "Stephen", "Anne", "Victoria", "Nicholas",
    "Ivan", "Francis", "Clement", "Urban", "Boniface", "Sixtus", "Felipe", "Juan", "Carlos", "Alfonso", "Fernando",
    "Isabella", "Ferdinand", "Otto", "Wilhelm", "Gustav", "Olaf", "Haakon"]),
  es: new Set(["Felipe", "Juan", "Pablo", "Carlos", "Alfonso", "Fernando", "Isabel", "Luis", "Enrique", "Pedro", "Jaime",
    "Benedicto", "Pío", "León", "Gregorio", "Clemente", "Inocencio", "Urbano", "Sixto", "Alejandro", "Napoleón",
    "Ricardo", "Eduardo", "Jorge", "Guillermo", "Catalina", "Federico", "María", "Francisco"]),
};
const FEMININE_ES = new Set(["Isabel", "Catalina", "María"]);
const ORDINAL_ES = ["", "primero", "segundo", "tercero", "cuarto", "quinto", "sexto", "séptimo", "octavo", "noveno",
  "décimo"];
const ROMAN_KEYWORDS_EN = ["chapter", "part", "volume", "book", "act", "scene", "section", "phase", "level", "season",
  "episode", "stage", "round", "grade", "class", "type", "title", "article", "appendix", "psalm", "war", "bowl"];
const DECADES_EN: Dict<string> = { 1: "tens", 2: "twenties", 3: "thirties", 4: "forties", 5: "fifties", 6: "sixties",
  7: "seventies", 8: "eighties", 9: "nineties" };
const FRACTIONS: Dict<Dict<string>> = {
  en: { "1/2": "one half", "1/3": "one third", "2/3": "two thirds", "1/4": "one quarter", "3/4": "three quarters",
    "1/8": "one eighth", "3/8": "three eighths", "5/8": "five eighths", "7/8": "seven eighths" },
  es: { "1/2": "un medio", "1/3": "un tercio", "2/3": "dos tercios", "1/4": "un cuarto", "3/4": "tres cuartos",
    "1/8": "un octavo", "3/8": "tres octavos", "5/8": "cinco octavos", "7/8": "siete octavos" },
};
// Currency: [singular, plural, minor singular, minor plural].
const CURRENCY_EN: Dict<string[]> = { "$": ["dollar", "dollars", "cent", "cents"], "€": ["euro", "euros", "cent", "cents"],
  "£": ["pound", "pounds", "penny", "pence"], "¥": ["yen", "yen", "", ""], "₹": ["rupee", "rupees", "paisa", "paise"],
  USD: ["US dollar", "US dollars", "cent", "cents"], EUR: ["euro", "euros", "cent", "cents"],
  GBP: ["pound", "pounds", "penny", "pence"] };
const CURRENCY_ES: Dict<string[]> = { "€": ["euro", "euros", "céntimo", "céntimos"],
  "£": ["libra", "libras", "penique", "peniques"], "¥": ["yen", "yenes", "", ""],
  USD: ["dólar estadounidense", "dólares estadounidenses", "centavo", "centavos"],
  EUR: ["euro", "euros", "céntimo", "céntimos"] };
const DOLLAR_ES = ["dólar", "dólares", "centavo", "centavos"];
const PESO_ES = ["peso", "pesos", "centavo", "centavos"];
const SCALES_EN: Dict<string> = { k: "thousand", thousand: "thousand", m: "million", mn: "million", million: "million",
  b: "billion", bn: "billion", billion: "billion", trillion: "trillion" };
type Unit = [string, string, boolean?];
// Units after a number: [singular, plural]. Matched case-sensitively, longest first.
const UNITS_EN: Dict<Unit> = {
  "km/h": ["kilometer per hour", "kilometers per hour"], kph: ["kilometer per hour", "kilometers per hour"],
  kmph: ["kilometer per hour", "kilometers per hour"],
  mph: ["mile per hour", "miles per hour"], "m/s": ["meter per second", "meters per second"],
  km: ["kilometer", "kilometers"], cm: ["centimeter", "centimeters"], mm: ["millimeter", "millimeters"],
  kg: ["kilogram", "kilograms"], mg: ["milligram", "milligrams"], lbs: ["pound", "pounds"],
  lb: ["pound", "pounds"], oz: ["ounce", "ounces"], mi: ["mile", "miles"], ft: ["foot", "feet"],
  yd: ["yard", "yards"], ml: ["milliliter", "milliliters"], mL: ["milliliter", "milliliters"],
  "°C": ["degree Celsius", "degrees Celsius"], "°F": ["degree Fahrenheit", "degrees Fahrenheit"],
  kWh: ["kilowatt hour", "kilowatt hours"], kW: ["kilowatt", "kilowatts"], kHz: ["kilohertz", "kilohertz"],
  MHz: ["megahertz", "megahertz"], GHz: ["gigahertz", "gigahertz"], Hz: ["hertz", "hertz"],
  dB: ["decibel", "decibels"], KB: ["kilobyte", "kilobytes"], kB: ["kilobyte", "kilobytes"],
  MB: ["megabyte", "megabytes"], GB: ["gigabyte", "gigabytes"], TB: ["terabyte", "terabytes"],
  Mbps: ["megabit per second", "megabits per second"], ms: ["millisecond", "milliseconds"],
  hrs: ["hour", "hours"], hr: ["hour", "hours"], fps: ["frame per second", "frames per second"],
  bpm: ["beat per minute", "beats per minute"], BPM: ["beat per minute", "beats per minute"],
  "°": ["degree", "degrees"],
};
// Units that need a space before them (too ambiguous when attached to the number).
const SPACED_UNITS_EN: Dict<Unit> = { m: ["meter", "meters"], g: ["gram", "grams"], min: ["minute", "minutes"],
  L: ["liter", "liters"], h: ["hour", "hours"] };
// [singular, plural, feminine]
const UNITS_ES: Dict<Unit> = {
  "km/h": ["kilómetro por hora", "kilómetros por hora", false], km: ["kilómetro", "kilómetros", false],
  cm: ["centímetro", "centímetros", false], mm: ["milímetro", "milímetros", false],
  kg: ["kilo", "kilos", false], mg: ["miligramo", "miligramos", false], ml: ["mililitro", "mililitros", false],
  "°C": ["grado Celsius", "grados Celsius", false], "°F": ["grado Fahrenheit", "grados Fahrenheit", false],
  kWh: ["kilovatio hora", "kilovatios hora", false], kHz: ["kilohercio", "kilohercios", false],
  Hz: ["hercio", "hercios", false], dB: ["decibelio", "decibelios", false],
  KB: ["kilobyte", "kilobytes", false], MB: ["megabyte", "megabytes", false],
  GB: ["gigabyte", "gigabytes", false], TB: ["terabyte", "terabytes", false],
  bpm: ["pulso por minuto", "pulsos por minuto", false], BPM: ["pulso por minuto", "pulsos por minuto", false],
  "°": ["grado", "grados", false],
};
const SPACED_UNITS_ES: Dict<Unit> = { m: ["metro", "metros", false], g: ["gramo", "gramos", false],
  l: ["litro", "litros", false], L: ["litro", "litros", false], h: ["hora", "horas", true],
  min: ["minuto", "minutos", false] };
const UNITS_MX: Dict<Unit> = { dB: ["decibel", "decibeles", false], Hz: ["hertz", "hertz", false],
  kHz: ["kilohertz", "kilohertz", false] };

// Abbreviations. Titles come before a name; street words after one.
const TITLES_EN: Dict<string> = { Mr: "Mister", Mrs: "Mrs", Ms: "Miz", Dr: "Doctor", Prof: "Professor",
  Rev: "Reverend", Hon: "Honorable", Gen: "General", Col: "Colonel", Capt: "Captain", Lt: "Lieutenant",
  Sgt: "Sergeant", Gov: "Governor", Sen: "Senator", Rep: "Representative", Pres: "President", Fr: "Father",
  Mt: "Mount", Ft: "Fort", St: "Saint" };
const TITLES_WITHOUT_DOT_EN = ["Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Mt", "Capt", "Sgt", "Lt"];
const STREETS_EN: Dict<string> = { St: "Street", Ave: "Avenue", Blvd: "Boulevard", Rd: "Road", Ln: "Lane",
  Dr: "Drive", Hwy: "Highway", Pkwy: "Parkway", Sq: "Square" };
const AFTER_NAME_EN: Dict<string> = { Jr: "Junior", Sr: "Senior", Inc: "Incorporated", Ltd: "Limited",
  Corp: "Corporation", Co: "Company", Bros: "Brothers" };
const BEFORE_NUMBER_EN: Dict<string> = { no: "number", nos: "numbers", fig: "figure", figs: "figures",
  vol: "volume", ch: "chapter", p: "page", pp: "pages", art: "article", sec: "section", est: "established",
  ext: "extension", tel: "telephone" };
const ANYWHERE_EN: Dict<string> = { approx: "approximately", dept: "department", govt: "government",
  misc: "miscellaneous", vs: "versus" };
const TITLES_ES: Dict<string> = { Sr: "señor", Sra: "señora", Srta: "señorita", Sres: "señores", Sras: "señoras",
  Dr: "doctor", Dra: "doctora", Lic: "licenciado", Lcda: "licenciada", Lda: "licenciada", Ing: "ingeniero",
  Prof: "profesor", Profa: "profesora", "Dña": "doña", Sto: "santo", Sta: "santa", Gral: "general",
  Av: "avenida", Avda: "avenida", Pza: "plaza", "Pº": "paseo", "c/": "calle", "C/": "calle" };
const ANYWHERE_ES: Dict<string> = { Ud: "usted", Uds: "ustedes", Vd: "usted", Vds: "ustedes",
  aprox: "aproximadamente", Aprox: "aproximadamente", dpto: "departamento", Dpto: "departamento",
  tel: "teléfono", Tel: "teléfono", "admón": "administración" };
const BEFORE_NUMBER_ES: Dict<string> = { "núm": "número", "Núm": "número", "nº": "número", "Nº": "número",
  "n.º": "número", No: "número", no: "número", "pág": "página", "págs": "páginas", "Pág": "página",
  cap: "capítulo", Cap: "capítulo", vol: "volumen", Vol: "volumen" };

// Brand names said the way the brand is said, in languages that would read them otherwise.
const BRAND: Dict<Dict<string>> = { es: { "loudness.ai": "Láudnes punto ei ai", loudness: "Láudnes" } };

// American pronunciations, in the British espeak symbol set (what the community US
// voices were trained on), for words whose British form these voices would copy.
export const US_WORDS: Dict<string> = {
  schedule: "skˈɛdʒuːl", schedules: "skˈɛdʒuːlz", scheduled: "skˈɛdʒuːld", scheduling: "skˈɛdʒuːlɪŋ",
  tomato: "təmˈeɪtəʊ", tomatoes: "təmˈeɪtəʊz", either: "ˈiːðə", neither: "nˈiːðə",
  privacy: "pɹˈaɪvəsi", vitamin: "vˈaɪtəmɪn", vitamins: "vˈaɪtəmɪnz", garage: "ɡəɹˈɑːʒ",
  garages: "ɡəɹˈɑːʒɪz", leisure: "lˈiːʒə", advertisement: "ˌadvətˈaɪzmənt",
  advertisements: "ˌadvətˈaɪzmənts", herb: "ˈɜːb", herbs: "ˈɜːbz", herbal: "ˈɜːbəl",
  oregano: "ɔːɹˈɛɡənəʊ", zebra: "zˈiːbɹə", yogurt: "jˈəʊɡət", mobile: "mˈəʊbəl", missile: "mˈɪsəl",
  missiles: "mˈɪsəlz", fertile: "fˈɜːtəl", hostile: "hˈɒstəl", fragile: "fɹˈadʒəl", futile: "fjˈuːtəl",
  sterile: "stˈɛɹəl", agile: "ˈadʒəl", docile: "dˈɒsəl", versatile: "vˈɜːsətəl",
  lieutenant: "luːtˈɛnənt", vase: "vˈeɪs", clerk: "klˈɜːk", process: "pɹˈɒsɛs", processes: "pɹˈɒsɛsɪz",
  processing: "pɹˈɒsɛsɪŋ", processed: "pɹˈɒsɛst", progress: "pɹˈɒɡɹɛs", patriot: "pˈeɪtɹiət",
  patriotic: "pˌeɪtɹiˈɒtɪk", dynasty: "dˈaɪnəsti", ballet: "balˈeɪ", glacier: "ɡlˈeɪʃə",
  laboratory: "lˈabɹətˌɔːɹi", inventory: "ˈɪnvəntˌɔːɹi", territory: "tˈɛɹɪtˌɔːɹi",
  territories: "tˈɛɹɪtˌɔːɹiz", dictionary: "dˈɪkʃənˌɛɹi", dictionaries: "dˈɪkʃənˌɛɹiz",
  secretary: "sˈɛkɹətˌɛɹi", necessary: "nˈɛsəsˌɛɹi", unnecessary: "ʌnnˈɛsəsˌɛɹi",
  military: "mˈɪlɪtˌɛɹi", library: "lˈaɪbɹˌɛɹi", libraries: "lˈaɪbɹˌɛɹiz", ordinary: "ˈɔːdɪnˌɛɹi",
  extraordinary: "ɛkstɹˈɔːdɪnˌɛɹi", temporary: "tˈɛmpəɹˌɛɹi", category: "kˈatəɡˌɔːɹi",
  categories: "kˈatəɡˌɔːɹiz", mandatory: "mˈandətˌɔːɹi", february: "fˈɛbɹuːˌɛɹi",
  january: "dʒˈanjuːˌɛɹi", primary: "pɹˈaɪmˌɛɹi", voluntary: "vˈɒlənˌtɛɹi", stationary: "stˈeɪʃənˌɛɹi",
  dietary: "dˈaɪətˌɛɹi", secondary: "sˈɛkənˌdɛɹi", preliminary: "pɹɪlˈɪmɪnˌɛɹi",
  vocabulary: "vəʊkˈabjʊlˌɛɹi", cemetery: "sˈɛmɪtˌɛɹi", monastery: "mˈɒnəstˌɛɹi", contrary: "kˈɒntɹˌɛɹi",
  arbitrary: "ˈɑːbɪtɹˌɛɹi", sanitary: "sˈanɪtˌɛɹi", legendary: "lˈɛdʒənˌdɛɹi",
  revolutionary: "ɹˌɛvəlˈuːʃənˌɛɹi", imaginary: "ɪmˈadʒɪnˌɛɹi", customary: "kˈʌstəmˌɛɹi",
  literary: "lˈɪtəɹˌɛɹi", observatory: "əbzˈɜːvətˌɔːɹi", dormitory: "dˈɔːmɪtˌɔːɹi",
  explanatory: "ɛksplˈanətˌɔːɹi", repository: "ɹɪpˈɒzɪtˌɔːɹi", "can't": "kˈant", half: "hˈaf",
  halves: "hˈavz", calf: "kˈaf", rather: "ɹˈaðə", banana: "bənˈanə", been: "bɪn", router: "ɹˈaʊtə",
  routers: "ɹˈaʊtəz", lever: "lˈɛvə", niche: "nˈɪtʃ", pasta: "pˈɑːstə", basil: "bˈeɪzəl",
  charade: "ʃəɹˈeɪd", depot: "dˈiːpəʊ", new: "nˈuː", news: "nˈuːz", knew: "nˈuː",
  student: "stˈuːdənt", students: "stˈuːdənts", tune: "tˈuːn", tunes: "tˈuːnz", duty: "dˈuːti",
  during: "dˈʊəɹɪŋ", tuesday: "tˈuːzdeɪ", due: "dˈuː", introduce: "ˌɪntɹədˈuːs",
  produce: "pɹədˈuːs", reduce: "ɹɪdˈuːs", avenue: "ˈavənuː", z: "zˈiː",
};

const byLengthDesc = (keys: string[]): string[] => [...keys].sort((a, b) => cpLength(b) - cpLength(a));
const keys = (d: object): string[] => Object.keys(d);

// Raw phoneme blocks for US_WORDS. Punctuation right after a word goes inside its block:
// Piper drops punctuation between a raw block and the following text.
const US_PATTERN = re("(?<![\\w@./'\\u2019-])(" + byLengthDesc(keys(US_WORDS)).map(escape).join("|")
  + ")(?![\\w@'\\u2019-]|\\.\\w)([,.;:!?)]*)", true);

const usBlock = (m: Match): string => "[[" + US_WORDS[m[1].toLowerCase().replace(/\u2019/g, "'")] + m[2] + "]]";

const EMOJI = re("[\\u{1F000}-\\u{1FAFF}\\u{2600}-\\u{27BF}\\u{2B00}-\\u{2BFF}\\u{1F1E6}-\\u{1F1FF}"
  + "\\u{E0020}-\\u{E007F}\\u{FE0E}\\u{FE0F}\\u{200D}\\u{20E3}]+");
const SENTENCE_END = re("[.!?。！？։۔؟।॥]+[\"'”’»)\\]]*(?=\\s|$)");
const CLOSERS = "\"'”’»)]";
const RUN_BREAK = re("[,;:—–،、，]$|^[—–-]$");
const MAX_RUN_WORDS = 28;

function ordinalEn(n: number): string {
  const suffix = n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Dict<string>)[n % 10] ?? "th";
  return `${n}${suffix}`;
}

function romanValue(s: string): number {
  const values: Dict<number> = { I: 1, V: 5, X: 10, L: 50, C: 100 };
  let total = 0;
  for (let i = 0; i < s.length; i++) {
    const v = values[s[i]];
    total += i + 1 < s.length && values[s[i + 1]] > v ? -v : v;
  }
  return total;
}

const validRoman = (s: string): boolean =>
  !!re("(?:C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})").fullmatch(s) && !!s;

// A number with thousands grouped by spaces, and an amount after a number.
const GROUPED = "\\d{1,3}(?:[ \\u00A0\\u202F]\\d{3})+(?!\\d)";
const AMOUNT_AFTER = re("\\s?(?:[$€£¥₹%]|euros?|d[oó]lar(?:es)?|pesos?|libras?|millones|mil)\\b", true);

/** '1,250.50' / '1.250,50' / '12.99' -> [whole, cents]. A last separator followed
 * by one or two digits is the decimal point; any other separator groups thousands. */
function parseAmount(text: string): [bigint, number | null] {
  const m = re("(.*?)(?:[.,](\\d{1,2}))?").fullmatch(text)!;
  const cents = m[2];
  const whole = re("[.,\\s  ]").sub(m[1], "") || "0";
  return [bigint(whole), cents ? int(cents.padEnd(2, "0")) : null];
}

// ---------------------------------------------------------------- rules

/** One line of text: rules claim spans of the original and say what to read instead. */
class Line {
  group = "";
  claims: Array<[number, number, string, string]> = [];
  constructor(readonly text: string, readonly base: number, readonly post: ((text: string) => string) | null,
    readonly disabled: Set<string>) {}

  free(s0: number, e0: number): boolean {
    return this.claims.every(([s, e]) => e0 <= s || e <= s0);
  }

  apply(pattern: ReturnType<typeof re>, replace: (m: Match) => Out): void {
    if (this.disabled.has(this.group)) return;
    for (const m of pattern.finditer(this.text)) {
      if (start(m) === end(m) || !this.free(start(m), end(m))) continue;
      const out = replace(m);
      if (out === null || out === undefined) continue;
      let [text, brk] = Array.isArray(out) ? out : [out, ""];
      let s = start(m);
      // A rule that keeps the first word ("Felipe VI" -> "Felipe sexto") claims
      // only the rest, so each written word keeps its own spoken words.
      const lead = re("\\S+\\s+").match(m[0]);
      if (lead && text.startsWith(lead[0]) && text.length > lead[0].length) {
        s += lead[0].length;
        text = text.slice(lead[0].length);
      }
      if (this.post && text && !text.includes("[[")) text = this.post(text);
      this.claims.push([s, end(m), text, brk]);
    }
  }

  build(): [string, number[], number[], Map<number, string>] {
    const chars: string[] = [];
    const starts: number[] = [];
    const ends: number[] = [];
    const breaks = new Map<number, string>();
    let pos = 0;
    const keep = (upto: number): void => {
      for (let i = pos; i < upto; i++) {
        chars.push(this.text[i]);
        starts.push(this.base + i);
        ends.push(this.base + i + 1);
      }
    };
    const sorted = [...this.claims].sort((a, b) => a[0] - b[0] || a[1] - b[1] || (a[2] < b[2] ? -1 : a[2] > b[2] ? 1 : 0));
    for (const [s, e, text, brk] of sorted) {
      keep(s);
      for (let k = 0; k < text.length; k++) {
        chars.push(text[k]);
        starts.push(this.base + s);
        ends.push(this.base + e);
      }
      if (brk) breaks.set(chars.length, brk);
      pos = e;
    }
    keep(this.text.length);
    return [chars.join(""), starts, ends, breaks];
  }
}

/** 14:10:30 as "14 hours 10 minutes and 30 seconds" (parts that are zero are left out). */
function clockWithSeconds(m: Match, hours: string[], minutes: string[], seconds: string[], joiner: string,
  one?: Dict<string>): string {
  const parts: string[] = [];
  const values = [int(m[1]), int(m[2]), int(m[3])];
  const names = [hours, minutes, seconds];
  for (let k = 0; k < 3; k++) {
    const value = values[k];
    if (value || (!parts.length && k === 0)) {
      const count = value === 1 ? (one ?? {})[names[k][0]] ?? "1" : String(value);
      parts.push(`${count} ${value === 1 ? names[k][0] : names[k][1]}`);
    }
  }
  return parts.length === 1 ? parts[0] : parts.slice(0, -1).join(" ") + ` ${joiner} ` + parts[parts.length - 1];
}

/** After an abbreviation's dot: does a new sentence start (capital letter or end of line)? */
function sentenceFollows(m: Match): boolean {
  const rest = m.input.slice(end(m));
  return !strip(rest) || !!re("\\s+[\"'“‘(]?[A-ZÁÉÍÓÚÑ]").match(rest);
}

/** An expanded abbreviation keeps its dot as a sentence end when a sentence follows. */
function expansion(word: string, m: Match, dotted: boolean): string | [string, string] {
  if (dotted && sentenceFollows(m)) return [word + ".", "sentence"];
  return word;
}

const withPrefix = (prefix: string, out: string | [string, string]): string | [string, string] =>
  Array.isArray(out) ? [prefix + out[0], out[1]] : prefix + out;

class Preparer {
  readonly lang: string;
  readonly region: string;
  readonly usWords: boolean;
  readonly spaced: boolean;
  readonly symbolAfter: ReturnType<typeof re> | null;
  readonly disabled: Set<string>;

  constructor(readonly locale: string, usWords: boolean, disabled: Iterable<string> = []) {
    this.disabled = new Set(disabled);
    const dash = locale.indexOf("-");
    this.lang = dash < 0 ? locale : locale.slice(0, dash);
    this.region = dash < 0 ? "" : locale.slice(dash + 1);
    this.usWords = usWords && locale === "en-US";
    this.spaced = !["zh", "ja", "th"].includes(this.lang); // words separated by spaces
    // A currency or unit after a number, for languages with money and unit rules.
    const tables: Dict<[Dict<Unit>, Dict<Unit>]> = {
      en: [UNITS_EN, SPACED_UNITS_EN],
      es: [{ ...UNITS_ES, ...(this.region === "MX" ? UNITS_MX : {}) }, SPACED_UNITS_ES],
    };
    const [units, spaced] = tables[this.lang] ?? [{}, {}];
    const names = byLengthDesc(keys(units)).map(escape).join("|");
    this.symbolAfter = keys(units).length
      ? re("\\s?(?:[$€£¥₹]|USD|EUR|GBP|" + names + ")(?![\\w/])|\\s(?:" + keys(spaced).map(escape).join("|")
        + ")(?![\\w/]|\\.\\w)")
      : null;
  }

  // ---- rules shared by every language
  genericRules(line: Line, firstLine: boolean): void {
    line.group = "addresses";
    line.apply(re("[\\w.+-]+@[\\w-]+(?:\\.[\\w-]+)+"), (m) => m[0]);
    line.apply(re("(?:https?://)(?:www\\.)?[^\\s]*[^\\s.,;:!?)\\]]"),
      (m) => rstrip(re("^https?://(?:www\\.)?").sub(m[0], ""), "/"));
    line.apply(re("(?<![\\w@.])www\\.([^\\s]*[^\\s.,;:!?)\\]])"), (m) => rstrip(m[1], "/"));
    line.apply(re("(?<![\\w@./])[A-Za-z0-9-]+(?:\\.[A-Za-z0-9-]+)*\\.(?:com|org|net|ai|io|co|uk|es|mx|de|fr|"
      + "it|app|dev|edu|gov|info|me|tv|us|ca|au|in|pk)\\b(?:/[^\\s]*[^\\s.,;:!?)\\]])?"), (m) => rstrip(m[0], "/"));
    line.group = "lists";
    if (firstLine) { // list markers and markdown headings at the start of a line
      line.apply(re("^\\s*#{1,6}\\s+"), () => "");
      line.apply(re("^\\s*(?:[-*•‣◦▪●·–—]+)\\s+"), () => "");
      line.apply(re("^\\s*(\\d{1,3})[.)]\\s+"), (m) => m[1] + ", ");
      line.apply(re("^\\s*([a-zA-Z])\\)\\s+"), () => "");
    }
  }

  lateGenericRules(line: Line): void {
    line.group = "markup";
    line.apply(re("(\\*\\*|__)(?=\\S)(.+?)(?<=\\S)\\1"), (m) => m[2]);
    line.apply(re("(?<![\\w*])\\*(?=\\S)([^*]+?)(?<=\\S)\\*(?![\\w*])"), (m) => m[1]);
    line.apply(re("(?<!\\w)[*`]+|[*`]+(?!\\w)"), () => "");
    line.apply(re("#(?=[^\\W\\d_])"), () => "");
    line.apply(EMOJI, () => "");
    line.group = "pauses";
    line.apply(re("(?:\\.\\s?){2,}\\.|…"), (m) => this.ellipsis(m));
    line.apply(re("(?<=\\S)\\s+[-–—]+\\s+(?=\\S)"), () => ", ");
    line.apply(re("(?<=\\w)[—](?=\\w)"), () => ", ");
  }

  ellipsis(m: Match): Out {
    const rest = m.input.slice(end(m));
    if (!strip(rest)) return [".", "sentence"];
    let nxt = first(lstrip(rest));
    if (nxt && "\"'“¿¡".includes(nxt)) { // an opening mark: the word after it decides
      nxt = codePointAt(lstrip(rest), 1) || nxt;
    }
    if (isspace(first(rest)) && nxt && isupper(nxt)) return [".", "sentence"];
    if ("?!.,;:".includes(nxt)) return "";
    return [",", "ellipsis"];
  }

  phoneRule(line: Line): void {
    if (this.lang === "zh" || this.lang === "ja") return;
    const plus = PLUS_WORD[this.lang] ?? "";
    line.group = "phone";
    const phone = (m: Match): Out => {
      const raw = m[0];
      const digits = re("\\D").sub(raw, "");
      const groups = re("\\d+").findall(raw);
      const n = cpLength(digits);
      if (!(n >= 7 && n <= 15) || re("\\d{4}-\\d{1,2}-\\d{1,2}|\\d{1,2}[-.]\\d{1,2}[-.]\\d{2,4}").fullmatch(raw)) return null;
      const prefixed = raw.startsWith("+") || raw.includes("(");
      const dashed = !!re("\\d[-.]\\d").search(raw);
      // Spain writes its nine-digit numbers in groups ("912 345 678", "91 234 56 78"); a round
      // number or an amount ("900 000 000 de personas", "600 000 000 euros") is not one.
      const shape = groups.map(cpLength).join(",");
      const spain = this.lang === "es" && n === 9 && "6789".includes(first(digits))
        && ["3,3,3", "2,3,2,2", "3,2,2,2"].includes(shape)
        && groups[groups.length - 1] !== "000" && !AMOUNT_AFTER.match(m.input, end(m));
      if (!(prefixed || spain || (dashed && groups.length >= 3) || (groups.length >= 3 && groups.some((g) => cpLength(g) === 4))
        || (dashed && groups.length === 2 && cpLength(groups[0]) === 3 && cpLength(groups[1]) === 4))) return null;
      const spoken = groups.map((g) => Array.from(g).join(" ")).join(", ");
      return (raw.startsWith("+") && plus ? `${plus} ` : "") + spoken;
    };
    line.apply(re("(?<![\\w+.,/])(?:\\+\\d{1,3}[\\s.-]?)?(?:\\(\\d{1,4}\\)[\\s.-]?)?\\d{2,4}(?:[\\s.-]\\d{2,4}){1,4}"
      + "(?![\\w.,/]\\d|\\d)"), phone);
  }

  numberRules(line: Line): void {
    line.group = "numbers";
    const symbol = this.symbolAfter;
    const grouped = (m: Match): Out => {
      // "$1 000" and "1 500 kg" are read by the money and unit rules.
      if (symbol && (re("[$€£¥₹]\\s?$").search(m.input.slice(0, start(m))) || symbol.match(m.input, end(m)))) return null;
      return re("\\s").sub(m[0], "");
    };
    // Thousands grouped with spaces ("10 000", "3 000 000"), read as one number.
    line.apply(re("(?<![\\d.,])\\d{1,3}(?:[ \\u00A0\\u202F]\\d{3})+(?!\\d)(?![.,]\\d)"), grouped);
    if (this.lang === "es") { // "10.º", "1.ª": espeak reads ordinals only without the dot
      line.apply(re("(\\d+)\\.([ºª])"), (m) => m[1] + m[2]);
    }
  }

  // ---- English
  englishRules(line: Line): void {
    const months = MONTHS.en;
    const us = this.region !== "GB" && this.region !== "IE";

    const year = (digits: string): string => {
      // American dates say years in pairs: "twenty twenty-six", "nineteen oh five".
      const y = int(digits);
      if (!(this.region === "US" && ((y >= 1100 && y <= 1999) || (y >= 2010 && y <= 2099)))) return digits;
      const d = Array.from(digits);
      const head = d.slice(0, 2).join("");
      const tail = d.slice(2);
      return tail.join("") === "00" ? `${head} hundred` : tail[0] === "0" ? `${head} oh ${tail[1]}` : `${head} ${tail.join("")}`;
    };

    const date = (day: number, month: number, yearDigits: string): Out => {
      if (!(month >= 1 && month <= 12 && day >= 1 && day <= 31)) return null;
      const said = year(yearDigits);
      return us ? `${months[month - 1]} ${ordinalEn(day)}, ${said}` : `${ordinalEn(day)} ${months[month - 1]} ${said}`;
    };

    line.group = "dates";
    line.apply(re("\\b(\\d{4})-(\\d{2})-(\\d{2})\\b"), (m) => date(int(m[3]), int(m[2]), m[1]));
    line.apply(re("\\b(\\d{4})/(\\d{1,2})/(\\d{1,2})\\b"), (m) => date(int(m[3]), int(m[2]), m[1]));
    line.apply(re("\\b(\\d{1,2})\\.(\\d{1,2})\\.(\\d{4})\\b"), (m) => date(int(m[1]), int(m[2]), m[3]));

    const slashDate = (m: Match): Out => {
      const a = int(m[1]);
      const b = int(m[2]);
      let y = m[3];
      if (cpLength(y) === 2) y = (int(y) < 50 ? "20" : "19") + y;
      let [month, day] = us && a <= 12 ? [a, b] : [b, a];
      if (month > 12) [month, day] = [day, month];
      return date(day, month, y);
    };
    line.apply(re("\\b(\\d{1,2})/(\\d{1,2})/(\\d{4}|\\d{2})\\b"), slashDate);

    const time = (m: Match): Out => {
      const hour = int(m[1]);
      const minute = m[2];
      const suffix = m[3] || "";
      const half = suffix ? ` ${suffix.replaceAll(".", "").replaceAll(" ", "").toUpperCase()}` : "";
      let said: string;
      if (minute === "00") {
        if (suffix) said = `${hour}${half}`;
        else if (hour === 12) said = this.region === "US" ? "noon" : "12 o'clock";
        else said = hour >= 13 || hour === 0 ? `${hour} hundred` : `${hour} o'clock`;
      } else if (minute.startsWith("0")) {
        said = `${hour} oh ${int(minute)}${half}`;
      } else if (suffix.endsWith(".")) {
        said = `${hour}:${minute}${half}`;
      } else {
        return null;
      }
      // "at 9:05 a.m. The…": the dot of "a.m." also ends the sentence.
      return expansion(said, m, suffix.endsWith("."));
    };
    line.group = "times";
    line.apply(re("\\b(\\d{1,2}):([0-5]\\d):([0-5]\\d)\\b"),
      (m) => clockWithSeconds(m, ["hour", "hours"], ["minute", "minutes"], ["second", "seconds"], "and"));
    line.apply(re("\\b([01]?\\d|2[0-3]):([0-5]\\d)(?!\\d)(\\s?[aApP]\\.?\\s?[mM]\\.?(?!\\w))?"), time);

    const dotTime = (m: Match): Out => {
      // "9.05 am", common in British writing: only with am/pm, so decimals stay decimals.
      const hour = int(m[1]);
      const minute = m[2];
      const suffix = m[3];
      const half = " " + suffix.replaceAll(".", "").replaceAll(" ", "").toUpperCase();
      const said = minute === "00" ? `${hour}${half}` : minute[0] === "0" ? `${hour} oh ${int(minute)}${half}`
        : `${hour}:${minute}${half}`;
      return expansion(said, m, suffix.endsWith("."));
    };
    line.apply(re("\\b(1[0-2]|0?[1-9])\\.([0-5]\\d)\\s?([aApP]\\.?\\s?[mM]\\.?)(?!\\w)"), dotTime);
    line.group = "money";
    this.moneyEn(line);

    const decade = (m: Match): Out => {
      const century = m[1];
      const tens = int(m[2]);
      if (tens === 0) {
        if (!century) return null;
        const c = Array.from(century);
        return c[1] !== "0" ? `${c.slice(0, 2).join("")} hundreds` : `${c[0]} thousands`;
      }
      return (century ? `${century} ` : "") + DECADES_EN[tens];
    };
    line.group = "decades";
    line.apply(re("(?<![\\w'])'?(\\d{2})?(\\d)0s\\b"), decade);
    line.group = "ranges";
    this.ranges(line);
    line.group = "fractions";
    this.fractions(line);
    line.group = "units";
    this.units(line, UNITS_EN, SPACED_UNITS_EN, (n, unit) => (n === "1" ? unit[0] : unit[1]));
    line.group = "roman";

    const romanKeyword = (m: Match): Out => (validRoman(m[2]) ? `${m[1]} ${romanValue(m[2])}` : null);
    line.apply(re("\\b(" + ROMAN_KEYWORDS_EN.join("|") + "|" + ROMAN_KEYWORDS_EN.map(title).join("|")
      + "|Vol\\.?|vol\\.?)\\s+([IVXLC]+)\\b(?!['’]?\\w)"), romanKeyword);

    const romanName = (m: Match): Out => {
      if (!validRoman(m[2])) return null;
      const value = romanValue(m[2]);
      return REGNAL_NAMES.en.has(m[1]) ? `${m[1]} the ${ordinalEn(value)}` : `${m[1]} ${value}`;
    };
    line.apply(re("\\b([A-Z][a-z]+)\\s+([IVX]{2,})\\b(?!['’]?[a-z])"), romanName);

    line.group = "dates";
    const short = keys(MONTH_ABBR_EN).join("|");
    const abbr = (s: string): string => months[MONTH_ABBR_EN[title(s)] - 1];
    const day = (s: string): boolean => int(s) >= 1 && int(s) <= 31;
    line.apply(re("\\b(" + short + ")\\.?\\s?(\\d{1,2})(?:st|nd|rd|th)?,?\\s(\\d{4})\\b", true),
      (m) => (day(m[2]) ? `${abbr(m[1])} ${ordinalEn(int(m[2]))}, ${year(m[3])}` : null));
    line.apply(re("\\b(" + short + ")\\.?\\s?(\\d{1,2})(?![\\d:]|st|nd|rd|th)\\b", true),
      (m) => (day(m[2]) ? `${abbr(m[1])} ${ordinalEn(int(m[2]))}` : null));
    line.apply(re("\\b(" + short + ")\\.?(?=\\s?\\d)", true), (m) => abbr(m[1]));
    const named = months.join("|");
    line.apply(re("\\b(" + named + ")\\s(\\d{1,2})(?:st|nd|rd|th)?,?\\s(\\d{4})\\b", true),
      (m) => (day(m[2]) ? `${m[1]} ${ordinalEn(int(m[2]))}, ${year(m[3])}` : null));
    line.apply(re("\\b(" + named + ")\\s(\\d{4})\\b", true), (m) => `${m[1]} ${year(m[2])}`);
    line.apply(re("\\b(" + named + ")\\s(\\d{1,2})(?![\\d:]|st|nd|rd|th)\\b", true),
      (m) => (day(m[2]) ? `${m[1]} ${ordinalEn(int(m[2]))}` : null));
    line.apply(re("\\b(\\d{1,2})\\s(" + named + ")\\b", true),
      (m) => (day(m[1]) ? `${ordinalEn(int(m[1]))} ${m[2]}` : null));
    line.group = "abbreviations";
    this.abbreviationsEn(line);
    line.group = "initials";
    this.initials(line);
    line.group = "symbols";
    line.apply(re("#\\s?(?=\\d)"), () => "number ");
    line.apply(re("~\\s?(?=\\d)"), () => "about ");
    line.apply(re("(?<![\\w/])N/A(?![\\w/])|(?<![\\w/])n/a(?![\\w/])"), () => "not applicable");
    line.apply(re("(?<![\\w/])w/(?=\\s)"), () => "with");
    line.group = "slashes";
    this.wordSlashes(line, "or", { "and/or": "and or" });
  }

  moneyEn(line: Line): void {
    const say = (symbol: string, amount: string, scale: string | undefined): Out => {
      const names = CURRENCY_EN[symbol];
      if (scale) {
        const word = SCALES_EN[scale.toLowerCase()];
        if (!word) return null;
        const number = re(",(?=\\d{3})|\\s").sub(amount, "");
        return `${number} ${word} ${names[1]}`;
      }
      const [whole, cents] = parseAmount(amount);
      const parts: string[] = [];
      if (whole || !cents) parts.push(`${whole} ${whole === 1n ? names[0] : names[1]}`);
      if (cents && names[2]) parts.push(`${cents} ${cents === 1 ? names[2] : names[3]}`);
      return parts.join(" and ");
    };
    const scale = "(?:\\s?(thousand|million|billion|trillion|bn|mn|[kKmMbB])\\b)?";
    const number = "(\\d{1,3}(?:,\\d{3})+(?:\\.\\d{1,2})?|" + GROUPED + "(?:\\.\\d{1,2})?|\\d+(?:[.,]\\d{1,2})?)";
    line.apply(re("(?<![\\w.,])([$€£¥₹])\\s?" + number + scale), (m) => say(m[1], m[2], m[3]));
    line.apply(re("(?<![\\w.,])" + number + "\\s?([€£¥₹]|USD|EUR|GBP)(?!\\w)"), (m) => say(m[2], m[1], undefined));
  }

  abbreviationsEn(line: Line): void {
    const starters = SENTENCE_STARTERS.en;

    const street = (m: Match): Out => {
      // "Elm St." / "5th Ave", but "St. Louis" is a saint: a following name decides.
      const following = re("\\s+([A-Z][a-z]+)").match(m.input.slice(end(m)));
      if (following && !starters.has(following[1])) return null;
      return withPrefix(`${m[1]} `, expansion(STREETS_EN[m[2]], m, !!m[3]));
    };
    line.apply(re("\\b([A-Z][a-z]+|\\d+(?:st|nd|rd|th))\\s(" + keys(STREETS_EN).join("|") + ")(\\.?)(?![\\w.])"), street);
    const dotless = TITLES_WITHOUT_DOT_EN.join("|");
    line.apply(re("\\b(" + keys(TITLES_EN).join("|") + ")\\.(?=\\s+[A-Z])"), (m) => TITLES_EN[m[1]]);
    line.apply(re("\\b(" + dotless + ")(?=\\s+[A-Z][a-z])"), (m) => TITLES_EN[m[1]]);
    // A person's title with a dot is never really a sentence end ("Mr. and Mrs. Jones").
    const people = keys(TITLES_EN).filter((t) => !["St", "Mt", "Ft"].includes(t)).join("|");
    line.apply(re("\\b(" + people + ")\\.(?!\\w)"), (m) => expansion(TITLES_EN[m[1]], m, true));

    const after = (m: Match): Out => withPrefix(m[1], expansion(AFTER_NAME_EN[m[2]], m, true));
    line.apply(re("(?<=\\S)(,?\\s)(" + keys(AFTER_NAME_EN).join("|") + ")\\.(?!\\w)"), after);
    line.apply(re("\\b(" + keys(BEFORE_NUMBER_EN).join("|") + "|" + keys(BEFORE_NUMBER_EN).map(title).join("|")
      + ")\\.\\s?(?=\\d)"), (m) => BEFORE_NUMBER_EN[m[1].toLowerCase()] + " ");
    line.apply(re("\\b(" + keys(ANYWHERE_EN).join("|") + "|" + keys(ANYWHERE_EN).map(title).join("|")
      + "|VS)(\\.?)(?!\\w)"),
    (m) => (m[2] || m[1].toLowerCase() === "vs" ? expansion(ANYWHERE_EN[m[1].toLowerCase()], m, !!m[2]) : null));
  }

  // ---- Spanish
  spanishRules(line: Line): void {
    const months = MONTHS.es;

    const date = (day: number, month: number, year: string): Out => {
      if (!(month >= 1 && month <= 12 && day >= 1 && day <= 31)) return null;
      return `${day === 1 ? "primero" : day} de ${months[month - 1]} de ${year}`;
    };
    line.group = "dates";
    line.apply(re("\\b(\\d{4})-(\\d{2})-(\\d{2})\\b"), (m) => date(int(m[3]), int(m[2]), m[1]));

    const numericDate = (m: Match): Out => {
      let year = m[3];
      if (cpLength(year) === 2) year = (int(year) < 50 ? "20" : "19") + year;
      return date(int(m[1]), int(m[2]), year);
    };
    line.apply(re("\\b(\\d{1,2})[/.](\\d{1,2})[/.](\\d{4}|\\d{2})\\b"), numericDate);

    const hourWord = (hour: number): string => (hour === 1 ? "una" : hour === 21 ? "veintiuna" : String(hour));

    const ampm = (m: Match): Out => {
      const hour = int(m[1]);
      const minute = m[2];
      const half = m[3].toLowerCase();
      if (!(hour >= 1 && hour <= 12)) return null;
      const when = half === "a" ? "de la mañana" : hour === 12 ? "del mediodía" : hour <= 7 ? "de la tarde" : "de la noche";
      return expansion(`${hourWord(hour)}${Preparer.minutesEs(minute)} ${when}`, m, m[0].endsWith("."));
    };
    line.group = "times";
    line.apply(re("\\b(\\d{1,2})(?::(\\d{2}))?\\s?([aApP])\\.?\\s?[mM]\\.?(?!\\w)"), ampm);

    const time = (m: Match): Out => {
      const hour = int(m[1]);
      const minute = m[2];
      if (minute === "00") return hour >= 13 || hour === 0 ? `${hourWord(hour)} horas` : `${hourWord(hour)} en punto`;
      return `${hourWord(hour)}${Preparer.minutesEs(minute)}`;
    };
    line.apply(re("\\b(\\d{1,2}):([0-5]\\d):([0-5]\\d)\\b"),
      (m) => clockWithSeconds(m, ["hora", "horas"], ["minuto", "minutos"], ["segundo", "segundos"], "y",
        { hora: "una", minuto: "un", segundo: "un" }));
    line.apply(re("\\b([01]?\\d|2[0-3]):([0-5]\\d)(?!\\d)"), time);
    line.group = "money";
    this.moneyEs(line);
    line.group = "ranges";
    this.ranges(line);
    line.group = "fractions";
    this.fractions(line);
    line.group = "units";
    const units = { ...UNITS_ES, ...(this.region === "MX" ? UNITS_MX : {}) };
    this.units(line, units, SPACED_UNITS_ES, (n, unit) => (n === "1" ? unit[0] : unit[1]), (unit) => (unit[2] ? "una" : "un"));

    const romanName = (m: Match): Out => {
      if (!validRoman(m[2]) || !REGNAL_NAMES.es.has(m[1])) return null;
      const value = romanValue(m[2]);
      if (value > 10) return `${m[1]} ${value}`;
      const word = ORDINAL_ES[value];
      return `${m[1]} ${FEMININE_ES.has(m[1]) ? word.slice(0, -1) + "a" : word}`;
    };
    line.group = "roman";
    line.apply(re("\\b([A-ZÁÉÍÓÚ][a-záéíóúñ]+)\\s+([IVX]+)\\b(?!['’]?[a-z])"), romanName);
    line.group = "abbreviations";
    line.apply(re("\\bEE\\.\\s?UU\\.|\\bEEUU\\b"), (m) => expansion("Estados Unidos", m, m[0].endsWith(".")));
    line.apply(re("(?<![\\w/])([cC]/)\\s?(?=[A-ZÁÉÍÓÚ])"), () => "calle ");
    const titles = keys(TITLES_ES).filter((k) => !k.endsWith("/")).map(escape).join("|");
    line.apply(re("(?<!\\w)(" + titles + ")\\.(?=\\s+[A-ZÁÉÍÓÚÑ])"), (m) => TITLES_ES[m[1]]);
    line.apply(re("(?<!\\w)(Av|Avda|Pza|Pº)\\.?(?=\\s+del?\\s)"), (m) => TITLES_ES[m[1]]);
    line.apply(re("(?<!\\w)(" + keys(BEFORE_NUMBER_ES).map(escape).join("|") + ")\\.?\\s?(?=\\d)"),
      (m) => (!["No", "no"].includes(m[1]) || rstrip(m[0]).endsWith(".") ? BEFORE_NUMBER_ES[m[1]] + " " : null));
    line.apply(re("(?<!\\w)(" + keys(ANYWHERE_ES).map(escape).join("|") + ")\\.(?!\\w)"),
      (m) => expansion(ANYWHERE_ES[m[1]], m, true));
    line.group = "initials";
    this.initials(line);
    line.group = "symbols";
    line.apply(re("#\\s?(?=\\d)"), () => "número ");
    line.apply(re("~\\s?(?=\\d)"), () => "unos ");
    line.group = "slashes";
    this.wordSlashes(line, "o", { "y/o": "y o" });
  }

  static minutesEs(minute: string | undefined): string {
    if (!minute || minute === "00") return "";
    return ({ 15: " y cuarto", 30: " y media" } as Dict<string>)[minute] ?? ` y ${int(minute)}`;
  }

  moneyEs(line: Line): void {
    const dollar = this.region === "MX" ? PESO_ES : DOLLAR_ES;
    const say = (symbol: string, amount: string): string => {
      const names = symbol === "$" ? dollar : CURRENCY_ES[symbol];
      const [whole, cents] = parseAmount(amount);
      let text = !whole && cents ? "" : `${whole === 1n ? "un" : whole} ${whole === 1n ? names[0] : names[1]}`;
      if (cents && names[2]) {
        const piece = `${cents === 1 ? "un" : cents} ${cents === 1 ? names[2] : names[3]}`;
        text = text ? `${text} con ${piece}` : piece;
      }
      return text;
    };
    const number = "(\\d{1,3}(?:[.,\\s]\\d{3})+(?:[.,]\\d{1,2})?|\\d+(?:[.,]\\d{1,2})?)";
    line.apply(re("(?<![\\w.,])([$€£¥])\\s?" + number + "(?![\\d])"), (m) => say(m[1], m[2]));
    line.apply(re("(?<![\\w.,])" + number + "\\s?([€£¥$]|USD|EUR)(?![\\w$])"), (m) => say(m[2], m[1]));
  }

  // ---- shared helpers
  ranges(line: Line): void {
    const word = RANGE_WORD[this.lang];
    if (word) {
      line.apply(re("(?<![\\w.,/:+-])(\\d{1,4})\\s?[-–]\\s?(\\d{1,4})(?![\\w.,/:-]?\\d)(?![\\w-])"),
        (m) => `${m[1]} ${word} ${m[2]}`);
    }
  }

  fractions(line: Line): void {
    const words = FRACTIONS[this.lang] ?? {};
    line.apply(re("(?<![\\w./])(\\d{1,3})/(\\d{1,3})(?![\\w/]|\\.\\d)"),
      (m) => words[`${int(m[1])}/${int(m[2])}`] || `${m[1]} ${m[2]}`);
  }

  units(line: Line, table: Dict<Unit>, spaced: Dict<Unit>, word: (n: string, unit: Unit) => string,
    one?: (unit: Unit) => string): void {
    const names = byLengthDesc(keys(table));
    const say = (m: Match, unit: Unit): string => {
      const number = re("\\s").sub(m[1], "");
      const single = re("0*1(?:[.,]0+)?").fullmatch(number) !== null;
      if (one && single) return `${one(unit)} ${unit[0]}`;
      return `${number} ${word(single ? "1" : number, unit)}`;
    };
    const pattern = names.map(escape).join("|");
    const number = "(?<![\\w.,])((?:" + GROUPED + "|\\d+)(?:[.,]\\d+)?)";
    line.apply(re(number + "\\s?(" + pattern + ")(?![\\w/])"), (m) => say(m, table[m[2]]));
    line.apply(re(number + "\\s(" + keys(spaced).join("|") + ")(?![\\w/]|\\.\\w)"), (m) => say(m, spaced[m[2]]));
  }

  initials(line: Line): void {
    // Initials before a name ("J. K. Rowling"), unless a sentence seems to start.
    const starters = SENTENCE_STARTERS[this.lang] ?? new Set<string>();
    line.apply(re("\\b((?:[A-Z]\\.\\s?){2,})(?=([A-Z][a-z]+))"),
      (m) => (starters.has(m[2]) ? null : re("[A-Z]").findall(m[1]).join(" ") + " "));
    for (const m of re("\\b[A-Z][a-z]+ ([A-Z])\\.(?= ([A-Z][a-z]+))").finditer(line.text)) {
      const s1 = end(m) - 2; // group 1 is the capital letter before the final dot
      if (!starters.has(m[2]) && line.free(s1, s1 + 2)) line.claims.push([s1, s1 + 2, m[1], ""]);
    }
  }

  wordSlashes(line: Line, joiner: string, fixed: Dict<string>): void {
    const slash = (m: Match): Out => {
      const token = m[0];
      if (token.toLowerCase() in fixed) return fixed[token.toLowerCase()];
      const [a, b] = token.split("/");
      if ((cpLength(a) === 1 && cpLength(b) === 1) || (isupper(a) && isupper(b))) return `${a} ${b}`;
      return `${a} ${joiner} ${b}`;
    };
    line.apply(re("(?<![\\w./@:-])[^\\W\\d_]+/[^\\W\\d_]+(?![\\w./@:-])"), slash);
  }

  lexicon(line: Line): void {
    line.group = "lexicon";
    const brand = BRAND[this.lang] ?? {};
    if (keys(brand).length) {
      const names = byLengthDesc(keys(brand));
      line.apply(re("(?<![\\w@./-])(" + names.map(escape).join("|") + ")(?![\\w@-]|\\.\\w)", true),
        (m) => brand[m[1].toLowerCase()]);
    }
    if (this.usWords) {
      line.apply(US_PATTERN, usBlock);
      line.apply(re("(?<=[A-Za-z]-)[Zz](?![\\w@-])([,.;:!?)]*)"), (m) => "[[" + US_WORDS.z + m[1] + "]]");
    }
  }

  static american(text: string): string {
    return US_PATTERN.sub(text, usBlock);
  }

  // ---- whole text
  prepare(text: string): Segment[] {
    const lines = Array.from(re("[^\\n]+").finditer(text), (m) => [start(m), end(m)]);
    const segments: Segment[] = [];
    lines.forEach(([s, e], index) => {
      const raw = text.slice(s, e);
      if (!strip(raw)) return;
      const line = new Line(raw, s, this.usWords ? Preparer.american : null, this.disabled);
      // People cannot send raw phonemes; brand names come before web addresses.
      line.apply(re("\\[\\[|\\]\\]"), (m) => m[0][0]);
      this.lexicon(line);
      this.genericRules(line, true);
      this.phoneRule(line);
      this.numberRules(line);
      if (this.lang === "en") this.englishRules(line);
      else if (this.lang === "es") this.spanishRules(line);
      else {
        line.group = "ranges";
        this.ranges(line);
      }
      this.lateGenericRules(line);
      const lineSegments = this.split(...line.build());
      if (!lineSegments.length) return;
      const following = index + 1 < lines.length ? text.slice(e, lines[index + 1][0]) : "";
      const last = lineSegments[lineSegments.length - 1];
      const kind = index + 1 === lines.length ? "" : (following.split("\n").length - 1 >= 2 ? "paragraph" : "line");
      lineSegments[lineSegments.length - 1] = { ...last, text: Preparer.terminate(last.text), pause: kind };
      segments.push(...lineSegments);
    });
    if (segments.length) segments[segments.length - 1] = { ...segments[segments.length - 1], pause: "" };
    return segments;
  }

  static terminate(text: string): string {
    const core = rstrip(text, CLOSERS + " ");
    const final = lastChar(core);
    if (core && isPunctuation(final) && !"-–—".includes(final)) return text;
    if ([",", "—", "–", "-"].some((c) => core.endsWith(c))) return rstrip(core, ",-–— ") + ".";
    return rstrip(text) + ".";
  }

  split(norm: string, starts: number[], ends: number[], breaks: Map<number, string>): Segment[] {
    const cuts = new Map(breaks);
    for (const m of SENTENCE_END.finditer(norm)) {
      const rest = lstrip(norm.slice(end(m)));
      if (m[0][0] === "." && islower(first(rest))) continue;
      if (!cuts.has(end(m))) cuts.set(end(m), "sentence");
    }
    const out: Segment[] = [];
    let prev = 0;
    for (const pos of [...[...cuts.keys()].sort((a, b) => a - b), norm.length]) {
      if (pos <= prev) continue;
      out.push(...this.piece(norm, starts, ends, prev, pos, cuts.get(pos) ?? ""));
      prev = pos;
    }
    return out;
  }

  /** A stretch between cuts, with long unpunctuated runs split at a comma pause. */
  piece(norm: string, starts: number[], ends: number[], a: number, b: number, kind: string): Segment[] {
    const words = Array.from(re("\\S+").finditer(norm.slice(a, b)), (m) => [start(m) + a, end(m) + a]);
    if (!words.length || !Array.from(norm.slice(a, b)).some(isalnumChar)) return [];
    const pieces: Array<[number, number]> = [];
    let run = 0;
    let firstWord = 0;
    const conj = CONJUNCTIONS[this.lang] ?? new Set<string>();
    if (this.spaced && !this.disabled.has("long-runs")) {
      for (let i = 0; i < words.length; i++) {
        const token = norm.slice(words[i][0], words[i][1]);
        run += 1;
        if (RUN_BREAK.search(token)) {
          run = 0;
        } else if (run > MAX_RUN_WORDS) {
          const startOfRun = i - run + 1;
          let choice = startOfRun + 18;
          let best = Infinity;
          for (let j = startOfRun + 10; j < Math.min(i + 1, startOfRun + 27); j++) {
            if (!conj.has(norm.slice(words[j][0], words[j][1]).toLowerCase())) continue;
            const score = Math.abs(j - startOfRun - 18);
            if (score < best) {
              best = score;
              choice = j;
            }
          }
          pieces.push([firstWord, choice]);
          firstWord = choice;
          run = i - choice + 1;
        }
      }
    }
    pieces.push([firstWord, words.length]);
    const segs: Segment[] = [];
    pieces.forEach(([w0, w1], n) => {
      const s = words[w0][0];
      const e = words[w1 - 1][1];
      let text = re("\\s+").sub(norm.slice(s, e), " ");
      const last = n + 1 === pieces.length;
      if (!last) text = rstrip(text) + ",";
      const tokens = words.slice(w0, w1).map(([ws, we]) =>
        [norm.slice(ws, we), Math.min(...starts.slice(ws, we)), Math.max(...ends.slice(ws, we))] as [string, number, number]);
      segs.push({ text, start: starts[s], end: ends[e - 1], pause: last ? kind : "clause", tokens });
    });
    return segs;
  }
}

/** Where each word's sounds begin: as close as possible to its expected share,
 * preferring gaps between spoken words, every word getting at least one sound
 * (a small dynamic programme around the proportional guess). */
export function cutPoints(counts: number[], totalSounds: number, wordStarts: Set<number>): number[] {
  const scale = totalSounds / counts.reduce((x, y) => x + y, 0);
  const expected = counts.map((count) => count * scale);
  const layers: Array<Map<number, [number, number | null]>> = [new Map([[0, [0.0, null]]])];
  let done = 0.0;
  for (let k = 1; k < counts.length; k++) {
    done += expected[k - 1];
    const low = Math.max(1, Math.trunc(done) - 8);
    const high = Math.min(totalSounds - 1, Math.trunc(done) + 9);
    const layer = new Map<number, [number, number | null]>();
    const previousLayer = layers[layers.length - 1];
    for (let cut = low; cut <= high; cut++) {
      let best: [number, number] | null = null;
      for (const [previous, [cost]] of previousLayer) {
        if (previous >= cut && totalSounds >= counts.length) continue;
        const step = cost + Math.abs((cut - previous) - expected[k - 1]) + (wordStarts.has(cut) ? 0.0 : 1.5);
        if (best === null || step < best[0]) best = [step, previous];
      }
      if (best) layer.set(cut, best);
    }
    if (layer.size) layers.push(layer);
    else {
      const top = Math.max(...previousLayer.keys());
      layers.push(new Map([[Math.min(totalSounds, top + 1), [0.0, top]]]));
    }
  }
  const lastLayer = layers[layers.length - 1];
  let last = -1;
  let bestCost = Infinity;
  for (const [cut, [cost]] of lastLayer) {
    const total = cost + Math.abs((totalSounds - cut) - expected[expected.length - 1]);
    if (total < bestCost) {
      bestCost = total;
      last = cut;
    }
  }
  const bounds = [totalSounds, last];
  for (let k = layers.length - 1; k >= 1; k--) bounds.push(layers[k].get(bounds[bounds.length - 1])![1] as number);
  return bounds.reverse();
}

/** Rule groups that can be switched off for research ablations. */
export const GROUPS = ["addresses", "lists", "lexicon", "phone", "numbers", "dates", "times", "money", "decades",
  "ranges", "fractions", "units", "roman", "abbreviations", "initials", "symbols", "slashes", "markup", "pauses",
  "long-runs"] as const;

/** Prepares text for a voice: returns the segments to speak, each with its pause and source span. */
export function prepare(text: string, locale: string, usWords = false, disabled: Iterable<string> = []): Segment[] {
  return new Preparer(locale, usWords, disabled).prepare(text);
}

/** American word forms as raw phoneme blocks, for US voices trained on British phonemes. */
export const american = (text: string): string => Preparer.american(text);
