"""Reference outputs from the Python edition, for the JavaScript edition's parity tests.

    .venv/bin/python scripts/make_js_fixtures.py <voice configs dir>

Writes javascript/test/fixtures/{textprep,frontend,configs,cutpoints}.json. Offsets are converted to UTF-16 code units (what
JavaScript strings use). The voice configs dir holds the <model>.onnx.json files of the catalogue (their phoneme maps).
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))

from local_tts import textprep  # noqa: E402
from local_tts.engine import clean_text  # noqa: E402
from piper import PiperVoice  # noqa: E402
from piper.config import PiperConfig  # noqa: E402
from piper.phonemize_espeak import ESPEAK_DATA_DIR  # noqa: E402

OUT = ROOT / "javascript" / "test" / "fixtures"
RESEARCH = ROOT.parent.parent / "research"
TTSPREP = ROOT.parent / "ttsprep"


def u16(text, index):
    return len(text[:index].encode("utf-16-le")) // 2


def segments(text, locale, us):
    out = []
    for s in textprep.prepare(text, locale, us):
        out.append([s.text, u16(text, s.start), u16(text, s.end), s.pause,
                    [[w, u16(text, a), u16(text, b)] for w, a, b in s.tokens]])
    return out


EN = ["the", "and", "we", "met", "today", "Dr.", "St.", "Mr.", "Mrs.", "Ms.", "Prof.", "Inc.", "Jr.", "Co.", "vs", "vs.",
      "approx.", "e.g.", "etc.", "Elm", "Ave", "5th", "5", "12", "2026", "1,250,000", "3.14", "10 000", "1/2", "3/4",
      "7/8", "4/5/2026", "12/31/99", "2026-10-16", "2026/1/5", "16.10.2026", "9:30", "9:05 a.m.", "10:00", "12:00",
      "0:00", "14:10:30", "7:45 PM", "9.05 am", "$5.50", "$1", "$0.99", "€12", "£3.20", "¥500", "₹99", "5 USD", "12 EUR",
      "$2.5m", "$3 bn", "$1,000", "$1 000", "20%", "5km", "60 km/h", "3 m", "5 min", "2 h", "100°C", "45°", "1 kg",
      "1.0 kg", "12 MB", "1990s", "'80s", "2000s", "1900s", "1-5", "10–20", "+1 555-123-4567", "(555) 123-4567",
      "555-1234", "Chapter IV", "chapter iv", "Henry VIII", "Louis XIV", "World War II", "Super Bowl LVII", "Jan 5",
      "March 3, 2026", "5 March", "Sept. 9th, 2025", "Dec 25", "May 2026", "#1", "~5", "N/A", "w/ them", "and/or",
      "A/B", "yes/no", "TCP/IP", "hello@loudness.ai", "https://loudness.ai/tools/", "www.example.com/path/",
      "loudness.ai", "**bold**", "*it*", "`code`", "#tag", "👍", "🇺🇸", "❤️", "...", "…", " - ", "—", "J. K. Rowling",
      "George W. Bush", "schedule", "tomato", "privacy", "can't", "Z", "Wi-Fi", "A-Z", "St. Louis", "Mt. Everest",
      "no. 5", "fig. 3", "p. 12", "pp. 4–7", "NASA", "U.S.", "a.m.", "x", "I", "It", "The", "\"quoted\"", "(aside)",
      "it's", "rock & roll", "[[hɛloʊ]]", "٣٤٥", "１２３", "日本", "Ελλάδα"]
ES = ["el", "la", "y", "de", "hoy", "Sr.", "Sra.", "Dra.", "Lic.", "EE. UU.", "EEUU", "c/ Mayor", "Av. de", "Avda. del",
      "núm. 5", "nº 7", "n.º 3", "No. 4", "no 4", "pág. 3", "págs. 4", "Ud.", "aprox.", "15/10/2026", "1/1/26",
      "2026-10-16", "10:30", "9 a.m.", "3 pm", "12 p.m.", "14:00", "21:15", "1:00", "0:30", "10:15:30", "5,50 €",
      "$100", "$1", "1.250,50 €", "€3", "10 USD", "¥5", "10.º", "1.ª", "Felipe VI", "Isabel II", "Juan XXIII",
      "Pío XII", "loudness.ai", "Loudness", "1/2", "3/4", "3–5", "5 km", "2 h", "1 h", "1 min", "100 kg", "3 m", "25 °C",
      "20 dB", "10 kHz", "912 345 678", "+34 912 345 678", "91 234 56 78", "900 000 000", "600 000 000 euros",
      "y/o", "sí/no", "#3", "~10", "Hola", "¿Qué?", "¡Bien!", "…", "...", "—", "señor", "usted", "1 000 000",
      "10 000", "Sto. Domingo", "Gral. San Martín", "Pza. Mayor", "Tel. 5", "admón.", "J. R. Jiménez", "👍"]
OTHER = {"de-DE": ["und", "aber", "der", "Tag", "1-5", "10–20", "12.10.2026", "3,5", "10 000", "z.B.", "Dr.", "😀"],
         "fr-FR": ["et", "mais", "le", "jour", "1-5", "10–20", "3,5", "10 000", "M.", "Mme", "« oui »", "😀"],
         "it-IT": ["e", "ma", "il", "giorno", "1-5", "3,5", "10 000", "perché", "😀"],
         "pt-BR": ["e", "mas", "o", "dia", "1-5", "3,5", "R$ 10", "então", "😀"],
         "nl-NL": ["en", "maar", "de", "dag", "1-5", "3,5", "omdat", "😀"],
         "zh-CN": ["你好", "世界", "。", "，", "1-5", "2026", "😀"],
         "ar-JO": ["مرحبا", "بالعالم", "؟", "،", "١٢٣", "😀"], "hi-IN": ["नमस्ते", "।", "१२३"]}
JOIN = [" ", " ", " ", " ", "", ". ", ", ", "! ", "? ", "; ", ": ", "\n", "\n\n", " - ", "... "]
STARTS = ["", "", "", "- ", "* ", "1. ", "2) ", "# ", "## ", "a) ", "• "]


def fuzz(rng, words, n):
    out = []
    for _ in range(n):
        parts = [rng.choice(STARTS)]
        for k in range(rng.randint(2, 14)):
            if k:
                parts.append(rng.choice(JOIN))
                if parts[-1].endswith("\n") and rng.random() < 0.3:
                    parts.append(rng.choice(STARTS))
            parts.append(rng.choice(words))
        if rng.random() < 0.7:
            parts.append(rng.choice([".", "!", "?", "", ",", "...", " —"]))
        out.append("".join(parts))
    return out


def text_corpus():
    rng = random.Random(20261016)
    cases = []
    for case in json.loads((TTSPREP / "evaluation" / "cases.json").read_text()):
        for us in (False, True) if case["locale"] == "en-US" else (False,):
            cases.append((case["text"], case["locale"], us))
            cases.append((case["reference"], case["locale"], us))
    nemo = TTSPREP / "evaluation" / "nemo-test-cases"
    for lang, locales in (("en", ("en-US", "en-GB")), ("es", ("es-ES", "es-MX"))):
        for f in sorted((nemo / lang).glob("*.txt")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if "~" in line:
                    for locale in locales:
                        cases.append((line.split("~")[0], locale, False))
    checks = RESEARCH / "tts-voice-sets-2026-10" / "text-checks"
    for name, locale in (("en.json", "en-US"), ("accent.json", "en-US"), ("long.json", "en-US"), ("es.json", "es-ES")):
        for text in json.loads((checks / name).read_text(encoding="utf-8")):
            cases.append((text, locale, locale == "en-US"))
            if locale == "en-US":
                cases.append((text, "en-GB", False))
    bench = json.loads((RESEARCH / "tts-voice-benchmark-2026-10" / "data" / "benchmark-texts.json").read_text())
    for lang, texts in bench.items():
        for text in texts:
            cases.append((text, {"en": "en-US", "es": "es-ES", "zh": "zh-CN", "no": "nb-NO"}.get(lang, lang), False))
    for text in fuzz(rng, EN, 3000):
        cases.append((text, rng.choice(["en-US", "en-US", "en-GB", "en-IE", "en-AU"]), rng.random() < 0.5))
    for text in fuzz(rng, ES, 2000):
        cases.append((text, rng.choice(["es-ES", "es-MX", "es-AR"]), False))
    for locale, words in OTHER.items():
        for text in fuzz(rng, words + EN[:40], 150):
            cases.append((text, locale, False))
    long_words = (EN[:12] * 40)
    for locale in ("en-US", "es-ES", "de-DE", "zh-CN"):
        for k in range(20):
            rng.shuffle(long_words)
            cases.append((" ".join(long_words[: 30 + 7 * k]), locale, False))
    return cases


def main():
    configs = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for text, locale, us in text_corpus():
        if (text, locale, us) in seen:
            continue
        seen.add((text, locale, us))
        rows.append({"t": text, "l": locale, "u": us, "s": segments(text, locale, us)})
    (OUT / "textprep.json").write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # Phonemes and ids, per catalogue model: what Piper (and the Python edition) would send to the model.
    bench = json.loads((RESEARCH / "tts-voice-benchmark-2026-10" / "data" / "benchmark-texts.json").read_text())
    stress = ["Hello, world! 3.14 and 1,250 apples; really? Yes: 50% (maybe).", "\"Quoted\" — dashes – and… ellipsis",
              "e.g. Dr. Smith vs. Mr. Jones, etc.", "12:30 2026-10-16 #5 @home", "A\tB C", ""]
    frontend, maps = [], {}
    for path in sorted(configs.glob("*.onnx.json")):
        model = path.name[: -len(".onnx.json")]
        raw = json.loads(path.read_text(encoding="utf-8"))
        maps[model] = {k: raw[k] for k in ("espeak", "phoneme_type", "phoneme_id_map", "vowel_clusters") if k in raw}
        cfg = PiperConfig.from_dict(raw)
        voice = PiperVoice(session=None, config=cfg, espeak_data_dir=ESPEAK_DATA_DIR)
        lang = model.split("_")[0]
        texts = list(bench.get({"nb": "no"}.get(lang, lang), bench["en"])) + stress
        locale = model.split("-")[0].replace("_", "-")
        if lang in ("en", "es"):
            texts += [s.text for t in [r["t"] for r in rows if r["l"] == locale][:150] for s in
                      textprep.prepare(clean_text(t, 100000) if t.strip() else "x", locale,
                                       locale == "en-US" and cfg.espeak_voice == "en")]
        id_map = cfg.phoneme_id_map
        for text in texts:
            sentences = voice.phonemize(text)
            ids = []
            for sentence in sentences:  # the engine's filtering (engine._sentences)
                ph = [p for i, p in enumerate(sentence) if p != " " or (i and sentence[i - 1] != " ")]
                ph = [p for p in ph if p in id_map]
                while ph and ph[0] == " ":
                    ph.pop(0)
                while ph and ph[-1] == " ":
                    ph.pop()
                ids.append(voice.phonemes_to_ids(ph) if ph else [])
            frontend.append({"m": model, "t": text, "p": ["".join(s) for s in sentences], "n": [len(s) for s in sentences],
                             "i": ids})
    (OUT / "frontend.json").write_text(json.dumps(frontend, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (OUT / "configs.json").write_text(json.dumps(maps, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    rng = random.Random(7)
    cuts = []
    for _ in range(400):
        counts = [rng.randint(1, 9) for _ in range(rng.randint(1, 14))]
        total = max(1, int(sum(counts) * rng.uniform(0.5, 1.6)))
        starts = sorted(rng.sample(range(total), k=min(total, rng.randint(1, len(counts) + 2))))
        cuts.append({"c": counts, "t": total, "w": starts, "r": textprep.cut_points(counts, total, set(starts))})
    (OUT / "cutpoints.json").write_text(json.dumps(cuts, separators=(",", ":")), encoding="utf-8")
    print(len(rows), "text cases,", len(frontend), "front-end cases,", len(cuts), "cut cases")


if __name__ == "__main__":
    main()
