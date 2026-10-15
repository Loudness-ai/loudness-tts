"""Repeatable end-to-end reference from the Python edition (random variation off), for the JavaScript edition's test.

    LOCAL_TTS_HOME=<cache with the voices> .venv/bin/python scripts/make_js_synthesis_fixture.py [audio dir]

Writes javascript/test/fixtures/synthesis.json (spoken text, timings in UTF-16 offsets, sample counts, loudness) and,
when an audio dir is given, the float32 samples of each case for a closer local comparison.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))

from local_tts import engine, loudness  # noqa: E402
from local_tts.downloader import downloaded_models  # noqa: E402

engine.CALM = 0.0
TEXTS = {
    "en_US-clean100-medium": ["Dr. Smith paid $5.50 on 4/5/2026. Call +1 555-123-4567, please.",
                              "Hello! The meeting starts at 9:30 AM, so please arrive a few minutes early.",
                              "Wait... what? Really?! Yes — absolutely.\n\nA new paragraph, with a schedule and privacy notes.",
                              "Great job 👍 see you soon",
                              "this sentence keeps going without any punctuation because people often paste long transcripts "
                              "or captions that were never punctuated and the voice has to read all of it in one breath which "
                              "can make the timing drift and the words run together so we want to know where it breaks"],
    "en_IE-jenny-medium": ["The 1990s, the 21st century, and chapter IV.", "It costs €12.99 with 20% VAT."],
    "de_DE-mls-medium": ["Guten Tag! Das Treffen beginnt am Montagmorgen, also komm bitte ein paar Minuten früher."],
    "en_US-libritts-high": ["Could you tell me where the nearest train station is?"],
}


def u16(text, i):
    return len(text[:i].encode("utf-16-le")) // 2


def main():
    audio_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    have = set(downloaded_models())
    tts = engine.TTS(offline=True, cache_size=4)
    out = []
    for model, texts in TEXTS.items():
        if model not in have:
            continue
        voice = next(v.id for v in __import__("local_tts").voices() if v.model == model and v.rank == min(
            x.rank for x in __import__("local_tts").voices() if x.model == model))
        for k, text in enumerate(texts):
            s = tts.synthesize(text, voice=voice)
            t = s.timings
            conv = lambda a, b: (u16(s.text, a), u16(s.text, b))  # noqa: E731
            timings = {"duration": t["duration"],
                       "segments": [[*conv(a, b), x, y] for a, b, x, y in t["segments"]],
                       "words": [[*conv(a, b), x, y] for a, b, x, y in t["words"]]}
            m = loudness.measure(s.samples, s.sample_rate)
            name = f"{model}-{k}"
            cfg = tts._load(s.voice).voice.config
            repeatable = cfg.noise_scale > 0.5 and cfg.noise_w_scale > 0.5  # voices at 0.5 or less keep their variation
            out.append({"name": name, "voice": voice, "text": text, "cleaned": s.text, "spoken": s.spoken,
                        "samples": int(s.samples.size), "rate": s.sample_rate, "timings": timings,
                        "lufs": m["integrated_lufs"], "peak": m["true_peak_dbtp"], "metrics": s.metrics["timings"],
                        "repeatable": repeatable})
            if audio_dir:
                audio_dir.mkdir(parents=True, exist_ok=True)
                s.samples.astype("<f4").tofile(audio_dir / f"{name}.f32")
    (ROOT / "javascript" / "test" / "fixtures" / "synthesis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                              encoding="utf-8")
    print(len(out), "cases")


if __name__ == "__main__":
    main()
