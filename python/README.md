# local-tts

**Fast, natural text to speech on your own computer or server.** Free, offline after the first voice download, no API key.
By [Loudness.ai](https://loudness.ai/).

- **1,683 ranked voices in 32 language variants:** every Piper voice that allows commercial use, scored for naturalness
  and clarity in [our open study](https://loudness.ai/research/piper-tts-voices-ranked/). `local-tts` picks the best voice
  for your language automatically and shows each voice's licence and credit line.
- **Reads like a person:** prices, dates, times, units, phone numbers, abbreviations and Roman numerals become words
  (English and Spanish), with natural pauses, calmer delivery and click-free joins
  ([how and how well](https://loudness.ai/research/piper-tts-text-normalization/)).
- **Ready to publish:** loudness normalized to −16 LUFS with true peak at or below −1.5 dBTP, word timings, SRT and VTT subtitles.
- **Fast on a CPU:** 16 to 21 times faster than real time with medium voices on an Apple M4, first streamed sentence in
  about 0.1 s. No PyTorch, no GPU, no FFmpeg needed (FFmpeg only adds MP3/Opus/AAC/FLAC output).
- **Works where you work:** Python library, command line, an **OpenAI-compatible speech server**, Docker, and (soon) npm.

```sh
pip install local-tts
local-tts "Dr. Smith paid $5.50 on 4/5/2026." -o speech.wav --srt speech.srt
```

The first run downloads the best English (US) voice (about 60 MB, checked against its published checksum). Try voices in
the browser first: [Piper Voice Explorer](https://huggingface.co/spaces/Loudness-ai/piper-voice-explorer) ·
[Loudness.ai Text to Speech](https://loudness.ai/tools/text-to-speech/).

## Python

```python
import local_tts

local_tts.speak("Hello! The meeting starts at 9:30 AM.", "hello.wav")          # best English (US) voice

speech = local_tts.synthesize("Llegamos el 15/10/2026 a las 10:30.", language="es-ES")
speech.save("es.wav")            # 16-bit WAV (speech.save(path, bit_depth=24) for 24-bit)
speech.save_srt("es.srt")        # subtitles; speech.vtt() for web players
print(speech.spoken)             # what the voice actually read
print(speech.voice.credit)       # credit line, when the voice's licence asks for one

for chunk in local_tts.stream("A long text is spoken sentence by sentence, so playback starts at once."):
    player.write(chunk.pcm())    # 16-bit mono PCM at chunk.sample_rate

local_tts.voices(language="de", sex="female")[:3]                            # best German female voices
```

For services, keep one engine so voices stay loaded:

```python
from local_tts import TTS

tts = TTS(language="en-US", threads=2, cache_size=4)
tts.preload()                                         # download and load now, not on the first request
speech = tts.synthesize(text, voice="en_US-libritts-high-s118", speed=1.1, pause="short", timeout=30)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `voice` | `"auto"` | voice id, model name, OpenAI voice name (`nova`, `onyx`, …), `"best:es-MX"` or a path to your own Piper `.onnx` |
| `language` | `en-US` | locale or language code used to pick the best voice |
| `speed` | 1.0 | 0.5 to 2.0 |
| `pause` | `"normal"` | `short`, `normal`, `long` or seconds between sentences |
| `expressiveness`, `rhythm` | 1.0 | 0.5 to 1.5: variation in sound and in timing |
| `prepare_text` | True | turn numbers, dates, money and abbreviations into words |
| `normalize`, `target_lufs`, `true_peak_db` | True, −16, −1.5 | loudness normalization |
| `cancel`, `timeout` | None | a `threading.Event` (or callable) to stop, and a time limit in seconds |

Word timings are estimated from the text by default; `pip install "local-tts[timings]"` makes them exact (taken from the
voice model's own sound durations).

## Command line

```sh
local-tts "Text to speak" -o out.wav [--voice ID] [--language es-ES] [--speed 1.1] [--srt out.srt] [--vtt out.vtt]
echo "Text from a pipe" | local-tts - > out.wav
local-tts voices --language fr --sex female --min-score 80
local-tts languages
local-tts download --language de-DE          # for offline machines and Docker images
local-tts doctor --synthesize                # check the installation
```

## OpenAI-compatible server

```sh
local-tts serve --port 8880                  # add --host 0.0.0.0 --api-key SECRET to serve other machines
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8880/v1", api_key="not-needed")
client.audio.speech.create(model="tts-1", voice="nova", input="Hello from my own server!",
                          response_format="wav").write_to_file("hello.wav")
```

Apps that let you set an OpenAI-compatible text-to-speech URL, such as Open WebUI, LibreChat and AnythingLLM, can use it
directly: base URL `http://127.0.0.1:8880/v1`, any model name, a voice id or an OpenAI voice name. `response_format`
`pcm` streams 24 kHz audio while it is made; `mp3`, `opus`, `aac` and `flac` need FFmpeg. `POST /v1/tts` returns JSON with
the audio, word timings, SRT and VTT. Text is never logged or stored.

## Docker

```sh
docker run -p 8880:8880 -v local-tts:/data ghcr.io/loudness-ai/local-tts
```

## Voices and licences

Every voice in the catalogue allows commercial use. About 1,600 ask for a credit line (CC BY, CC BY-SA, Apache 2.0),
51 need nothing, two are copyleft and 17 have unclear terms; `local-tts` never picks those last two groups on its own.
`voice.licence`, `voice.use` and `voice.credit` tell you what applies. Voice files come from their original sources (the
official Piper voices at a pinned revision, or the community author) and are verified before use. The ranking and its
method: [Open Piper TTS Voices Ranked](https://loudness.ai/research/piper-tts-voices-ranked/), with the data on
[Hugging Face](https://huggingface.co/datasets/Loudness-ai/piper-voices-ranked).

## Privacy

Speech is made on your machine. The only network requests are voice downloads (one per voice, from Hugging Face or the
voice author). Set `LOCAL_TTS_OFFLINE=1` to forbid them; `LOCAL_TTS_HOME` moves the voice cache.

## Licence and credits

GPL-3.0-or-later, because it builds on [Piper](https://github.com/OHF-Voice/piper1-gpl) and espeak-ng. The text
preparation layer (`local_tts/textprep.py`) is Apache-2.0 and can be reused on its own. Voices keep their own licences.
If local-tts helps your project, a link to [loudness.ai](https://loudness.ai/) is appreciated (not required).

```bibtex
@software{local_tts,
  author = {Shehzad, Asad},
  title = {local-tts: natural text to speech on your own computer or server},
  publisher = {Loudness.ai},
  year = {2026},
  url = {https://github.com/Loudness-ai/local-tts}
}
```

More from Loudness.ai: [free online audio tools](https://loudness.ai/tools/) ·
[Audio Kit](https://github.com/Loudness-ai/loudness-audio-kit) (local noise removal and loudness normalization) ·
[Research & Benchmarks](https://loudness.ai/research/) · [Hugging Face](https://huggingface.co/Loudness-ai)
