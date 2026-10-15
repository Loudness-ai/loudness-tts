# loudness-tts for JavaScript and TypeScript

**Fast, natural text to speech in Node, Bun, Deno and the browser.** Free, offline after the first voice download, no
API key, no native add-ons. By [Loudness.ai](https://loudness.ai/).

- **1,683 ranked voices in 32 language variants:** every Piper voice that allows commercial use, scored for naturalness
  and clarity in [our open study](https://loudness.ai/research/piper-tts-voices-ranked/). `loudness-tts` picks the best voice
  for your language automatically and shows each voice's licence and credit line.
- **Reads like a person:** prices, dates, times, units, phone numbers, abbreviations and Roman numerals become words
  (English and Spanish), with natural pauses, calmer delivery and click-free joins
  ([how and how well](https://loudness.ai/research/piper-tts-text-normalization/)).
- **Ready to publish:** loudness normalized to −16 LUFS with true peak at or below −1.5 dBTP, exact word timings, SRT and
  VTT subtitles.
- **Fast everywhere:** about 20 times faster than real time in Node on an Apple M4 and 15 times in Chrome; a stream's
  first sentence arrives in under 0.2 s. WebAssembly only (ONNX Runtime Web and espeak-ng), so it installs the same on
  every system.
- **The same speech as the Python edition:** the same phonemes, text preparation and word timings, checked by tests on
  8,609 texts and every voice model.

```sh
npm install loudness-tts
npx loudness-tts "Dr. Smith paid \$5.50 on 4/5/2026." -o speech.wav --srt speech.srt
```

> **The npm package is coming soon.** Until then, install the file attached to the
> [latest release](https://github.com/Loudness-ai/loudness-tts/releases/latest):
> `npm install https://github.com/Loudness-ai/loudness-tts/releases/download/v0.1.1/loudness-tts-0.1.1.tgz`.

The first run downloads the best English (US) voice (about 78 MB, checked against its published checksum). Voices are
kept in `~/.cache/loudness-tts`, shared with the Python edition. Try voices first:
[Piper Voice Explorer](https://huggingface.co/spaces/Loudness-ai/piper-voice-explorer) ·
[Loudness.ai Text to Speech](https://loudness.ai/tools/text-to-speech/).

## Node, Bun, Deno

```js
import { speak, synthesize, stream, voices } from "loudness-tts";   // Deno: "npm:loudness-tts"

await speak("Hello! The meeting starts at 9:30 AM.", "hello.wav");   // best English (US) voice

const speech = await synthesize("Llegamos el 15/10/2026 a las 10:30.", { language: "es-ES" });
await speech.save("es.wav");           // 16-bit WAV (speech.save(path, 24) for 24-bit); speech.wav() gives the bytes
await speech.saveSrt("es.srt");        // subtitles; speech.vtt() for web players
console.log(speech.spoken);            // what the voice actually read
console.log(speech.voice.credit);      // credit line, when the voice's licence asks for one

for await (const chunk of stream("A long text is spoken sentence by sentence, so playback starts at once.")) {
  player.write(chunk.pcm());           // 16-bit mono PCM at chunk.sampleRate
}

voices({ language: "en-US", sex: "female" }).slice(0, 3);   // best English (US) female voices
```

For services, keep one engine so voices stay loaded:

```js
import { TTS } from "loudness-tts";

const tts = new TTS({ language: "en-US", threads: 2, cacheSize: 4 });
await tts.preload();                                          // download and load now, not on the first request
const speech = await tts.synthesize(text, { voice: "en_US-libritts-high-s118", speed: 1.1, pause: "short",
  signal: AbortSignal.timeout(30_000) });
```

| Option | Default | Meaning |
| --- | --- | --- |
| `voice` | `"auto"` | voice id, model name, OpenAI voice name (`nova`, `onyx`, …), `"best:es-MX"` or a path or URL to your own Piper `.onnx` |
| `language` | `en-US` | locale or language code used to pick the best voice |
| `speed` | 1.0 | 0.5 to 2.0 |
| `pause` | `"normal"` | `short`, `normal`, `long` or seconds between sentences |
| `expressiveness`, `rhythm` | 1.0 | 0.5 to 1.5: variation in sound and in timing |
| `prepareText` | true | turn numbers, dates, money and abbreviations into words |
| `normalize`, `targetLufs`, `truePeakDb` | true, −16, −1.5 | loudness normalization |
| `signal` | none | an `AbortSignal` to stop (`AbortSignal.timeout(ms)` for a time limit) |

Errors are `TTSError` with a stable `code` (`VOICE_NOT_FOUND`, `OFFLINE`, `TEXT_TOO_LONG`, `CANCELLED`, `TIMEOUT`, …).
TypeScript types are included.

## In the browser

The same package runs in a web page, with your bundler (Vite, webpack, esbuild, Next.js, …):

```js
import { speak, synthesize } from "loudness-tts";

await speak("Hello from this page!");                       // plays it
const speech = await synthesize("Download me.");
audio.src = URL.createObjectURL(speech.blob());              // a WAV Blob
```

Speech is made on the visitor's device; nothing is sent to a server. Voices download once from Hugging Face and stay in
the browser's cache. In a browser, `auto` picks the best voice under 80 MB (`en_US-bryce-medium` for English, 64 MB),
and only voices whose host allows web pages to download them (`voice.browser`). espeak-ng's data and ONNX Runtime's
WebAssembly load from jsDelivr by default; serve them yourself with `configure({ espeakDataUrl, onnxWasmUrl })`. Serve
your page with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` to let
inference use several threads.

## Command line and OpenAI-compatible server

```sh
npx loudness-tts "Text to speak" -o out.wav [--voice ID] [--language es-ES] [--speed 1.1] [--srt out.srt] [--vtt out.vtt]
echo "Text from a pipe" | npx loudness-tts - > out.wav
npx loudness-tts voices --language en-US --sex female --min-score 95
npx loudness-tts download --language de-DE        # for offline machines and Docker images
npx loudness-tts serve --port 8880                # add --host 0.0.0.0 --api-key SECRET to serve other machines
```

The server speaks OpenAI's speech API (`POST /v1/audio/speech`), so apps such as Open WebUI, LibreChat and AnythingLLM
can use it with base URL `http://127.0.0.1:8880/v1`. `response_format` `pcm` streams 24 kHz audio while it is made; `mp3`,
`opus`, `aac` and `flac` need FFmpeg. `POST /v1/tts` returns JSON with the audio, word timings, SRT and VTT. Text is never
logged or stored. In code: `import { serve } from "loudness-tts"; await serve({ port: 8880 })`.

## Voices and licences

Every voice in the catalogue allows commercial use. About 1,600 ask for a credit line (CC BY, CC BY-SA, Apache 2.0),
51 need nothing, two are copyleft and 17 have unclear terms; `loudness-tts` never picks those last two groups on its own.
`voice.licence`, `voice.use` and `voice.credit` tell you what applies. Voice files come from their original sources (the
official Piper voices at a pinned revision, or the community author) and are verified before use. Ranking and method:
[Open Piper TTS Voices Ranked](https://loudness.ai/research/piper-tts-voices-ranked/), data on
[Hugging Face](https://huggingface.co/datasets/Loudness-ai/piper-voices-ranked).

## Privacy

Speech is made on your machine or in the visitor's browser. The only network requests are voice downloads (one per
voice) and, in browsers, the engine files from jsDelivr unless you serve them. Set `LOUDNESS_TTS_OFFLINE=1` (or
`new TTS({ offline: true })`) to forbid downloads; `LOUDNESS_TTS_HOME` moves the voice cache.

## Licence and credits

GPL-3.0-or-later, because it builds on [Piper](https://github.com/OHF-Voice/piper1-gpl) and espeak-ng (compiled to
WebAssembly from the same commit Piper uses). The text preparation layer (`import { prepare } from "loudness-tts/textprep"`)
is Apache-2.0 and can be reused on its own. Voices keep their own licences. If loudness-tts helps your project, a link to
[loudness.ai](https://loudness.ai/) is appreciated (not required).

Python edition: `pip install loudness-tts` ([PyPI](https://pypi.org/project/loudness-tts/)). More from Loudness.ai:
[free online audio tools](https://loudness.ai/tools/) · [Audio Kit](https://github.com/Loudness-ai/loudness-audio-kit) ·
[Research & Benchmarks](https://loudness.ai/research/) · [Hugging Face](https://huggingface.co/Loudness-ai)
