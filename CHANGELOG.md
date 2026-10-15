# Changelog

## 0.1.1 (2026-10-16)

- JavaScript in browsers: espeak-ng's data now loads from this release on GitHub (through jsDelivr), so web pages work
  without any setup before the npm package exists.
- The JavaScript package file is attached to every GitHub release (`npm install <file URL>`).
- The Python package page links to the JavaScript edition on GitHub.

## 0.1.0 (2026-10-16)

First release, in two editions that produce the same speech: Python (PyPI) and JavaScript/TypeScript (npm) for Node,
Bun, Deno and browsers. The JavaScript edition runs on WebAssembly (espeak-ng built from Piper's commit, ONNX Runtime
Web) with no native add-ons; its phonemes, text preparation and word timings are tested against the Python edition.

- Text to speech with every commercially usable Piper voice (1,683 voices, 32 language variants), ranked by
  Loudness.ai's open voice study; automatic best-voice choice per language; licences and credit lines built in.
- Verified voice downloads (size and checksum) from the original sources, then fully offline.
- Text preparation for English and Spanish (money, dates, times, units, phone numbers, abbreviations, Roman numerals),
  natural pauses, calmer delivery, click-free joins, American pronunciations for US voices trained on British phonemes.
- Loudness normalization (−16 LUFS, true peak ≤ −1.5 dBTP) without FFmpeg; word timings; SRT and VTT subtitles.
- Streaming by sentence, cancellation, timeouts, bounded text length.
- Command line (`say`, `voices`, `languages`, `download`, `serve`, `cache`, `doctor`).
- OpenAI-compatible speech server (`/v1/audio/speech`, streamed PCM, optional API key and CORS) and a JSON endpoint
  with timings and subtitles.
