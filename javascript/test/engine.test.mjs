// Synthesis with real voices. Runs when en_US-clean100-medium is in the voice cache (LOUDNESS_TTS_HOME); CI downloads it
// first with `loudness-tts download`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { TTS, downloadedModels, loudness, TTSError } from "../dist/node.js";

const EN = "en_US-clean100-medium";
const ready = (await downloadedModels()).includes(EN);
const tts = new TTS({ offline: true });

test("speech is normalized, timed and subtitled", { skip: !ready && `${EN} is not in the cache` }, async () => {
  await tts.preload(); // speed is measured on a loaded voice
  await tts.synthesize("Warm up.");
  const s = await tts.synthesize("Dr. Smith paid $5.50 on 4/5/2026. Call +1 555-123-4567, please.");
  const m = loudness.measure(s.samples, s.sampleRate);
  assert.ok(Math.abs(m.integratedLufs + 16) <= 0.3, `loudness ${m.integratedLufs}`);
  assert.ok(m.truePeakDbtp <= -1.4, `true peak ${m.truePeakDbtp}`);
  assert.ok(s.duration > 3 && s.metrics.realTimeFactor < 1);
  assert.equal(s.metrics.timings, "exact");
  const words = s.timings.words.map(([a, b]) => s.text.slice(a, b));
  assert.ok(words.includes("$5.50") && words.includes("4/5/2026."), words.join(" "));
  const starts = s.timings.words.map((w) => w[2]);
  assert.deepEqual(starts, [...starts].sort((a, b) => a - b));
  assert.ok(s.srt().startsWith("1\n") && s.vtt().startsWith("WEBVTT"));
  const wav = s.wav();
  assert.equal(new TextDecoder().decode(wav.subarray(0, 4)), "RIFF");
});

test("text preparation changes what is said", { skip: !ready }, async () => {
  const raw = await tts.synthesize("It costs $5.50.", { prepareText: false });
  const prepared = await tts.synthesize("It costs $5.50.");
  assert.equal(raw.spoken, "It costs $5.50.");
  assert.equal(prepared.spoken, "It costs 5 dollars and 50 cents.");
});

test("streaming yields sentences in order", { skip: !ready }, async () => {
  const chunks = [];
  for await (const chunk of tts.stream("First sentence. Second sentence! Third?")) chunks.push(chunk);
  assert.ok(chunks.length >= 3 && chunks.at(-1).final);
  assert.deepEqual(chunks.map((c) => c.index), chunks.map((_, i) => i));
});

test("options are checked and work can be stopped", { skip: !ready }, async () => {
  await assert.rejects(tts.synthesize("Hello", { speed: 5 }), (e) => e instanceof TTSError && e.code === "OPTION_INVALID");
  const stop = new AbortController();
  stop.abort();
  await assert.rejects(tts.synthesize("Hello there.", { signal: stop.signal }), (e) => e.code === "CANCELLED");
  const long = Array.from({ length: 40 }, (_, i) => `This is sentence number ${i + 1}.`).join(" ");
  const later = new AbortController();
  setTimeout(() => later.abort(), 30);
  await assert.rejects(tts.synthesize(long, { signal: later.signal }), (e) => e.code === "CANCELLED");
  await assert.rejects(tts.synthesize(long, { signal: AbortSignal.timeout(30) }), (e) => e.code === "TIMEOUT");
});
