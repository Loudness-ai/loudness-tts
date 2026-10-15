// A quick end-to-end check for other runtimes (Bun, Deno): node test/smoke.mjs / bun test/smoke.mjs /
// deno run -A test/smoke.mjs. Needs en_US-clean100-medium in the voice cache (LOCAL_TTS_HOME).
import { TTS, loudness, version } from "local-tts";

const tts = new TTS({ offline: true });
const speech = await tts.synthesize("Dr. Smith paid $5.50 on 4/5/2026.");
const m = loudness.measure(speech.samples, speech.sampleRate);
if (speech.spoken !== "Doctor Smith paid 5 dollars and 50 cents on April 5th, 20 26." || Math.abs(m.integratedLufs + 16) > 0.3
  || speech.metrics.timings !== "exact" || !speech.srt().startsWith("1\n")) {
  throw new Error(`unexpected result: ${JSON.stringify(speech.toJSON()).slice(0, 400)}`);
}
console.log(`local-tts ${version}: ${speech.duration.toFixed(2)} s of speech in ${speech.metrics.seconds} s (${speech.voice.id})`);
