// End-to-end parity with the Python edition, with random variation off so both are repeatable: the same spoken text,
// the same number of samples, the same word timings, the same loudness. Cases whose voice is not in the cache are
// skipped. Set LOCAL_TTS_PY_AUDIO to a folder of the Python edition's samples for a sample-level comparison.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { TTS, downloadedModels, loudness } from "../dist/node.js";

const cases = JSON.parse(readFileSync(new URL("./fixtures/synthesis.json", import.meta.url), "utf8"));
const have = new Set(await downloadedModels());
const tts = new TTS({ offline: true, cacheSize: 4 });
tts.calm = 0;

for (const c of cases) {
  const model = c.name.replace(/-\d+$/, "");
  test(`same speech as the Python edition: ${c.name}`, { skip: !have.has(model) && `${model} is not in the cache` }, async () => {
    const s = await tts.synthesize(c.text, { voice: c.voice });
    assert.equal(s.text, c.cleaned);
    assert.equal(s.spoken, c.spoken);
    assert.equal(s.metrics.timings, c.metrics);
    if (!c.repeatable) return; // a voice that keeps its own random variation: timing differs on every run
    assert.equal(s.samples.length, c.samples);
    assert.deepEqual({ duration: s.timings.duration, segments: s.timings.segments, words: s.timings.words }, c.timings);
    const m = loudness.measure(s.samples, s.sampleRate);
    assert.ok(Math.abs(m.integratedLufs - c.lufs) < 0.02, `loudness ${m.integratedLufs} vs ${c.lufs}`);
    assert.ok(Math.abs(m.truePeakDbtp - c.peak) < 0.05, `true peak ${m.truePeakDbtp} vs ${c.peak}`);
    const dir = process.env.LOCAL_TTS_PY_AUDIO;
    if (dir && existsSync(`${dir}/${c.name}.f32`)) {
      const ref = new Float32Array(readFileSync(`${dir}/${c.name}.f32`).buffer.slice(0));
      let worst = 0;
      for (let i = 0; i < ref.length; i++) worst = Math.max(worst, Math.abs(ref[i] - s.samples[i]));
      assert.ok(worst < 2e-3, `largest sample difference ${worst}`);
      console.log(`${c.name}: largest sample difference ${worst.toExponential(2)}`);
    }
  });
}
