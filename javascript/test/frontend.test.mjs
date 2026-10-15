import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import "../dist/node.js";
import { phonemizeForVoice } from "../dist/espeak.js";
import { phonemesToIds, speakablePhonemes } from "../dist/engine.js";

const load = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8"));

test("phonemes and model ids match Piper for every catalogue voice", async () => {
  const configs = load("configs.json");
  const cases = load("frontend.json");
  let failures = 0;
  for (const c of cases) {
    const config = configs[c.m];
    const clusters = config.vowel_clusters?.length ? new Set(config.vowel_clusters.map((v) => v.join("\0"))) : undefined;
    const sentences = await phonemizeForVoice(config, c.t, clusters);
    const got = {
      p: sentences.map((s) => s.join("")),
      n: sentences.map((s) => s.length),
      i: sentences.map((s) => {
        const ph = speakablePhonemes(s, config.phoneme_id_map);
        return ph.length ? phonemesToIds(ph, config.phoneme_id_map) : [];
      }),
    };
    try {
      assert.deepEqual(got, { p: c.p, n: c.n, i: c.i });
    } catch {
      if (failures++ < 5) console.log(JSON.stringify({ model: c.m, text: c.t, want: c.p, got: got.p }));
    }
  }
  assert.equal(failures, 0, `${failures} of ${cases.length} cases differ`);
});
