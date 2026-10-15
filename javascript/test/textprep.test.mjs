import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { prepare, cutPoints } from "../dist/textprep.js";

const load = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8"));

test("text preparation matches the Python edition", () => {
  const cases = load("textprep.json");
  let failures = 0;
  for (const c of cases) {
    const got = prepare(c.t, c.l, c.u).map((s) => [s.text, s.start, s.end, s.pause, s.tokens]);
    try {
      assert.deepEqual(got, c.s);
    } catch (error) {
      if (failures++ < 8) console.log(JSON.stringify({ text: c.t, locale: c.l, us: c.u, want: c.s, got }));
    }
  }
  assert.equal(failures, 0, `${failures} of ${cases.length} cases differ`);
});

test("word timing cut points match the Python edition", () => {
  for (const c of load("cutpoints.json")) assert.deepEqual(cutPoints(c.c, c.t, new Set(c.w)), c.r);
});
