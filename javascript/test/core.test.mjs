// Unit tests that need no voice model: catalogue, loudness, WAV, subtitles, MD5, downloads, CLI.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync, existsSync } from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import {
  AUTO_USE, LocalTTSError, NodeStore, findVoice, languages, loudness, openaiVoiceMap, resample, srt, voices, vtt, wavBytes,
} from "../dist/node.js";
import { Md5 } from "../dist/md5.js";
import { withAlignmentOutput } from "../dist/alignment.js";

test("catalogue size and fields", () => {
  const every = voices();
  assert.equal(every.length, 1683);
  assert.ok(every.every((v) => ["none", "credit", "copyleft", "unclear"].includes(v.use)));
  const locales = new Set(every.map((v) => v.locale));
  for (const l of ["en-US", "en-GB", "es-ES", "es-MX", "de-DE", "fr-FR"]) assert.ok(locales.has(l), l);
  assert.equal(new Set(every.map((v) => v.language)).size, 32); // language variants as ranked in the study
  assert.equal(languages().length, 31); // locales (English US and the US/UK mix share en-US)
});

test("auto picks the best voice with a clear licence", () => {
  for (const lang of ["en-US", "es", "de-DE", "fr-FR"]) {
    const v = findVoice("auto", lang);
    assert.ok(AUTO_USE.includes(v.use));
    assert.equal(v.overall, Math.max(...voices({ language: lang, use: AUTO_USE }).map((x) => x.overall)));
  }
  const web = findVoice("auto", "en-US", true);
  assert.ok(web.browser && web.locale === "en-US");
});

test("voices by id, model and OpenAI name; filters and credit", () => {
  const best = voices("en-US")[0];
  assert.equal(findVoice(best.id).id, best.id);
  assert.equal(findVoice("en_US-libritts-high").model, "en_US-libritts-high");
  assert.equal(findVoice("nova").sex, "F");
  assert.equal(findVoice("onyx").sex, "M");
  for (const name of ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]) assert.ok(name in openaiVoiceMap());
  assert.throws(() => findVoice("no-such-voice"), (e) => e instanceof LocalTTSError && e.code === "VOICE_NOT_FOUND");
  const f = voices({ language: "en-US", sex: "female", minScore: 95 });
  assert.ok(f.length && f.every((v) => v.sex === "F" && v.overall >= 95));
  assert.ok(findVoice("en_IE-jenny-medium").credit.includes("Jenny"));
  assert.equal(findVoice(voices({ use: "none" })[0].id).credit, null);
});

const sine = (rate, seconds = 5, amp = 1, freq = 997) =>
  Float32Array.from({ length: Math.trunc(seconds * rate) }, (_, i) => amp * Math.sin(2 * Math.PI * freq * i / rate));

test("BS.1770 reference tone reads -3.01 LUFS at any rate", () => {
  for (const rate of [16000, 22050, 48000]) {
    assert.ok(Math.abs(loudness.integratedLoudness(sine(rate), rate) + 3.01) < 0.05, String(rate));
  }
});

test("normalize reaches the target and limits peaks", () => {
  const rate = 22050;
  const x = sine(rate, 5, 0.05);
  x.fill(0.9, 5000, 5100);
  const { audio, normalized, limited } = loudness.normalize(x, rate, -14, -1.5);
  const m = loudness.measure(audio, rate);
  assert.ok(Math.abs(m.integratedLufs + 14) <= 0.15 && m.truePeakDbtp <= -1.4 && normalized && limited);
  assert.equal(loudness.normalize(new Float32Array(22050), 22050).normalized, false);
});

test("WAV files, resampling and subtitles", () => {
  const x = sine(22050, 0.1, 0.5);
  for (const depth of [16, 24]) {
    const wav = wavBytes(x, 22050, depth);
    const view = new DataView(wav.buffer);
    assert.deepEqual([view.getUint16(34, true), view.getUint32(24, true), view.getUint32(40, true)], [depth, 22050, x.length * depth / 8]);
  }
  assert.equal(resample(new Float32Array(22050), 22050, 24000).length, 24000);
  const text = "Hello there, my friend. This is a second sentence that is quite a bit longer than the first one, so it splits.";
  const timings = { version: 1, text, duration: 9000, segments: [[0, 23, 0, 1500], [24, 111, 1800, 8500]], words: [] };
  for (const [a, b, t0, t1] of timings.segments) {
    const found = [...text.slice(a, b).matchAll(/\S+/g)];
    found.forEach((m, i) => timings.words.push([a + m.index, a + m.index + m[0].length,
      t0 + i * (t1 - t0) / found.length, t0 + (i + 1) * (t1 - t0) / found.length]));
  }
  assert.ok(srt(timings).startsWith("1\n00:00:00,000 --> "));
  assert.ok(vtt(timings).startsWith("WEBVTT\n\n00:00:00.000 --> "));
  assert.ok(srt(timings).includes("Hello there, my friend."));
});

test("MD5 matches Node's", () => {
  for (const size of [0, 1, 55, 56, 63, 64, 65, 1000, 100_003]) {
    const data = Uint8Array.from({ length: size }, (_, i) => (i * 31 + 7) & 255);
    const chunked = new Md5();
    for (let i = 0; i < size; i += 777) chunked.update(data.subarray(i, i + 777));
    assert.equal(chunked.hex(), createHash("md5").update(data).digest("hex"), String(size));
  }
});

test("alignment patch leaves non-models alone", () => {
  assert.equal(withAlignmentOutput(new Uint8Array([1, 2, 3])), null);
});

test("downloads are checked against their checksum", async () => {
  const payload = Buffer.alloc(5000, 7);
  const server = http.createServer((req, res) => res.end(payload));
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${server.address().port}/m.onnx`;
  const root = mkdtempSync(path.join(os.tmpdir(), "ltts-"));
  const store = new NodeStore();
  try {
    const good = { url, size: payload.length, md5: createHash("md5").update(payload).digest("hex") };
    const data = await store.file("m", "m.onnx", good, { root });
    assert.equal(data.length, payload.length);
    assert.equal(readFileSync(path.join(root, "voices", "m", "m.onnx.ok"), "utf8"), `md5:${good.md5}`);
    await assert.rejects(store.file("n", "n.onnx", { ...good, md5: "0".repeat(32) }, { root }), (e) => e.code === "CHECKSUM_MISMATCH");
    assert.ok(!existsSync(path.join(root, "voices", "n", "n.onnx")));
    await assert.rejects(store.file("o", "o.onnx", good, { root, offline: true }), (e) => e.code === "OFFLINE");
  } finally {
    server.close();
  }
});

const cli = promisify(execFile);
const run = (...args) => cli(process.execPath, [fileURLToPath(new URL("../dist/cli.js", import.meta.url)), ...args]).catch((e) => e);

test("command line", async () => {
  const listed = await run("voices", "--language", "it", "--json");
  assert.equal(JSON.parse(listed.stdout)[0].locale, "it-IT");
  const failed = await run("say", "hello", "--voice", "nope", "-o", "x.wav", "--json");
  assert.equal(failed.code, 1);
  assert.equal(JSON.parse(failed.stdout).error.code, "VOICE_NOT_FOUND");
  assert.match((await run("--version")).stdout, /^local-tts \d/);
});
