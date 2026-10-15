// The OpenAI-compatible server. Runs when en_US-clean100-medium is in the voice cache.
import { test } from "node:test";
import assert from "node:assert/strict";
import { TTS, createServer, downloadedModels } from "../dist/node.js";

const ready = (await downloadedModels()).includes("en_US-clean100-medium");

test("OpenAI-compatible speech server", { skip: !ready && "en_US-clean100-medium is not in the cache" }, async () => {
  const server = createServer({ tts: new TTS({ offline: true }), apiKey: "k", cors: "*" });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  const auth = { Authorization: "Bearer k", "Content-Type": "application/json" };
  try {
    assert.equal((await fetch(`${url}/v1/models`)).status, 401);
    assert.equal((await (await fetch(`${url}/health`)).json()).status, "ok");
    const models = await (await fetch(`${url}/v1/models`, { headers: auth })).json();
    assert.ok(models.data.some((m) => m.id === "tts-1"));

    let r = await fetch(`${url}/v1/audio/speech`, { method: "POST", headers: auth,
      body: JSON.stringify({ model: "tts-1", input: "Hello there.", voice: "nova", response_format: "wav" }) });
    assert.equal(r.status, 200);
    assert.equal(r.headers.get("content-type"), "audio/wav");
    assert.ok(r.headers.get("x-voice-id"));
    assert.equal(new TextDecoder().decode((await r.arrayBuffer()).slice(0, 4)), "RIFF");

    r = await fetch(`${url}/v1/audio/speech`, { method: "POST", headers: auth,
      body: JSON.stringify({ input: "One. Two. Three.", response_format: "pcm" }) });
    const pcm = await r.arrayBuffer();
    assert.equal(r.headers.get("content-type"), "audio/pcm");
    assert.ok(pcm.byteLength > 24000 && pcm.byteLength % 2 === 0);

    const j = await (await fetch(`${url}/v1/tts`, { method: "POST", headers: auth, body: JSON.stringify({ text: "It costs $5." }) })).json();
    assert.ok(j.audio && j.srt.startsWith("1\n") && j.timings.words.length === 3 && j.voice.credit);

    r = await fetch(`${url}/v1/audio/speech`, { method: "POST", headers: auth, body: JSON.stringify({ input: "Hi", voice: "nope" }) });
    assert.equal(r.status, 400);
    assert.equal((await r.json()).error.code, "VOICE_NOT_FOUND");
  } finally {
    server.close();
  }
});
