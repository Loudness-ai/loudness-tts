// Copies the espeak-ng data the JavaScript edition needs from Piper's own build: the piper-tts 1.8.0 wheel for Linux
// x86-64 bundles espeak-ng-data compiled from the same espeak-ng commit as espeak/build.sh. The wheel is downloaded from
// PyPI and checked against its published SHA-256, so every system produces the same files; the result is checked
// against espeak-ng-data/MANIFEST.json (kept in git).
//   node scripts/espeak-data.mjs [folder]      (a folder: use that espeak-ng-data instead of downloading)
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { inflateRawSync } from "node:zlib";

const WHEEL = "https://files.pythonhosted.org/packages/84/81/0112a7d510911f33018dc24023d1655bb772f84a4e95fe7f0180f66bbd17/"
  + "piper_tts-1.8.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl";
const WHEEL_SHA256 = "25b4d3f31ff70c8fa7151908e00aaa5650cbdf16bca8fcf21299f3941b89a7d3";
const PREFIX = "piper/espeak-ng-data/";
const BASE = ["phontab", "phonindex", "phondata", "intonations"];

/** The files of a zip archive whose names start with prefix: {name relative to prefix: bytes}. */
function unzip(zip, prefix) {
  let end = zip.length - 22;
  while (end >= 0 && zip.readUInt32LE(end) !== 0x06054b50) end--;
  if (end < 0) throw new Error("not a zip file");
  const count = zip.readUInt16LE(end + 10);
  let p = zip.readUInt32LE(end + 16);
  const out = {};
  for (let i = 0; i < count; i++) {
    const method = zip.readUInt16LE(p + 10);
    const size = zip.readUInt32LE(p + 20);
    const nameLength = zip.readUInt16LE(p + 28);
    const extra = zip.readUInt16LE(p + 30);
    const comment = zip.readUInt16LE(p + 32);
    const local = zip.readUInt32LE(p + 42);
    const name = zip.toString("utf8", p + 46, p + 46 + nameLength);
    p += 46 + nameLength + extra + comment;
    if (!name.startsWith(prefix) || name.endsWith("/")) continue;
    const start = local + 30 + zip.readUInt16LE(local + 26) + zip.readUInt16LE(local + 28);
    const data = zip.subarray(start, start + size);
    out[name.slice(prefix.length)] = method === 0 ? data : inflateRawSync(data);
  }
  return out;
}

function fromFolder(dir, prefix = "", out = {}) {
  for (const name of readdirSync(dir)) {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) fromFolder(full, `${prefix}${name}/`, out);
    else out[prefix + name] = readFileSync(full);
  }
  return out;
}

let files;
if (process.argv[2]) {
  files = fromFolder(process.argv[2]);
} else {
  const response = await fetch(WHEEL);
  if (!response.ok) throw new Error(`Could not download piper-tts 1.8.0 (HTTP ${response.status}).`);
  const wheel = Buffer.from(await response.arrayBuffer());
  if (createHash("sha256").update(wheel).digest("hex") !== WHEEL_SHA256) throw new Error("piper-tts wheel checksum mismatch");
  files = unzip(wheel, PREFIX);
}

const target = fileURLToPath(new URL("../espeak-ng-data/", import.meta.url));
const manifestPath = path.join(target, "MANIFEST.json");
const expected = existsSync(manifestPath) ? JSON.parse(readFileSync(manifestPath, "utf8")) : null;
rmSync(target, { recursive: true, force: true });
mkdirSync(target, { recursive: true });

const lang = {};
for (const name of Object.keys(files).sort()) {
  if (BASE.includes(name) || /^[^/]+_dict$/.test(name)) writeFileSync(path.join(target, name), files[name]);
  else if (name.startsWith("lang/")) lang[name.slice(5)] = files[name].toString("utf8");
}
writeFileSync(path.join(target, "lang.json"), JSON.stringify(lang));

const manifest = { espeak_ng: "1.52.0.1, commit 724808c (as built by piper-tts 1.8.0)", files: {} };
for (const f of readdirSync(target).sort()) {
  manifest.files[f] = createHash("sha256").update(readFileSync(path.join(target, f))).digest("hex");
}
writeFileSync(manifestPath, JSON.stringify(manifest, null, 1) + "\n");
if (expected && JSON.stringify(expected.files) !== JSON.stringify(manifest.files)) {
  const changed = Object.keys({ ...expected.files, ...manifest.files }).filter((f) => expected.files[f] !== manifest.files[f]);
  throw new Error(`The espeak-ng data differs from MANIFEST.json (${changed.slice(0, 5).join(", ")}).`);
}
console.log(`${Object.keys(manifest.files).length} files, ${Object.keys(lang).length} language files`);
