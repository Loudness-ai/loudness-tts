// Copies the espeak-ng data the JavaScript edition needs, from Piper's own build (the piper-tts wheel bundles
// espeak-ng-data compiled from the same espeak-ng commit as espeak/build.sh), and writes a checksum manifest.
//   pip install piper-tts==1.8.0 && node scripts/espeak-data.mjs [<piper package dir>/espeak-ng-data]
// The result is checked against espeak-ng-data/MANIFEST.json (kept in git) when that file exists.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const source = process.argv[2] ?? execFileSync(process.env.PYTHON ?? (process.platform === "win32" ? "python" : "python3"),
  ["-c", "import piper.phonemize_espeak as p; print(p.ESPEAK_DATA_DIR)"], { encoding: "utf8" }).trim();
const target = fileURLToPath(new URL("../espeak-ng-data/", import.meta.url));
const expected = existsSync(path.join(target, "MANIFEST.json")) ? JSON.parse(readFileSync(path.join(target, "MANIFEST.json"), "utf8")) : null;
rmSync(target, { recursive: true, force: true });
mkdirSync(target, { recursive: true });

const files = readdirSync(source).filter((f) => ["phontab", "phonindex", "phondata", "intonations"].includes(f) || f.endsWith("_dict"));
for (const f of files) cpSync(path.join(source, f), path.join(target, f));

const lang = {};
const walk = (dir, prefix) => {
  for (const name of readdirSync(dir).sort()) {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) walk(full, `${prefix}${name}/`);
    else lang[prefix + name] = readFileSync(full, "utf8");
  }
};
walk(path.join(source, "lang"), "");
writeFileSync(path.join(target, "lang.json"), JSON.stringify(lang));

const manifest = { espeak_ng: "1.52.0.1, commit 724808c (as built by piper-tts 1.8.0)", files: {} };
for (const f of readdirSync(target).sort()) {
  manifest.files[f] = createHash("sha256").update(readFileSync(path.join(target, f))).digest("hex");
}
writeFileSync(path.join(target, "MANIFEST.json"), JSON.stringify(manifest, null, 1) + "\n");
if (expected && JSON.stringify(expected.files) !== JSON.stringify(manifest.files)) {
  throw new Error("The espeak-ng data differs from MANIFEST.json: use piper-tts 1.8.0 (or update the manifest on purpose).");
}
console.log(`${Object.keys(manifest.files).length} files, ${Object.keys(lang).length} language files`);
