// Runs every *.test.mjs file with Node's test runner (the same on every Node version and system).
import { spawnSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const dir = fileURLToPath(new URL(".", import.meta.url));
const files = readdirSync(dir).filter((f) => f.endsWith(".test.mjs")).sort().map((f) => dir + f);
process.exitCode = spawnSync(process.execPath, ["--test", ...files], { stdio: "inherit" }).status ?? 1;
