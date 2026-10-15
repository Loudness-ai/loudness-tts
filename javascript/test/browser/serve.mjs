// A static server for the browser test page, with the headers that let WebAssembly use threads (cross-origin
// isolation). node test/browser/serve.mjs [port] -> http://127.0.0.1:8792/test/browser/
import { createReadStream, statSync } from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
  ".wasm": "application/wasm", ".css": "text/css" };
http.createServer((req, res) => {
  const file = path.join(root, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (!file.startsWith(root)) return res.writeHead(403).end();
  let target = file;
  try {
    if (statSync(target).isDirectory()) target = path.join(target, "index.html");
    statSync(target);
  } catch {
    return res.writeHead(404).end();
  }
  res.writeHead(200, { "Content-Type": types[path.extname(target)] ?? "application/octet-stream",
    "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "cross-origin", "Cache-Control": "no-store" });
  createReadStream(target).pipe(res);
}).listen(Number(process.argv[2] ?? 8792), "127.0.0.1");
