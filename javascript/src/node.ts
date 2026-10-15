// SPDX-License-Identifier: GPL-3.0-or-later
// The Node entry point (also used by Bun and Deno). Everything works as in browsers, plus files: voices are kept in a
// cache folder shared with the Python edition, speech can be saved, and the OpenAI-compatible server is available.
import "./node-platform.js";

export * from "./index.js";
export { cacheDir, downloadedModels, removeModel, NodeStore } from "./node-platform.js";
export { serve, createServer, type ServeOptions } from "./server.js";
export { encode, ffmpegPath, CONTENT_TYPES, PCM_RATE } from "./formats.js";
