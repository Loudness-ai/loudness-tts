// SPDX-License-Identifier: GPL-3.0-or-later
// MD5 (RFC 1321), only to check downloads against the checksums the voice catalogue publishes. WebCrypto has no MD5.

const S = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
  14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10,
  15, 21];
const K = Array.from({ length: 64 }, (_, i) => Math.floor(Math.abs(Math.sin(i + 1)) * 2 ** 32) >>> 0);

/** Incremental MD5: update() with chunks, then hex(). */
export class Md5 {
  private h = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];
  private buffer = new Uint8Array(64);
  private buffered = 0;
  private length = 0;
  private readonly words = new Uint32Array(16);

  update(data: Uint8Array): this {
    this.length += data.length;
    let i = 0;
    if (this.buffered) {
      const take = Math.min(64 - this.buffered, data.length);
      this.buffer.set(data.subarray(0, take), this.buffered);
      this.buffered += take;
      i = take;
      if (this.buffered < 64) return this;
      this.block(this.buffer, 0);
      this.buffered = 0;
    }
    for (; i + 64 <= data.length; i += 64) this.block(data, i);
    if (i < data.length) {
      this.buffer.set(data.subarray(i), 0);
      this.buffered = data.length - i;
    }
    return this;
  }

  hex(): string {
    const bits = this.length * 8;
    const tail = new Uint8Array(((this.buffered + 8) >> 6 << 6) + 64);
    tail.set(this.buffer.subarray(0, this.buffered));
    tail[this.buffered] = 0x80;
    const view = new DataView(tail.buffer);
    view.setUint32(tail.length - 8, bits >>> 0, true);
    view.setUint32(tail.length - 4, Math.floor(bits / 2 ** 32), true);
    for (let i = 0; i < tail.length; i += 64) this.block(tail, i);
    return this.h.map((x) => Array.from({ length: 4 }, (_, k) => ((x >>> (8 * k)) & 255).toString(16).padStart(2, "0")).join("")).join("");
  }

  private block(data: Uint8Array, offset: number): void {
    const w = this.words;
    for (let j = 0; j < 16; j++) {
      const o = offset + j * 4;
      w[j] = data[o] | (data[o + 1] << 8) | (data[o + 2] << 16) | (data[o + 3] << 24);
    }
    let [a, b, c, d] = this.h;
    for (let j = 0; j < 64; j++) {
      let f: number;
      let g: number;
      if (j < 16) { f = (b & c) | (~b & d); g = j; }
      else if (j < 32) { f = (d & b) | (~d & c); g = (5 * j + 1) % 16; }
      else if (j < 48) { f = b ^ c ^ d; g = (3 * j + 5) % 16; }
      else { f = c ^ (b | ~d); g = (7 * j) % 16; }
      const t = d;
      d = c;
      c = b;
      const x = (a + f + K[j] + w[g]) | 0;
      b = (b + ((x << S[j]) | (x >>> (32 - S[j])))) | 0;
      a = t;
    }
    this.h[0] = (this.h[0] + a) | 0;
    this.h[1] = (this.h[1] + b) | 0;
    this.h[2] = (this.h[2] + c) | 0;
    this.h[3] = (this.h[3] + d) | 0;
  }
}
