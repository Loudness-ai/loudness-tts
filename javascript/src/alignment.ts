// SPDX-License-Identifier: GPL-3.0-or-later
// Exact word timings: marks the voice model's w_ceil tensor (the per-sound sample counts of its Ceil node) as an extra
// graph output, as Piper's patch_voice_with_alignment.py does, by editing the ONNX protobuf directly (no onnx package).

interface Field {
  field: number;
  start: number; // where the tag begins
  from: number; // payload span (length-delimited fields)
  to: number;
}

function varint(buf: Uint8Array, pos: number): [number, number] {
  let result = 0;
  let scale = 1;
  let byte: number;
  do {
    byte = buf[pos++];
    result += (byte & 0x7f) * scale;
    scale *= 128;
  } while (byte & 0x80);
  return [result, pos];
}

function encodeVarint(n: number): number[] {
  const out: number[] = [];
  while (n > 127) {
    out.push((n % 128) | 128);
    n = Math.floor(n / 128);
  }
  out.push(n);
  return out;
}

function fields(buf: Uint8Array, from: number, to: number): Field[] {
  const out: Field[] = [];
  let p = from;
  while (p < to) {
    const start = p;
    let tag: number;
    [tag, p] = varint(buf, p);
    const wire = tag & 7;
    let a = p;
    let b = p;
    if (wire === 2) {
      let length: number;
      [length, p] = varint(buf, p);
      a = p;
      b = p + length;
      p = b;
    } else if (wire === 0) {
      [, p] = varint(buf, p);
    } else if (wire === 5) {
      p += 4;
    } else if (wire === 1) {
      p += 8;
    } else {
      throw new Error(`unsupported protobuf wire type ${wire}`);
    }
    out.push({ field: Math.floor(tag / 8), start, from: a, to: b });
  }
  return out;
}

const text = (buf: Uint8Array, f: Field): string => new TextDecoder().decode(buf.subarray(f.from, f.to));

/** The model with its Ceil output added, or null when it has no single Ceil tensor or already reports it. */
export function withAlignmentOutput(model: Uint8Array): Uint8Array | null {
  let graph: Field | undefined;
  try {
    graph = fields(model, 0, model.length).find((f) => f.field === 7); // ModelProto.graph
  } catch {
    return null;
  }
  if (!graph) return null;
  const ceil = new Set<string>();
  const outputs = new Set<string>();
  for (const f of fields(model, graph.from, graph.to)) {
    if (f.field === 1) { // GraphProto.node: NodeProto (output = 2, op_type = 4)
      const node = fields(model, f.from, f.to);
      if (node.some((x) => x.field === 4 && text(model, x) === "Ceil")) {
        node.filter((x) => x.field === 2).forEach((x) => ceil.add(text(model, x)));
      }
    } else if (f.field === 12) { // GraphProto.output: ValueInfoProto (name = 1)
      fields(model, f.from, f.to).filter((x) => x.field === 1).forEach((x) => outputs.add(text(model, x)));
    }
  }
  if (ceil.size !== 1) return null;
  const name = [...ceil][0];
  if (outputs.has(name)) return null;
  const nameBytes = new TextEncoder().encode(name);
  const info = [0x0a, ...encodeVarint(nameBytes.length), ...nameBytes];
  const extra = Uint8Array.from([0x62, ...encodeVarint(info.length), ...info]); // field 12, length-delimited
  const graphLength = graph.to - graph.from + extra.length;
  const head = Uint8Array.from([0x3a, ...encodeVarint(graphLength)]); // field 7, length-delimited
  const out = new Uint8Array(graph.start + head.length + graphLength + (model.length - graph.to));
  out.set(model.subarray(0, graph.start), 0);
  out.set(head, graph.start);
  out.set(model.subarray(graph.from, graph.to), graph.start + head.length);
  out.set(extra, graph.start + head.length + (graph.to - graph.from));
  out.set(model.subarray(graph.to), graph.start + head.length + graphLength);
  return out;
}
