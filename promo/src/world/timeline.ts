import { E, clamp, smoothstep } from "../lib/math";

export type Stage = { a: number; b: number; mix: number; size: number; turb: number; reveal: number; dim: number; pulse: number };

// world time f: 0 = start of volume 2
const SEG: { f0: number; f1: number; a: number; b: number }[] = [
  { f0: 66, f1: 94, a: 0, b: 1 },
  { f0: 106, f1: 134, a: 1, b: 2 },
  { f0: 146, f1: 174, a: 2, b: 3 },
  { f0: 186, f1: 230, a: 3, b: 5 },
  { f0: 304, f1: 344, a: 5, b: 6 },
  { f0: 412, f1: 474, a: 6, b: 7 },
];

export function stageAt(f: number): Stage {
  let a = 0, b = 0, mix = 0;
  for (const s of SEG) {
    if (f >= s.f0) {
      if (f < s.f1) { a = s.a; b = s.b; mix = E.inOutQuad(clamp((f - s.f0) / (s.f1 - s.f0))); break; }
      a = s.b; b = s.b; mix = 0;
    }
  }
  const reveal = clamp((f - 4) / 56);
  const beat = Math.exp(-6 * ((f % 15) / 15));
  const size = a >= 7 || b >= 7 ? 0.044 : 0.05;
  const turb = 0.75;
  const dim = 1.0 - 0.8 * smoothstep(440, 520, f);
  return { a, b, mix, size, turb, reveal: f < 62 ? E.outCubic(reveal) : 1, dim, pulse: beat * smoothstep(222, 240, f) * (1 - smoothstep(300, 310, f)) };
}
