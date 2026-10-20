import { mulberry32, gauss } from "../lib/math";

// Hydrogen orbitals sampled as point clouds (|psi|^2), atomic units.
// Radial parts are the textbook closed forms; angular parts are real harmonics normalised to max |Y| = 1.
const radial = (n: number, l: number, r: number) => {
  const e = Math.exp(-r / n);
  if (n === 1 && l === 0) return 2 * Math.exp(-r);
  if (n === 2 && l === 0) return (1 / (2 * Math.SQRT2)) * (2 - r) * e;
  if (n === 2 && l === 1) return (1 / (2 * Math.sqrt(6))) * r * e;
  if (n === 3 && l === 0) return (2 / (81 * Math.sqrt(3))) * (27 - 18 * r + 2 * r * r) * e;
  if (n === 3 && l === 1) return (8 / (27 * Math.sqrt(6))) * r * (1 - r / 6) * e;
  if (n === 3 && l === 2) return (4 / (81 * Math.sqrt(30))) * r * r * e;
  throw new Error("orbital not tabulated");
};

export type OrbKind = "s" | "pz" | "dz2" | "dxy";
const angular = (kind: OrbKind, c: number, ph: number) => {
  switch (kind) {
    case "s": return 1;
    case "pz": return c;
    case "dz2": return 0.5 * (3 * c * c - 1);
    case "dxy": return (1 - c * c) * Math.sin(2 * ph);
  }
};

export type Orbital = { n: number; l: number; kind: OrbKind; target: number };

/** returns positions (n*3), sign (n), radial fraction (n) with the 95th-percentile radius scaled to `target`. */
export function sampleOrbital(orb: Orbital, count: number, seed: number) {
  const rnd = mulberry32(seed);
  const rmax = 7 * orb.n * orb.n;
  const steps = 4000;
  const cdf = new Float64Array(steps + 1);
  for (let i = 1; i <= steps; i++) {
    const r = (i / steps) * rmax;
    const R = radial(orb.n, orb.l, r);
    cdf[i] = cdf[i - 1] + r * r * R * R;
  }
  const total = cdf[steps];
  const sample = () => {
    const u = rnd() * total;
    let lo = 0, hi = steps;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (cdf[m] < u) lo = m; else hi = m;
    }
    return ((lo + (u - cdf[lo]) / Math.max(1e-30, cdf[hi] - cdf[lo])) / steps) * rmax;
  };
  const pos = new Float32Array(count * 3);
  const sign = new Float32Array(count);
  const rs = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const r = sample();
    let c = 0, ph = 0, Y = 0;
    for (let tries = 0; tries < 200; tries++) {
      c = rnd() * 2 - 1;
      ph = rnd() * Math.PI * 2;
      Y = angular(orb.kind, c, ph);
      if (rnd() < Y * Y) break;
    }
    const s = Math.sqrt(Math.max(0, 1 - c * c));
    pos[i * 3] = r * s * Math.cos(ph);
    pos[i * 3 + 1] = r * c; // z axis of the orbital points up (y in the scene)
    pos[i * 3 + 2] = r * s * Math.sin(ph);
    const R = radial(orb.n, orb.l, r);
    sign[i] = (R * Y >= 0 ? 1 : -1);
    rs[i] = r;
  }
  // scale so that 95th percentile radius -> target
  const sorted = Array.from(rs).sort((a, b) => a - b);
  const r95 = sorted[Math.floor(count * 0.95)];
  const k = orb.target / r95;
  for (let i = 0; i < count * 3; i++) pos[i] *= k;
  for (let i = 0; i < count; i++) rs[i] = Math.min(1, (rs[i] * k) / orb.target);
  void gauss;
  return { pos, sign, rs };
}
