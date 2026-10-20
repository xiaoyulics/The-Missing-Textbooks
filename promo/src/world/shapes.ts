import { gauss, mulberry32, TAU } from "../lib/math";
import { Orbital, sampleOrbital } from "./orbitals";
import { heightUV, U0, U1, V0, V1, worldX, worldZ } from "./terrain";

export const N = 60000;

export const RING_RADII = [0.55, 1.55, 2.85, 4.4];

// Balmer series: lambda_n = 364.56 n^2 / (n^2 - 4), n = 3, 4, ...
export const BALMER = Array.from({ length: 14 }, (_, i) => {
  const n = i + 3;
  const lam = (364.56 * n * n) / (n * n - 4);
  return { n, lam };
});
export const SPEC_X0 = -6.2, SPEC_X1 = 6.2, SPEC_Y0 = -1.7, SPEC_H = 3.9;
export const specX = (lam: number) => SPEC_X0 + ((SPEC_X1 - SPEC_X0) * (lam - 364.56)) / (656.2 - 364.56);
const SPEC_HEIGHT = [1.0, 0.64, 0.46, 0.34, 0.27, 0.22, 0.18, 0.15, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06];
const SPEC_COL: [number, number, number][] = [
  [1.0, 0.3, 0.24], [0.18, 0.88, 1.0], [0.36, 0.48, 1.0], [0.64, 0.36, 1.0],
];
export const specColor = (i: number): [number, number, number] =>
  i < 4 ? SPEC_COL[i] : [0.62 - i * 0.012, 0.42, 1.0];

// double descent: theory curve (ridgeless least squares, schematic), mapped to the screen plane
export const CURVE_X0 = -6.2, CURVE_X1 = 6.2, CURVE_GMAX = 4.4;
export const curveX = (g: number) => CURVE_X0 + ((CURVE_X1 - CURVE_X0) * g) / CURVE_GMAX;
export const risk = (g: number, s2 = 0.25) => {
  const V = g < 1 ? (s2 * g) / (1 - g) : s2 / (g - 1);
  const Bi = 0.85 * Math.exp(-2.2 * g);
  return V + Bi;
};
export const curveY = (g: number) => -2.3 + 4.7 * (1 - Math.exp(-Math.min(risk(g), 5) * 0.55));

export type WorldAttrs = {
  pos: Float32Array[];
  signs: Float32Array; // 4 per particle: sign for stages 1..4
  tone: Float32Array; // 4 per particle
  colB: Float32Array; // spectrum colour
  rand: Float32Array; // 4 per particle: delay, size, phase, aux
};

export const ORBITALS: Orbital[] = [
  { n: 1, l: 0, kind: "s", target: 1.7 },
  { n: 2, l: 1, kind: "pz", target: 2.7 },
  { n: 3, l: 2, kind: "dz2", target: 3.4 },
  { n: 3, l: 2, kind: "dxy", target: 3.4 },
];

export function buildWorld(): WorldAttrs {
  const rnd = mulberry32(2026);
  const pos = Array.from({ length: 8 }, () => new Float32Array(N * 3));
  const signs = new Float32Array(N * 4);
  const tone = new Float32Array(N * 4);
  const colB = new Float32Array(N * 3);
  const rand = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    rand[i * 4] = rnd();
    rand[i * 4 + 1] = rnd();
    rand[i * 4 + 2] = rnd();
    rand[i * 4 + 3] = rnd();
  }

  // stage 0: Bohr rings in the horizontal plane + a bright nucleus
  const wsum = RING_RADII.reduce((a, b) => a + b, 0);
  for (let i = 0; i < N; i++) {
    const u = rnd();
    let r: number;
    if (u < 0.03) r = Math.abs(gauss(rnd)) * 0.05;
    else {
      let t = ((u - 0.03) / 0.97) * wsum, k = 0;
      while (k < 3 && t > RING_RADII[k]) { t -= RING_RADII[k]; k++; }
      r = RING_RADII[k] + gauss(rnd) * 0.011;
    }
    const a = rnd() * TAU;
    pos[0][i * 3] = r * Math.cos(a);
    pos[0][i * 3 + 1] = gauss(rnd) * (r < 0.1 ? 0.05 : 0.008);
    pos[0][i * 3 + 2] = r * Math.sin(a);
  }

  // stages 1..4: orbitals
  ORBITALS.forEach((orb, k) => {
    const s = sampleOrbital(orb, N, 100 + k);
    pos[k + 1].set(s.pos);
    for (let i = 0; i < N; i++) {
      signs[i * 4 + k] = s.sign[i];
      tone[i * 4 + k] = 0.45 + 0.55 * Math.pow(1 - s.rs[i] * 0.85, 1.2);
    }
  });

  // stage 5: Balmer spectrum bars (screen plane)
  const hs = SPEC_HEIGHT;
  const weights = hs.map((h) => h);
  const wtot = weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < N; i++) {
    const u = rnd();
    if (u < 0.07) {
      // baseline axis
      pos[5][i * 3] = SPEC_X0 - 0.4 + rnd() * (SPEC_X1 - SPEC_X0 + 0.8);
      pos[5][i * 3 + 1] = SPEC_Y0 + gauss(rnd) * 0.008;
      pos[5][i * 3 + 2] = gauss(rnd) * 0.02;
      colB[i * 3] = 0.45; colB[i * 3 + 1] = 0.6; colB[i * 3 + 2] = 0.75;
      continue;
    }
    let t = ((u - 0.07) / 0.93) * wtot, k = 0;
    while (k < hs.length - 1 && t > weights[k]) { t -= weights[k]; k++; }
    const x = specX(BALMER[k].lam);
    const h = hs[k] * SPEC_H;
    pos[5][i * 3] = x + gauss(rnd) * (k < 4 ? 0.022 : 0.013);
    pos[5][i * 3 + 1] = SPEC_Y0 + Math.pow(rnd(), 0.9) * h;
    pos[5][i * 3 + 2] = gauss(rnd) * 0.03;
    const c = specColor(k);
    colB[i * 3] = c[0]; colB[i * 3 + 1] = c[1]; colB[i * 3 + 2] = c[2];
  }

  // stage 6: the double-descent curve
  for (let i = 0; i < N; i++) {
    const t = rnd();
    // denser sampling around the peak
    const g = t < 0.55 ? 0.03 + t * 1.6 : 0.03 + 0.88 + Math.pow((t - 0.55) / 0.45, 1.3) * (CURVE_GMAX - 0.91);
    const x = curveX(g), y = curveY(g);
    const nrm = gauss(rnd) * (0.03 + 0.012 * rnd());
    pos[6][i * 3] = x + gauss(rnd) * 0.012;
    pos[6][i * 3 + 1] = y + nrm;
    pos[6][i * 3 + 2] = gauss(rnd) * 0.05;
  }

  // stage 7: the loss landscape as a cloud of points on the surface
  for (let i = 0; i < N; i++) {
    const u = U0 + rnd() * (U1 - U0), v = V0 + rnd() * (V1 - V0);
    pos[7][i * 3] = worldX(u);
    pos[7][i * 3 + 1] = heightUV(u, v) + 0.015;
    pos[7][i * 3 + 2] = worldZ(v);
  }
  return { pos, signs, tone, colB, rand };
}
