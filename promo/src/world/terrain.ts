// The loss landscape for volume 3: a Rosenbrock-style banana valley seen through log(1 + f).
export const B = 10;
export const fRos = (u: number, v: number) => (1 - u) ** 2 + B * (v - u * u) ** 2;
export const gradRos = (u: number, v: number): [number, number] => [
  -2 * (1 - u) - 4 * B * u * (v - u * u),
  2 * B * (v - u * u),
];
export const U0 = -1.95, U1 = 2.05, V0 = -0.3, V1 = 2.8;
export const HK = 0.5;
export const heightUV = (u: number, v: number) => HK * Math.log1p(fRos(u, v));
export const worldX = (u: number) => (u - 0.1) * 2.1;
export const worldZ = (v: number) => -(v - 1.15) * 2.1;
export const uOfX = (x: number) => x / 2.1 + 0.1;
export const vOfZ = (z: number) => -z / 2.1 + 1.15;

export type Racer = { name: string; latin: string; note: string; color: string; path: [number, number][] };

const START: [number, number] = [-1.5, 1.2];

function simulate(step: (x: number[], st: any, i: number) => number[], N: number) {
  let x = [...START];
  const st: any = {};
  const path: [number, number][] = [[x[0], x[1]]];
  for (let i = 0; i < N; i++) {
    x = step(x, st, i);
    path.push([x[0], x[1]]);
  }
  return path;
}

export const STEPS = 700;

function lcg(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}
const gaussR = (r: () => number) => Math.sqrt(-2 * Math.log(Math.max(1e-12, r()))) * Math.cos(2 * Math.PI * r());

export function buildRacers(): Racer[] {
  const noise = lcg(7);
  const sgd = simulate((x) => {
    const g = gradRos(x[0], x[1]);
    return [x[0] - 0.0058 * (g[0] + 1.3 * gaussR(noise)), x[1] - 0.0058 * (g[1] + 1.3 * gaussR(noise))];
  }, STEPS);
  const mom = simulate((x, st) => {
    const g = gradRos(x[0], x[1]);
    st.m = st.m || [0, 0];
    st.m = [0.9 * st.m[0] + g[0], 0.9 * st.m[1] + g[1]];
    return [x[0] - 0.002 * st.m[0], x[1] - 0.002 * st.m[1]];
  }, STEPS);
  const adam = simulate((x, st, i) => {
    const g = gradRos(x[0], x[1]);
    st.m = st.m || [0, 0]; st.v = st.v || [0, 0];
    st.m = st.m.map((m: number, k: number) => 0.9 * m + 0.1 * g[k]);
    st.v = st.v.map((v: number, k: number) => 0.999 * v + 0.001 * g[k] * g[k]);
    const t = i + 1;
    return x.map((xx, k) => xx - (0.05 * (st.m[k] / (1 - 0.9 ** t))) / (Math.sqrt(st.v[k] / (1 - 0.999 ** t)) + 1e-8));
  }, STEPS);
  const lion = simulate((x, st) => {
    const g = gradRos(x[0], x[1]);
    st.m = st.m || [0, 0];
    const c = st.m.map((m: number, k: number) => Math.sign(0.9 * m + 0.1 * g[k]));
    st.m = st.m.map((m: number, k: number) => 0.99 * m + 0.01 * g[k]);
    return [x[0] - 0.018 * c[0], x[1] - 0.018 * c[1]];
  }, STEPS);
  return [
    { name: "SGD", latin: "Gradientus stochasticus", note: "walks like it has had a few", color: "#ff6b5e", path: sgd },
    { name: "Momentum", latin: "Impetus gravis", note: "overshoots, then thinks better of it", color: "#ffb454", path: mom },
    { name: "Adam", latin: "Adamus adaptivus", note: "rescales every coordinate", color: "#46e0ff", path: adam },
    { name: "Lion", latin: "Leo signum", note: "listens only to the sign", color: "#c78bff", path: lion },
  ];
}

let racersCache: Racer[] | null = null;
export const RACERS = () => (racersCache ??= buildRacers());

export const RACE_F0 = 580; // world frame at which the race starts (step 0)
export const RACE_F1 = 752; // ... and at which step STEPS is reached
export const stepAt = (f: number) => {
  const t = Math.min(1, Math.max(0, (f - RACE_F0) / (RACE_F1 - RACE_F0)));
  const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  return STEPS * (0.12 * t + 0.88 * e);
};

/** world position of racer `i` at fractional step `s` (interpolating the path) */
export function racerPos(i: number, s: number, lift = 0.05): [number, number, number] {
  const path = RACERS()[i].path;
  const k = Math.min(path.length - 2, Math.max(0, Math.floor(s)));
  const t = Math.min(1, Math.max(0, s - k));
  const u = path[k][0] + (path[k + 1][0] - path[k][0]) * t;
  const v = path[k][1] + (path[k + 1][1] - path[k][1]) * t;
  return [worldX(u), heightUV(u, v) + lift, worldZ(v)];
}
export const MIN_POS: [number, number, number] = [worldX(1), 0.0, worldZ(1)];
export const START_POS = (): [number, number, number] => {
  const p = RACERS()[0].path[0];
  return [worldX(p[0]), heightUV(p[0], p[1]), worldZ(p[1])];
};
