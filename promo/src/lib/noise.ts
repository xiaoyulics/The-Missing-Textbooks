// Small value-noise helpers for particles (deterministic).
import { hash1 } from "./math";

export const noise1 = (x: number, seed = 0) => {
  const i = Math.floor(x), f = x - i;
  const u = f * f * (3 - 2 * f);
  return hash1(i + seed * 17.13) * (1 - u) + hash1(i + 1 + seed * 17.13) * u;
};

const h2 = (x: number, y: number) => {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
};
export const noise2 = (x: number, y: number) => {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
  const a = h2(ix, iy), b = h2(ix + 1, iy), c = h2(ix, iy + 1), d = h2(ix + 1, iy + 1);
  return a * (1 - ux) * (1 - uy) + b * ux * (1 - uy) + c * (1 - ux) * uy + d * ux * uy;
};
/** curl of a scalar noise field: divergence-free swirl */
export const curl2 = (x: number, y: number, e = 0.01): [number, number] => {
  const dy = (noise2(x, y + e) - noise2(x, y - e)) / (2 * e);
  const dx = (noise2(x + e, y) - noise2(x - e, y)) / (2 * e);
  return [dy, -dx];
};
