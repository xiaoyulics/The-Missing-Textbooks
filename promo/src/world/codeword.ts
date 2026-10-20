import data from "./codeword.json";
import { mulberry32 } from "../lib/math";

/**
 * The emblem of volume 5: the Hadamard codeword of 1985 (2048 bits), displayed in 32 rows of 64 in the order
 * fixed by NumPy's default_rng(1985).permutation(2048) -- the same call as figures/scripts/00-emblem.py in
 * the book, so the grid and the ringed triple match the title page cell for cell.
 */
export const U = data.U as number;
export const COLS = 64;
export const ROWS = 32;
export const N = COLS * ROWS;
const perm = data.perm as number[];

const parity = (v: number) => {
  v ^= v >> 8; v ^= v >> 4; v ^= v >> 2; v ^= v >> 1;
  return v & 1;
};
/** the codeword at position x: parity of the bits that x and U share */
export const f = (x: number) => parity(x & U);

/** where[x] = display position (row-major) of codeword position x */
export const where = new Int16Array(N);
perm.forEach((x, p) => { where[x] = p; });

/** the bit shown at display position p (honest codeword) */
export const honestBit = (p: number) => f(perm[p]);
export const posOf = (x: number): [number, number] => { const p = where[x]; return [p % COLS, Math.floor(p / COLS)]; };

/** the triple on the title page */
export const COVER_TRIPLE = [45, 673, 45 ^ 673] as const;

// ------------------------------------------------------------------ the prover's lie, and the verifier's tests
/** positions flipped by a cheating prover (a spread-thin lie: each position independently with probability DELTA) */
export const DELTA = 0.1;
export function liePositions(seed: number) {
  const r = mulberry32(seed);
  const flipped = new Uint8Array(N); // indexed by codeword position x
  for (let x = 0; x < N; x++) if (r() < DELTA) flipped[x] = 1;
  return flipped;
}

export type Test = { x: number; y: number; z: number; bits: [number, number, number]; accepted: boolean };
/** one linearity test on the word `flipped` (all zeros = the honest word): read f(x), f(y), f(x^y), accept iff f(x)+f(y) = f(x^y) */
export function runTest(x: number, y: number, flipped: Uint8Array | null): Test {
  const z = x ^ y;
  const b = (t: number) => f(t) ^ (flipped ? flipped[t] : 0);
  const bits: [number, number, number] = [b(x), b(y), b(z)];
  return { x, y, z, bits, accepted: (bits[0] ^ bits[1]) === bits[2] };
}

/**
 * The tests shown in the film. Test 0 is the title page's triple. Tests 0..HONEST-1 run on the honest word;
 * from test HONEST on, the prover has cheated. The lie is chosen (by scanning seeds) so that the first rejection
 * is test number FIRST_REJECT, which lands on a beat; everything else is the real test on real bits.
 */
export const HONEST = 24;
export const FIRST_REJECT = 29;
const TOTAL_TESTS = 140;

function build() {
  for (let seed = 1; seed < 4000; seed++) {
    const lie = liePositions(seed);
    const r = mulberry32(1000 + seed);
    const tests: Test[] = [];
    for (let i = 0; i < TOTAL_TESTS; i++) {
      let x: number, y: number;
      if (i === 0) { x = COVER_TRIPLE[0]; y = COVER_TRIPLE[1]; }
      else { x = Math.floor(r() * N); y = Math.floor(r() * N); }
      tests.push(runTest(x, y, i < HONEST ? null : lie));
    }
    const firstBad = tests.findIndex((t) => !t.accepted);
    // the honest tests must all accept (they do, by linearity), and the first rejection must be the one we want
    if (firstBad === FIRST_REJECT && tests.slice(0, HONEST).every((t) => t.accepted)) return { lie, tests, seed };
  }
  throw new Error("no seed found");
}
const built = build();
export const lie = built.lie;
export const tests = built.tests;
export const LIE_SEED = built.seed;
/** the bit actually on the wall at display position p after the prover has cheated */
export const cheatedBit = (p: number) => honestBit(p) ^ lie[perm[p]];
export const isLie = (p: number) => lie[perm[p]] === 1;
