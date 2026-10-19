"""The linearity test of Blum, Luby and Rubinfeld on functions at controlled distance from linear.

Chapter 2 (Spot Checks), "Running the protocol". For f: F_2^l -> F_2 the test draws x, y uniformly
and accepts if f(x) + f(y) = f(x + y). The chapter proves that it rejects with probability at least
delta and at least 3 delta - 6 delta^2, where delta is the distance from f to the nearest linear
function. This script runs the test on three families of functions on F_2^l, l = 14:

  random     a linear function with a uniformly random set D of positions flipped (|D| = delta 2^l);
             the exact rejection probability is close to (1 - (1 - 2 delta)^3)/2
  hyperplane a linear function with a random set D flipped inside the hyperplane {x : x_1 = 1};
             the rejection probability is exactly 3 delta - 6 delta^2 (the second bound is attained)
  bent       x_1 x_2 + ... + x_{2k-1} x_{2k} plus a linear function, k = 1..7: distance
             1/2 - 2^-(k+1), rejection 1/2 - 2^-(2k+1) (the first bound is approached as k grows)

For each function it computes the exact distance and rejection probability by the Fourier formula
Pr[reject] = 1/2 - 1/2 sum_u F^(u)^3 (fast Walsh-Hadamard transform), and measures the rejection
rate by running the test TRIALS times (Clopper-Pearson 95% intervals).

Run from the volume root:  python3 figures/scripts/02-spotchecks-blr.py   (under a second)
Writes figures/data/02-spotchecks-blr-{curves,random,hyperplane,bent}.dat.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

N_VARS = 14
TRIALS = 20000
SEED = 2
U = 0b10110110001101          # the linear function h_u(x) = u . x that is corrupted
OUT = "figures/data/02-spotchecks-blr"


def parity(v: np.ndarray) -> np.ndarray:
    """Parity of the bits of each entry of a uint64 array."""
    v = v.astype(np.uint64).copy()
    for s in (32, 16, 8, 4, 2, 1):
        v ^= v >> np.uint64(s)
    return (v & np.uint64(1)).astype(np.uint8)


def fourier(f: np.ndarray) -> np.ndarray:
    """Fourier coefficients F^(u) = E_x[(-1)^(f(x) + u.x)] of a 0/1 table, by the fast transform."""
    a = 1.0 - 2.0 * f.astype(float)
    h = 1
    while h < len(a):
        a = a.reshape(-1, 2, h)
        a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], axis=1).reshape(-1)
        h *= 2
    return a / len(a)


def exact(f: np.ndarray) -> tuple[float, float]:
    """(distance to the nearest linear function, rejection probability of the test)."""
    c = fourier(f)
    return (1.0 - c.max()) / 2.0, 0.5 - 0.5 * float(np.sum(c ** 3))


def measure(f: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Run the test TRIALS times; return the rejection rate and its 95% interval."""
    gen = pk.rng(seed)
    n = len(f)
    x = gen.integers(0, n, TRIALS)
    y = gen.integers(0, n, TRIALS)
    k = int(np.sum(f[x] ^ f[y] ^ f[x ^ y]))
    lo, hi = pk.clopper_pearson(k, TRIALS)
    return k / TRIALS, lo, hi


def main() -> None:
    gen = pk.rng(SEED)
    N = 2 ** N_VARS
    idx = np.arange(N, dtype=np.uint64)
    hu = parity(idx & np.uint64(U))
    header = ["delta", "exact", "measured", "lo", "hi", "elo", "ehi"]

    def row(f, seed):
        d, rej = exact(f)
        r, lo, hi = measure(f, seed)
        return [d, rej, r, lo, hi, r - lo, hi - r]

    # random corruptions
    rows = []
    for i, delta in enumerate([0.01, 0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]):
        f = hu.copy()
        f[gen.choice(N, int(round(delta * N)), replace=False)] ^= 1
        rows.append(row(f, 100 + i))
    pk.write_dat(OUT + "-random.dat", header, rows)
    print("random corruptions: delta(exact)  rejection exact  measured [95% CI]")
    for r in rows:
        print(f"  {r[0]:.4f}  {r[1]:.4f}  {r[2]:.4f} [{r[3]:.4f}, {r[4]:.4f}]   (1-(1-2d)^3)/2 = {(1 - (1 - 2 * r[0]) ** 3) / 2:.4f}")
    worst_random = min(r[1] - max(r[0], 3 * r[0] - 6 * r[0] ** 2) for r in rows)

    # corruptions confined to the hyperplane x_1 = 1 (the lowest variable is bit 0)
    half = idx[(idx & np.uint64(1)) == 1].astype(np.int64)
    rows_h = []
    for i, delta in enumerate([0.02, 0.05, 0.1, 0.15, 0.2, 0.25]):
        f = hu.copy()
        f[gen.choice(half, int(round(delta * N)), replace=False)] ^= 1
        rows_h.append(row(f, 200 + i))
    pk.write_dat(OUT + "-hyperplane.dat", header, rows_h)
    print("hyperplane-confined corruptions: delta  exact  3d-6d^2  measured [95% CI]")
    for r in rows_h:
        print(f"  {r[0]:.4f}  {r[1]:.5f}  {3 * r[0] - 6 * r[0] ** 2:.5f}  {r[2]:.4f} [{r[3]:.4f}, {r[4]:.4f}]")

    # bent functions: inner product on 2k variables, plus h_u
    rows_b = []
    for k in range(1, N_VARS // 2 + 1):
        ip = np.zeros(N, dtype=np.uint8)
        for i in range(k):
            ip ^= (((idx >> np.uint64(2 * i)) & np.uint64(1)) & ((idx >> np.uint64(2 * i + 1)) & np.uint64(1))).astype(np.uint8)
        rows_b.append(row(ip ^ hu, 300 + k))
    pk.write_dat(OUT + "-bent.dat", header, rows_b)
    print("bent functions: k  delta  exact rejection  measured [95% CI]   predicted 1/2-2^-(k+1), 1/2-2^-(2k+1)")
    for k, r in enumerate(rows_b, 1):
        print(f"  {k}  {r[0]:.5f}  {r[1]:.5f}  {r[2]:.4f} [{r[3]:.4f}, {r[4]:.4f}]   {0.5 - 2.0 ** -(k + 1):.5f}, {0.5 - 2.0 ** -(2 * k + 1):.5f}")

    # theory curves
    d = np.linspace(0.0, 0.5, 101)
    pk.write_dat(OUT + "-curves.dat", ["delta", "fourier", "counting", "random"],
                 zip(d, d, np.maximum(3 * d - 6 * d ** 2, 0.0), (1 - (1 - 2 * d) ** 3) / 2))

    # checks: no exact or measured rejection rate below the proved bounds
    allrows = rows + rows_h + rows_b
    gap = min(r[1] - max(r[0], 3 * r[0] - 6 * r[0] ** 2) for r in allrows)
    above = all(r[4] >= max(r[0], 3 * r[0] - 6 * r[0] ** 2) - 1e-12 for r in allrows)
    print(f"smallest (exact rejection - proved bound) over all {len(allrows)} functions: {gap:.2e}"
          f" (random corruptions alone: {worst_random:.4f})")
    print(f"every upper 95% limit lies on or above the bound: {above}")


if __name__ == "__main__":
    main()
