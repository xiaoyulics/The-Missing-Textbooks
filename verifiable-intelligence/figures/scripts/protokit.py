"""protokit -- the shared kit for the "Running the protocol" sections of Verifiable Intelligence.

Every chapter runs its protocol: an honest prover and one or more cheating provers interact with
the verifier many times, and the measured acceptance rates are compared with the bounds the
chapter proves. This module holds what several chapters need:

  rng(seed)                      a NumPy generator (always pass an explicit seed)
  PRIME, Field(p)                arithmetic in the prime field F_p, vectorized over NumPy arrays
  Field.lagrange_eval(ys, r)     the polynomial of degree <= len(ys)-1 through (0,ys[0]),(1,ys[1]),...
                                 evaluated at r
  Field.mle_eval(table, point)   the multilinear extension of a table on {0,1}^l at a point of F^l
  Field.poly_eval(coeffs, r)     a polynomial given by coefficients (lowest degree first) at r
  acceptance(trial, n, seed)     run trial(gen) -> bool n times; return (rate, lo, hi) with a
                                 Clopper-Pearson 95% interval
  clopper_pearson(k, n)          the exact binomial 95% interval for k successes in n trials
  write_dat(path, header, rows)  whitespace-separated data with a header row, for pgfplots

Conventions: scripts live in figures/scripts/NN-key-name.py, run from the volume root, write to
figures/data/NN-key-name*.dat, print every number the chapter quotes, and finish in at most about
two minutes on a laptop CPU. Field elements are Python ints or int64 NumPy arrays reduced mod p;
p < 2^31 so that products fit in int64.

Self-test:  python3 figures/scripts/protokit.py
"""
from __future__ import annotations

import os
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import stats

# 2^31 - 1, a Mersenne prime: products of two reduced elements fit in a signed 64-bit integer
PRIME = 2_147_483_647


def rng(seed: int) -> np.random.Generator:
    """A NumPy generator with an explicit seed (never use global random state)."""
    return np.random.default_rng(seed)


class Field:
    """The prime field F_p. Elements are ints (or int64 arrays) in [0, p)."""

    def __init__(self, p: int = PRIME):
        if p < 2 or p >= 2 ** 31:
            raise ValueError("use a prime p with 2 <= p < 2^31")
        self.p = p

    # --- elementwise arithmetic (ints or int64 arrays) ---
    def add(self, a, b):
        return (np.asarray(a, dtype=np.int64) + b) % self.p

    def sub(self, a, b):
        return (np.asarray(a, dtype=np.int64) - b) % self.p

    def mul(self, a, b):
        return (np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64)) % self.p

    def inv(self, a: int) -> int:
        a = int(a) % self.p
        if a == 0:
            raise ZeroDivisionError("0 has no inverse")
        return pow(a, self.p - 2, self.p)

    def random(self, gen: np.random.Generator, size=None):
        """Uniform field element(s)."""
        return gen.integers(0, self.p, size=size, dtype=np.int64)

    # --- polynomials ---
    def poly_eval(self, coeffs: Sequence[int], r: int) -> int:
        """sum_k coeffs[k] r^k (Horner)."""
        acc = 0
        for c in reversed(list(coeffs)):
            acc = (acc * r + int(c)) % self.p
        return acc

    def lagrange_eval(self, ys: Sequence[int], r: int) -> int:
        """Value at r of the unique polynomial of degree <= d with value ys[i] at i = 0..d."""
        d = len(ys) - 1
        r = int(r) % self.p
        total = 0
        for i in range(d + 1):
            num, den = 1, 1
            for j in range(d + 1):
                if j != i:
                    num = num * ((r - j) % self.p) % self.p
                    den = den * ((i - j) % self.p) % self.p
            total = (total + int(ys[i]) * num % self.p * self.inv(den)) % self.p
        return total

    def mle_eval(self, table: Sequence[int], point: Sequence[int]) -> int:
        """Multilinear extension of table (length 2^l, index bits b_1..b_l with b_1 the most
        significant) at point (r_1, ..., r_l): fold one variable at a time."""
        vals = np.asarray(table, dtype=np.int64) % self.p
        l = len(point)
        if vals.shape[0] != 2 ** l:
            raise ValueError("table length must be 2^len(point)")
        for r in point:
            half = vals.shape[0] // 2
            lo, hi = vals[:half], vals[half:]
            vals = (lo + (hi - lo) % self.p * (int(r) % self.p)) % self.p
        return int(vals[0])


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval for a binomial proportion with k successes in n trials."""
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def acceptance(trial: Callable[[np.random.Generator], bool], n: int, seed: int) -> tuple[float, float, float]:
    """Run trial(gen) n times with one generator seeded by seed; return the acceptance rate and
    its Clopper-Pearson 95% interval."""
    gen = rng(seed)
    k = sum(1 for _ in range(n) if trial(gen))
    lo, hi = clopper_pearson(k, n)
    return k / n, lo, hi


def write_dat(path: str, header: Iterable[str], rows: Iterable[Iterable]) -> None:
    """Write whitespace-separated columns with a header row (pgfplots reads it by column name)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = list(header)
    with open(path, "w") as fh:
        fh.write(" ".join(header) + "\n")
        for row in rows:
            row = list(row)
            if len(row) != len(header):
                raise ValueError("row length does not match header")
            fh.write(" ".join(_fmt(v) for v in row) + "\n")


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


def _self_test() -> None:
    F = Field(101)
    # Lagrange: the polynomial 3 + 2x + x^2 through 0..2, evaluated at 7
    ys = [F.poly_eval([3, 2, 1], i) for i in range(3)]
    assert F.lagrange_eval(ys, 7) == (3 + 14 + 49) % 101
    # MLE agrees with the table on the cube and is affine in each variable
    table = [5, 9, 2, 7]          # f(b1, b2) with index 2*b1 + b2
    assert [F.mle_eval(table, [b1, b2]) for b1 in (0, 1) for b2 in (0, 1)] == table
    v = [F.mle_eval(table, [3, r]) for r in range(4)]
    assert (v[2] - 2 * v[1] + v[0]) % 101 == 0
    # acceptance and intervals
    rate, lo, hi = acceptance(lambda g: g.random() < 0.25, 4000, seed=1)
    assert lo < 0.25 < hi and abs(rate - 0.25) < 0.03
    assert clopper_pearson(0, 100)[0] == 0.0 and clopper_pearson(100, 100)[1] == 1.0
    big = Field()
    a, b = big.random(rng(0), 3), big.random(rng(1), 3)
    assert np.all(big.mul(a, b) == (a.astype(object) * b.astype(object)) % PRIME)
    print("protokit self-test passed")


if __name__ == "__main__":
    _self_test()
