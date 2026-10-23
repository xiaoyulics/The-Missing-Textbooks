"""07-catalogue-formulas.py -- the formulas of Chapter 7 (A Catalogue), checked against each other.

Run from the volume root:  python figures/scripts/07-catalogue-formulas.py   (about 15 seconds)

Prints every number the chapter quotes from formulas:
  * the Aoyagi--Watanabe learning coefficient of reduced-rank regression (as restated by Hayashi and
    Watanabe 2017, eq. (6); multiplicity 2 in the odd sub-case, as in Watanabe 2013, Table 3) and its
    checks: the rank-one theorem (H = 1), the reduction to a zero truth, Watanabe's Table 3, Drton and
    Plummer's Table 1;
  * Aoyagi's deep-linear-network formula (restated in Aoyagi 2025, Definition 4) and its agreement with
    the reduced-rank formula at depth 2 and with the chain;
  * the one-input tanh-network formula (Aoyagi 2019, end of Section 4);
  * the volume function of rank-one regression with M = N = 3 and the uniform prior on the box
    [-1, 1]^6, exactly, with a Monte Carlo check, and the constant kappa of its tempered mean;
  * the learning coefficients lambda_ij of 3 x 3 reduced-rank regression and the singular-BIC example.
Writes figures/data/07-catalogue-table.dat (the small instances of the catalogue table).
"""
import math
import os
import sys
from fractions import Fraction as Fr
from itertools import combinations

import numpy as np
from scipy import integrate, optimize, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402


def aoyagi_watanabe(M, N, H, r):
    """(lambda, m) of reduced-rank regression y = BAx, B in R^{M x H}, A in R^{H x N}, true rank r."""
    assert 0 <= r <= min(M, N, H)
    if N + r <= M + H and M + r <= N + H and H + r <= M + N:
        s = 2 * (H + r) * (M + N) - (M - N) ** 2 - (H + r) ** 2
        return (Fr(s, 8), 1) if (M + N + H + r) % 2 == 0 else (Fr(s + 1, 8), 2)
    if M + H < N + r:
        return Fr(H * M - H * r + N * r, 2), 1
    if N + H < M + r:
        return Fr(H * N - H * r + M * r, 2), 1
    return Fr(M * N, 2), 1                       # M + N < H + r


def deep_linear(widths, r):
    """(lambda, m) of y = A1 A2 ... AL x with Aj of size widths[j-1] x widths[j] (output width first)
    and a true product of rank r: Aoyagi (2024) in the form of Aoyagi (2025, Definition 4, Remark 1)."""
    Ms = [h - r for h in widths]
    base = Fr(-r * r + r * (widths[0] + widths[-1]), 2)
    if min(Ms) == 0:
        return base, 1
    found = set()
    idx = range(len(Ms))
    for k in range(2, len(Ms) + 1):
        for S in combinations(idx, k):
            ell, tot = k - 1, sum(Ms[s] for s in S)
            out = [t for t in idx if t not in S]
            if (all(Ms[s] < Ms[t] for s in S for t in out) and all(tot >= ell * Ms[s] for s in S)
                    and all(tot < ell * Ms[t] for t in out)):
                Mint = -(-tot // ell)             # the integer with Mint - 1 < tot/ell <= Mint
                a = tot - (Mint - 1) * ell
                lam = (base + Fr(a * (ell - a), 4 * ell) + Fr(tot * tot, 4 * ell)
                       - Fr(sum(Ms[s] ** 2 for s in S), 4))
                found.add((lam, a * (ell - a) + 1))
    assert len(found) == 1, (widths, r, found)
    return found.pop()


def tanh_one_input(H, M=1, Q=2):
    """Aoyagi (2019): sum_h a_h tanh(b_h x) with one input, M outputs, truth 0 (Q = 2 for tanh)."""
    k = max(i for i in range(0, H + 2) if 2 * H >= M * (i * (i - 1) * Q + 2 * i))
    lam = Fr(M * Q * k * (k + 1) + 2 * H, 4 * (1 + k * Q))
    return lam, (2 if 2 * H == M * (k * (k - 1) * Q + 2 * k) else 1)


def fmt(q):
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def main():
    # ---------------------------------------------------------------- reduced-rank regression
    tuples = [(M, N, H, r) for M in range(1, 10) for N in range(1, 10) for H in range(1, 10)
              for r in range(0, min(M, N, H) + 1)]
    for M, N, H, r in tuples:
        lam, m = aoyagi_watanabe(M, N, H, r)
        if H == 1 and r == 0:
            assert (lam, m) == (Fr(min(M, N), 2), 2 if M == N else 1)
        if H == 1 and r == 1:
            assert (lam, m) == (Fr(M + N - 1, 2), 1)
        sub = aoyagi_watanabe(M - r, N - r, H - r, 0) if min(M, N, H) > r else (Fr(0), 1)
        assert (lam, m) == (Fr(M * r + N * r - r * r, 2) + sub[0], sub[1])
        assert deep_linear((M, H, N), r) == (lam, m)
    print(f"checked {len(tuples)} tuples (M, N, H <= 9): H = 1 agrees with the rank-one theorem and the "
          "Morse-Bott value; the reduction to a zero truth holds; Aoyagi's deep formula at depth 2 agrees")
    for L in range(1, 9):
        assert deep_linear((1,) * (L + 1), 0) == (Fr(1, 2), L)
    print("the chain y = a1...aL x: (1/2, L) from Aoyagi's deep formula, L = 1..8")
    for h in range(1, 7):
        for L in range(h, 13):
            assert deep_linear((h,) * (L + 1), 0) == (Fr(h * (h + 1), 4), h * (L - h) + 1), (h, L)
    print("equal widths h, depth L >= h, truth 0: (h(h+1)/4, h(L-h)+1) for h <= 6, L <= 12;",
          "width 3, depths 3, 4, 5:", [deep_linear((3,) * (L + 1), 0) for L in (3, 4, 5)],
          "; width 3, depth 2:", deep_linear((3, 3, 3), 0))
    for (M, N, H, r) in ((4, 4, 2, 1), (3, 5, 2, 1), (2, 3, 2, 1)):
        print(f"   ({M},{N},{H},{r}): {aoyagi_watanabe(M, N, H, r)}; reduced part {aoyagi_watanabe(M - r, N - r, H - r, 0)}")
    print("sub-model bound H min(M,N)/2 at the zero truth, M = N = 3:",
          [(H, fmt(aoyagi_watanabe(3, 3, H, 0)[0]), fmt(Fr(3 * H, 2))) for H in (1, 2, 3)])
    print("Watanabe 2013 Table 3 (M = N = 6, r = 3), H = 3..6:",
          [(fmt(aoyagi_watanabe(6, 6, H, 3)[0]), aoyagi_watanabe(6, 6, H, 3)[1]) for H in range(3, 7)])
    print("Drton-Plummer Table 1 (5 x 3):", {(i, j): fmt(aoyagi_watanabe(5, 3, i, j)[0]) + ("(m=2)" if aoyagi_watanabe(5, 3, i, j)[1] == 2 else "")
                                            for i in range(1, 4) for j in range(0, i + 1)})
    print("3 x 3 reduced-rank regression, lambda_ij (model rank i, true rank j):")
    for i in range(1, 4):
        print("   i =", i, [(fmt(aoyagi_watanabe(3, 3, i, j)[0]), aoyagi_watanabe(3, 3, i, j)[1]) for j in range(0, i + 1)])

    # ---------------------------------------------------------------- the catalogue table
    rows = []

    def add(name, d, lam, m):
        rows.append((name, d, float(Fr(d, 2)), float(lam), m))
        print(f"   {name:28s} d = {d:2d}  d/2 = {fmt(Fr(d, 2)):5s}  lambda = {fmt(lam):5s}  m = {m}")

    print("catalogue table (small instances):")
    for (H, r) in ((1, 0), (1, 1), (2, 0), (2, 1), (3, 1)):
        lam, m = aoyagi_watanabe(3, 3, H, r)
        add(f"RRR_3x3_H{H}_r{r}", H * 6, lam, m)
    for H in (4, 6):
        lam, m = aoyagi_watanabe(6, 6, H, 3)
        add(f"RRR_6x6_H{H}_r3", H * 12, lam, m)
    for widths in ((1, 1, 1, 1), (2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2, 2), (3, 3, 3), (3, 3, 3, 3)):
        lam, m = deep_linear(widths, 0)
        d = sum(widths[i] * widths[i + 1] for i in range(len(widths) - 1))
        add("DLN_" + "-".join(map(str, widths)) + "_r0", d, lam, m)
    for H in (1, 2, 3, 4):
        lam, m = tanh_one_input(H)
        add(f"tanh_H{H}", 2 * H, lam, m)
    add("RBM_M3_N1_a0", 3, Fr(3, 4), 1)              # Aoyagi 2010, Theorems 2 and 3
    add("RBM_M4_N2_a0", 8, Fr(8, 4), 1)              # Theorem 3: MN/4, M > N + 1
    add("NaiveBayes_n3_type2", 7, Fr(4, 2), 1)       # Rusakov-Geiger Theorem 4(c): (n+1)/2
    add("NaiveBayes_n2_indep", 5, Fr(3, 2), 3)       # Theorem 4(e): 3/2 with +2 lnln N
    add("mixture_3par_truth_N(mu,1)", 3, Fr(3, 4), 1)  # Drton-Plummer Example 2.1
    add("book_mixture", 2, Fr(1, 2), 2)
    sk.write_dat("figures/data/07-catalogue-table.dat", ["name", "d", "half_d", "lam", "m"], rows)

    # RBM Theorem 3 cross-check by the Newton polyhedron for M = 3, N = 1: the face x+y+z = 4
    print("RBM M=3, N=1, truth 0: K ~ b1^2 b2^2 + b1^2 b3^2 + b2^2 b3^2, Newton distance t = 4/3 -> lambda = 3/4")

    # ---------------------------------------------------------------- rank-one regression, box prior
    J1, err = integrate.dblquad(
        lambda th, ph: math.log(1.0 / max(math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
        * math.sin(th), 0, math.pi / 2, 0, math.pi / 2, epsabs=1e-12, epsrel=1e-10)
    c2 = math.pi / 3 * J1 + math.pi ** 2 / 36
    C = 2 ** 1.5 * math.pi ** 2 / 24
    print(f"rank-one 3x3, uniform prior on [-1,1]^6: J1 = E[|a|^-3; |a| > 1] = {J1:.6f}; "
          f"V(eps) = (2 eps)^(3/2) [(pi^2/24) log(1/(2 eps)) + {c2:.4f}] for eps <= 1/2; "
          f"leading constant pi^2 sqrt2 / 12 = {C:.4f}")
    gen = sk.rng(707)
    hits, total = np.zeros(3), 0
    eps = np.array([1e-2, 1e-3, 1e-4])
    for _ in range(10):
        a = gen.uniform(-1, 1, (2_000_000, 3))
        b = gen.uniform(-1, 1, (2_000_000, 3))
        K = 0.5 * np.sum(a * a, 1) * np.sum(b * b, 1)
        hits += (K[None, :] < eps[:, None]).sum(axis=1)
        total += len(K)
    for e, h in zip(eps, hits):
        lo, hi = sk.clopper_pearson(int(h), total)
        print(f"   eps = {e:g}: exact V = {(2 * e) ** 1.5 * (math.pi ** 2 / 24 * math.log(1 / (2 * e)) + c2):.4e}, "
              f"Monte Carlo {h / total:.4e} [{lo:.4e}, {hi:.4e}], leading term {C * e ** 1.5 * math.log(1 / e):.4e}")
    # the ball prior (exact, Computation 7.1): V = delta^3 (1 + 3 log(1/delta)) for M = N = 3 and
    # V = 3 delta^2 - 2 delta^3 for (M, N) = (3, 2), delta = sqrt(2 eps); Monte Carlo check of both
    r3, r3b = gen.random(4_000_000) ** (1 / 3), gen.random(4_000_000) ** (1 / 3)
    r2 = gen.random(4_000_000) ** (1 / 2)
    for e in (1e-2, 1e-4):
        dlt = math.sqrt(2 * e)
        v33, v32 = dlt ** 3 * (1 + 3 * math.log(1 / dlt)), 3 * dlt ** 2 - 2 * dlt ** 3
        print(f"   ball prior, eps = {e:g}: M=N=3 V = {v33:.4e} (Monte Carlo {np.mean(0.5 * r3 ** 2 * r3b ** 2 < e):.4e}); "
              f"(M,N)=(3,2) V = {v32:.4e} (Monte Carlo {np.mean(0.5 * r2 ** 2 * r3 ** 2 < e):.4e})")
    # the constant of Theorem 7.1(ii) for (M, N) = (3, 2) on [-1, 1]^5: C = (pi/2) E|b|^-2, b uniform on the cube
    Eb2, _ = integrate.dblquad(
        lambda th, ph: math.sin(th) / max(math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)),
        0, math.pi / 2, 0, math.pi / 2, epsabs=1e-12, epsrel=1e-10)
    print(f"   (M,N)=(3,2), box prior: E|b|^-2 = {Eb2:.4f}, C = (pi/2) E|b|^-2 = {math.pi / 2 * Eb2:.4f}")
    kappa = 8 * J1 / math.pi - math.log(2) - special.digamma(1.5)
    print(f"kappa (3x3 rank one, box) = 8 J1/pi - log 2 - psi(3/2) = {kappa:.4f};  "
          f"kappa (product model) = gamma + log 2 = {np.euler_gamma + math.log(2):.4f}")

    def two_temp(lam, m, n, kap):
        lb = math.log(n)
        b1, b2 = 1 / lb, 1.5 / lb
        E = lambda b: (lam - (m - 1) / (math.log(n * b) + kap)) / b
        return (E(b1) - E(b2)) / (1 / b1 - 1 / b2)

    for n in (1e3, 1e4, 1e5):
        print(f"   n = {n:.0e}: predicted two-temperature value, rank one 3x3 zero truth {two_temp(1.5, 2, n, kappa):.3f} "
              f"(digests' rule lambda - 1/log n = {1.5 - 1 / math.log(n):.3f}); product model {two_temp(0.5, 2, n, np.euler_gamma + math.log(2)):.3f}"
              f" ({0.5 - 1 / math.log(n):.3f}); H=2, r=1 with kappa = 0: {two_temp(3.5, 2, n, 0.0):.3f}")

    # ---------------------------------------------------------------- the singular BIC, worked example
    def sbic_gap(T, n):
        """sBIC(M1) - sBIC(M0) for M0 = {C = 0} and M1 = {rank C <= 1}, 3 x 3, with
        lambda_00 = 0, lambda_10 = 3/2 (m = 2), lambda_11 = 5/2 (m = 1); T = twice the max log-LR."""
        A = math.exp(T / 2)
        L11, L10 = A * n ** -2.5, A * n ** -1.5 * math.log(n)
        b, c = 1.0 - L11, L10
        return math.log(0.5 * (-b + math.sqrt(b * b + 4 * c)))

    for n in (100, 1000):
        Tstar = optimize.brentq(lambda T: sbic_gap(T, n), 0, 200)
        print(f"sBIC example, n = {n}: M1 preferred when T > {Tstar:.2f}; BIC with dim 5 needs T > {5 * math.log(n):.2f}, "
              f"with d = 6 needs T > {6 * math.log(n):.2f}")
        for T in (0, 10, 20, 30, 40):
            print(f"   T = {T:2d}: sBIC gap {sbic_gap(T, n):+.2f}; BIC gap (dim 5) {T / 2 - 2.5 * math.log(n):+.2f}; "
                  f"pure singular formula {T / 2 - 1.5 * math.log(n) + math.log(math.log(n)):+.2f}")


if __name__ == "__main__":
    main()
