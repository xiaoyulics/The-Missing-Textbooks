"""Chapter 1 (freeentropy): the Curie-Weiss free entropy at finite N against Laplace's bound and the
saddle-point corrections.  Run from the volume root:  python figures/scripts/01-freeentropy-curieweiss.py  (~5 s)

Model:  Z_N(beta, h) = sum_{s in {-1,1}^N} exp(beta S^2/(2N) + h S),  S = sum_i s_i,
computed exactly as a sum over the N + 1 values of S (log-sum-exp of log-binomials).
Theorem thm:freeentropy:curieweiss:  phi <= (1/N) log Z_N <= phi + log(N+1)/N  (beta >= 0), with
phi(beta, h) = max_m [s(m) + beta m^2/2 + h m] = max_mu [log 2cosh(beta mu + h) - beta mu^2/2].
Physical predictions for N((1/N) log Z_N - phi) (Laplace's method on the Hubbard-Stratonovich integral):
  * a nondegenerate saddle: log k - (1/2) log(1 - beta(1 - m*^2)), k = number of maximizers (2 at h = 0, beta > 1);
  * the critical point beta = 1, h = 0 (quartic saddle):  (1/4) log N + log c,
    c = 2 * 12^{1/4} Gamma(5/4) / sqrt(2 pi).
Data: figures/data/01-freeentropy-cw.dat (N, gaps times N, predictions, the bound log(N+1)).
Also printed: the second moment of SK at h = 0, E Z^2/(E Z)^2 = exp(-beta^2/2) Z_N^CW(beta^2, 0)/2^N,
and the Gibbs magnetization <m>_N at h = 0.2, beta = 1.5 against m*(beta, h).
"""
import math
import os
import sys

import numpy as np
from scipy import optimize
from scipy.special import gammaln, logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA = os.path.join(ROOT, "figures", "data")


def logZ(N, beta, h):
    k = np.arange(N + 1)
    m = (2 * k - N) / N
    return float(logsumexp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1) + N * (beta * m * m / 2 + h * m)))


def magnetization(N, beta, h):
    k = np.arange(N + 1)
    m = (2 * k - N) / N
    lw = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1) + N * (beta * m * m / 2 + h * m)
    w = np.exp(lw - logsumexp(lw))
    return float(np.sum(w * m))


def mstar(beta, h):
    """The maximizer of s(m) + beta m^2/2 + h m with the sign of h (the positive one at h = 0, beta > 1)."""
    if h == 0 and beta <= 1:
        return 0.0
    return rk.root(lambda m: math.tanh(beta * m + abs(h)) - m, 1e-12, 1.0) * (1 if h >= 0 else -1)


def phi(beta, h):
    m = mstar(beta, h)
    return math.log(2 * math.cosh(beta * m + h)) - beta * m * m / 2


def entropy(m):
    out = 0.0
    for p in ((1 + m) / 2, (1 - m) / 2):
        if p > 0:
            out -= p * math.log(p)
    return out


def main():
    c = 2 * 12 ** 0.25 * math.gamma(1.25) / math.sqrt(2 * math.pi)
    cases = [(0.5, 0.0), (1.0, 0.0), (1.5, 0.0)]
    info = {}
    for beta, h in cases:
        m = mstar(beta, h)
        p = phi(beta, h)
        # the two variational formulas agree (Legendre duality), and the maximizer solves m = tanh(beta m + h)
        p2 = entropy(m) + beta * m * m / 2 + h * m
        x = np.linspace(-1 + 1e-12, 1 - 1e-12, 200001)
        ent = -(1 + x) / 2 * np.log((1 + x) / 2) - (1 - x) / 2 * np.log((1 - x) / 2)
        brute = float(np.max(ent + beta * x * x / 2 + h * x))
        k = 2 if (h == 0 and beta > 1) else 1
        pred = math.log(k) - 0.5 * math.log(1 - beta * (1 - m * m)) if not (beta == 1 and h == 0) else float("nan")
        info[(beta, h)] = (p, m, pred)
        print(f"beta={beta}, h={h}: phi = {p:.10f} (entropy form {p2:.10f}, grid max {brute:.10f}), m* = {m:.8f}, "
              f"Gaussian saddle constant = {pred:.6f}")
    print(f"critical constant log c = {math.log(c):.6f}  (c = {c:.6f})")
    Ns = np.unique(np.round(np.logspace(1, 6, 41)).astype(int))
    rows = []
    for N in Ns:
        row = [int(N)]
        for beta, h in cases:
            p, m, pred = info[(beta, h)]
            gap = logZ(int(N), beta, h) - N * p
            assert -1e-9 <= gap <= math.log(N + 1) + 1e-9, (N, beta, gap)
            if beta == 1 and h == 0:
                pred = 0.25 * math.log(N) + math.log(c)
            row += [gap, pred]
        row.append(math.log(N + 1))
        rows.append(row)
    rk.write_dat(os.path.join(DATA, "01-freeentropy-cw.dat"),
                 ["N", "gap05", "pred05", "gap10", "pred10", "gap15", "pred15", "bound"], rows)
    print("N, N(f_N - phi) for beta = 0.5, 1, 1.5 (h = 0), and log(N+1):")
    for row in rows:
        if row[0] in (10, 100, 1000, 10000, 100000, 1000000):
            print(f"  N={row[0]:7d}: {row[1]:.5f} (pred {row[2]:.5f})  {row[3]:.5f} (pred {row[4]:.5f})"
                  f"  {row[5]:.5f} (pred {row[6]:.5f})  bound {row[7]:.4f}")
    # near-critical spontaneous magnetization m* ~ sqrt(3 (beta - 1))
    for beta in [1.1, 1.01]:
        print(f"beta={beta}: m* = {mstar(beta, 0.0):.5f}, sqrt(3(beta-1)) = {math.sqrt(3 * (beta - 1)):.5f}")
    # Gibbs magnetization converges to m*(beta, h) for h != 0 (Corollary cor:freeentropy:magnetization)
    beta, h = 1.5, 0.2
    print(f"beta={beta}, h={h}: m* = {mstar(beta, h):.6f}; <m>_N = "
          + ", ".join(f"{magnetization(N, beta, h):.6f} (N={N})" for N in (10, 100, 1000, 10000)))
    # SK second moment through Curie-Weiss at beta^2 (Computation comp:freeentropy:secondmoment)
    for b in (0.5, 0.8, 1.5):
        logs = [-b * b / 2 + logZ(N, b * b, 0.0) - N * math.log(2) for N in (16, 200, 2000)]
        lim = math.exp(-b * b / 2) / math.sqrt(1 - b * b) if b < 1 else float("inf")
        print(f"SK second-moment ratio E Z^2/(E Z)^2 at beta={b}: N=16 {math.exp(logs[0]):.5g}, N=200 "
              f"{math.exp(min(logs[1], 700)):.5g}; (1/N) log ratio at N=2000: {logs[2] / 2000:.5f}; limit {lim:.5f};"
              f"  (1/N) log of ratio -> phi_CW(beta^2) - log 2 = {phi(b * b, 0.0) - math.log(2):.5f}")


if __name__ == "__main__":
    main()
