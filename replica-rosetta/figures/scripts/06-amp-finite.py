"""Chapter 6 (Message Passing): the finite-N approach to state evolution at a fixed iteration, and a
universality check with a non-Gaussian matrix.

Z2 synchronization, lam = 1.5, warm start from a side channel with SNR g0 = 0.1 (the setting of the
chapter's theorem, first step without Onsager term).  At the fixed iteration t = 3 the theorem gives
<m^3, x>/N -> q_3 = 1 - mmse(gamma_3) in probability.  For N = 500, 1000, 2000, 4000 (seeds
8000 + N + r) we report the mean and the root-mean-square deviation of the
overlap from q_3, and sqrt(N) times the RMS deviation, which stays bounded if the fluctuations are
of order N^{-1/2}.  (R = 200, 100, 60, 30 for the four sizes.)
Universality: the same AMP with W replaced by a symmetric matrix with independent entries +-1/sqrt(N)
on and above the diagonal (N = 3000, R = 20, seeds 9000 + r); the chapter's proof does not apply
(it needs Gaussian entries), Chen--Lam (2021) prove that state evolution still holds.
Intervals: Student-t 95%.  Writes figures/data/06-amp-finite.dat.  Run from the volume root (~50 s).
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk

LAM, G0, TFIX = 1.5, 0.1, 3


def q_of(g):
    return 1.0 - rk.mmse_rademacher(g, n=201) if g > 0 else 0.0


def se(g0, T):
    g = [g0]
    for _ in range(T):
        g.append(LAM ** 2 * q_of(g[-1]))
    return np.array(g)


def goe32(N, gen):
    """GOE(N) in single precision (off-diagonal variance 1/N, diagonal 2/N), to keep memory low."""
    G = gen.standard_normal((N, N), dtype=np.float32)
    W = G + G.T
    W *= np.float32(1.0 / math.sqrt(2.0 * N))
    return W


def rademacher_wigner(N, gen):
    """Symmetric matrix with independent entries +-1/sqrt(N) on and above the diagonal."""
    S = gen.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(N, N))
    U = np.triu(S)
    W = U + np.triu(U, 1).T
    W *= np.float32(1.0 / math.sqrt(N))
    return W


def run(W, x, gen, T):
    """Warm-start Bayes-AMP; returns the overlaps <m^t, x>/N, t = 0..T."""
    N = x.size
    ytil = math.sqrt(G0) * x + gen.standard_normal(N)
    h, mp = math.sqrt(G0) * ytil, np.zeros(N)
    out = []
    for _ in range(T + 1):
        m = np.tanh(h)
        out.append(m @ x / N)
        Am = (W @ m.astype(np.float32)).astype(float) + (LAM / N) * x * (x @ m)
        h, mp = LAM * Am - LAM ** 2 * (1 - m @ m / N) * mp, m
    return np.array(out)


def main():
    t0 = time.time()
    g = se(G0, 12)
    q = np.array([q_of(v) for v in g])
    print("SE overlaps q_t (t=0..12):", np.round(q, 4))
    rows = []
    for N, R in ((500, 200), (1000, 100), (2000, 60), (4000, 30)):
        dev = []
        for r in range(R):
            gen = rk.rng(8000 + N + r)
            x = gen.choice([-1.0, 1.0], size=N)
            W = goe32(N, gen)
            dev.append(run(W, x, gen, TFIX)[TFIX] - q[TFIX])
            del W
        dev = np.array(dev)
        mean, half = rk.mean_ci(dev)
        rms = math.sqrt(np.mean(dev ** 2))
        rows.append((N, R, q[TFIX] + mean, half, rms, math.sqrt(N) * rms))
        print("N=%4d R=%3d: overlap at t=3 %.4f +- %.4f (SE %.4f); RMS deviation %.4f; sqrt(N)*RMS %.3f"
              % (N, R, q[TFIX] + mean, half, q[TFIX], rms, math.sqrt(N) * rms))
    rk.write_dat(os.path.join("figures", "data", "06-amp-finite.dat"),
                 ["N", "R", "overlap", "ci", "rms", "sqrtNrms"], rows)
    # universality: Rademacher entries
    N, R, T = 3000, 20, 12
    ov_g, ov_r = [], []
    for r in range(R):
        gen = rk.rng(9000 + r)
        x = gen.choice([-1.0, 1.0], size=N)
        W = rademacher_wigner(N, gen)
        ov_r.append(run(W, x, gen, T))
        del W
        W = goe32(N, gen)
        ov_g.append(run(W, x, gen, T))
        del W
    ov_r, ov_g = np.array(ov_r), np.array(ov_g)
    print("universality, N=%d, R=%d: t, overlap with +-1 entries, with Gaussian entries, SE" % (N, R))
    worst = 0.0
    for t in range(T + 1):
        mr, hr = rk.mean_ci(ov_r[:, t])
        mg, hg = rk.mean_ci(ov_g[:, t])
        worst = max(worst, abs(mr - q[t]))
        print("  %2d  %.4f +- %.4f   %.4f +- %.4f   %.4f" % (t, mr, hr, mg, hg, q[t]))
    print("largest |mean overlap (+-1 entries) - SE| over t = 0..12: %.4f" % worst)
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
