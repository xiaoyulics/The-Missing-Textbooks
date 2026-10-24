"""Chapter 6 (Message Passing): Bayes-AMP for Z2 synchronization against state evolution.

Model (OUTLINE.md): A = (lam/N) x x^T + W, W ~ GOE(N), x uniform on {-1,+1}^N, lam = 1.5, N = 3000.
AMP in the kit's convention (rosettakit.amp_z2):
    y^{t+1} = A f_t(y^t) - b_t f_{t-1}(y^{t-1}),  f_t(y) = tanh(lam y),  b_t = mean f_t'(y^t),
and the field h^t = lam y^t (Celentano--Fan--Mei's variable).  State evolution (the chapter's
corollary on Bayes-AMP for Z2 synchronization, and Montanari--Venkataramanan's theorem):
    gamma_{t+1} = lam^2 (1 - mmse(gamma_t)),   h^t ~ gamma_t x + sqrt(gamma_t) Z,
    cos(m^t, x) -> sqrt(q_t),  q_t = 1 - mmse(gamma_t).
Five runs on each matrix:
  warm      warm start from a side channel ytil = sqrt(g0) x + g, m^0 = tanh(sqrt(g0) ytil), g0 = 0.1,
            first step without Onsager term (the setting of the chapter's theorem); gamma_0 = g0;
  warm_no   the same iteration without any Onsager term, y^{t+1} = A f_t(y^t);
  spec      spectral start (Montanari--Venkataramanan): y^0 = sqrt(N) phi_1, h^0 = lam sqrt(lam^2-1) y^0,
            m^0 = tanh(h^0), virtual previous estimate m^{-1} = h^0/lam^2 (the convention that matches
            MV21's E{L f(X_{-1})} = 1/lam); gamma_0 = lam^2 - 1;
  spec_mis  the same with m^{-1} = lam h^0, as printed in Celentano--Fan--Mei arXiv v3, (SI);
  spec_no   spectral start, no Onsager term at all.
R = 24 matrices, seeds 6000 + r.  Intervals: Student-t 95% over the R matrices.
Writes figures/data/06-amp-iterations.dat and prints the numbers quoted in the text.
Run from the volume root: python figures/scripts/06-amp-iterations.py   (about 30 s on an idle laptop)
"""
import math
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk

LAM, N, T, R, G0 = 1.5, 3000, 12, 24, 0.1
OUT = os.path.join("figures", "data", "06-amp-iterations.dat")


def q_of(g):
    """q(gamma) = 1 - mmse(gamma) = E tanh(gamma + sqrt(gamma) Z) for the Rademacher prior."""
    return 1.0 - rk.mmse_rademacher(g, n=201) if g > 0 else 0.0


def se(g0, T):
    """State evolution gamma_0 = g0, gamma_{t+1} = lam^2 q(gamma_t)."""
    g = [g0]
    for _ in range(T):
        g.append(LAM ** 2 * q_of(g[-1]))
    return np.array(g)


def amp(matvec, x, h0, m_prev, onsager=True, T=T):
    """AMP in h = lam y variables: h^{t+1} = lam A m^t - lam^2 (1 - Q(m^t)) m^{t-1}, m^t = tanh(h^t).
    Returns per-iteration rows (cos(m^t, x), <m^t, x>/N, <h^t, x>/N, residual variance of h^t)."""
    n = x.size
    h, mp = h0.copy(), m_prev.copy()
    rows = []
    for _ in range(T + 1):
        m = np.tanh(h)
        mu = h @ x / n
        rows.append((m @ x / (np.linalg.norm(m) * math.sqrt(n)), m @ x / n, mu, np.mean((h - mu * x) ** 2)))
        if onsager:
            h, mp = LAM * matvec(m) - LAM ** 2 * (1.0 - m @ m / n) * mp, m
        else:
            h, mp = LAM * matvec(m), m
    return np.array(rows)


def main():
    t_start = time.time()
    names = ["warm", "warm_no", "spec", "spec_mis", "spec_no"]
    res = {k: [] for k in names}
    eig = []
    for r in range(R):
        gen = rk.rng(6000 + r)
        x = gen.choice([-1.0, 1.0], size=N)
        W = rk.goe(N, gen)
        matvec = lambda v: W @ v + (LAM / N) * x * (x @ v)
        # warm start from an independent side channel (covered by the chapter's theorem)
        ytil = math.sqrt(G0) * x + gen.standard_normal(N)
        hw = math.sqrt(G0) * ytil
        res["warm"].append(amp(matvec, x, hw, np.zeros(N)))
        res["warm_no"].append(amp(matvec, x, hw, np.zeros(N), onsager=False))
        if r == 0:   # the kit's amp_z2 is the same iteration (y = h / lam): check that the estimates agree
            Afull = W + (LAM / N) * np.outer(x, x)
            est = rk.amp_z2(Afull, LAM, np.tanh(hw), T)
            h, mp = hw.copy(), np.zeros(N)
            for _ in range(T):
                m = np.tanh(h)
                h, mp = LAM * matvec(m) - LAM ** 2 * (1 - m @ m / N) * mp, m
            assert np.max(np.abs(est[-1] - np.tanh(h))) < 1e-8, "kit amp_z2 differs"
            del Afull
        # spectral start
        op = LinearOperator((N, N), matvec=matvec, dtype=float)
        val, vec = eigsh(op, k=1, which="LA", tol=1e-10)
        phi = vec[:, 0] * (1.0 if vec[:, 0] @ x >= 0 else -1.0)
        eig.append((val[0], (phi @ x) ** 2 / N))
        h0 = LAM * math.sqrt(LAM ** 2 - 1) * math.sqrt(N) * phi
        res["spec"].append(amp(matvec, x, h0, h0 / LAM ** 2))
        res["spec_mis"].append(amp(matvec, x, h0, LAM * h0))
        res["spec_no"].append(amp(matvec, x, h0, np.zeros(N), onsager=False))
        del W
    gw, gs = se(G0, T), se(LAM ** 2 - 1, T)
    stats = {}
    for k in names:
        a = np.array(res[k])          # R x (T+1) x 4
        stats[k] = [[rk.mean_ci(a[:, t, j]) for j in range(4)] for t in range(T + 1)]
    header = ["t", "se_warm", "se_spec", "gam_warm", "gam_spec"]
    for k in names:
        header += [k + "_cos", k + "_ci", k + "_coef", k + "_var"]
    rows = []
    for t in range(T + 1):
        row = [t, math.sqrt(q_of(gw[t])), math.sqrt(q_of(gs[t])), gw[t], gs[t]]
        for k in names:
            s = stats[k][t]
            row += [s[0][0], s[0][1], s[2][0], s[3][0]]
        rows.append(row)
    rk.write_dat(OUT, header, rows)

    ev = np.array(eig)
    print("N=%d lam=%.2f R=%d: top eigenvalue %.4f (limit %.4f), spectral cos^2 %.4f +- %.4f (limit %.4f)"
          % (N, LAM, R, ev[:, 0].mean(), rk.bbp_eigenvalue(LAM), ev[:, 1].mean(), rk.mean_ci(ev[:, 1])[1],
             rk.bbp_overlap(LAM)))
    gstar = rk.root(lambda g: LAM ** 2 * q_of(g) - g, 0.1, 10.0)
    print("SE fixed point gamma* = %.6f, q* = %.6f, cos* = %.6f; spectral cos = %.6f"
          % (gstar, gstar / LAM ** 2, math.sqrt(gstar / LAM ** 2), math.sqrt(1 - LAM ** -2)))
    for k in names:
        G = gw if k.startswith("warm") else gs
        print("\n%s:  t  cos (SE)  |  coef (SE gamma)  var" % k)
        for t in range(T + 1):
            s = stats[k][t]
            print("   %2d  %.4f +- %.4f (%.4f) | %.3f (%.3f) %.3f" % (
                t, s[0][0], s[0][1], math.sqrt(q_of(G[t])), s[2][0], G[t], s[3][0]))
    # how many of the SE values fall inside the intervals (theorem-covered runs and the spectral run)
    for k, G in (("warm", gw), ("spec", gs)):
        inside = sum(abs(stats[k][t][0][0] - math.sqrt(q_of(G[t]))) <= stats[k][t][0][1] for t in range(T + 1))
        print("%s: SE inside the 95%% interval at %d of %d iterations" % (k, inside, T + 1))
    # paired comparison of AMP and the naive iteration on the same matrices, at the last iteration
    for a, b in (("spec", "spec_no"), ("warm", "warm_no")):
        d = np.array(res[a])[:, T, 0] - np.array(res[b])[:, T, 0]
        md, hd = rk.mean_ci(d)
        print("cos(%s) - cos(%s) at t=%d: %.4f +- %.4f (paired, %d matrices)" % (a, b, T, md, hd, R))
    # predicted statistics of the naive second iterate from the warm start (exercise)
    q0, q1 = q_of(gw[0]), q_of(gw[1])
    c = LAM ** 2 * (1 - q1)
    print("naive iterate t=2 from warm start: predicted coef %.4f var %.4f" % (gw[2] + c * q0, gw[2] + c * c * q0 * (1 - q0)))
    print("elapsed %.1f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
