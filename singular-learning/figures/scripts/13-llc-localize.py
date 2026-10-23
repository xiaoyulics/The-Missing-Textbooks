"""Chapter 13 (llc): what a Gaussian localizer measures.

Rank-one regression y = b (a^T x) + noise, a in R^N, b in R^M, x ~ N(0, I_N), truth 0, so that
K(a, b) = |a|^2 |b|^2 / 2. At w* = (a*, 0) with a* != 0 the local learning coefficient is M/2 (K is a
nondegenerate quadratic in b there); at the origin it is min(M, N)/2. The localized "population skeleton"
of Lau et al.'s estimator, t E[K] under the density prop. to exp(-t K(w) - gamma |w - w*|^2 / 2), t = n beta,
reduces to a one-dimensional integral (the chapter's proposition): integrating b out,
    t E[K] = (M/2) E[ t|a|^2 / (t|a|^2 + gamma) ],  a with density prop. to (t|a|^2+gamma)^(-M/2) e^{-gamma|a-a*|^2/2},
and the remaining integral over a in R^N is reduced to its radius. The ReLU neuron at (1, 0) is the case
M = N = 1 with K = a^2 b^2 / 4.

Computed: (1) the skeleton against gamma at t = n/log n, n = 10^4, for the ReLU neuron at (1,0) and for
rank-one regression with (M, N) = (3, 3) and (3, 2) at a* = e1; (2) the skeleton against t for (M, N) = (3, 2)
at gamma = 1, 10, 30; (3) SGLD estimates (Lau et al.'s update, noiseless labels, minibatches of 500,
step size 1e-4, 30000 steps of which the first 15000 are discarded, 3 data sets x 4 chains) for (M, N) = (3, 2)
at six values of gamma, with Student-t 95% intervals over the data-set means.

Outputs: figures/data/13-llc-localize-gamma.dat, figures/data/13-llc-localize-t.dat,
figures/data/13-llc-localize-sgld.dat. Run from the volume root: python figures/scripts/13-llc-localize.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy import integrate, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

N_DATA, BATCH = 10_000, 500
SEED = 1303


def skeleton(t, gamma, astar, M, N, c=0.5):
    """t E[K] for K = c |a|^2 |b|^2 under exp(-t K - gamma |w - (a*, 0)|^2 / 2), a in R^N, b in R^M."""
    k = 2 * c * t

    def log_ang(rho):
        z = gamma * rho * astar
        base = -gamma * (rho - astar) ** 2 / 2
        if N == 1:
            return base + math.log1p(math.exp(-2 * z))
        if N == 2:
            return base + math.log(2 * math.pi * special.i0e(z))
        if z < 1e-8:
            return -gamma * (rho * rho + astar * astar) / 2 + math.log(4 * math.pi)
        return base + math.log(4 * math.pi) + math.log(-math.expm1(-2 * z)) - math.log(2 * z)

    def logw(rho):
        return (N - 1) * math.log(rho) + log_ang(rho) - (M / 2) * math.log(k * rho * rho + gamma)

    top = astar + 12 / math.sqrt(gamma) + 2
    ref = max(logw(r) for r in np.concatenate([np.geomspace(1e-12, 1, 400), np.linspace(1, top, 300)]))
    w = lambda r: math.exp(logw(r) - ref) if r > 0 else 0.0
    f = lambda r: w(r) * k * r * r / (k * r * r + gamma)
    s = math.sqrt(gamma / k)
    pts = sorted({p for p in (0.0, s / 100, s, 10 * s, max(astar - 6 / math.sqrt(gamma), 0.0), astar,
                              astar + 6 / math.sqrt(gamma), top) if p <= top})
    D = sum(integrate.quad(w, u, v, limit=500, epsabs=0, epsrel=1e-11)[0] for u, v in zip(pts[:-1], pts[1:]))
    Nm = sum(integrate.quad(f, u, v, limit=500, epsabs=0, epsrel=1e-11)[0] for u, v in zip(pts[:-1], pts[1:]))
    return (M / 2) * Nm / D


def limit_value(M, N):
    """The fixed-gamma limit of the proposition: min(M, N)/2."""
    return min(M, N) / 2


def sgld_rankone(M, N, gammas, eps, T, burn, R, C, gen):
    """Lau et al.'s SGLD estimator for y = b a^T x at w* = (e1, 0), noiseless labels (y = 0)."""
    nbeta = N_DATA / math.log(N_DATA)
    J = N_DATA // BATCH
    G = len(gammas)
    Cn = R * G * C
    Sxx = np.empty((Cn, J, N, N)); Fxx = np.empty((Cn, N, N))
    for r in range(R):
        x = gen.standard_normal((N_DATA, N))
        for g in range(G * C):
            i = r * G * C + g
            perm = gen.permutation(N_DATA).reshape(J, BATCH)
            xb = x[perm]
            Sxx[i] = np.einsum("jnp,jnq->jpq", xb, xb) / BATCH
            Fxx[i] = x.T @ x / N_DATA
    gam = np.tile(np.repeat(np.asarray(gammas, float), C), R)[:, None]
    a = np.zeros((Cn, N)); a[:, 0] = 1.0
    b = np.zeros((Cn, M))
    a0, b0 = a.copy(), b.copy()
    he, se = 0.5 * eps, math.sqrt(eps)
    acc = np.zeros(Cn); cnt = 0
    for t in range(T):
        S = Sxx[:, t % J]
        Sa = np.einsum("cpq,cq->cp", S, a)                 # S a
        aSa = np.einsum("cp,cp->c", a, Sa)
        bb = np.einsum("cm,cm->c", b, b)
        ga = bb[:, None] * Sa                               # grad_a of (1/2)|b|^2 a^T S a
        gb = aSa[:, None] * b
        if t >= burn:
            aFa = np.einsum("cp,cpq,cq->c", a, Fxx, a)
            acc += 0.5 * bb * aFa; cnt += 1
        a = a + he * (-nbeta * ga - gam * (a - a0)) + se * gen.standard_normal(a.shape)
        b = b + he * (-nbeta * gb - gam * (b - b0)) + se * gen.standard_normal(b.shape)
    est = nbeta * acc / cnt                                 # L_n(w*) = 0 with noiseless labels
    rows = []
    for gi, g in enumerate(gammas):
        per = [est[r * G * C + gi * C: r * G * C + (gi + 1) * C].mean() for r in range(R)]
        mu, lo, hi = sk.mean_ci(per)
        rows.append([g, mu, mu - lo, hi - mu])
    return rows


def main():
    t0 = time.perf_counter()
    n = N_DATA
    t = n / math.log(n)
    gammas = np.geomspace(0.1, 1000, 41)
    rows = []
    for g in gammas:
        rows.append([g, skeleton(t, g, 1.0, 1, 1, c=0.25), skeleton(t, g, 1.0, 3, 3), skeleton(t, g, 1.0, 3, 2)])
    sk.write_dat("figures/data/13-llc-localize-gamma.dat", ["gamma", "relu", "r33", "r32"], rows)
    for g in (0.3, 1, 3, 10, 30, 100, 300):
        print(f"t={t:.1f} gamma={g:>5}: relu(1,0) {skeleton(t, g, 1.0, 1, 1, c=0.25):.4f}   "
              f"rank-one 3x3 {skeleton(t, g, 1.0, 3, 3):.4f}   rank-one M=3,N=2 {skeleton(t, g, 1.0, 3, 2):.4f}")
    best = max(rows, key=lambda r: r[3])
    print(f"rank-one M=3,N=2: maximum of the skeleton over gamma is {best[3]:.4f} at gamma = {best[0]:.1f}")
    trows = []
    for tt in np.geomspace(10, 1e12, 45):
        trows.append([tt] + [skeleton(tt, g, 1.0, 3, 2) for g in (1.0, 10.0, 30.0)])
    sk.write_dat("figures/data/13-llc-localize-t.dat", ["t", "g1", "g10", "g30"], trows)
    for tt in (1e2, 1e3, 1e4, 1e6, 1e8, 1e10, 1e12):
        print(f"M=3,N=2, t={tt:.0e}: gamma=1 {skeleton(tt, 1.0, 1.0, 3, 2):.4f}  gamma=10 {skeleton(tt, 10.0, 1.0, 3, 2):.4f}"
              f"  gamma=30 {skeleton(tt, 30.0, 1.0, 3, 2):.4f}   (fixed-gamma limit {limit_value(3, 2)})")
    gen = sk.rng(SEED)
    sg = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    srows = sgld_rankone(3, 2, sg, 1e-4, 30_000, 15_000, 3, 4, gen)
    for (g, mu, em, ep), gg in zip(srows, sg):
        print(f"SGLD M=3,N=2 gamma={g:>5}: {mu:.3f} [{mu - em:.3f}, {mu + ep:.3f}]   skeleton {skeleton(t, gg, 1.0, 3, 2):.3f}")
    sk.write_dat("figures/data/13-llc-localize-sgld.dat", ["gamma", "est", "em", "ep"], srows)
    print(f"n = {n}, t = n/log n = {t:.1f}; seed {SEED}; total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
