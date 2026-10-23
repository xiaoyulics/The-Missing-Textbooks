"""Chapter 10 (WAIC): WAIC, leave-one-out cross-validation and AIC against the Bayes generalization
loss for the mixture (1-a) N(0,1) + a N(b,1), a in [0,1], b in [-1,1], uniform prior, truth N(0,1).

For each n and each of R data sets x_1..x_n ~ N(0,1) the posterior is computed by midpoint
quadrature on a grid in (tau, eta) with a = tau^3, b = eta^3 (so that the grid is dense near the
cross a = 0 or b = 0, where the posterior lives); the convergence is checked by doubling the grid.
Every per-observation quantity is a smooth function of the observation x alone,
    g_w(x) = log p(x|w) - log phi(x) = log(1 + a (exp(b x - b^2/2) - 1)),
so the posterior averages E_w[e^{g}], E_w[e^{-g}], E_w[g], E_w[g^2] are computed at 96 Chebyshev
nodes on [-7, 7] and interpolated to the data (the data never leave [-7, 7]; the script checks it);
sums over the data of g_w use the Chebyshev moments sum_i T_k(x_i / 7). Agreement with direct
evaluation is checked once. Per data set the script records (realizable truth, so L(w0) = S):
    Bg = n (G_n - S)        Bayes generalization error, G_n - S = -int phi(x) log E_w[e^{g_w(x)}] dx
    Te = n (T_n - S_n)      Bayes training error
    V  = V_n                functional variance
    Ce = n (LOO - S_n)      leave-one-out error, LOO = (1/n) sum_i log E_w[1/p(x_i|w)]
    We = Te + V             n (WAIC - S_n)
    dl = n^2 (LOO - WAIC)
    Ab = Te + 2             n (T_n + d/n - S_n): the penalty d/n of AIC on the Bayes training loss
    Ak = n K_n(w_ml) + 2    n (AIC - S_n), Akaike's AIC = L_n(w_ml) + d/n with the maximum-likelihood fit
Because E[S_n] = S, the bias of an estimate Est of G_n is E[n (Est - S_n) - Bg]; using the errors
removes the O(n^{-1/2}) fluctuation of S_n - S, which is common to all estimates.
To locate the growth of n^2 E[LOO - WAIC], the script also prints, at n = 1e4, the per-observation
contribution as a function of x, the share of the 1% most extreme observations, and, at every n, the
per-observation function averaged over x ~ N(0,1) instead of over the data (with the part from |x| > 4).

Run from the volume root:  python figures/scripts/10-waic-mixture.py
Writes figures/data/10-waic-mixture.dat and figures/data/10-waic-mixture-scatter.dat.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import optimize, special

import sltkit as sk

NS = (100, 300, 1000, 3000, 10000)
R = 500                     # data sets per n
GRID = (96, 192)            # (tau, eta) midpoints
GRID2 = (192, 384)          # doubled grid for the convergence check
NCHECK = 20                 # data sets per n recomputed on the doubled grid
SCATTER_N = 10000
SEED0 = 1000                # data seed for sample size n: SEED0 + n
X = 7.0                     # Chebyshev interval [-X, X]
K = 96                      # Chebyshev nodes
D = 2                       # number of parameters

XK = X * np.cos(np.pi * (np.arange(K) + 0.5) / K)


def cheb_T(z):
    """Chebyshev polynomials T_0..T_{K-1} at z (shape (m,)), returned with shape (m, K)."""
    T = np.empty((K, len(z)))
    T[0] = 1.0
    T[1] = z
    for k in range(2, K):
        np.multiply(z, T[k - 1], out=T[k])
        T[k] *= 2.0
        T[k] -= T[k - 2]
    return T.T


DM = (2.0 / K) * cheb_T(XK / X)     # values at the nodes -> Chebyshev coefficients
DM[:, 0] *= 0.5
GLX, GLW = np.polynomial.legendre.leggauss(200)
GX = X * GLX                         # nodes and weights for int phi(x) f(x) dx on [-X, X]
GWT = X * GLW * np.exp(-0.5 * GX * GX) / math.sqrt(2 * math.pi)
TG = cheb_T(GX / X)
TAIL = np.abs(GX) > 4.0


def gfun(a, b, x):
    """log p(x|a,b) - log phi(x) for parameter arrays a, b (m,) and points x (k,): shape (m, k)."""
    return np.log1p(a[:, None] * np.expm1(b[:, None] * x[None, :] - 0.5 * (b * b)[:, None]))


class MixtureGrid:
    """Midpoint grid in (tau, eta), a = tau^3, b = eta^3, with the prior density 1/2 on [0,1]x[-1,1]."""

    def __init__(self, na, nb, store=True, chunk=4096):
        tau = (np.arange(na) + 0.5) / na
        eta = -1.0 + (np.arange(nb) + 0.5) * 2.0 / nb
        A, B = np.meshgrid(tau ** 3, eta ** 3, indexing="ij")
        self.a, self.b = A.ravel(), B.ravel()
        self.logw = np.log(np.outer(3 * tau ** 2 / na, 3 * eta ** 2 * 2.0 / nb).ravel() * 0.5)
        self.G = np.empty((len(self.a), K))
        for s in range(0, len(self.a), chunk):
            self.G[s:s + chunk] = gfun(self.a[s:s + chunk], self.b[s:s + chunk], XK)
        self.C = self.G @ DM                    # Chebyshev coefficients of g_w, one row per grid point
        self.store = store
        if store:
            self.F = np.concatenate([np.exp(self.G), np.exp(-self.G), self.G, self.G * self.G], axis=1)

    def posterior_averages(self, p, chunk=8192):
        if self.store:
            return (p @ self.F).reshape(4, K)
        H = np.zeros((4, K))
        for s in range(0, len(p), chunk):
            Gc, pc = self.G[s:s + chunk], p[s:s + chunk]
            H += np.stack([pc @ np.exp(Gc), pc @ np.exp(-Gc), pc @ Gc, pc @ (Gc * Gc)])
        return H

    def analyse(self, x):
        n = len(x)
        if np.max(np.abs(x)) >= X:
            raise ValueError("an observation left [-X, X]")
        Tdt = cheb_T(x / X).T                   # (K, n), contiguous
        M = Tdt.sum(axis=1)
        nK = -(self.C @ M)                      # n K_n(w) on the grid
        lw = self.logw - nK
        lw -= special.logsumexp(lw)
        p = np.exp(lw)
        H = self.posterior_averages(p)
        h1n, hm1n, m1n, m2n = H
        coef = np.stack([h1n, hm1n, m2n - m1n * m1n]) @ DM
        h1, hm1, v = coef @ Tdt                 # at the data: E_w[e^g], E_w[e^-g], Var_w[g]
        Te = -np.sum(np.log(h1))                # n (T_n - S_n)
        Ce = np.sum(np.log(hm1))                # n (LOO - S_n)
        V = np.sum(v)
        dli = n * n * (np.log(hm1) + np.log(h1) - v)          # n^2 (LOO - WAIC), observation by observation
        dl = float(np.sum(dli)) / n
        Bg = -n * float(GWT @ np.log(TG @ coef[0]))
        imin = int(np.argmin(nK))
        top = np.argsort(-np.abs(x))[:max(1, n // 100)]        # the 1% most extreme observations
        share = float(np.sum(dli[top]) / np.sum(dli)) if np.sum(dli) != 0 else float("nan")
        # the same per-observation function n^2 d(x), averaged over x ~ N(0,1) instead of over the data
        # (the observation's own effect on the posterior ignored), and the part of it from |x| > 4
        hg, hmg, vg = coef @ TG.T
        prof = n * n * (np.log(hmg) + np.log(hg) - vg)
        integ, tail = float(GWT @ prof), float(GWT[TAIL] @ prof[TAIL])
        return dict(Bg=Bg, Te=Te, V=V, Ce=Ce, dl=dl, M=M, nKgrid=nK[imin], w0=(self.a[imin], self.b[imin]),
                    mass_arm_a=float(p[self.a < 0.1].sum()), share=share, coef=coef, integ=integ, tail=tail)


def energy_cheb(w, M):
    """n K_n(a, b) and its gradient from the Chebyshev moments of the data."""
    a, b = float(w[0]), float(w[1])
    e = np.exp(b * XK - 0.5 * b * b)
    den = 1.0 + a * (e - 1.0)
    g = np.log(den)
    ga = (e - 1.0) / den
    gb = a * (XK - b) * e / den
    DMM = DM @ M
    return -float(g @ DMM), np.array([-float(ga @ DMM), -float(gb @ DMM)])


def mle_energy(M, w0):
    """min over [0,1]x[-1,1] of n K_n, started at the best grid point (n K_n = 0 on a = 0)."""
    res = optimize.minimize(energy_cheb, np.array(w0), args=(M,), method="L-BFGS-B", jac=True,
                            bounds=[(0.0, 1.0), (-1.0, 1.0)], options=dict(ftol=1e-15, gtol=1e-10))
    return min(res.fun, energy_cheb(w0, M)[0], 0.0), res.x


def direct_check(x, grid):
    """The same quantities by direct evaluation of g_w(x_i) on the grid (no interpolation)."""
    g = gfun(grid.a, grid.b, x)
    lw = grid.logw + g.sum(axis=1)
    lw -= special.logsumexp(lw)
    p = np.exp(lw)
    h1, hm1, m1, m2 = p @ np.exp(g), p @ np.exp(-g), p @ g, p @ (g * g)
    v = m2 - m1 * m1
    n = len(x)
    Bg = -n * float(GWT @ np.log(p @ np.exp(gfun(grid.a, grid.b, GX))))
    return dict(Bg=Bg, Te=-np.sum(np.log(h1)), V=v.sum(), Ce=np.sum(np.log(hm1)),
                dl=n * np.sum(np.log(hm1) + np.log(h1) - v))


def ci(v):
    m, lo, hi = sk.mean_ci(v)
    return m, m - lo, hi - m


def main():
    t_start = time.time()
    grid = MixtureGrid(*GRID)
    grid2 = MixtureGrid(*GRID2, store=False)
    print(f"grid {GRID}: {len(grid.a)} points; check grid {GRID2}: {len(grid2.a)} points")

    # one-time check: Chebyshev interpolation against direct evaluation (n = 300, seed 7)
    xc = sk.rng(7).standard_normal(300)
    r1, r2 = grid.analyse(xc), direct_check(xc, grid)
    print("interpolation check (n=300): max |difference| over Bg, Te, V, Ce, dl = %.2e"
          % max(abs(r1[k] - r2[k]) for k in ("Bg", "Te", "V", "Ce", "dl")))

    rows, scatter = [], []
    for n in NS:
        t0 = time.time()
        gen = sk.rng(SEED0 + n)
        out = {k: [] for k in ("Bg", "Te", "V", "Ce", "dl", "Ak", "arm", "share", "integ", "tail")}
        profile = []
        dmax = {k: 0.0 for k in ("Bg", "Te", "V", "Ce", "dl")}
        xmax = 0.0
        for r in range(R):
            x = gen.standard_normal(n)
            xmax = max(xmax, float(np.max(np.abs(x))))
            res = grid.analyse(x)
            for k in ("Bg", "Te", "V", "Ce", "dl"):
                out[k].append(res[k])
            emin, _ = mle_energy(res["M"], res["w0"])
            out["Ak"].append(emin + D)
            out["arm"].append(res["mass_arm_a"])
            out["share"].append(res["share"])
            out["integ"].append(res["integ"])
            out["tail"].append(res["tail"])
            if n == SCATTER_N and r < 3:                       # n^2 d(x) at x = 0..5 for three data sets
                xs = np.arange(6.0)
                c = res["coef"]
                h1x, hm1x, vx = c @ cheb_T(xs / X).T
                profile.append(n * n * (np.log(hm1x) + np.log(h1x) - vx))
            if r < NCHECK:
                res2 = grid2.analyse(x)
                for k in dmax:
                    dmax[k] = max(dmax[k], abs(res2[k] - res[k]))
        o = {k: np.array(v) for k, v in out.items()}
        We = o["Te"] + o["V"]
        Ab = o["Te"] + D
        bias_w, bias_a, bias_k = We - o["Bg"], Ab - o["Bg"], o["Ak"] - o["Bg"]
        corr = float(np.corrcoef(o["Bg"], o["Ce"])[0, 1])
        z, se = math.atanh(corr), 1.0 / math.sqrt(R - 3)
        row = [n]
        for v in (o["Bg"], o["V"], bias_w, bias_a, bias_k, o["dl"], o["Bg"] + o["Ce"]):
            row += list(ci(v))
        row += [corr, math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se), float(np.median(np.abs(o["dl"]))),
                float(np.mean(np.abs(o["dl"])))]
        rows.append(row)
        print(f"n={n:6d} R={R} ({time.time() - t0:5.1f}s, max|x|={xmax:.2f}): "
              f"nE[G-S]={row[1]:.3f}+-{row[2]:.3f}  E[V]={row[4]:.3f}+-{row[5]:.3f}  "
              f"nE[WAIC-G]={row[7]:+.3f}+-{row[8]:.3f}  nE[T+d/n-G]={row[10]:.3f}+-{row[11]:.3f}  "
              f"nE[AIC-G]={row[13]:.3f}+-{row[14]:.3f}")
        print(f"          n^2 E[LOO-WAIC]={row[16]:.4f}+-{row[17]:.4f} (median |.| {row[25]:.4f})  "
              f"nE[Bg+Cv]={row[19]:.3f}+-{row[20]:.3f}  corr(Bg,Cv)={corr:.4f} [{row[23]:.4f},{row[24]:.4f}]  "
              f"E[n(L_n(w0)-L_n(w_ml))]={-np.mean(o['Ak'] - D):.3f}  mass(a<0.1)={np.mean(o['arm']):.3f}")
        print(f"          grid doubling ({NCHECK} sets): max|diff| nBg {dmax['Bg']:.1e} nTe {dmax['Te']:.1e} "
              f"V {dmax['V']:.1e} nCe {dmax['Ce']:.1e} n^2(LOO-WAIC) {dmax['dl']:.1e}")
        ig = ci(o["integ"])
        print(f"          n^2 (LOO-WAIC) per observation averaged over x ~ N(0,1) instead of the data: "
              f"mean {ig[0]:.3f}+-{ig[1]:.3f}; share from |x| > 4: {np.mean(o['tail']) / ig[0]:.2f}")
        if n == SCATTER_N:
            scatter = np.column_stack([o["Bg"], o["Ce"], We])
            print("          per-observation n^2 (LOO - WAIC) at x = 0, 1, ..., 5 (three data sets):")
            for prof in profile:
                print("            " + "  ".join(f"{v:.3g}" for v in prof)
                      + "   ratios x=3->4->5: %.1f %.1f" % (prof[4] / prof[3], prof[5] / prof[4]))
            print(f"          share of n^2(LOO-WAIC) from the 1% of observations with largest |x|: "
                  f"median {np.median(o['share']):.2f}, mean {np.mean(o['share']):.2f}")
    header = ["n", "bg", "bgm", "bgp", "v", "vm", "vp", "w", "wm", "wp", "ab", "abm", "abp", "ak", "akm", "akp",
              "dl", "dlm", "dlp", "sum", "summ", "sump", "corr", "corrlo", "corrhi", "dlmed", "dlabs"]
    sk.write_dat("figures/data/10-waic-mixture.dat", header, rows)
    sk.write_dat("figures/data/10-waic-mixture-scatter.dat", ["bg", "cv", "w"], scatter)
    print(f"total {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
