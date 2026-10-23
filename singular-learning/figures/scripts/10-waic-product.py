"""Chapter 10 (WAIC): WAIC, leave-one-out cross-validation and AIC against the Bayes generalization
loss for the product model y = abx + N(0,1), x ~ N(0,1), uniform prior on [-1,1]^2, truth ab = 0;
and, for comparison, leave-one-out minus WAIC for the regular line y = ax (flat prior, exact).

The likelihood depends on (a, b) only through c = ab, whose prior density is rho(c) = (1/2) log(1/|c|);
the posterior of c is rho(c) exp(-S (c - chat)^2 / 2) with S = sum x_i^2, chat = sum x_i y_i / S.
For this density every posterior average needed here has a closed form in terms of
    h0(mu) = E log|mu + U|,  U ~ N(0,1)   (a Poisson mixture of digamma values; h0' is Dawson's function),
because int (1/2) log(1/|c|) exp(-S (c - m)^2 / 2) dc = (1/2) sqrt(2 pi / S) [ (1/2) log S - h0(sqrt(S) m) ]
(the truncation |c| <= 1 is neglected, which needs the posterior and its tilts to stay many standard
deviations inside |c| < 1: the script uses the closed form for n >= 1000, checks that condition for every
data set, and uses Gauss-Legendre quadrature in c for n = 100 and 300). Tilting the posterior by
p(x_i, y_i | c)^alpha changes (S, chat) to (S + alpha x_i^2, (S chat + alpha x_i y_i)/(S + alpha x_i^2)),
which gives E_w[p_i] (alpha = 1) and E_w[1/p_i] (alpha = -1) exactly; V_n uses the posterior cumulants
of c, which are derivatives of log[(1/2) log S - h0(sqrt(S) m)]. The script checks the closed forms
against direct quadrature in c, and the whole computation against the kit's exact posterior sampler.
Per data set it records the same quantities as 10-waic-mixture.py (see there).

Run from the volume root:  python figures/scripts/10-waic-product.py
Writes figures/data/10-waic-product.dat, 10-waic-product-scatter.dat and 10-waic-regline.dat.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import special

import sltkit as sk

NS = (100, 300, 1000, 3000, 10000, 30000, 100000)
R = 500
SEED0 = 2000                 # data seed for sample size n: SEED0 + n
REG_SEED0 = 3000             # regular line: REG_SEED0 + n
SCATTER_N = 10000
DIRECT_MAX = 300             # quadrature in c up to this n; the closed form above it
D = 2
GHX, GHW = np.polynomial.hermite_e.hermegauss(64)
GHW = GHW / math.sqrt(2 * math.pi)
GX, GY = [g.ravel() for g in np.meshgrid(GHX, GHX, indexing="ij")]
GW = np.outer(GHW, GHW).ravel()
_keep = GW > 1e-20                      # nodes beyond x^2 + y^2 ~ 90 carry weight < 1e-20: dropped
GX, GY, GW = GX[_keep], GY[_keep], GW[_keep]
TAYLOR_ORDER = 16


def h0_series(mu):
    """E log|mu + U| = (1/2)[log 2 + sum_j Poisson(j; mu^2/2) psi(j + 1/2)] (log of a noncentral chi^2_1)."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    lam = 0.5 * mu * mu
    jmax = int(lam.max() + 12 * math.sqrt(lam.max()) + 30)
    j = np.arange(jmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = -lam[:, None] + j[None, :] * np.log(lam)[:, None] - special.gammaln(j + 1)[None, :]
    logp[:, 0] = -lam                                   # j = 0 term, also when lam = 0
    if np.any(lam == 0):
        logp[lam == 0, 1:] = -np.inf
    return 0.5 * (math.log(2) + np.exp(logp) @ special.digamma(j + 0.5))


def h0_derivatives(mu, order):
    """h0^(k)(mu), k = 0..order: h0' = sqrt(2) Dawson(mu/sqrt2), h0'' = 1 - mu h0',
    h0^(k+2) = -mu h0^(k+1) - k h0^(k) for k >= 1."""
    d = [float(h0_series(mu)[0]), math.sqrt(2) * float(special.dawsn(mu / math.sqrt(2)))]
    d.append(1 - mu * d[1])
    for k in range(1, order - 1):
        d.append(-mu * d[k + 1] - k * d[k])
    return np.array(d[:order + 1])


def h0_near(mu0, derivs, mu):
    """h0(mu) by its Taylor series around mu0 (used when every |mu - mu0| < 1)."""
    delta = mu - mu0
    out = np.full_like(delta, derivs[0])
    term = np.ones_like(delta)
    for k in range(1, len(derivs)):
        term = term * delta / k
        out += derivs[k] * term
    return out


def h0_at(mu, mu0, derivs):
    delta = np.abs(mu - mu0)
    out = np.empty_like(mu)
    near = delta < 0.75
    if np.any(near):
        out[near] = h0_near(mu0, derivs, mu[near])
    if np.any(~near):
        out[~near] = h0_series(mu[~near])
    return out


class Posterior:
    """The posterior of c for data summaries S = sum x^2, chat = sum xy / S."""

    def __init__(self, S, chat):
        self.S, self.m = S, chat
        self.mu = math.sqrt(S) * chat
        self.der = h0_derivatives(self.mu, TAYLOR_ORDER)
        self.u0 = 0.5 * math.log(S) - self.der[0]

    def log_tilt(self, s, t, alpha):
        """log E[exp(alpha (c s - c^2 t / 2))] for arrays s, t, and a mask of the points where the
        tilted Gaussian factor is at least 8 standard deviations inside |c| < 1 (elsewhere the
        neglected truncation would matter and the value is not used)."""
        Sp = self.S + alpha * t
        mp = (self.S * self.m + alpha * s) / Sp
        ok = (1.0 - np.abs(mp)) * np.sqrt(Sp) > 8.0
        up = 0.5 * np.log(Sp) - h0_at(np.where(ok, np.sqrt(Sp) * mp, self.mu), self.mu, self.der)
        val = 0.5 * (Sp * mp * mp - self.S * self.m * self.m) - 0.5 * np.log(Sp / self.S) + np.log(up / self.u0)
        return np.where(ok, val, 0.0), ok

    def cumulants(self):
        """Mean, variance, third and fourth cumulants of c."""
        S, d = self.S, self.der
        u = self.u0
        u1, u2, u3, u4 = -math.sqrt(S) * d[1], -S * d[2], -S ** 1.5 * d[3], -S ** 2 * d[4]
        l1 = u1 / u
        l2 = u2 / u - l1 ** 2
        l3 = u3 / u - 3 * u2 * u1 / u ** 2 + 2 * l1 ** 3
        l4 = u4 / u - 4 * u3 * u1 / u ** 2 - 3 * (u2 / u) ** 2 + 12 * u2 * u1 ** 2 / u ** 3 - 6 * l1 ** 4
        return self.m + l1 / S, 1 / S + l2 / S ** 2, l3 / S ** 3, l4 / S ** 4


def analyse(x, y):
    n = len(x)
    S = float(x @ x)
    Sxy = float(x @ y)
    post = Posterior(S, Sxy / S)
    s, t = x * y, x * x
    lp1, ok1 = post.log_tilt(s, t, 1.0)           # log E_w[p_i] - log q_i
    lm1, ok2 = post.log_tilt(s, t, -1.0)          # log E_w[1/p_i] + log q_i
    if not (ok1.all() and ok2.all()):
        raise ValueError("a tilted posterior reaches |c| = 1: the closed form does not apply")
    mc, vc, k3, k4 = post.cumulants()
    cov_c_c2 = k3 + 2 * vc * mc
    var_c2 = k4 + 4 * k3 * mc + 2 * vc * vc + 4 * vc * mc * mc
    vi = s * s * vc - s * t * cov_c_c2 + 0.25 * t * t * var_c2
    lg, okg = post.log_tilt(GX * GY, GX * GX, 1.0)
    if GW[~okg].sum() > 1e-15:
        raise ValueError("quadrature nodes with non-negligible weight reach |c| = 1")
    Bg = -n * float(GW @ lg)
    c_ml = min(1.0, max(-1.0, Sxy / S))
    return dict(Bg=Bg, Te=-np.sum(lp1), V=float(np.sum(vi)), Ce=float(np.sum(lm1)),
                dl=n * float(np.sum(lm1 + lp1 - vi)), Ak=0.5 * S * c_ml ** 2 - Sxy * c_ml + D)


def direct(x, y):
    """The same quantities by Gauss-Legendre quadrature in c (graded toward the log singularity at 0)."""
    S = float(x @ x)
    chat = float(x @ y) / S
    sig = 1 / math.sqrt(S)
    lo, hi = max(-1.0, chat - 14 * sig), min(1.0, chat + 14 * sig)
    glx, glw = np.polynomial.legendre.leggauss(120)
    uu = 0.5 * (glx + 1)
    cs, ws = [], []
    for (u0, u1) in ([(lo, 0.0), (0.0, hi)] if lo < 0 < hi else [(lo, hi)]):
        if u0 == 0.0 or u1 == 0.0:
            h = u1 - u0
            sgn = 1.0 if u1 > 0 else -1.0
            cs.append(sgn * h * uu ** 4)
            ws.append(0.5 * glw * 4 * h * uu ** 3)
        else:
            cs.append(0.5 * (u0 + u1) + 0.5 * (u1 - u0) * glx)
            ws.append(0.5 * (u1 - u0) * glw)
    c, w = np.concatenate(cs), np.concatenate(ws)
    lw = np.log(w) + np.log(0.5 * np.log(1 / np.abs(c))) - 0.5 * S * (c - chat) ** 2
    lw -= special.logsumexp(lw)
    p = np.exp(lw)
    ll = c[:, None] * (x * y)[None, :] - 0.5 * c[:, None] ** 2 * (x * x)[None, :]
    lp1 = special.logsumexp(ll, axis=0, b=p[:, None])
    lm1 = special.logsumexp(-ll, axis=0, b=p[:, None])
    vi = p @ (ll * ll) - (p @ ll) ** 2
    lg = c[:, None] * (GX * GY)[None, :] - 0.5 * c[:, None] ** 2 * (GX * GX)[None, :]
    n = len(x)
    Bg = -n * float(GW @ special.logsumexp(lg, axis=0, b=p[:, None]))
    c_ml = min(1.0, max(-1.0, chat))
    return dict(Bg=Bg, Te=-np.sum(lp1), V=vi.sum(), Ce=np.sum(lm1), dl=n * np.sum(lm1 + lp1 - vi),
                Ak=0.5 * S * c_ml ** 2 - float(x @ y) * c_ml + D)


def regular_line(x, y):
    """Regular line y = ax, flat prior: exact WAIC and LOO with the leverages h_i = x_i^2 / S."""
    S = x @ x
    r = y - (x @ y) / S * x
    h = x * x / S
    V = np.sum(r * r * h + 0.5 * h * h)
    d = np.sum(-0.5 * np.log1p(-h * h) - 0.5 * h * h + r * r * h ** 3 / (1 - h * h))   # n (LOO - WAIC)
    return V, d


def ci(v):
    m, lo, hi = sk.mean_ci(v)
    return m, m - lo, hi - m


def main():
    t_start = time.time()
    # check 1: closed forms against quadrature in c
    worst = 0.0
    for n in (1000, 10000):
        for rep in range(5):
            g = sk.rng(17 + n + rep)
            x, y = g.standard_normal(n), g.standard_normal(n)
            a, b = analyse(x, y), direct(x, y)
            worst = max(worst, max(abs(a[k] - b[k]) for k in ("Bg", "Te", "V", "Ce", "dl", "Ak")))
    print(f"closed form vs quadrature in c (n = 1e3, 1e4; 5 data sets each): max |difference| = {worst:.1e}")
    # check 2: the kit's exact posterior sampler (20000 draws) and sk.waic, n = 1000
    n = 1000
    x, y = sk.PRODUCT.sample([0.0, 0.0], n, seed=11)
    x = x[:, 0]
    sk.PRODUCT.fit(x, y)
    draws = sk.sample_product_posterior(sk.PRODUCT, 20000, seed=12)
    c = draws[:, 0] * draws[:, 1]
    ll = -0.5 * math.log(2 * math.pi) - 0.5 * (y[None, :] - c[:, None] * x[None, :]) ** 2
    W, T, V = sk.waic(ll)
    Sn = float(np.mean(0.5 * math.log(2 * math.pi) + 0.5 * y * y))
    r = direct(x, y)
    print(f"kit's exact sampler (20000 draws) vs exact averages (quadrature in c), n = 1000: n(WAIC - S_n) {n * (W - Sn):.4f} vs "
          f"{r['Te'] + r['V']:.4f}; n(T_n - S_n) {n * (T - Sn):.4f} vs {r['Te']:.4f}; V_n {V:.4f} vs {r['V']:.4f}")

    rows, regrows, scatter = [], [], []
    for n in NS:
        t0 = time.time()
        gen = sk.rng(SEED0 + n)
        out = {k: [] for k in ("Bg", "Te", "V", "Ce", "dl", "Ak")}
        for _ in range(R):
            x, y = gen.standard_normal(n), gen.standard_normal(n)
            res = direct(x, y) if n <= DIRECT_MAX else analyse(x, y)
            for k in out:
                out[k].append(res[k])
        o = {k: np.array(v) for k, v in out.items()}
        We, Ab = o["Te"] + o["V"], o["Te"] + D
        corr = float(np.corrcoef(o["Bg"], o["Ce"])[0, 1])
        z, se = math.atanh(corr), 1.0 / math.sqrt(R - 3)
        L = math.log(n)
        row = [n]
        for v in (o["Bg"], o["V"], We - o["Bg"], Ab - o["Bg"], o["Ak"] - o["Bg"], o["dl"], o["Bg"] + o["Ce"]):
            row += list(ci(v))
        row += [corr, math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se), float(np.median(np.abs(o["dl"]))),
                float(np.mean(np.abs(o["dl"]))), 0.5 - 1 / L, 1 - 1 / L, 1 + 1 / L]
        rows.append(row)
        print(f"n={n:6d} R={R} ({time.time() - t0:5.1f}s): nE[G-S]={row[1]:.3f}+-{row[2]:.3f} (1/2-1/log n={0.5 - 1 / L:.3f})  "
              f"E[V]={row[4]:.3f}+-{row[5]:.3f} (1-1/log n={1 - 1 / L:.3f})  nE[WAIC-G]={row[7]:+.3f}+-{row[8]:.3f}  "
              f"nE[T+d/n-G]={row[10]:.3f}+-{row[11]:.3f}  nE[AIC-G]={row[13]:.3f}+-{row[14]:.3f}")
        print(f"          n^2 E[LOO-WAIC]={row[16]:.4f}+-{row[17]:.4f} (median |.| {row[25]:.4f}, mean |.| {row[26]:.4f})  "
              f"nE[Bg+Cv]={row[19]:.3f}+-{row[20]:.3f}  corr(Bg,Cv)={corr:.4f} [{row[23]:.4f},{row[24]:.4f}]  "
              f"sd(nBg)={np.std(o['Bg'], ddof=1):.3f} sd(nCv)={np.std(o['Ce'], ddof=1):.3f}")
        if n == SCATTER_N:
            scatter = np.column_stack([o["Bg"], o["Ce"], We])
        # the regular line at the same n
        g2 = sk.rng(REG_SEED0 + n)
        rv = np.array([regular_line(g2.standard_normal(n), g2.standard_normal(n)) for _ in range(R)])
        d3 = rv[:, 1] * n * n                         # n^3 (LOO - WAIC)
        regrows.append([n, *ci(rv[:, 0]), *ci(d3), float(np.mean(np.abs(d3)))])
        print(f"          regular line: E[V]={regrows[-1][1]:.4f}  n^3 E[LOO-WAIC]={regrows[-1][4]:.3f}+-{regrows[-1][5]:.3f}")
    header = ["n", "bg", "bgm", "bgp", "v", "vm", "vp", "w", "wm", "wp", "ab", "abm", "abp", "ak", "akm", "akp",
              "dl", "dlm", "dlp", "sum", "summ", "sump", "corr", "corrlo", "corrhi", "dlmed", "dlabs",
              "bgpred", "vpred", "abpred"]
    sk.write_dat("figures/data/10-waic-product.dat", header, rows)
    sk.write_dat("figures/data/10-waic-product-scatter.dat", ["bg", "cv", "w"], scatter)
    sk.write_dat("figures/data/10-waic-regline.dat", ["n", "v", "vm", "vp", "d3", "d3m", "d3p", "d3abs"], regrows)
    print(f"total {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
