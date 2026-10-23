"""Chapter 13 (llc): SGLD estimates of the local learning coefficient on models whose value is known.

The estimator is Lau et al.'s (AISTATS 2025, Definition 2 and Algorithm 1): from w*, run SGLD on the
localized tempered posterior prop. to exp(-n beta L_n(w) - gamma |w - w*|^2 / 2), beta = 1/log n,
    w <- w + (eps/2) (-n beta grad L_B(w) - gamma (w - w*)) + N(0, eps I),
with minibatches B of 500 examples (one shuffle per chain, then the minibatches in turn), discard the
first half of the chain, and report n beta (mean of L_n(w_t) - L_n(w*)) ("full-batch read-out", their
eq. 13) and the same with the minibatch losses L_B(w_t) recycled ("recycled", their Algorithm 1).
As in their deep-linear-network experiments the labels are noiseless (y = f(x, w_true)), n = 10^4,
x ~ N(0, I), gamma = 1 unless stated. Every model here is linear in a feature vector phi(x), so all
losses come from (minibatch) sufficient statistics and many chains run at once in numpy.

Cases (lambda and m from Chapters 2, 7 and Aoyagi's formula; r = rank of the Hessian of K at w*):
  regular plane y = a x1 + b x2 at 0; ReLU neuron y = a relu(bx) at (0,0) and (1,0); rank-one regression
  y = b a^T x (M = N = 3) at 0 and at (e1, 0); deep linear networks at 0 with zero truth; one deep linear
  network with a rank-one truth at the "special" point of Hitchcock and Hoogland (2025).
For each case and step size: mean estimate over R data sets x C chains with a Student-t 95% interval over
the R data-set means, the recycled estimate, the MALA acceptance diagnostic of Lau et al. (App. H; full-batch
gradients, every 25th post-burn-in step), and the fraction of chains that diverged. The step size selected
by the diagnostic is the largest one whose mean diagnostic is at least 0.9.

For the regular plane the step-size bias of the Langevin discretization is known exactly: on a Gaussian
target of precision kappa the chain's stationary variance is 1/(kappa (1 - eps kappa/4)), so the estimate is
(n beta/(n beta + gamma)) / (1 - eps (n beta + gamma)/4); the script prints it next to the measurement.

Outputs: figures/data/13-llc-sgld-sweep.dat, figures/data/13-llc-sgld-table.dat,
figures/data/13-llc-sgld-ula.dat (the exact discretization bias for the regular plane).
Run from the volume root: python figures/scripts/13-llc-sgld.py   (about 1.5 minutes)
"""
import itertools
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np
from scipy import integrate, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

N_DATA = 10_000
BATCH = 500
EPS = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
R, C = 4, 3                       # data sets, chains per data set (per step size)
T, BURN = 12_000, 6_000
MALA_EVERY = 25
SEED = 1302


# ------------------------------------------------------------------------------------------------
# Models: y = G(w) phi(x) + noise, G(w) of shape (chains, M, P)
# ------------------------------------------------------------------------------------------------
class DLN:
    """Deep linear network y = W_L ... W_1 x with widths (H_0, ..., H_L); rank-one regression is (N, 1, M)."""

    def __init__(self, widths):
        self.H = list(widths)
        self.shapes = [(self.H[i + 1], self.H[i]) for i in range(len(self.H) - 1)]
        self.offs = np.cumsum([0] + [a * b for a, b in self.shapes])
        self.d, self.P, self.M, self.inp = int(self.offs[-1]), self.H[0], self.H[-1], self.H[0]

    def feats(self, x):
        return x

    def mats(self, w):
        return [w[:, self.offs[k]:self.offs[k + 1]].reshape(len(w), *s) for k, s in enumerate(self.shapes)]

    def G(self, w):
        Ws = self.mats(w)
        G = Ws[0]
        for Wk in Ws[1:]:
            G = Wk @ G
        return G

    def vjp(self, w, dG):
        Ws, L, Cn = self.mats(w), len(self.shapes), len(w)
        pre = [np.broadcast_to(np.eye(self.H[0]), (Cn, self.H[0], self.H[0]))]
        for k in range(L):
            pre.append(Ws[k] @ pre[k])
        suf = [None] * (L + 1)
        suf[L] = np.broadcast_to(np.eye(self.H[-1]), (Cn, self.H[-1], self.H[-1]))
        for k in range(L - 1, -1, -1):
            suf[k] = suf[k + 1] @ Ws[k]
        out = np.empty_like(w)
        for k in range(L):
            g = np.swapaxes(suf[k + 1], 1, 2) @ dG @ np.swapaxes(pre[k], 1, 2)
            out[:, self.offs[k]:self.offs[k + 1]] = g.reshape(Cn, -1)
        return out


class ReluNeuron:
    """y = a relu(b x): on b >= 0 the mean is (ab) x 1{x > 0}, on b < 0 it is (ab) x 1{x < 0}."""
    d, P, M, inp = 2, 2, 1, 1

    def feats(self, x):
        x = x[:, 0]
        return np.stack([np.where(x > 0, x, 0.0), np.where(x < 0, x, 0.0)], axis=1)

    def G(self, w):
        a, b = w[:, 0], w[:, 1]
        pos = b >= 0
        return np.stack([np.where(pos, a * b, 0.0), np.where(pos, 0.0, a * b)], axis=1)[:, None, :]

    def vjp(self, w, dG):
        a, b = w[:, 0], w[:, 1]
        g = np.where(b >= 0, dG[:, 0, 0], dG[:, 0, 1])
        return np.stack([g * b, g * a], axis=1)


def loss_grad(model, w, Spp, Spy, Syy, grad=True):
    """L = (1/2)[tr(G Spp G^T) - 2 tr(G Spy) + Syy] from sufficient statistics (per chain)."""
    G = model.G(w)
    GS = G @ Spp
    L = 0.5 * (np.einsum("cmp,cmp->c", GS, G) - 2.0 * np.einsum("cmp,cpm->c", G, Spy) + Syy)
    if not grad:
        return L, None
    return L, model.vjp(w, GS - np.swapaxes(Spy, 1, 2))


# ------------------------------------------------------------------------------------------------
# Population quantities: the Hessian rank at w*, and the population skeleton of the estimator
# ------------------------------------------------------------------------------------------------
def hessian_rank(model, wstar, Gstar, Sigma):
    """Rank of the Hessian of K(w) = (1/2) tr((G - G*) Sigma (G - G*)^T) at w*, by central differences."""
    d, h = model.d, 1e-5
    S = Sigma[None]

    def grad(w):
        return loss_grad(model, w[None], S, (Sigma @ Gstar.T)[None], np.array([np.trace(Gstar @ Sigma @ Gstar.T)]))[1][0]
    H = np.empty((d, d))
    for i in range(d):
        e = np.zeros(d); e[i] = h
        H[:, i] = (grad(wstar + e) - grad(wstar - e)) / (2 * h)
    H = (H + H.T) / 2
    ev = np.linalg.eigvalsh(H)
    return int(np.sum(ev > 1e-6 * max(1.0, np.abs(ev).max()))), H


def skeleton_rankone(t, gamma, astar, M, N, c=0.5):
    """t E[K] for K = c |a|^2 |b|^2 (a in R^N, b in R^M) under exp(-tK - gamma|w - (a*,0)|^2/2):
    b integrated out exactly, a reduced to its radius (Proposition in the chapter)."""
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
    ref = max(logw(r) for r in np.concatenate([np.geomspace(1e-9, 1, 300), np.linspace(1, top, 300)]))
    w = lambda r: math.exp(logw(r) - ref) if r > 0 else 0.0
    f = lambda r: w(r) * k * r * r / (k * r * r + gamma)
    s = math.sqrt(gamma / k)
    pts = sorted({p for p in (0.0, s / 100, s, 10 * s, max(astar - 6 / math.sqrt(gamma), 0.0), astar,
                              astar + 6 / math.sqrt(gamma), top) if p <= top})
    D = sum(integrate.quad(w, u, v, limit=500, epsabs=0, epsrel=1e-11)[0] for u, v in zip(pts[:-1], pts[1:]))
    Nm = sum(integrate.quad(f, u, v, limit=500, epsabs=0, epsrel=1e-11)[0] for u, v in zip(pts[:-1], pts[1:]))
    return (M / 2) * Nm / D


# ------------------------------------------------------------------------------------------------
# Aoyagi's formula for deep linear networks (Aoyagi 2024; Aoyagi 2025, Definition 4)
# ------------------------------------------------------------------------------------------------
def aoyagi(H, r):
    D = [h - r for h in H]
    if min(D) == 0:
        return Fraction(-r * r + r * (H[0] + H[-1]), 2), 1
    idx = range(len(D))
    for k in range(2, len(D) + 1):
        for S in itertools.combinations(idx, k):
            ell, ins = k - 1, [D[i] for i in S]
            outs = [D[i] for i in idx if i not in S]
            tot = sum(ins)
            if outs and not max(ins) < min(outs):
                continue
            if not all(tot >= ell * x for x in ins) or (outs and not all(tot < ell * x for x in outs)):
                continue
            q = -(-tot // ell)
            a = tot - (q - 1) * ell
            lam = (Fraction(-r * r + r * (H[0] + H[-1]), 2) + Fraction(a * (ell - a), 4 * ell)
                   + Fraction(tot * tot, 4 * ell) - Fraction(sum(x * x for x in ins), 4))
            return lam, a * (ell - a) + 1
    raise ValueError("no admissible index set")


# ------------------------------------------------------------------------------------------------
# SGLD
# ------------------------------------------------------------------------------------------------
def run_case(model, wstar, Gstar, gamma, gen):
    """All step sizes at once: chains indexed (data set, step size, chain)."""
    nE = len(EPS)
    Cn = R * nE * C
    J = N_DATA // BATCH
    Spp = np.empty((Cn, J, model.P, model.P)); Spy = np.empty((Cn, J, model.P, model.M)); Syy = np.empty((Cn, J))
    Fpp = np.empty((Cn, model.P, model.P)); Fpy = np.empty((Cn, model.P, model.M)); Fyy = np.empty(Cn)
    for r in range(R):
        x = gen.standard_normal((N_DATA, model.inp))
        phi = model.feats(x)
        y = phi @ Gstar.T                                     # noiseless labels
        sl = slice(r * nE * C, (r + 1) * nE * C)
        Fpp[sl], Fpy[sl], Fyy[sl] = phi.T @ phi / N_DATA, phi.T @ y / N_DATA, np.sum(y * y) / N_DATA
        for c in range(nE * C):
            perm = gen.permutation(N_DATA).reshape(J, BATCH)
            pb, yb = phi[perm], y[perm]
            i = r * nE * C + c
            Spp[i] = np.einsum("jnp,jnq->jpq", pb, pb) / BATCH
            Spy[i] = np.einsum("jnp,jnm->jpm", pb, yb) / BATCH
            Syy[i] = np.einsum("jnm,jnm->j", yb, yb) / BATCH
    eps = np.tile(np.repeat(EPS, C), R)
    nbeta = N_DATA / math.log(N_DATA)
    ws = np.repeat(wstar[None], Cn, axis=0).astype(float)
    w = ws.copy()
    Lstar, _ = loss_grad(model, ws, Fpp, Fpy, Fyy, grad=False)
    he, se = 0.5 * eps[:, None], np.sqrt(eps)[:, None]
    acc_f = np.zeros(Cn); acc_b = np.zeros(Cn); cnt = 0
    mala_sum = np.zeros(Cn); mala_cnt = 0
    alive = np.ones(Cn, bool)
    for t in range(T):
        j = t % J
        Lb, gb = loss_grad(model, w, Spp[:, j], Spy[:, j], Syy[:, j])
        wn = w + he * (-nbeta * gb - gamma * (w - ws)) + se * gen.standard_normal(w.shape)
        if t >= BURN:
            Lf, _ = loss_grad(model, w, Fpp, Fpy, Fyy, grad=False)
            acc_f += Lf; acc_b += Lb; cnt += 1
            if (t - BURN) % MALA_EVERY == 0:
                L0, g0 = loss_grad(model, w, Fpp, Fpy, Fyy)
                L1, g1 = loss_grad(model, wn, Fpp, Fpy, Fyy)
                lp0 = -nbeta * L0 - 0.5 * gamma * np.sum((w - ws) ** 2, 1)
                lp1 = -nbeta * L1 - 0.5 * gamma * np.sum((wn - ws) ** 2, 1)
                q01 = -np.sum((wn - w - he * (-nbeta * g0 - gamma * (w - ws))) ** 2, 1) / (2 * eps)
                q10 = -np.sum((w - wn - he * (-nbeta * g1 - gamma * (wn - ws))) ** 2, 1) / (2 * eps)
                la = lp1 + q10 - lp0 - q01
                mala_sum += np.exp(np.minimum(0.0, np.where(np.isfinite(la), la, -np.inf)))
                mala_cnt += 1
        bad = ~np.isfinite(wn).all(1) | (np.abs(wn).max(1) > 1e3)
        alive &= ~bad
        wn[bad] = ws[bad]
        w = wn
    est_f = nbeta * (acc_f / cnt - Lstar)
    est_b = nbeta * (acc_b / cnt - Lstar)
    return eps, est_f, est_b, mala_sum / mala_cnt, alive


def summarize(eps, est_f, est_b, mala, alive):
    out = []
    for e in EPS:
        sel = eps == e
        f = est_f[sel].reshape(R, C); b = est_b[sel].reshape(R, C); a = alive[sel].reshape(R, C)
        fm = np.array([f[i][a[i]].mean() if a[i].any() else np.nan for i in range(R)])
        bm = np.array([b[i][a[i]].mean() if a[i].any() else np.nan for i in range(R)])
        fm, bm = fm[np.isfinite(fm)], bm[np.isfinite(bm)]
        mu, lo, hi = sk.mean_ci(fm) if len(fm) > 1 else (np.nan, np.nan, np.nan)
        out.append(dict(eps=e, mean=mu, lo=lo, hi=hi, rec=float(np.mean(bm)) if len(bm) else np.nan,
                        rec_sd=float(np.std(b[a], ddof=1)) if a.sum() > 1 else np.nan,
                        fb_sd=float(np.std(f[a], ddof=1)) if a.sum() > 1 else np.nan,
                        mala=float(np.mean(mala[sel])), div=float(1 - alive[sel].mean())))
    return out


def main():
    t0 = time.perf_counter()
    gen = sk.rng(SEED)
    nbeta = N_DATA / math.log(N_DATA)
    E11 = np.zeros((3, 3)); E11[0, 0] = 1.0
    cases = []
    reg = DLN((2, 1))                                     # y = a x1 + b x2: a 1 x 2 "network"
    cases.append(("plane", "regular plane", reg, np.zeros(2), np.zeros((1, 2)), 1.0, Fraction(1), 1))
    relu = ReluNeuron()
    for wst, gam, key in (((0.0, 0.0), 1.0, "relu0"), ((1.0, 0.0), 1.0, "relu1"), ((1.0, 0.0), 10.0, "relu1g")):
        cases.append((key, "ReLU neuron", relu, np.array(wst), np.zeros((1, 2)), gam, Fraction(1, 2),
                      2 if wst == (0.0, 0.0) else 1))
    r1 = DLN((3, 1, 3))
    e1 = np.zeros(6); e1[0] = 1.0                         # a = e1 (first layer), b = 0
    cases.append(("rank0", "rank-one 3x3", r1, np.zeros(6), np.zeros((3, 3)), 1.0, Fraction(3, 2), 2))
    cases.append(("rank1", "rank-one 3x3", r1, e1, np.zeros((3, 3)), 1.0, Fraction(3, 2), 1))
    cases.append(("rank1g", "rank-one 3x3", r1, e1, np.zeros((3, 3)), 10.0, Fraction(3, 2), 1))
    for H in ((2, 2, 2), (3, 3, 3), (3, 3, 3, 3), (4, 3, 2, 4), (3, 4, 5, 3), (2, 2, 2, 2, 2)):
        net = DLN(H)
        lam, m = aoyagi(H, 0)
        cases.append(("dln" + "".join(map(str, H)), "DLN " + "-".join(map(str, H)), net, np.zeros(net.d),
                      np.zeros((H[-1], H[0])), 1.0, lam, m))
    net = DLN((3, 3, 3))
    lam, m = aoyagi((3, 3, 3), 1)
    wsp = np.concatenate([E11.ravel(), E11.ravel()])      # special point W1 = W2 = E11 (Hitchcock-Hoogland)
    cases.append(("dln333r1", "DLN 3-3-3, rank 1", net, wsp, E11, 1.0, lam, m))

    sweep_rows, table_rows = [], []
    for k, (key, label, model, wstar, Gstar, gamma, lam, m) in enumerate(cases):
        tc = time.perf_counter()
        Sigma = np.diag([0.5, 0.5]) if isinstance(model, ReluNeuron) else np.eye(model.P)
        r, _ = hessian_rank(model, wstar, Gstar, Sigma)
        if isinstance(model, ReluNeuron):          # K = a^2 b^2 / 4 exactly: Hessian diag(b^2/2, a^2/2) at b=0
            r = int(wstar[0] != 0)
        d = model.d
        if key == "plane":
            skel = nbeta / (nbeta + gamma)
        elif isinstance(model, ReluNeuron):
            skel = skeleton_rankone(nbeta, gamma, abs(wstar[0]), 1, 1, c=0.25)
        elif key.startswith("rank"):
            skel = skeleton_rankone(nbeta, gamma, float(np.linalg.norm(wstar[:3])), 3, 3)
        else:
            skel = float("nan")
        res = summarize(*run_case(model, wstar, Gstar, gamma, gen))
        sel = [x for x in res if x["mala"] >= 0.9 and x["div"] == 0]
        chosen = max(sel, key=lambda x: x["eps"]) if sel else res[0]
        fixed = [x for x in res if x["eps"] == 1e-4][0]
        lamf = float(lam)
        print(f"[{k}] {label} at {key}: d={d} r={r} bounds [{r / 2:.2f}, {r / 2 + (d - r) / 4:.2f}] lambda={lam} m={m}"
              f" d/2={d / 2} gamma={gamma} skeleton={skel:.3f}  ({time.perf_counter() - tc:.1f} s)")
        for x in res:
            extra = ""
            if key == "plane":
                kap = nbeta + gamma
                extra = f"  [exact ULA prediction {skel / (1 - x['eps'] * kap / 4):.3f}]"
            print(f"      eps={x['eps']:.0e}: {x['mean']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] rel.err {(x['mean'] - lamf) / lamf:+.3f}"
                  f"  recycled {x['rec']:.3f} (chain sd {x['rec_sd']:.3f} vs full-batch {x['fb_sd']:.3f})"
                  f"  MALA {x['mala']:.3f}  diverged {x['div']:.2f}{extra}")
            sweep_rows.append([k, x["eps"], x["mean"], x["mean"] - x["lo"], x["hi"] - x["mean"],
                               (x["mean"] - lamf) / lamf, (x["lo"] - lamf) / lamf, (x["hi"] - lamf) / lamf,
                               x["rec"], x["mala"], x["div"]])
        print(f"      selected eps={chosen['eps']:.0e}: {chosen['mean']:.3f} [{chosen['lo']:.3f}, {chosen['hi']:.3f}]")
        table_rows.append([k, key, d, r, r / 2, r / 2 + (d - r) / 4, lamf, m, d / 2, gamma, skel,
                           fixed["mean"], fixed["lo"], fixed["hi"], fixed["mala"],
                           chosen["eps"], chosen["mean"], chosen["lo"], chosen["hi"], chosen["mala"]])
    sk.write_dat("figures/data/13-llc-sgld-sweep.dat",
                 ["case", "eps", "est", "ep", "em", "rel", "rello", "relhi", "rec", "mala", "div"], sweep_rows)
    sk.write_dat("figures/data/13-llc-sgld-table.dat",
                 ["case", "key", "d", "r", "lower", "upper", "lambda", "m", "half_d", "gamma", "skeleton",
                  "est4", "lo4", "hi4", "mala4", "eps", "est", "lo", "hi", "mala"], table_rows)
    kap = nbeta + 1.0
    sk.write_dat("figures/data/13-llc-sgld-ula.dat", ["eps", "rel"],
                 [[e, (nbeta / kap) / (1 - e * kap / 4) - 1.0] for e in np.geomspace(8e-6, 1.25e-3, 40)])
    # wide format for the step-size figure: relative error and its interval for selected cases
    keys = [c[0] for c in cases]
    show = ["plane", "rank0", "dln3333", "dln333", "dln22222"]
    header, wide = ["eps"], []
    for s in show:
        header += [s, s + "lo", s + "hi"]
    for j, e in enumerate(EPS):
        row = [e]
        for s in show:
            rr = sweep_rows[keys.index(s) * len(EPS) + j]
            row += [rr[5], rr[5] - rr[6], rr[7] - rr[5]]
        wide.append(row)
    sk.write_dat("figures/data/13-llc-sgld-eps.dat", header, wide)
    print(f"n={N_DATA}, n beta={nbeta:.1f}, batch {BATCH}, {T} steps ({BURN} burn-in), R={R} data sets x C={C}"
          f" chains per step size, seed {SEED}; total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
