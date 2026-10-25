"""Chapter 11 (classification): the max-margin classifier with a teacher.

Theorem 11.2: for alpha < alpha_MLE(r), the maximal margin kappa_n = max_{|theta|=1} min_i y_i <x_i, theta>
tends to kappa*(alpha), the cosine of the max-margin direction with the teacher to c*, and its test error to
Err* = P(c* V + sqrt(1 - c*^2) Z < 0), where kappa* is the unique kappa > 0 with
    min_t F_kappa(t) = 1/alpha,   F_kappa(t) = E (kappa sqrt(1 + t^2) - t V - Z)_+^2,
and c* = t*/sqrt(1 + t*^2) for the minimizer t* (V = Y U, P(Y = 1 | U) = f(r U)).
For the noiseless teacher (f = 1{t > 0}) V = |U| and Err* = arccos(c*)/pi: the replica prediction of
Opper, Kinzel, Kleinz and Nehl (1990) for the perceptron of optimal stability. For random labels c* = 0
and alpha = 1/E (Z + kappa*)_+^2 (Gardner's capacity).

Computes
  (1) kappa*, c*, Err* on a grid of alpha for the noiseless teacher, the logistic teacher with r = sqrt(5)
      (alpha < alpha_MLE = 3.0714), and random labels; the large-alpha constants alpha Err*, and their
      limit 3/(sqrt(2 pi) G(s*)) (see large_alpha_constant);
  (2) the Bayes-optimal error of the noiseless teacher (Opper--Haussler; the replica formula proved by
      Barbier et al. 2019): q = qh/(1 + qh), qh = (alpha/(pi (1 - q))) E[exp(-q xi^2/(1 - q))/Phi(u xi)],
      u = sqrt(q/(1 - q)), error arccos(sqrt q)/pi;
  (3) a simulation at d = 200: the hard-margin SVM through the origin (min |w|^2/2 subject to y_i <x_i, w> >= 1)
      solved in the dual with L-BFGS-B, 10 data sets per alpha; measured margin min_i y_i <x_i, w>/|w| and test
      error Err(w/|w|) (exact, from the cosine with the teacher). Student-t 95% intervals.
Writes figures/data/11-classification-maxmargin-{curve,logistic,sim-noiseless,sim-logistic}.dat.
Runtime under a minute.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # shared machine: one BLAS thread per script
os.environ.setdefault("OMP_NUM_THREADS", "1")
import math
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import optimize, special

import rosettakit as rk

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")
SQ2PI = math.sqrt(2 * math.pi)
_xl, _wl = np.polynomial.legendre.leggauss(400)


def v_nodes(r, link, L=12.0):
    """Nodes/weights of V = Y U: density (f(r v) + 1 - f(-r v)) phi(v), split at v = 0."""
    vs, ws = [], []
    for lo, hi in ((-L, 0.0), (0.0, L)):
        v = 0.5 * (hi - lo) * _xl + 0.5 * (hi + lo)
        if link == "noiseless":
            dens = 2.0 * (v > 0)
        elif link == "random":
            dens = np.ones_like(v)
        else:
            dens = special.expit(r * v) + 1.0 - special.expit(-r * v)
        vs.append(v)
        ws.append(0.5 * (hi - lo) * _wl * dens * np.exp(-0.5 * v * v) / SQ2PI)
    return np.concatenate(vs), np.concatenate(ws)


def gpos(u):
    """E (u - Z)_+^2 = (1 + u^2) Phi(u) + u phi(u)."""
    return (1 + u * u) * special.ndtr(u) + u * np.exp(-0.5 * u * u) / SQ2PI


class Teacher:
    def __init__(self, link, r=1.0):
        self.V, self.W = v_nodes(r, link)
        self.link = link

    def F(self, kappa, t):
        return float(np.sum(self.W * gpos(kappa * math.sqrt(1 + t * t) - t * self.V)))

    def tmin(self, kappa):
        ts = np.concatenate([np.linspace(-4, 4, 81), np.geomspace(4.2, 400, 50)])
        vals = [self.F(kappa, t) for t in ts]
        i = int(np.argmin(vals))
        a, b = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
        res = optimize.minimize_scalar(lambda t: self.F(kappa, t), bounds=(a, b), method="bounded",
                                       options={"xatol": 1e-13})
        return res.fun, res.x

    def err(self, c):
        """P(c V + sqrt(1 - c^2) Z < 0)."""
        if c >= 1.0:
            return float(np.sum(self.W * (self.V < 0)))
        return float(np.sum(self.W * special.ndtr(-c * self.V / math.sqrt(1 - c * c))))

    def maxmargin(self, alpha):
        """(kappa*, c*, Err*) or None if alpha >= alpha_MLE (not separable)."""
        m = lambda k: self.tmin(k)[0] - 1.0 / alpha
        if m(0.0) >= 0:
            return None
        hi = 1.0
        while m(hi) < 0:
            hi *= 2
        k = optimize.brentq(m, 0.0, hi, xtol=1e-14)
        t = self.tmin(k)[1]
        c = t / math.sqrt(1 + t * t)
        return k, c, self.err(c)


_tl, _twl = np.polynomial.legendre.leggauss(2000)
TT, TW = 40.0 * _tl, 40.0 * _twl
RATIO = np.exp(-TT * TT - np.log(2 * np.pi) - special.log_ndtr(TT))   # phi(t)^2 / Phi(t)


def bayes_error(alpha):
    """Bayes-optimal error of the noiseless spherical teacher (state evolution of the replica formula)."""
    q = 0.5
    for _ in range(20000):
        u = math.sqrt(q / (1 - q))
        dens = np.exp(-0.5 * (TT / u) ** 2) / (SQ2PI * u)
        qh = 2 * alpha / (1 - q) * float(np.sum(TW * RATIO * dens))
        qn = qh / (1 + qh)
        if abs(qn - q) < 1e-15:
            q = qn
            break
        q = 0.5 * q + 0.5 * qn
    return math.acos(math.sqrt(q)) / math.pi


def large_alpha_constant():
    """Limit of alpha Err* (noiseless teacher) as alpha -> infinity. For large t only |U| ~ 1/t matters, so
    F_kappa(t) ~ sqrt(2/pi) G(kappa t)/(3 t) with G(s) = E (s - Z)_+^3 = (s^3 + 3 s) Phi(s) + (s^2 + 2) phi(s);
    minimizing gives s G'(s) = G(s), and Err* ~ 1/(pi t*) gives alpha Err* -> 3/(sqrt(2 pi) G(s*))."""
    G = lambda s: (s ** 3 + 3 * s) * special.ndtr(s) + (s * s + 2) * math.exp(-0.5 * s * s) / SQ2PI
    G1 = lambda s: 3 * ((1 + s * s) * special.ndtr(s) + s * math.exp(-0.5 * s * s) / SQ2PI)
    s = optimize.brentq(lambda s: s * G1(s) - G(s), 0.1, 3.0, xtol=1e-15)
    return 3 / (SQ2PI * G(s)), s


def svm_origin(Z):
    """Hard-margin SVM through the origin for the rows z_i of Z; returns w (dual by L-BFGS-B)."""
    K = Z @ Z.T
    n = Z.shape[0]
    fg = lambda a: (0.5 * a @ K @ a - a.sum(), K @ a - 1.0)
    res = optimize.minimize(fg, np.full(n, 1e-3), jac=True, method="L-BFGS-B", bounds=[(0, None)] * n,
                            options={"maxiter": 100000, "maxfun": 200000, "ftol": 1e-16, "gtol": 1e-11})
    return Z.T @ res.x


def main():
    t0 = time.time()
    noiseless, logistic, randl = Teacher("noiseless"), Teacher("logit", math.sqrt(5.0)), Teacher("random")
    print("== noiseless teacher: kappa*, c*, Err*, Bayes ==")
    for a in [0.5, 1.0, 2.0, 4.0, 10.0]:
        k, c, e = noiseless.maxmargin(a)
        print("alpha=%5.1f kappa*=%.5f c*=%.5f Err*=%.5f Bayes=%.5f" % (a, k, c, e, bayes_error(a)))
    for a in [50.0, 100.0, 200.0]:
        k, c, e = noiseless.maxmargin(a)
        print("alpha=%5.0f: alpha Err* = %.4f, alpha Bayes = %.4f, alpha kappa* = %.4f" % (a, a * e, a * bayes_error(a), a * k))
    print("limit alpha -> infinity: alpha Err* -> %.5f (s* = %.6f)" % large_alpha_constant())
    print("random labels alpha=1: kappa* = %.5f (Gardner: alpha_c(kappa*) = %.6f)" % (
        randl.maxmargin(1.0)[0], rk.gardner_capacity(randl.maxmargin(1.0)[0])))
    rows = []
    for a in np.concatenate([np.linspace(0.2, 2.0, 19), np.linspace(2.25, 10.0, 32)]):
        k, c, e = noiseless.maxmargin(a)
        lg = logistic.maxmargin(a) if a < 3.07 else None
        kl, el = (lg[0], lg[2]) if lg else (float("nan"), float("nan"))
        kr = randl.maxmargin(a)[0] if a < 2.0 else float("nan")
        rows.append((a, k, c, e, bayes_error(a), kl, el, kr))
    # the logistic curve close to its endpoint alpha_MLE = 3.0714, where kappa* -> 0
    for a in [2.5, 2.75, 2.9, 3.0, 3.05, 3.07]:
        lg = logistic.maxmargin(a)
        print("logistic r=sqrt5 alpha=%.2f:" % a, "kappa*=%.5f c*=%.5f Err*=%.5f" % lg if lg else "not separable")
    lg_rows = []
    for a in np.concatenate([np.linspace(0.2, 2.9, 28), [2.95, 3.0, 3.03, 3.05, 3.06, 3.07]]):
        lg = logistic.maxmargin(a)
        if lg:
            lg_rows.append((a, lg[0], lg[1], lg[2]))
    rk.write_dat(os.path.join(DATA, "11-classification-maxmargin-curve.dat"),
                 ["alpha", "kappa", "cos", "err", "bayes", "kappalog", "errlog", "kapparand"], rows)
    rk.write_dat(os.path.join(DATA, "11-classification-maxmargin-logistic.dat"),
                 ["alpha", "kappa", "cos", "err"], lg_rows)
    print("Bayes error of the logistic teacher, E f(-r|U|) = %.4f" % float(
        np.sum(randl.W * special.expit(-math.sqrt(5.0) * np.abs(randl.V)))))
    print("theory done (%.0f s)" % (time.time() - t0))

    d, reps = 200, 10
    gen = rk.rng(11004)
    sim = []
    for name, teach, alphas in (("noiseless", noiseless, [0.5, 1.0, 2.0, 3.0, 4.0]),
                                ("logistic", logistic, [0.5, 1.0, 1.5, 2.0, 2.5])):
        for a in alphas:
            n = int(round(a * d))
            ms, es = [], []
            for _ in range(reps):
                X = gen.standard_normal((n, d))
                if name == "noiseless":
                    y = np.sign(X[:, 0])
                else:
                    y = np.where(gen.random(n) < special.expit(math.sqrt(5.0) * X[:, 0]), 1.0, -1.0)
                Z = y[:, None] * X
                w = svm_origin(Z)
                wn = w / np.linalg.norm(w)
                ms.append(float(np.min(Z @ wn)))
                es.append(teach.err(float(np.clip(wn[0], -1, 1))))
            mm, mh = rk.mean_ci(ms)
            em, eh = rk.mean_ci(es)
            k, c, e = teach.maxmargin(a)
            sim.append((0 if name == "noiseless" else 1, a, mm, mh, em, eh, k, e))
            print("%s alpha=%.1f (n=%d): margin %.4f +- %.4f (kappa* %.4f), error %.4f +- %.4f (Err* %.4f)  (%.0f s)" % (
                name, a, n, mm, mh, k, em, eh, e, time.time() - t0))
    hdr = ["alpha", "margin", "marginh", "err", "errh", "kappapred", "errpred"]
    rk.write_dat(os.path.join(DATA, "11-classification-maxmargin-sim-noiseless.dat"), hdr,
                 [row[1:] for row in sim if row[0] == 0])
    rk.write_dat(os.path.join(DATA, "11-classification-maxmargin-sim-logistic.dat"), hdr,
                 [row[1:] for row in sim if row[0] == 1])
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
