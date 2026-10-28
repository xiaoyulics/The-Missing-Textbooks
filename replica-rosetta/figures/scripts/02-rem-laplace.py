"""Exact finite-N expectations of the random energy model by the Laplace transform (Chapter 2).

The partition function Z_N = sum_{i<=M} w_i, w_i = exp(c X_i), c = beta sqrt(N), M = 2^N, X_i iid N(0,1),
is a sum of M independent terms, so E exp(-t Z_N) = L(t)^M with L(t) = E exp(-t w). Every quantity
below is an exact one-dimensional integral against L(t)^M, evaluated by quadrature in log space
(no simulation). With t = exp(-tau) and g(tau) = 1 - L(exp(-tau)) = E[1 - exp(-exp(cX - tau))]:

  E log Z     = -gamma + int (1[tau > 0] - L^M) dtau                 (Frullani: log z = int (e^-t - e^-tz) dt/t)
  E Y_N       = M int E[u^2 e^-u] L^{M-1} dtau,  u = exp(cX - tau)    (1/Z^2 = int t e^{-tZ} dt; Mezard--Montanari ch. 5)
  E Z^n       = n / Gamma(1-n) int (1 - L^M) e^{n tau} dtau,  0 < n < 1
  E Z^n       = sum over set partitions of the n replicas,  n = 1, 2, 3, ...

The planted model of the decoding problem (one level shifted by s = lambda sqrt N) uses the same
integrals with one factor L replaced by its shifted version.

This file is a module (loaded by the other 02-rem-*.py scripts with importlib); run it directly for a
check against simulation (about 20 s): python figures/scripts/02-rem-laplace.py
"""
import math
import os
import sys

import numpy as np
from scipy import special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

LOG2 = math.log(2.0)
BETA_C = math.sqrt(2.0 * LOG2)
EULER = 0.57721566490153286
LOGPHI0 = -0.5 * math.log(2.0 * math.pi)
OFF = np.arange(-16.0, 16.0 + 1e-9, 0.025)        # x-grid around the peak of each inner integrand
HX = OFF[1] - OFF[0]


def phi_rem(beta):
    """Derrida's free entropy: log 2 + beta^2/2 below beta_c, beta beta_c above."""
    return rk.rem_free_entropy(beta)


def phi_moment(n, beta):
    """The limit varphi(n) = lim N^-1 log E Z_N^n of theorem thm:rem:moments: n phi(beta) for n <= n_c,
    log 2 + n^2 beta^2 / 2 for n >= n_c, with m = beta_c / beta and n_c = max(m, m^2)."""
    m = BETA_C / beta
    nc = max(m, m * m)
    return n * phi_rem(beta) if n <= nc else LOG2 + 0.5 * n * n * beta * beta


def _log_F(y):
    """log(1 - exp(-e^y)), stable for all real y."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    lo = y < -30.0
    out[lo] = y[lo]
    hi = y > 4.0
    with np.errstate(over="ignore"):
        out[hi] = np.log1p(-np.exp(-np.exp(np.minimum(y[hi], 700.0))))
    mid = ~(lo | hi)
    out[mid] = np.log(-np.expm1(-np.exp(y[mid])))
    return out


def _log_trap(logf, h):
    """log of the trapezoid integral of exp(logf) along the last axis."""
    m = np.max(logf, axis=-1, keepdims=True)
    e = np.exp(logf - m)
    s = np.sum(e, axis=-1) - 0.5 * (e[..., 0] + e[..., -1])
    return m[..., 0] + np.log(s * h)


def _offsets(c):
    """x-grid around the peak: the factors in exp(cX - tau) vary on the scale 1/c, so the spacing is
    min(0.025, 0.25/c) (checked: E Y_N at beta = 3, N = 320 is stable to 2e-7 below this spacing)."""
    hx = min(HX, 0.25 / c)
    return np.arange(-16.0, 16.0 + 1e-9, hx), hx


def _chunked(fun, tau, c):
    off, hx = _offsets(c)
    step = max(1, int(3e6 // len(off)))
    return np.concatenate([fun(tau[i:i + step], off, hx) for i in range(0, len(tau), step)])


def log_g(tau, c):
    """log g(tau) = log E[1 - exp(-exp(cX - tau))] for an array of tau."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))

    def one(t, off, hx):
        x0 = np.clip(np.minimum(c, t / c), 0.0, None)
        X = x0[:, None] + off[None, :]
        return _log_trap(LOGPHI0 - 0.5 * X * X + _log_F(c * X - t[:, None]), hx)
    return _chunked(one, tau, c)


def log_hk(tau, c, k):
    """log E[u^k e^{-u}], u = exp(cX - tau), k = 1 or 2."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))

    def one(t, off, hx):
        x0 = np.clip(np.minimum(k * c, (t + math.log(k)) / c), 0.0, None)
        X = x0[:, None] + off[None, :]
        y = c * X - t[:, None]
        return _log_trap(LOGPHI0 - 0.5 * X * X + k * y - np.exp(np.minimum(y, 700.0)), hx)
    return _chunked(one, tau, c)


def M_log_L(tau, c, logM):
    """M log L(e^{-tau}) = M log(1 - g(tau)), stable when g is tiny or M is huge."""
    lg = log_g(tau, c)
    out = np.empty_like(lg)
    small = lg < -18.0
    out[small] = -np.exp(logM + lg[small]) * (1.0 + 0.5 * np.exp(lg[small]))
    big = ~small
    out[big] = math.exp(logM) * np.log1p(-np.exp(lg[big])) if logM < 700 else -np.inf
    return out


def _find_tau(c, logM, target, lo=-60.0):
    """tau with log(M g(tau)) = target, by bisection (log(Mg) decreases in tau)."""
    hi = 4.0 * c * c + 4.0 * logM / LOG2 + 300.0
    f = lambda t: logM + log_g(np.array([t]), c)[0] - target
    if f(lo) < 0:
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


def _simpson(y, h):
    if len(y) % 2 == 0:
        y = y[:-1]
    return h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))


def _grid(a, b, h):
    n = max(2, int(math.ceil((b - a) / h / 2.0)) * 2)
    return np.linspace(a, b, n + 1)


def E_logZ(beta, N, h=0.1):
    """E log Z_N(beta), exact up to quadrature."""
    c, logM = beta * math.sqrt(N), N * LOG2
    ta = max(_find_tau(c, logM, math.log(60.0)), 0.0)
    tb = _find_tau(c, logM, math.log(1e-20))
    tau = _grid(ta, tb, h)
    val = ta + _simpson(-np.expm1(M_log_L(tau, c, logM)), tau[1] - tau[0])
    if ta == 0.0:
        t2 = np.linspace(-40.0, 0.0, 4001)
        val -= _simpson(np.exp(M_log_L(t2, c, logM)), t2[1] - t2[0])
    return val - EULER


def E_Y(beta, N, h=0.1):
    """E Y_N = E sum_s G_N(s)^2, exact up to quadrature."""
    c, logM = beta * math.sqrt(N), N * LOG2
    logM1 = math.log(2.0 ** N - 1.0) if N < 60 else logM
    ta = _find_tau(c, logM, math.log(logM + 60.0))
    tb = _find_tau(c, logM, math.log(1e-25))
    tau = _grid(ta, tb, h)
    return _simpson(np.exp(logM + log_hk(tau, c, 2) + M_log_L(tau, c, logM1)), tau[1] - tau[0])


def _log_Jn(n, loga):
    """log J_n(a), J_n(a) = int_0^a (1 - e^{-s}) s^{-n-1} ds (0 < n < 1), from log a."""
    loga = np.asarray(loga, dtype=float)
    out = np.empty_like(loga)
    small = loga < math.log(1e-2)
    ls = loga[small]
    s = np.zeros_like(ls)
    for k in range(1, 12):
        s += (-1) ** (k + 1) * np.exp((k - 1) * ls) / (math.factorial(k) * (k - n))
    out[small] = (1.0 - n) * ls + np.log(s)
    big = ~small
    ab = np.exp(np.minimum(loga[big], 700.0))
    out[big] = np.log((math.gamma(1.0 - n) * special.gammainc(1.0 - n, ab) - (-np.expm1(-ab)) * ab ** (-n)) / n)
    return out


def log_EZn_frac(n, beta, N, h=0.1):
    """log E Z_N^n for 0 < n < 1, exact up to quadrature."""
    c, logM = beta * math.sqrt(N), N * LOG2
    ta = _find_tau(c, logM, math.log(60.0))
    tb = _find_tau(c, logM, math.log(1e-22))
    tau = _grid(ta, tb, h)
    ml = M_log_L(tau, c, logM)
    lf = np.where(ml < -1e-8, np.log(-np.expm1(ml)), np.log(np.maximum(-ml, 1e-300))) + n * tau
    mx = lf.max()
    part = mx + math.log(_simpson(np.exp(lf - mx), tau[1] - tau[0]))
    left = n * ta - math.log(n)                    # 1 - L^M = 1 to machine precision below ta
    x = np.arange(-12.0, max(c, tb / c, n * c) + 16.0, 0.01)   # above tb, 1 - L^M = M g to 1e-22
    tail = float(_log_trap((logM + LOGPHI0 - 0.5 * x * x + n * c * x + _log_Jn(n, c * x - tb))[None, :], 0.01)[0])
    return math.log(n) - special.gammaln(1.0 - n) + float(special.logsumexp([part, left, tail]))


def log_EZ_int(n, beta, N):
    """log E Z_N^n for a positive integer n: sum over set partitions of the replicas,
    E Z^n = sum_pi (2^N)_{|pi|} exp(beta^2 N sum_B |B|^2 / 2)."""
    def partitions(k, largest=None):
        largest = k if largest is None else largest
        if k == 0:
            yield []
            return
        for first in range(min(k, largest), 0, -1):
            for rest in partitions(k - first, first):
                yield [first] + rest
    terms = []
    for sizes in partitions(n):
        mult = {}
        for s in sizes:
            mult[s] = mult.get(s, 0) + 1
        lognum = math.lgamma(n + 1) - sum(math.lgamma(s + 1) for s in sizes) - sum(math.lgamma(v + 1) for v in mult.values())
        logfall = sum(N * LOG2 + math.log1p(-j * 2.0 ** (-N)) for j in range(len(sizes)))
        terms.append(lognum + logfall + 0.5 * beta * beta * N * sum(s * s for s in sizes))
    return float(special.logsumexp(terms))


# ------------------------------------------------------------------------------------------------
# The decoding problem: one planted level y* = lambda sqrt N + X* among M - 1 levels X_s
# ------------------------------------------------------------------------------------------------


def map_success(lam, N):
    """P(MAP decoding succeeds) = E Phi(lambda sqrt N + X)^{M-1}, exact up to quadrature."""
    x = np.arange(-12.0, 12.0, 0.002)
    logM1 = math.log(2.0 ** N - 1.0)
    lf = LOGPHI0 - 0.5 * x * x + math.exp(logM1) * special.log_ndtr(lam * math.sqrt(N) + x)
    return float(np.exp(_log_trap(lf[None, :], x[1] - x[0])[0]))


def E_logZ_planted(lam, N, h=0.1):
    """E log Z(y) in the planted model at the Bayes temperature beta = lambda."""
    c, s = lam * math.sqrt(N), lam * math.sqrt(N)
    logM1 = math.log(2.0 ** N - 1.0) if N < 60 else N * LOG2
    ta = max(min(_find_tau(c, logM1, math.log(60.0)), c * s - 12.0 * c), 0.0)
    tb = max(_find_tau(c, logM1, math.log(1e-20)), c * s + c * c / 2.0 + 60.0)
    tau = _grid(ta, tb, h)
    with np.errstate(divide="ignore"):                          # log(1 - g*(tau)); -inf where g* = 1
        lstar = np.log1p(-np.minimum(np.exp(log_g(tau - c * s, c)), 1.0))
    integrand = -np.expm1(lstar + M_log_L(tau, c, logM1))
    return ta + _simpson(integrand, tau[1] - tau[0]) - EULER


def mutual_information(lam, N):
    """I(sigma*; y) / N = log 2 + lambda^2 - E log Z(y) / N (corollary cor:rem:decoding)."""
    return LOG2 + lam * lam - E_logZ_planted(lam, N) / N


if __name__ == "__main__":
    import time
    t0 = time.time()
    gen = rk.rng(202)
    print("check: exact E log Z_N / N against simulation (95% Student-t intervals)")
    for N, R in [(8, 4000), (12, 600)]:
        for beta in [0.6, 1.2, 2.0]:
            sims = [special.logsumexp(beta * math.sqrt(N) * gen.standard_normal(2 ** N)) / N for _ in range(R)]
            mu, half = rk.mean_ci(sims)
            ex = E_logZ(beta, N) / N
            print(f"  N={N:2d} beta={beta:.1f}: exact {ex:.5f}  simulated {mu:.5f} +- {half:.5f}  {'ok' if abs(ex - mu) < 2 * half else 'CHECK'}")
    print("check: exact E Y_N against simulation")
    for N, R in [(10, 3000)]:
        for beta in [1.0, 2.0]:
            ys = []
            for _ in range(R):
                lw = beta * math.sqrt(N) * gen.standard_normal(2 ** N)
                ys.append(math.exp(special.logsumexp(2 * (lw - special.logsumexp(lw)))))
            mu, half = rk.mean_ci(ys)
            print(f"  N={N} beta={beta}: exact {E_Y(beta, N):.5f}  simulated {mu:.5f} +- {half:.5f}")
    print("check: fractional moment n -> 1 and integer moment n = 1 (N = 20)")
    for beta in [0.8, 2.0]:
        print(f"  beta={beta}: n=0.999 {log_EZn_frac(0.999, beta, 20) / 20:.5f}   n=1 {log_EZ_int(1, beta, 20) / 20:.5f}")
    print("check: planted model at N = 10, lambda = 1.2: MAP success and E log Z against simulation")
    lam, N, R = 1.2, 10, 4000
    succ, lz = [], []
    for _ in range(R):
        X = gen.standard_normal(2 ** N)
        X[0] += lam * math.sqrt(N)
        succ.append(float(np.argmax(X) == 0))
        lz.append(special.logsumexp(lam * math.sqrt(N) * X))
    print(f"  success: exact {map_success(lam, N):.4f}  simulated {np.mean(succ):.4f} (n={R})")
    mu, half = rk.mean_ci(lz)
    print(f"  E log Z: exact {E_logZ_planted(lam, N):.4f}  simulated {mu:.4f} +- {half:.4f}")
    print(f"done in {time.time() - t0:.1f} s")
