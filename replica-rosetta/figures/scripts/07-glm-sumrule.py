"""Chapter 7 (glm): a Monte Carlo check of the sum rule (Theorem "sum rule").

Rademacher teacher, probit channel P(y|s) = Phi(y s/sqrt(Delta)), Delta = 1/2, d = 6, n = 7.  For a path
(q(t), r(t)) with R1 = int r, R2 = int q, the theorem says
    f_d = psi(R1(1)) + (n/d) Psi(R2(1)) - (1/2) int_0^1 r q dt + (1/2) int_0^1 E<(A_t - r)(Q - q)>_t dt
exactly, for every d.  Both sides are estimated by Monte Carlo with exact enumeration of the 2^d sign
vectors; the integral over u_mu in the Gibbs measure is done in closed form (a probit channel convolved
with a Gaussian is a probit channel), and the t-integral by 6-point Gauss--Legendre quadrature.
Psi for the probit channel: 2 E[Phi(bV) log Phi(bV)], b = sqrt(q/(1 + Delta - q)).
Seed 2026.  Runtime: about 50 s on an idle laptop (several times more on a loaded machine).  Prints the two
sides and the remainder for three paths, and writes them to figures/data/07-glm-sumrule.dat
(path 0 is the left side f_d; columns: potential part, remainder, 95% half-width, total).
"""
import math
import os
import sys
import time

import numpy as np
from scipy import integrate, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

T0 = time.time()
gen = rk.rng(2026)
d, n, Delta = 6, 7, 0.5
WALL = (1 - 2 * ((np.arange(2 ** d)[:, None] >> np.arange(d)) & 1)).astype(float)
logPhi = special.log_ndtr


def psi(r):
    return rk.gauss(lambda z: np.logaddexp(r + math.sqrt(r) * z, -r - math.sqrt(r) * z) - math.log(2), 161) - r / 2


def Psi(q):
    b = math.sqrt(q / (1 + Delta - q))
    return rk.gauss(lambda v: 2 * np.exp(logPhi(b * v)) * logPhi(b * v), 161)


def lhs(N, chunk=20000):
    vals = []
    for _ in range(N // chunk):
        ws = gen.choice([-1.0, 1.0], size=(chunk, d))
        X = gen.standard_normal((chunk, n, d))
        S = np.einsum("cnd,cd->cn", X, ws) / math.sqrt(d)
        Y = np.sign(S + math.sqrt(Delta) * gen.standard_normal((chunk, n)))
        a = np.einsum("cnd,wd->cwn", X, WALL) / math.sqrt(d)
        logL = logPhi(Y[:, None, :] * a / math.sqrt(Delta)).sum(axis=2)
        vals.append(special.logsumexp(logL, axis=1) - d * math.log(2))
    v = np.concatenate(vals) / d
    return v.mean(), 1.96 * v.std() / math.sqrt(len(v))


def remainder_at(t, q, r, R1, R2, N, chunk=15000):
    vals = []
    b2 = t - R2
    for _ in range(N // chunk):
        ws = gen.choice([-1.0, 1.0], size=(chunk, d))
        X = gen.standard_normal((chunk, n, d))
        V = gen.standard_normal((chunk, n))
        U = gen.standard_normal((chunk, n))
        S = math.sqrt(1 - t) * np.einsum("cnd,cd->cn", X, ws) / math.sqrt(d) + math.sqrt(R2) * V + math.sqrt(b2) * U
        Y = np.sign(S + math.sqrt(Delta) * gen.standard_normal((chunk, n)))
        Yp = math.sqrt(R1) * ws + gen.standard_normal((chunk, d))
        a = math.sqrt(1 - t) * np.einsum("cnd,wd->cwn", X, WALL) / math.sqrt(d) + math.sqrt(R2) * V[:, None, :]
        sd = math.sqrt(Delta + b2)
        c = Y[:, None, :] * a / sd
        logw = logPhi(c).sum(axis=2) + math.sqrt(R1) * (Yp @ WALL.T)
        logw -= logw.max(axis=1, keepdims=True)
        pi = np.exp(logw)
        pi /= pi.sum(axis=1, keepdims=True)
        Qw = (ws @ WALL.T) / d
        g_st = Y[:, None, :] * np.exp(-0.5 * c * c - 0.5 * math.log(2 * math.pi) - logPhi(c)) / sd   # <u'(s)> given w
        cS = Y * S / math.sqrt(Delta)
        g_te = Y * np.exp(-0.5 * cS * cS - 0.5 * math.log(2 * math.pi) - logPhi(cS)) / math.sqrt(Delta)
        vals.append(np.einsum("cn,cwn,cw->c", g_te, g_st, pi * (Qw - q)) / d - r * np.sum(pi * (Qw - q), axis=1))
    v = np.concatenate(vals)
    return v.mean(), 1.96 * v.std() / math.sqrt(len(v))


def rhs(qf, rf, N_node=30000, nodes=6):
    tx, tw = np.polynomial.legendre.leggauss(nodes)
    tx, tw = 0.5 * (tx + 1), 0.5 * tw
    R1 = lambda t: integrate.quad(rf, 0, t)[0]
    R2 = lambda t: integrate.quad(qf, 0, t)[0]
    rem, err2 = 0.0, 0.0
    for t, w in zip(tx, tw):
        m, h = remainder_at(t, qf(t), rf(t), R1(t), R2(t), N_node)
        rem += w * m
        err2 += (w * h) ** 2
    main = psi(R1(1)) + (n / d) * Psi(R2(1)) - 0.5 * integrate.quad(lambda t: rf(t) * qf(t), 0, 1)[0]
    return main, 0.5 * rem, 0.5 * math.sqrt(err2)


L, hL = lhs(140000)
print(f"d = {d}, n = {n}, probit Delta = {Delta}")
print(f"left side  f_d = {L:.4f} +- {hL:.4f}")
rows = [[0, float("nan"), float("nan"), hL, L]]
for k, (name, qf, rf) in enumerate([("q = 0, r = 0", lambda t: 0.0, lambda t: 0.0),
                                    ("q = 0.7, r = 0.3", lambda t: 0.7, lambda t: 0.3),
                                    ("q = 0.2 + 0.5 t, r = 1 - 0.6 t", lambda t: 0.2 + 0.5 * t, lambda t: 1.0 - 0.6 * t)]):
    main, rem, h = rhs(qf, rf)
    rows.append([k + 1, main, rem, h, main + rem])
    print(f"path {name:32s}: potential part {main:.4f}, remainder {rem:+.4f} +- {h:.4f}, right side {main + rem:.4f}")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rk.write_dat(os.path.join(ROOT, "figures", "data", "07-glm-sumrule.dat"), ["path", "potential", "remainder", "ci", "total"], rows)
print(f"runtime {time.time() - T0:.1f} s")
