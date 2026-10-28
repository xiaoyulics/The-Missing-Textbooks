"""Chapter 7 (glm): GAMP on the teacher--student perceptron against its state evolution.

Data: x_mu ~ N(0, I_d), y_mu = sgn(<x_mu, w*>/sqrt(d)), mu <= n = round(alpha d), d = 2000.
Teachers: Ising (w* uniform on {-1,1}^d) and spherical (w* ~ N(0, I_d)).
Bayes-optimal GAMP (Rangan 2011), with F = X/sqrt(d), started from the prior mean (w_hat = 0, v = 1):
    V     = mean(v)
    omega = F w_hat - V g_prev                                   (Onsager term)
    g     = d/d omega log Phi(y omega/sqrt V),  dg = -g (g + omega/V)
    A     = -(n/d) mean(dg),   B = F^T g + A w_hat
    w_hat = tanh(B), v = 1 - w_hat^2        (Ising prior)
    w_hat = B/(1+A), v = 1/(1+A)            (Gaussian prior)
State evolution (the same recursion as in 07-glm-potential.py): q_0 = 0, r_t = alpha 2Psi'(q_t),
q_{t+1} = E tanh(r_t + sqrt(r_t) Z) (Ising) or r_t/(1+r_t) (Gaussian).  It predicts
    <w_hat_t, w*>/d -> q_t,  |w_hat_t|^2/d -> q_t,  and the test error of sgn<x, w_hat_t>:
    (1/pi) arccos( <w_hat_t, w*> / (|w_hat_t| |w*|) ) -> (1/pi) arccos sqrt(q_t).
Outputs: figures/data/07-glm-gamp-alpha.dat (final errors against alpha, Ising),
         figures/data/07-glm-gamp-iter.dat  (overlap against iteration, Ising, alpha = 1.3, 1.47, 1.6),
         figures/data/07-glm-gamp-sph.dat   (spherical teacher).
Seeds 7000 + 100 k + s (Ising), 9000 + 100 k + s (spherical).  The matrix is stored in single
precision for speed.  Runtime: about 25 s on an idle laptop.
"""
import math
import os
import sys
import time

import numpy as np
from scipy import special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")
D = 2000
TMAX = 150

# ---------------------------------------------------------------------------------------------
# State evolution (the closed forms of 07-glm-potential.py)
# ---------------------------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(1600)


def EV_of(g, a):
    if a < 1.0:
        return rk.gauss(lambda v: g(a * v), 201)
    x, w = 40.0 * _GL_X, 40.0 * _GL_W
    return float(np.sum(w * g(x) * np.exp(-0.5 * (x / a) ** 2) / (a * math.sqrt(2 * math.pi))))


def dPsi2(q):
    if q <= 0:
        return 2 / math.pi
    a = math.sqrt(q / (1 - q))
    return EV_of(lambda x: np.exp(-x * x - special.log_ndtr(x)), a) / (math.pi * (1 - q))


def se(alpha, prior, T):
    qs = [0.0]
    for _ in range(T):
        q = qs[-1]
        if q > 1 - 1e-12:
            qs.append(1.0)
            continue
        r = alpha * dPsi2(q)
        if prior == "ising":
            qn = rk.gauss(lambda z: np.tanh(r + math.sqrt(r) * z), 201) if r < 300 else 1.0
        else:
            qn = r / (1 + r)
        qs.append(min(qn, 1.0))
    return np.array(qs)


# ---------------------------------------------------------------------------------------------
# GAMP
# ---------------------------------------------------------------------------------------------
def gamp(F, y, prior, wstar, T=TMAX, tol=1e-9):
    """Run GAMP; stop when the iterates stop moving (mean squared change < tol), or when GAMP's own
    variance estimate V = mean(v) drops below 1e-8 (the estimate has frozen to a point of the cube:
    continuing would only divide by V, and with finitely many patterns the empirical A would then
    collapse), or after T iterations."""
    n, d = F.shape
    w = np.zeros(d)
    v = np.ones(d)
    g_prev = np.zeros(n)
    over, errs = [0.0], [0.5]
    for t in range(T):
        if float(v.mean()) < 1e-8:
            break
        V = float(v.mean())
        omega = (F @ w.astype(np.float32)).astype(np.float64) - V * g_prev
        u = y * omega / math.sqrt(V)
        ratio = np.exp(-0.5 * u * u - 0.5 * math.log(2 * math.pi) - special.log_ndtr(u))   # phi(u)/Phi(u)
        g = y * ratio / math.sqrt(V)
        dg = -g * (g + omega / V)
        A = -float(dg.sum()) / d
        B = (F.T @ g.astype(np.float32)).astype(np.float64) + A * w
        if prior == "ising":
            w_new = np.tanh(B)
            v = 1 - w_new ** 2
        else:
            w_new = B / (1 + A)
            v = np.full(d, 1 / (1 + A))
        change = float(np.mean((w_new - w) ** 2))
        w, g_prev = w_new, g
        m = float(w @ wstar) / d
        cos = float(w @ wstar) / (np.linalg.norm(w) * np.linalg.norm(wstar) + 1e-300)
        over.append(m)
        errs.append(math.acos(max(-1.0, min(1.0, cos))) / math.pi)
        if change < tol:
            break
    return np.array(over), np.array(errs), w


def instance(alpha, prior, seed):
    gen = rk.rng(seed)
    n = int(round(alpha * D))
    if prior == "ising":
        wstar = gen.choice([-1.0, 1.0], size=D)
    else:
        wstar = gen.standard_normal(D)
    F = gen.standard_normal((n, D)) / math.sqrt(D)
    y = np.sign(F @ wstar)
    return F.astype(np.float32), y, wstar


# ---------------------------------------------------------------------------------------------
# 1. Ising teacher: sweep in alpha
# ---------------------------------------------------------------------------------------------
alphas = [0.5, 0.8, 1.0, 1.2, 1.3, 1.4, 1.47, 1.55, 1.6, 1.8, 2.0]
SEEDS = 8
rows = []
traces = {}
print(f"Ising teacher, d = {D}, {SEEDS} instances per alpha, at most {TMAX} iterations")
for k, a in enumerate(alphas):
    errs_fin, over_fin, iters, exact = [], [], [], 0
    hist = []
    for s in range(SEEDS):
        F, y, ws = instance(a, "ising", 7000 + 100 * k + s)
        ov, er, w = gamp(F, y, "ising", ws)
        errs_fin.append(er[-1])
        over_fin.append(ov[-1])
        iters.append(len(ov) - 1)
        exact += int(np.all(np.sign(w) == ws))
        hist.append(ov)
    q_se = se(a, "ising", 400)
    e_se = math.acos(math.sqrt(min(q_se[-1], 1.0))) / math.pi
    me, he = rk.mean_ci(errs_fin)
    mq, hq = rk.mean_ci(over_fin)
    rows.append([a, me, he, mq, hq, e_se, q_se[-1], exact])
    print(f"  alpha={a:4.2f}: GAMP error {me:.4f} +- {he:.4f}, overlap {mq:.4f} +- {hq:.4f}, iterations {min(iters)}-{max(iters)},"
          f" exact recovery {exact}/{SEEDS};  SE fixed point: q={q_se[-1]:.5f}, error {e_se:.5f}")
    if a in (1.3, 1.47, 1.6):
        traces[a] = (hist, q_se)
rk.write_dat(os.path.join(DATA, "07-glm-gamp-alpha.dat"),
             ["alpha", "err", "errci", "over", "overci", "errSE", "qSE", "exact"], rows)

# iteration traces: mean overlap over the instances (padding finished runs with their last value)
T_tr = 60
trows = []
for t in range(T_tr + 1):
    row = [t]
    for a in (1.3, 1.47, 1.6):
        hist, q_se = traces[a]
        vals = [h[t] if t < len(h) else h[-1] for h in hist]
        m, hci = rk.mean_ci(vals)
        row += [m, hci, q_se[t]]
    trows.append(row)
rk.write_dat(os.path.join(DATA, "07-glm-gamp-iter.dat"),
             ["t", "m130", "ci130", "se130", "m147", "ci147", "se147", "m160", "ci160", "se160"], trows)
for a in (1.3, 1.47, 1.6):
    hist, q_se = traces[a]
    for t in (1, 2, 5, 10, 20, 40, 60):
        vals = [h[t] if t < len(h) else h[-1] for h in hist]
        print(f"  trace alpha={a}: t={t:3d} GAMP overlap {np.mean(vals):.4f} (min {np.min(vals):.4f}, max {np.max(vals):.4f}), SE {q_se[t]:.4f}")

# ---------------------------------------------------------------------------------------------
# 2. Spherical teacher
# ---------------------------------------------------------------------------------------------
print("Spherical teacher")
srows = []
for k, a in enumerate([0.5, 1.0, 2.0, 4.0, 8.0]):
    errs_fin = []
    for s in range(3):
        F, y, ws = instance(a, "sph", 9000 + 100 * k + s)
        ov, er, w = gamp(F, y, "sph", ws)
        errs_fin.append(er[-1])
    q_se = se(a, "sph", 400)[-1]
    e_se = math.acos(math.sqrt(q_se)) / math.pi
    me, he = rk.mean_ci(errs_fin)
    srows.append([a, me, he, e_se])
    print(f"  alpha={a:4.1f}: GAMP error {me:.4f} +- {he:.4f};  Bayes (SE fixed point) {e_se:.5f}")
rk.write_dat(os.path.join(DATA, "07-glm-gamp-sph.dat"), ["alpha", "err", "errci", "errSE"], srows)
print(f"runtime {time.time() - T0:.1f} s")
