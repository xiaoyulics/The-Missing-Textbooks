#!/usr/bin/env python3
"""Chapter 6 (adaptive), figure fig:adaptive:regret.

Online logistic regression with sparse binary features whose frequencies follow a power law,
p_i = i^(-alpha).  At step t the learner sees x_t in {0,1}^d (feature i present with probability
p_i, independently), predicts with theta_t, and suffers f_t(theta) = log(1+exp(-y_t <x_t,theta>)),
where y_t = +-1 is drawn from a logistic teacher with weights +-1.  The feasible set is the box
[-1,1]^d (l_inf-diameter D_inf = 2, Euclidean diameter D_2 = 2 sqrt(d)).

Two learners, both started at 0 and both projected on the box by clipping:
  * diagonal AdaGrad with eta = D_inf/sqrt(2)                 (Theorem thm:adaptive:adagrad),
  * online gradient descent with the scalar adaptive step
    eta_t = D_2 / (sqrt(2) ||g_{1:t}||_F)                       (the one-coordinate case).
Regret is measured against the best fixed theta in the box in hindsight (L-BFGS-B).

The theory curves are the two regret bounds in expectation, with |g_{t,i}| <= x_{t,i} and Jensen:
  E R_AdaGrad <= sqrt(2) D_inf sum_i sqrt(p_i T),   E R_OGD <= sqrt(2) D_2 sqrt(T sum_i p_i).

Writes figures/data/06-adaptive-regret-sim.dat (simulation, means over seeds) and
figures/data/06-adaptive-regret-bounds.dat (bounds on a dense grid of d).
Deterministic seeds; about one minute on one core.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize, Bounds
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
T = 10000
R = 1.0                      # box [-R, R]^d
ALPHAS = (1.5, 2.0)
DIMS = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
SEEDS = (0, 1, 2)


def logloss(z):
    return np.logaddexp(0.0, -z)


def make_data(rng, d, alpha):
    p = np.minimum(1.0, np.arange(1, d + 1, dtype=float) ** (-alpha))
    rows, cols = [], []
    for i in range(d):                       # feature by feature: cheap for sparse features
        idx = np.nonzero(rng.random(T) < p[i])[0]
        rows.append(idx)
        cols.append(np.full(len(idx), i))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(T, d))
    teacher = rng.choice([-1.0, 1.0], d)
    y = np.where(rng.random(T) < expit(X @ teacher), 1.0, -1.0)
    return X, y, p


def online(X, y, method):
    """Cumulative loss of the online learner; only active coordinates move."""
    d = X.shape[1]
    th = np.zeros(d)
    s2 = np.zeros(d)
    tot = 0.0
    cum = 0.0
    eta_ada = 2 * R / np.sqrt(2.0)               # D_inf / sqrt(2)
    eta_ogd = 2 * R * np.sqrt(d) / np.sqrt(2.0)  # D_2 / sqrt(2)
    ip, ix = X.indptr, X.indices
    for t in range(T):
        idx = ix[ip[t]:ip[t + 1]]
        z = y[t] * th[idx].sum()
        cum += logloss(z)
        if len(idx) == 0:
            continue
        gv = -y[t] * expit(-z)                   # gradient entry on every active coordinate
        if method == "adagrad":
            s2[idx] += gv * gv
            th[idx] = np.clip(th[idx] - eta_ada * gv / np.sqrt(s2[idx]), -R, R)
        else:
            tot += gv * gv * len(idx)
            th[idx] = np.clip(th[idx] - eta_ogd * gv / np.sqrt(tot), -R, R)
    return cum


def hindsight(X, y):
    XT = X.T.tocsr()

    def f(th):
        z = y * (X @ th)
        return np.sum(logloss(z)), XT @ (-y * expit(-z))
    d = X.shape[1]
    res = minimize(f, np.zeros(d), jac=True, method="L-BFGS-B",
                   bounds=Bounds(-R * np.ones(d), R * np.ones(d)),
                   options={"maxiter": 20000, "ftol": 1e-13, "gtol": 1e-9})
    return res.fun


def bounds(d, alpha):
    p = np.minimum(1.0, np.arange(1, d + 1, dtype=float) ** (-alpha))
    b_ada = np.sqrt(2.0) * (2 * R) * np.sum(np.sqrt(p * T))
    b_ogd = np.sqrt(2.0) * (2 * R * np.sqrt(d)) * np.sqrt(T * p.sum())
    return b_ada, b_ogd


def main():
    sim_rows = []
    for d in DIMS:
        row = [d]
        for alpha in ALPHAS:
            ra, ro = [], []
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * seed + int(10 * alpha) + 7 * d)
                X, y, _ = make_data(rng, d, alpha)
                best = hindsight(X, y)
                ra.append(online(X, y, "adagrad") - best)
                ro.append(online(X, y, "ogd") - best)
            ma, mo = np.mean(ra), np.mean(ro)
            ba, bo = bounds(d, alpha)
            row += [ma, mo, mo / ma, bo / ba]
            print(f"d={d:6d} alpha={alpha}: regret AdaGrad {ma:8.1f}  OGD {mo:8.1f}  "
                  f"ratio {mo / ma:5.2f}  bound ratio {bo / ba:5.2f}", flush=True)
        sim_rows.append(row)
    with open(os.path.join(DATA, "06-adaptive-regret-sim.dat"), "w") as fh:
        fh.write("d regA15 regO15 ratio15 bratio15 regA20 regO20 ratio20 bratio20\n")
        for r in sim_rows:
            fh.write(" ".join(f"{v:.6g}" for v in r) + "\n")
    dense = np.unique(np.round(np.logspace(1, 4, 61)).astype(int))
    with open(os.path.join(DATA, "06-adaptive-regret-bounds.dat"), "w") as fh:
        fh.write("d bndA15 bndO15 bratio15 bndA20 bndO20 bratio20\n")
        for d in dense:
            vals = [d]
            for alpha in ALPHAS:
                ba, bo = bounds(d, alpha)
                vals += [ba, bo, bo / ba]
            fh.write(" ".join(f"{v:.6g}" for v in vals) + "\n")


if __name__ == "__main__":
    main()
