#!/usr/bin/env python3
"""Chapter 6 (adaptive), figure fig:adaptive:reddi.

Reddi's counterexample with period n: on Theta = [-1, 1] the gradient is C = 2(n-1) at steps
t = 1, n+1, 2n+1, ... and -1 at all other steps, so the minimizer of the average loss is -1.
Adam with beta1 = 0 converges to +1 (the worst point) when the drift per period
    kappa_n(beta2, C) = -C/sqrt(v_1) + sum_{k=2}^{n} 1/sqrt(v_k),
    v_k = 1 + rho beta2^(k-1) (C^2 - 1),  rho = (1-beta2)/(1-beta2^n),
is positive, and to -1 when it is negative (Proposition prop:adaptive:reddi).

Theory: the memory 1/(1-beta2) at which kappa = 0, for each n (exact, by root finding), and its
large-n asymptote n/(2 y), where y > 0 solves e^y - 1 = 2y.
Simulation: Adam as implemented in common libraries (beta1 = 0, bias correction, epsilon = 1e-8,
step 0.5/sqrt(t), projection on [-1, 1] by clipping), started at 0, on a grid of (n, memory);
a run counts as a failure if theta_T > 0 after T = max(2e5, 300 n) steps.

Writes figures/data/06-adaptive-reddi-boundary.dat and figures/data/06-adaptive-reddi-runs.dat.
Deterministic; about half a minute on one core.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def kappa(n, beta2, C):
    if beta2 == 0.0:
        v = np.array([C * C] + [1.0] * (n - 1))
    else:
        rho = (1 - beta2) / (1 - beta2 ** n)
        k = np.arange(1, n + 1)
        v = 1 + rho * beta2 ** (k - 1) * (C * C - 1)
    g = np.array([C] + [-1.0] * (n - 1))
    return -np.sum(g / np.sqrt(v))


def critical_memory(n, C):
    f = lambda lm: kappa(n, 1 - np.exp(-lm), C)      # lm = log(memory)
    return np.exp(brentq(f, np.log(1.0 + 1e-9), np.log(1e12), xtol=1e-12))


def adam_final(n, mems, C, T, eta0=0.5, eps=1e-8):
    beta2 = 1 - 1 / np.asarray(mems, float)
    th = np.zeros(len(beta2))
    v = np.zeros(len(beta2))
    pw = np.ones(len(beta2))
    for t in range(1, T + 1):
        g = C if (t - 1) % n == 0 else -1.0
        v = beta2 * v + (1 - beta2) * g * g
        pw = pw * beta2
        vhat = v / (1 - pw)
        th = np.clip(th - (eta0 / np.sqrt(t)) * g / (np.sqrt(vhat) + eps), -1.0, 1.0)
    return th


def main():
    y_star = brentq(lambda y: np.exp(y) - 1 - 2 * y, 0.1, 10.0)
    ns = np.unique(np.round(np.logspace(np.log10(3), np.log10(3000), 70)).astype(int))
    with open(os.path.join(DATA, "06-adaptive-reddi-boundary.dat"), "w") as fh:
        fh.write("n memcrit memasym\n")
        for n in ns:
            fh.write(f"{n} {critical_memory(n, 2.0 * (n - 1)):.6g} {n / (2 * y_star):.6g}\n")
    grid_n = (3, 5, 10, 20, 50, 100, 200, 500, 1000)
    mems = 10.0 ** (0.25 * np.arange(17))          # memory 1 (sign descent) to 1e4
    rows = []
    for n in grid_n:
        C = 2.0 * (n - 1)
        th = adam_final(n, mems, C, max(200000, 300 * n))
        mc = critical_memory(n, C)
        for m, x in zip(mems, th):
            pred = "fail" if m < mc else "ok"
            seen = "fail" if x > 0 else "ok"
            rows.append((n, m, x, m if seen == "ok" else np.nan, m if seen == "fail" else np.nan))
            if pred != seen:
                print(f"disagreement: n={n} memory={m:.3g} critical={mc:.3g} theta_T={x:+.3f}")
        print(f"n={n:5d}: critical memory {mc:9.3f}; theta_T =",
              " ".join(f"{x:+.2f}" for x in th), flush=True)
    with open(os.path.join(DATA, "06-adaptive-reddi-runs.dat"), "w") as fh:
        fh.write("n mem thetaT ok fail\n")
        for r in rows:
            fh.write(" ".join("nan" if (isinstance(v, float) and np.isnan(v)) else f"{v:.6g}"
                              for v in r) + "\n")
    print(f"asymptote: memory = n/(2y) = {1 / (2 * y_star):.4f} n, y = {y_star:.6f}")


if __name__ == "__main__":
    main()
