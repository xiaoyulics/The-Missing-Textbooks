"""Figure 2 of Chapter 1 (Double Descent): test error of ridge regression in the
proportional limit (ridge theorem and optimal-ridge proposition) versus gamma = p/n, for
several ridge parameters and for the optimal one, compared with simulations at
n = 200.

Model: x ~ N(0, I_p), y = <w_star, x> + noise, ||w_star||^2 = 1, sigma^2 = 0.25.
Estimator: w_lambda = (X^T X + n lambda I)^{-1} X^T y (lambda = 0: minimum norm).
Theory: test error = sigma^2 + r^2 (v - lambda v') + sigma^2 gamma v', where v is
the positive root of gamma v^2 + (1 - gamma + lambda) v - lambda = 0.

Writes figures/data/01-doubledescent-ridge-theory.dat and
figures/data/01-doubledescent-ridge-mc.dat. Deterministic; about 20 s on a laptop.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # small matrices: threads only add overhead
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
R2, S2 = 1.0, 0.25
LAMBDAS = [0.01, 0.1]
N = 200
TRIALS = 500

def v_root(lam, g):
    b = 1 - g + lam
    return (np.sqrt(b * b + 4 * g * lam) - b) / (2 * g)

def v_prime(lam, g):
    v = v_root(lam, g)
    return (1 - v) / (2 * g * v + 1 - g + lam)

def ridge_test_error(lam, g):
    if lam == 0:
        if abs(g - 1) < 1e-12:
            return np.inf
        return S2 + (S2 * g / (1 - g) if g < 1 else R2 * (1 - 1 / g) + S2 / (g - 1))
    v, vp = v_root(lam, g), v_prime(lam, g)
    return S2 + R2 * (v - lam * vp) + S2 * g * vp

def optimal_test_error(g):
    return ridge_test_error(S2 * g / R2, g)

def mc(rng, p, lam, trials):
    w = np.zeros(p); w[0] = np.sqrt(R2)
    e = np.empty(trials)
    for t in range(trials):
        X = rng.standard_normal((N, p))
        y = X @ w + np.sqrt(S2) * rng.standard_normal(N)
        if p <= N:
            wh = np.linalg.solve(X.T @ X + N * lam * np.eye(p), X.T @ y)
        else:
            wh = X.T @ np.linalg.solve(X @ X.T + N * lam * np.eye(N), y)
        e[t] = S2 + np.sum((wh - w) ** 2)
    return e.mean(), e.std(ddof=1) / np.sqrt(trials)

def fmt(v):
    return "nan" if not np.isfinite(v) else f"{v:.6g}"

def main():
    rng = np.random.default_rng(1017)
    gs = np.round(np.arange(0.02, 4.0001, 0.02), 4)
    with open(os.path.join(DATA, "01-doubledescent-ridge-theory.dat"), "w") as f:
        f.write("gamma ridgeless " + " ".join(f"lam{l:g}".replace(".", "p") for l in LAMBDAS) + " optimal\n")
        for g in gs:
            vals = [ridge_test_error(0.0, g)] + [ridge_test_error(l, g) for l in LAMBDAS] + [optimal_test_error(g)]
            f.write(f"{g:g} " + " ".join(fmt(v) for v in vals) + "\n")
    grid = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    with open(os.path.join(DATA, "01-doubledescent-ridge-mc.dat"), "w") as f:
        f.write("gamma lam0p01 lam0p01se lam0p1 lam0p1se optimal optimalse\n")
        for g in grid:
            p = int(round(g * N))
            row = [g]
            for lam in (0.01, 0.1, S2 * p / (N * R2)):   # the last one is lambda* = sigma^2 p/(n r^2)
                m, se = mc(rng, p, lam, TRIALS)
                row += [m, se]
            f.write(" ".join(fmt(v) for v in row) + "\n")
            th = [ridge_test_error(0.01, g), ridge_test_error(0.1, g), optimal_test_error(g)]
            print(f"gamma={g}: " + "  ".join(f"{t:.4f} vs {row[1+2*i]:.4f}+-{row[2+2*i]:.4f}" for i, t in enumerate(th)))

if __name__ == "__main__":
    main()
