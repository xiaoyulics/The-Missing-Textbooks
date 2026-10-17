"""Figure of Chapter 6 (Lazy Training): how far the weights and the kernel move, versus width.

Same training set and network as 06-lazy-modes.py (n = 32 equally spaced points on the
circle with a bias coordinate, tanh, first layer trained, antisymmetric initialization so
that f(.; W_0) = 0), with labels y_i = cos t_i + cos 2t_i, trained by full-batch gradient
descent with step eta = 0.02 for time T = 25 (1250 steps).

Measured, for widths m = 2^5, ..., 2^14 and 4 seeds per width (mean reported):
  relW  = ||W_T - W_0||_F / ||W_0||_F       (relative weight change)
  dW    = ||W_T - W_0||_F                   (absolute weight change)
  maxw  = max_r ||w_r(T) - w_r(0)||          (largest move of one neuron)
  relK  = ||K_T - K_0||_op / ||K_0||_op     (relative change of the tangent kernel on the training set)
  loss  = ||r_T|| / ||r_0||
Theory: the linearized model with the infinite-width kernel K_inf, run by the same gradient
descent, moves its parameters by D = sqrt( sum_k c_k^2 (1 - (1 - eta lam_k)^s)^2 / lam_k ),
c_k^2 = ||projection of y on the eigenspace k||^2 (the RKHS norm of the fitted part of y),
and E||W_0||_F^2 = 3m, so the predicted relative change is D / sqrt(3m).

Writes figures/data/06-lazy-width.dat. Deterministic (seeds 0..3); about 40 seconds.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N, ETA, T = 32, 0.02, 25.0
FREQS = [1, 2]
WIDTHS = [2 ** j for j in range(5, 15)]
SEEDS = range(4)

z, wq = np.polynomial.hermite_e.hermegauss(80)
wq = wq / wq.sum()

def dtanh(u):
    return 1.0 / np.cosh(u) ** 2

def kappa(rho):
    s = np.sqrt(max(1.0 - rho * rho, 0.0))
    return float(np.sum(wq[:, None] * wq[None, :] * dtanh(z[:, None]) * dtanh(rho * z[:, None] + s * z[None, :])))

theta = 2 * np.pi * np.arange(N) / N
Xt = np.stack([np.cos(theta), np.sin(theta), np.ones(N)], 1) / np.sqrt(2)
G = Xt @ Xt.T
Kinf = G * np.vectorize(kappa)(G)
y = sum(np.cos(k * theta) for k in FREQS)
STEPS = int(round(T / ETA))

def kernel(W, a, m):
    D = dtanh(Xt @ W.T)                       # n x m; a_r^2 = 1
    return G * (D @ D.T) / m

def run(m, seed):
    rng = np.random.default_rng(seed)
    h = m // 2
    W = rng.standard_normal((h, 3))
    W = np.vstack([W, W])
    a = np.r_[np.ones(h), -np.ones(h)]
    W0 = W.copy()
    K0 = kernel(W, a, m)
    for _ in range(STEPS):
        P = Xt @ W.T
        r = (np.tanh(P) @ a) / np.sqrt(m) - y
        W -= ETA * (((r[:, None] * dtanh(P)) * a[None, :]).T @ Xt) / np.sqrt(m)
    r = (np.tanh(Xt @ W.T) @ a) / np.sqrt(m) - y
    KT = kernel(W, a, m)
    dW = np.linalg.norm(W - W0)
    return dict(relW=dW / np.linalg.norm(W0), dW=dW, maxw=np.max(np.linalg.norm(W - W0, axis=1)),
                relK=np.linalg.norm(KT - K0, 2) / np.linalg.norm(K0, 2), loss=np.linalg.norm(r) / np.linalg.norm(y))

def theory_displacement():
    ev, V = np.linalg.eigh(Kinf)
    c = V.T @ y
    keep = c ** 2 > 1e-20 * (y @ y)
    return np.sqrt(np.sum(c[keep] ** 2 * (1 - (1 - ETA * ev[keep]) ** STEPS) ** 2 / ev[keep]))

def main():
    D = theory_displacement()
    print("predicted displacement D =", D, " (infinite time: %.4f)" % np.sqrt(y @ np.linalg.solve(Kinf, y)))
    keys = ["relW", "dW", "maxw", "relK", "loss"]
    with open(os.path.join(DATA, "06-lazy-width.dat"), "w") as fh:
        fh.write("m " + " ".join(keys) + " relWtheory\n")
        for m in WIDTHS:
            res = [run(m, s) for s in SEEDS]
            mean = {k: np.mean([r[k] for r in res]) for k in keys}
            fh.write("%d " % m + " ".join("%.6e" % mean[k] for k in keys) + " %.6e\n" % (D / np.sqrt(3 * m)))
            print(m, " ".join("%s=%.4g" % (k, mean[k]) for k in keys), "theory relW=%.4g" % (D / np.sqrt(3 * m)))

if __name__ == "__main__":
    main()
