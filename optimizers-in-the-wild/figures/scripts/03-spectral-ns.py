"""Chapter 3 figure: what k Newton-Schulz-type iterations do to the singular values of a matrix.

Three odd-polynomial iterations X <- a X + b (X X^T) X + c (X X^T)^2 X, started at X_0 = G / ||G||_F:
  cubic   (a, b, c) = (3/2, -1/2, 0) at every step (the classical Newton-Schulz iteration for the polar factor);
  muon    (a, b, c) = (3.4445, -4.7750, 2.0315) at every step (Muon's quintic, from the reference code
          github.com/KellerJordan/Muon, muon.py);
  pe      Polar Express: a different quintic at each step, computed by the recipe of the authors' code
          (github.com/NoahAmsel/PolarExpress, polar_express.py): minimax odd quintic on [l, u] by a
          simplified Remez algorithm, l = 1e-3, safety factor 1.01, cushion 0.02; the input is divided by
          1.01 ||G||_F, as in that code. Re-implemented here from the published description.

Theory (curves): the composition p_k o ... o p_1 applied to the normalized singular value s = sigma/||G||_F
(divided by 1.01 for Polar Express), for k = 1, ..., 5. By the odd-polynomial lemma of the chapter, the k-th
iterate is U diag(p_k o ... o p_1 (s_i)) V^T exactly.

Simulation (points): six test matrices of size 64 x 128 with prescribed singular values and Haar-random
singular vectors (seed 3), iterated in bfloat16 exactly as in Muon's code (cast to bfloat16, divide by the
Frobenius norm + 1e-7, then A = X X^T, B = b A + c A A, X = a X + B X), for k = 1..5 steps. The point plotted for
the singular pair (u_i, v_i) of G is u_i^T X_k v_i. The same iteration in float64 is also run, and its maximal
deviation from the curves is printed (it is of order 1e-7, the size of the 1e-7 in the normalization).

Output: figures/data/03-spectral-ns-curves.dat (s, cub1..cub5, muon1..muon5, pe1..pe5)
        figures/data/03-spectral-ns-points.dat (s, the same columns, bfloat16 measurements)
"""
import os
from math import inf, sqrt

import numpy as np
import torch

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def optimal_quintic(l, u):
    """Minimax odd quintic a x + b x^3 + c x^5 approximating 1 on [l, u] (simplified Remez)."""
    if 1 - 5e-6 <= l / u:
        return (15 / 8) / u, (-10 / 8) / u ** 3, (3 / 8) / u ** 5
    q, r = (3 * l + u) / 4, (l + 3 * u) / 4
    E, old_E = inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        M = np.array([[l, l ** 3, l ** 5, 1], [q, q ** 3, q ** 5, -1],
                      [r, r ** 3, r ** 5, 1], [u, u ** 3, u ** 5, -1]])
        a, b, c, E = np.linalg.solve(M, np.ones(4))
        q, r = np.sqrt((-3 * b + np.array([-1, 1]) * sqrt(9 * b ** 2 - 20 * a * c)) / (10 * c))
    return float(a), float(b), float(c)


def polar_express_coefficients(l=1e-3, num_iters=10, safety_eps=1e-2, cushion=0.02):
    u, safety, out = 1.0, 1 + safety_eps, []
    for it in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion * u), u)
        p = np.polynomial.Polynomial([0, a, 0, b, 0, c])
        if cushion * u > l:
            p *= 2 / (p(l) + p(u))
        if it < num_iters - 1:
            p = p(np.polynomial.Polynomial([0, 1 / safety]))
        out.append(tuple(float(x) for x in p.coef[1::2]))
        l = p(l)
        u = 2 - l
    return out


METHODS = {
    "cub": ([(1.5, -0.5, 0.0)] * 5, 1.0),
    "muon": ([(3.4445, -4.7750, 2.0315)] * 5, 1.0),
    "pe": (polar_express_coefficients()[:5], 1.01),
}


def compose(coeffs, s):
    out = []
    for a, b, c in coeffs:
        s = a * s + b * s ** 3 + c * s ** 5
        out.append(s.copy())
    return out


def haar(k, rng):
    Q, R = np.linalg.qr(rng.standard_normal((k, k)))
    return Q * np.sign(np.diag(R))


def test_matrices(rng, m=64, n=128):
    """Matrices whose normalized singular values cover [1e-4, 1]; returns (G, U, V, s, idx) with the
    designated singular values s (normalized by ||G||_F) and their positions idx."""
    tiny_all = np.logspace(-4, np.log10(0.06), 30)
    mids = [None, 0.12, 0.25, 0.4, 0.55, 0.7]
    mats = []
    for j, mid in enumerate(mids):
        tiny = tiny_all[j::len(mids)]
        vals = list(tiny) + ([mid] if mid is not None else [])
        filler = sqrt(1.0 - sum(v * v for v in vals))
        vals = [filler] + vals
        sig = np.zeros(m)
        sig[:len(vals)] = vals
        U = haar(m, rng)
        V = haar(n, rng)[:, :m]
        G = (U * sig) @ V.T
        mats.append((G, U, V, np.array(vals) / np.linalg.norm(G), np.arange(len(vals))))
    return mats


def iterate(G, coeffs, pre, dtype):
    if dtype == "bf16":
        X = torch.tensor(G, dtype=torch.float32).bfloat16()
    else:
        X = torch.tensor(G, dtype=torch.float64)
    X = X / (X.norm() * pre + 1e-7)
    out = []
    for a, b, c in coeffs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        out.append(X.double().numpy())
    return out


def main():
    s_grid = np.logspace(-4, 0, 241)
    cols = ["s"]
    table = [s_grid]
    for name, (coeffs, pre) in METHODS.items():
        for k, y in enumerate(compose(coeffs, s_grid / pre), 1):
            cols.append(f"{name}{k}")
            table.append(y)
    with open(os.path.join(DATA, "03-spectral-ns-curves.dat"), "w") as f:
        f.write(" ".join(cols) + "\n")
        for row in np.array(table).T:
            f.write(" ".join(f"{v:.6e}" for v in row) + "\n")

    rng = np.random.default_rng(3)
    mats = test_matrices(rng)
    s_all = np.concatenate([m[3] for m in mats])
    order = np.argsort(s_all)
    pts = {"s": s_all[order]}
    for name, (coeffs, pre) in METHODS.items():
        meas = {"bf16": [[] for _ in range(5)], "f64": [[] for _ in range(5)]}
        for G, U, V, s, idx in mats:
            for dtype in ("bf16", "f64"):
                Xs = iterate(G, coeffs, pre, dtype)
                for k in range(5):
                    d = np.einsum("ij,ik,kj->j", U[:, idx], Xs[k], V[:, idx])
                    meas[dtype][k].append(d)
        theory = compose(coeffs, s_all / pre)
        for k in range(5):
            bf = np.concatenate(meas["bf16"][k])
            f64 = np.concatenate(meas["f64"][k])
            pts[f"{name}{k + 1}"] = bf[order]
            print(f"{name} k={k + 1}: max |float64 - theory| = {np.abs(f64 - theory[k]).max():.1e}, "
                  f"max |bfloat16 - theory| = {np.abs(bf - theory[k]).max():.1e}")
    with open(os.path.join(DATA, "03-spectral-ns-points.dat"), "w") as f:
        f.write(" ".join(pts.keys()) + "\n")
        for row in np.array(list(pts.values())).T:
            f.write(" ".join(f"{v:.6e}" for v in row) + "\n")
    print("Polar Express coefficients used:")
    for cf in METHODS["pe"][0]:
        print("   (%.4f, %.4f, %.4f)" % cf)


if __name__ == "__main__":
    main()
