"""Chapter 13 (kacrice): the exact finite-N formula against a direct count of critical points.

For the pure spherical 3-spin at N = 3,...,7 we find every critical point of one random Hamiltonian by
Newton's method, started first from the endpoints of a gradient-descent sweep (which reaches the local
minima) and then from random starts, in batches, until three consecutive batches find nothing new; the
Morse identity sum_k (-1)^k Crt_k = 1 + (-1)^(N-1) is checked on every instance.  (Without the descent
sweep, random-start Newton missed about 6% of the minima at N = 7.)  The averages are compared with the
exact formula (Theorem thm:kacrice:exact),
    E Crt_{N,k}(R) = 2 sqrt(2/p) (p-1)^{N/2} E exp(-N (p-2) lambda_(k)^2 / (4p)),
evaluated by Monte Carlo over GOE(N) (book normalization; 400000 matrices per N).

Writes figures/data/13-kacrice-count.dat (N, formula and count for all critical points and for minima,
with 95% Student-t intervals over instances) and prints the index-by-index table.
Seeds: instances 1300 + N, GOE Monte Carlo 77.  About two minutes on a laptop.
"""
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
P = 3


def sym3(G):
    T = G.copy()
    for ax in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        T += G.transpose(ax)
    return T / 6.0


def newton_batch(T, S, rng, iters=45, X=None):
    """Critical points of F(x) = T(x,x,x) on the unit sphere from S random starts (or from the rows of X):
    Newton on T(x,x,.) - mu x = 0, (|x|^2 - 1)/2 = 0.  Returns the converged points."""
    N = T.shape[0]
    T2 = T.reshape(N, N * N)
    if X is None:
        X = rng.standard_normal((S, N))
    S = len(X)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Txx = (X @ T2).reshape(S, N, N)
    mu = np.einsum("si,sij,sj->s", X, Txx, X)
    I = np.eye(N)
    for _ in range(iters):
        Txx = (X @ T2).reshape(S, N, N)                       # T(x,.,.)
        Tx = np.einsum("sij,sj->si", Txx, X)                   # T(x,x,.)
        Fv = np.concatenate([Tx - mu[:, None] * X, 0.5 * (np.sum(X * X, 1) - 1)[:, None]], 1)
        Jm = np.zeros((S, N + 1, N + 1))
        Jm[:, :N, :N] = 2 * Txx - mu[:, None, None] * I
        Jm[:, :N, N] = -X
        Jm[:, N, :N] = X
        d = np.linalg.solve(Jm, -Fv[:, :, None])[:, :, 0]
        d /= np.maximum(1.0, np.linalg.norm(d, axis=1, keepdims=True) / 0.5)   # damp long steps
        X = X + d[:, :N]
        mu = mu + d[:, N]
    Txx = (X @ T2).reshape(S, N, N)
    Tx = np.einsum("sij,sj->si", Txx, X)
    res = np.linalg.norm(Tx - mu[:, None] * X, axis=1) + np.abs(np.sum(X * X, 1) - 1)
    ok = np.isfinite(res) & (res < 1e-10)
    X = X[ok] / np.linalg.norm(X[ok], axis=1, keepdims=True)
    return np.concatenate([X, -X])          # F is odd: -x is critical whenever x is


def classify(T, X):
    """Critical values and indices (number of negative eigenvalues of the Riemannian Hessian)."""
    N = T.shape[0]
    T2 = T.reshape(N, N * N)
    Txx = (X @ T2).reshape(len(X), N, N)
    F = np.einsum("si,sij,sj->s", X, Txx, X)
    P = np.eye(N)[None] - X[:, :, None] * X[:, None, :]
    H = P @ (6 * Txx - 3 * F[:, None, None] * np.eye(N)[None]) @ P + 1e3 * X[:, :, None] * X[:, None, :]
    ev = np.linalg.eigvalsh(H)
    return F, np.sum(ev < 0, axis=1)


def descent_starts(T, rng, S, steps=500, eta=0.05):
    """Endpoints of projected gradient descent on F from S random starts (near local minima)."""
    N = T.shape[0]
    T2 = T.reshape(N, N * N)
    X = rng.standard_normal((S, N))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for _ in range(steps):
        Tx = np.einsum("sij,sj->si", (X @ T2).reshape(S, N, N), X)
        F = np.sum(Tx * X, 1)
        X -= eta * 3 * (Tx - F[:, None] * X)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def all_critical_points(T, rng, batch, maxbatches=16):
    """Newton runs from the endpoints of a gradient-descent sweep (which finds the local minima, and by
    oddness the maxima) and then from random starts, in batches, until three consecutive batches find no
    new critical value and, for odd N, the Morse identity sum_k (-1)^k Crt_k = 2 holds (distinct
    critical points have distinct critical values almost surely)."""
    N = T.shape[0]
    chi = 1 + (-1) ** (N - 1)
    vals = np.empty(0)
    pts = np.empty((0, N))
    empty = 0
    for b in range(maxbatches):
        X = newton_batch(T, batch, rng, X=descent_starts(T, rng, batch) if b == 0 else None)
        if len(X):
            T2 = T.reshape(N, N * N)
            F = np.einsum("si,sij,sj->s", X, (X @ T2).reshape(len(X), N, N), X)
            o = np.argsort(F)
            F, X = F[o], X[o]
            keep = np.r_[True, np.diff(F) > 1e-8]
            F, X = F[keep], X[keep]
            if len(vals):
                pos = np.searchsorted(vals, F)
                dist = np.minimum(np.abs(F - vals[np.clip(pos - 1, 0, len(vals) - 1)]),
                                  np.abs(F - vals[np.clip(pos, 0, len(vals) - 1)]))
                new = dist > 1e-8
            else:
                new = np.ones(len(F), bool)
        else:
            new = np.zeros(0, bool)
        if new.any():
            vals = np.concatenate([vals, F[new]])
            pts = np.concatenate([pts, X[new]])
            o = np.argsort(vals)
            vals, pts = vals[o], pts[o]
            empty = 0
        else:
            empty += 1
        if empty >= 3:
            if N % 2 == 0 or np.sum((-1.0) ** classify(T, pts)[1]) == chi:
                break
    F, idx = classify(T, pts)
    return F, idx


def formula(N, M, rng):
    """E Crt_{N,k}(R), k = 0..N-1, by Monte Carlo over GOE(N), with Monte Carlo standard errors."""
    a = (P - 2) / (4 * P)
    pref = 2 * math.sqrt(2 / P) * (P - 1) ** (N / 2)
    vals = []
    for chunk in range(M // 50000):
        G = rng.standard_normal((50000, N, N)) / math.sqrt(N)
        ev = np.linalg.eigvalsh((G + G.transpose(0, 2, 1)) / math.sqrt(2))
        vals.append(np.exp(-N * a * ev ** 2))
    W = pref * np.concatenate(vals)                  # (M, N): weights of the ordered eigenvalues
    return W.mean(0), W.std(0) / math.sqrt(len(W)), W.sum(1).mean(), W.sum(1).std() / math.sqrt(len(W))


def main():
    t0 = time.time()
    rows = []
    reps = {3: 200, 4: 150, 5: 120, 6: 100, 7: 80}
    for N in (3, 4, 5, 6, 7):
        rng = rk.rng(1300 + N)
        tot, mins, byk, bad = [], [], np.zeros(N), 0
        for r in range(reps[N]):
            T = sym3(rng.standard_normal((N, N, N)))
            F, idx = all_critical_points(T, rng, batch=40 * N)
            if np.sum((-1.0) ** idx) != 1 + (-1) ** (N - 1):
                bad += 1
            tot.append(len(F))
            mins.append(int(np.sum(idx == 0)))
            byk += np.bincount(idx, minlength=N)[:N]
        fk, fk_se, ft, ft_se = formula(N, 400000, rk.rng(77))
        mt, ht = rk.mean_ci(tot)
        mm, hm = rk.mean_ci(mins)
        print(f"N={N}: {reps[N]} instances, Morse failures {bad}, max count {max(tot)} (bound {2*(2**N-1)})")
        print(f"   all critical points: formula {ft:.3f} +- {ft_se:.3f} (MC s.e.), count {mt:.3f} +- {ht:.3f}")
        print(f"   minima:              formula {fk[0]:.4f} +- {fk_se[0]:.4f},       count {mm:.4f} +- {hm:.4f}")
        print("   by index k: formula", " ".join(f"{v:.3f}" for v in fk), "| count", " ".join(f"{v:.3f}" for v in byk / reps[N]))
        rows.append([N, ft, fk[0], mt, ht, mm, hm])
        print(f"   ({time.time()-t0:.1f}s)")
    rk.write_dat(os.path.join(OUT, "13-kacrice-count.dat"),
                 ["N", "formula_all", "formula_min", "count_all", "half_all", "count_min", "half_min"], rows)


if __name__ == "__main__":
    main()
