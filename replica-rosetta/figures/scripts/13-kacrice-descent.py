"""Chapter 13 (kacrice): where gradient descent stops on the pure spherical 3-spin.

Projected gradient descent (step 0.15) on H(s) = N^{-1} sum_{ijk} J_ijk s_i s_j s_k on |s|^2 = N
(E H(s) H(s') = N R^3, the book's normalization), from uniform random starts, until
|grad H| < 1e-5 sqrt(N); K independent Hamiltonians x R starts per size (N = 25, 50, 75, 100, 200;
K = 10, 8, 6, 5, 4; R = 8, except R = 4 at N = 200, where the dense tensor makes a run cost about
20 seconds).  The 95% intervals are
Student-t intervals over the K per-Hamiltonian means.  At N = 100 the Riemannian Hessian at every endpoint,
divided by sqrt(p(p-1)), is compared with the semicircle on [-2 - t, 2 - t], t = 2u/E_inf, that the Gaussian
regression lemma predicts at a point of energy u, and with the threshold semicircle on [0, 4].

Writes figures/data/13-kacrice-gd.dat, 13-kacrice-hessian.dat and 13-kacrice-hessian-sc.dat.
Seeds 130 + N.  About two minutes on a laptop; BLAS limited to two threads.  The data files
13-kacrice-hessian*.dat are not plotted; the chapter quotes the printed Hessian summary.
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
EINF = 2 * math.sqrt((P - 1) / P)


def make_T(N, rng):
    G = rng.standard_normal((N, N, N))
    T = G.copy()
    for ax in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        T += G.transpose(ax)
    return T / 6.0


def descend(T, S, eta=0.15, maxsteps=8000, tol=1e-5):
    N = T.shape[0]
    T2 = T.reshape(N, N * N)
    R = S.shape[0]
    done = np.zeros(R, bool)
    for _ in range(maxsteps):
        Tss = ((S[:, :, None] * S[:, None, :]).reshape(R, N * N)) @ T2.T     # T(., s, s)
        G = 3.0 * Tss / N
        G -= (np.sum(G * S, 1) / N)[:, None] * S
        done |= np.linalg.norm(G, axis=1) / math.sqrt(N) < tol
        if done.all():
            break
        upd = ~done
        S[upd] -= eta * G[upd]
        S[upd] *= (math.sqrt(N) / np.linalg.norm(S[upd], axis=1))[:, None]
    E = np.sum(Tss * S, 1) / N / N
    return S, E, done


def hessian_spectrum(T, s):
    """Eigenvalues of the Riemannian Hessian of H at s (sphere of radius sqrt N), tangent directions only."""
    N = T.shape[0]
    Ts = (s @ T.reshape(N, N * N)).reshape(N, N)          # T(s, ., .)
    H = float(s @ Ts @ s) / N
    Pm = np.eye(N) - np.outer(s, s) / N
    M = Pm @ (6 * Ts / N - 3 * H / N * np.eye(N)) @ Pm + 1e3 * np.outer(s, s) / N
    return np.linalg.eigvalsh(M)[:-1], H / N


def main():
    t0 = time.time()
    rows, spec, energies = [], [], []
    plan = {25: (10, 8), 50: (8, 8), 75: (6, 8), 100: (5, 8), 200: (4, 4)}
    for N, (K, R) in plan.items():
        rng = rk.rng(130 + N)
        means, allE, unconv = [], [], 0
        for _ in range(K):
            T = make_T(N, rng)
            S = rng.standard_normal((R, N))
            S *= (math.sqrt(N) / np.linalg.norm(S, axis=1))[:, None]
            S, E, done = descend(T, S)
            unconv += int((~done).sum())
            means.append(float(E.mean()))
            allE.extend(E.tolist())
            if N == 100:
                for r in range(R):
                    ev, u = hessian_spectrum(T, S[r])
                    spec.append(ev / math.sqrt(P * (P - 1)))
                    energies.append(u)
        m, h = rk.mean_ci(means)
        rows.append([N, m, h, float(np.min(allE)), float(np.max(allE))])
        print(f"N={N}: gradient-descent endpoint energy {m:.4f} +- {h:.4f} (95% t over {K} Hamiltonians, {R} starts each);"
              f" range [{min(allE):.4f}, {max(allE):.4f}]; unconverged {unconv}  ({time.time()-t0:.1f}s)")
    rk.write_dat(os.path.join(OUT, "13-kacrice-gd.dat"), ["N", "energy", "half", "emin", "emax"], rows)
    ev = np.concatenate(spec)
    ubar = float(np.mean(energies))
    t = 2 * ubar / EINF
    edges = np.linspace(-0.5, 4.5, 51)
    hist, _ = np.histogram(ev, bins=edges, density=True)
    mids = 0.5 * (edges[1:] + edges[:-1])
    rk.write_dat(os.path.join(OUT, "13-kacrice-hessian.dat"), ["x", "density"],
                 [[float(a), float(b)] for a, b in zip(mids, hist)])
    fine = np.linspace(-0.5, 4.5, 401)
    sc = lambda y: np.sqrt(np.maximum(4 - y * y, 0.0)) / (2 * math.pi)
    rk.write_dat(os.path.join(OUT, "13-kacrice-hessian-sc.dat"), ["x", "energy", "threshold"],
                 [[float(a), float(sc(a + t)), float(sc(a - 2))] for a in fine])
    mins = np.array([s.min() for s in spec])
    print(f"Hessians at the N=100 endpoints ({len(spec)} minima): mean energy {ubar:.4f}, t = {t:.4f};"
          f" semicircle at that energy on [{-2 - t:.4f}, {2 - t:.4f}]; mean eigenvalue {ev.mean():.4f}"
          f" (predicted {-t:.4f}); smallest eigenvalue per minimum: median {np.median(mins):.4f},"
          f" range [{mins.min():.4f}, {mins.max():.4f}]; largest eigenvalue median"
          f" {np.median([s.max() for s in spec]):.4f}")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
