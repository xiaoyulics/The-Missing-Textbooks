"""Chapter 13 (kacrice): gradient descent on the likelihood landscape of tensor PCA (p = 3).

H_{N,lam}(s) = H_N(s) - lam N m(s)^3,  m(s) = <s, x*>/N,  |s|^2 = |x*|^2 = N,  H_N the pure 3-spin
(E H_N(s) H_N(s') = N R^3).  Minimizing H_{N,lam} is maximum likelihood.  For each lam we run projected
gradient descent (step 0.15 for lam <= 2 and 0.3/lam above, where the spike stiffens the landscape) from
  - uniform random starts (|m(s_0)| of order N^{-1/2}), and
  - warm starts with m(s_0) = 0.5 (the rest of s_0 uniform on the orthogonal sphere),
on K = 2 independent noise tensors (shared across lam) with R = 5 starts of each kind per tensor, and
record the final overlap (descent stops when |grad| < 1e-4 sqrt N or after 5000 steps).  The prediction is the overlap m*(lam) of the good critical point (largest root of
m^2 (1 - m^2) = 1/(3 lam^2), Ben Arous--Mei--Montanari--Nica Prop. 2), which for lam > lam_ML is also the
overlap of the maximum likelihood estimator (Jagannath--Lopatto--Miolane Thm 1.2); it exists for
lam >= lam_c = 2/sqrt 3.

Writes figures/data/13-kacrice-spikedgd.dat (lam, warm-start mean overlap with 95% t interval over
the warm runs, random-start success fraction m > 0.5 with Clopper--Pearson interval) and
figures/data/13-kacrice-mstar.dat (the curve m*(lam)).  N = 80, seed 1331.  The warm-start interval is a
Student-t interval over the 10 warm runs.  About a minute on a laptop.
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


def make_T(N, rng):
    G = rng.standard_normal((N, N, N))
    T = G.copy()
    for ax in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        T += G.transpose(ax)
    return T / 6.0


def descend(T, v, lam, S, eta=0.15, maxsteps=5000, tol=1e-4):
    N = T.shape[0]
    T2 = T.reshape(N, N * N)
    R = S.shape[0]
    done = np.zeros(R, bool)
    for _ in range(maxsteps):
        Tss = ((S[:, :, None] * S[:, None, :]).reshape(R, N * N)) @ T2.T
        m = S @ v / N
        G = 3.0 * Tss / N - 3.0 * lam * (m ** 2)[:, None] * v[None, :]
        G -= (np.sum(G * S, 1) / N)[:, None] * S
        done |= np.linalg.norm(G, axis=1) / math.sqrt(N) < tol
        if done.all():
            break
        upd = ~done
        S[upd] -= eta * G[upd]
        S[upd] *= (math.sqrt(N) / np.linalg.norm(S[upd], axis=1))[:, None]
    return S @ v / N, done


def m_star(lam):
    if lam < 2 / math.sqrt(3):
        return float("nan")
    return math.sqrt((1 + math.sqrt(1 - 4 / (3 * lam * lam))) / 2)


def main():
    t0 = time.time()
    N, K, R, m0 = 80, 2, 5, 0.5
    rng = rk.rng(1331)
    v = np.zeros(N)
    v[0] = math.sqrt(N)                        # x*; the noise is rotation invariant
    tensors = [make_T(N, rng) for _ in range(K)]
    lams = [0.6, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 4.0, 6.0]
    rows = []
    for lam in lams:
        warm_means, succ, n_rand, unconv = [], 0, 0, 0
        for T in tensors:
            S = rng.standard_normal((2 * R, N))
            S *= (math.sqrt(N) / np.linalg.norm(S, axis=1))[:, None]
            for r in range(R, 2 * R):
                w = S[r].copy()
                w[0] = 0.0
                w *= math.sqrt(N * (1 - m0 ** 2)) / np.linalg.norm(w)
                w[0] = m0 * math.sqrt(N)
                S[r] = w
            m, done = descend(T, v, lam, S, eta=0.15 if lam <= 2 else 0.3 / lam)
            unconv += int((~done).sum())
            warm_means.extend(m[R:].tolist())
            succ += int(np.sum(m[:R] > 0.5))
            n_rand += R
        wm, wh = rk.mean_ci(warm_means)
        lo, hi = rk.clopper_pearson(succ, n_rand)
        frac = succ / n_rand
        rows.append([lam, wm, wh, frac, frac - lo, hi - frac, m_star(lam) if lam >= 2 / math.sqrt(3) else -1])
        print(f"lam={lam:.2f}: warm-start overlap {wm:.4f} +- {wh:.4f} (runs {min(warm_means):.3f} to"
              f" {max(warm_means):.3f}, |m| <= {max(abs(x) for x in warm_means):.3f}); m* = {m_star(lam):.4f};"
              f" random starts reaching m > 0.5: {succ}/{n_rand}; unconverged {unconv}  ({time.time()-t0:.1f}s)")
    rk.write_dat(os.path.join(OUT, "13-kacrice-spikedgd.dat"),
                 ["lam", "warm", "warmhalf", "frac", "fraclo", "frachi", "mstar"], rows)
    curve = [[float(l), m_star(float(l))] for l in np.linspace(2 / math.sqrt(3) + 1e-12, 6.5, 300)]
    rk.write_dat(os.path.join(OUT, "13-kacrice-mstar.dat"), ["lam", "mstar"], curve)
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
