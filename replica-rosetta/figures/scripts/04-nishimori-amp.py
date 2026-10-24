"""Chapter 4 (nishimori): Bayes-AMP for Z2 synchronization at N = 2000 against the MMSE curve.

Model (book normalization): A = (lam/N) x* x*^T + W, W ~ GOE(N), x* uniform on {-1,+1}^N.
Bayes-AMP (Chapter 6; Montanari--Venkataramanan 2021, Sec. 2.4), with m^k = tanh(h^k), Q(m) = |m|^2/N,
    h^{k+1} = lam A m^k - lam^2 (1 - Q(m^k)) m^{k-1}.
Initialization for lam > 1: spectral, h^0 = sqrt(N lam^2 (lam^2 - 1)) v_1 (top eigenvector, unit norm) and
the virtual previous iterate m^{-1} = h^0 / lam^2 (digest 02: the convention under which the iterates
follow state evolution from the first step).  For lam <= 1 the top eigenvector carries no information;
there AMP starts from h^0 = 0.1 g with g ~ N(0, I_N) independent of A, and m^{-1} = 0.
State evolution: h^k ~ gamma_k x* + sqrt(gamma_k) G, gamma_{k+1} = lam^2 E tanh(gamma_k + sqrt(gamma_k) Z),
gamma_0 = lam^2 - 1 (spectral start); its fixed point is gamma* = lam^2 q* with q* the RS overlap.
The global sign of the estimate is irrelevant: the script reports the matrix squared error of
xhat xhat^T with xhat = m^K (K = 40 iterations),
    MSE = (1/N^2) |x* x*^T - xhat xhat^T|_F^2 = 1 - 2 <xhat, x*>^2/N^2 + |xhat|^4/N^2,
and, for comparison, that of the spectral estimator (1 - 1/lam^2) N v_1 v_1^T, whose limit is the
proved curve 1 - (1 - 1/lam^2)^2 (BBP, Chapter 5).  The prediction for AMP is 1 - q*^2, the MMSE
(a theorem for every lam != 1: Deshpande--Abbe--Montanari 2016; Montanari--Venkataramanan 2021).

Writes figures/data/04-nishimori-amp.dat (lam, mean and 95% half-width of the AMP and spectral MSE,
RS prediction).  Seeds 4000 + 100 k + r for the k-th lam and repetition r.  Runtime: about 80 s.
Run from the volume root:  python figures/scripts/04-nishimori-amp.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk

OUT = os.path.join("figures", "data")
N = 2000
LAMS = [0.6, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0]
REPS = 10
K = 40


def g(gam):
    return 0.0 if gam <= 0 else rk.gauss(lambda z: np.tanh(gam + math.sqrt(gam) * z), 201)


def qstar(lam):
    return 0.0 if lam <= 1 else rk.root(lambda q: g(lam * lam * q) - q, 1e-12, 1.0)


def matrix_mse(xhat, xs):
    n = xs.size
    return 1.0 - 2.0 * float(xhat @ xs) ** 2 / n ** 2 + float(xhat @ xhat) ** 2 / n ** 2


def run(lam, seed):
    gen = rk.rng(seed)
    xs = gen.choice([-1.0, 1.0], size=N)
    A = rk.spiked_wigner(xs, lam, gen)
    val, vec = eigsh(A, k=1, which="LA", tol=1e-10, v0=np.ones(N) / math.sqrt(N))
    v = vec[:, 0]
    spec_scale = max(1 - 1 / lam ** 2, 0.0)
    mse_spec = matrix_mse(math.sqrt(spec_scale * N) * v, xs)
    if lam > 1:
        h = math.sqrt(N * lam ** 2 * (lam ** 2 - 1)) * v
        m_prev = h / lam ** 2
    else:
        h = 0.1 * gen.standard_normal(N)
        m_prev = np.zeros(N)
    traj = []
    for _ in range(K):
        m = np.tanh(h)
        traj.append(float(m @ xs) / N)
        h, m_prev = lam * (A @ m) - lam ** 2 * (1 - float(m @ m) / N) * m_prev, m
    m = np.tanh(h)
    return matrix_mse(m, xs), mse_spec, float(val[0]), abs(float(m @ xs)) / N, traj


def main():
    t0 = time.time()
    rows = []
    for k, lam in enumerate(LAMS):
        q = qstar(lam)
        res = [run(lam, 4000 + 100 * k + r) for r in range(REPS)]
        amp = [r[0] for r in res]
        spec = [r[1] for r in res]
        ma, ha = rk.mean_ci(amp)
        ms, hs = rk.mean_ci(spec)
        top = np.mean([r[2] for r in res])
        ov = np.mean([r[3] for r in res])
        rows.append((lam, ma, ha, ms, hs, 1 - q * q, 1 - max(1 - 1 / lam ** 2, 0) ** 2))
        print(f"lam={lam:4.2f}: AMP MSE={ma:.4f}+-{ha:.4f}  (RS 1-q*^2={1 - q * q:.4f})  |<m,x>|/N={ov:.4f} (q*={q:.4f})"
              f"   spectral MSE={ms:.4f}+-{hs:.4f} (limit {1 - max(1 - 1 / lam ** 2, 0) ** 2:.4f})  top eig={top:.4f}")
        if lam == 1.5:
            # state evolution from the spectral start against one run's overlaps
            gam = lam ** 2 - 1
            se = []
            for _ in range(8):
                se.append(g(gam))      # E[x tanh(h)] = g(gamma) is the overlap of m^k with the truth
                gam = lam ** 2 * g(gam)
            print("   SE overlaps     ", " ".join(f"{s:.4f}" for s in se))
            print("   AMP overlaps (r0)", " ".join(f"{s:.4f}" for s in res[0][4][:8]))
    rk.write_dat(os.path.join(OUT, "04-nishimori-amp.dat"),
                 ["lam", "amp", "ampci", "spec", "specci", "rs", "speclim"], rows)
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
