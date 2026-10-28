"""Chapter 14 (dmft): direct simulation of Langevin dynamics and gradient flow on the spherical 3-spin model.

Book normalization: H(s) = N^{-1} sum_{ijk} G_ijk s_i s_j s_k with i.i.d. N(0,1) entries, so that
E H(s) H(s') = N (s.s'/N)^3; dynamics ds = -grad H dt + sqrt(2T) dB projected on |s|^2 = N, start uniform on
the sphere. One step of length dt: a Heun step for the drift and a Gaussian increment for the noise, then
renormalization to the sphere. Renormalizing shrinks both increments by (1 + a^2)^{-1/2}, where a^2 is the noise
variance per coordinate, which would lower the temperature by a factor of about 1 - T dt; the step therefore uses
a^2 = 2T dt/(1 - 2T dt) and the drift step dt/sqrt(1 - 2T dt), so that the increments after renormalization have
the right size. (Without this correction the late-time energy at T = 1.2 is biased downward by a few
hundredths at dt = 0.025.)
For each disorder sample the three temperatures T = 1.2, 0.7, 0 share the disorder and the starting point and
use independent noises. N = 100 with 12 samples (seeds 1430 + k) and N = 50 with 24 samples (seeds 2440 + k).
Compared in the chapter with the CK equations integrated by 14-dmft-ck.py.
Writes figures/data/14-dmft-pspin-sim.dat.
"""
import math
import os
import sys
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")   # the machine is shared: two BLAS threads avoid oversubscription
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")
TEMPS = np.array([1.2, 0.7, 0.0])
DT, TMAX = 0.025, 20.0
TS = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0)


def run_sample(N, seed):
    gen = rk.rng(seed)
    G = gen.standard_normal((N, N, N))
    Gs = (G + G.transpose(1, 0, 2) + G.transpose(2, 0, 1)).reshape(N, N * N)   # grad_i = (1/N) Gs (s x s)
    del G
    K = len(TEMPS)

    def grad(S):
        return Gs @ np.einsum("ik,jk->ijk", S, S).reshape(N * N, K) / N

    s0 = gen.standard_normal(N)
    S = np.repeat(s0[:, None], K, axis=1)
    S *= math.sqrt(N) / np.linalg.norm(S, axis=0)
    rec = {int(round(t / DT)): t for t in TS}
    out = np.zeros((len(TS), K))
    j = 0
    amp = np.sqrt(2 * TEMPS * DT / (1 - 2 * TEMPS * DT))      # noise per coordinate, corrected for renormalization
    hd = DT / np.sqrt(1 - 2 * TEMPS * DT)                       # drift step, corrected likewise
    for k in range(1, int(round(TMAX / DT)) + 1):
        g = grad(S)
        Sp = S - hd * g
        Sp *= math.sqrt(N) / np.linalg.norm(Sp, axis=0)
        S = S - 0.5 * hd * (g + grad(Sp)) + amp * gen.standard_normal((N, K))
        S *= math.sqrt(N) / np.linalg.norm(S, axis=0)
        if k in rec:
            out[j] = np.sum(S * grad(S), axis=0) / (3 * N)     # Euler: <s, grad H> = 3 H
            j += 1
    return out


def main():
    t0 = time.time()
    # finite-difference check of the gradient at N = 30
    gen = rk.rng(1)
    n = 30
    G = gen.standard_normal((n, n, n))
    Gs = (G + G.transpose(1, 0, 2) + G.transpose(2, 0, 1)).reshape(n, n * n)
    Hf = lambda s: np.einsum("ijk,i,j,k->", G, s, s, s) / n
    s, d = gen.standard_normal(n), gen.standard_normal(n)
    fd = (Hf(s + 1e-5 * d) - Hf(s - 1e-5 * d)) / 2e-5
    print("gradient check: finite difference %.8f, formula %.8f" % (fd, float(Gs @ np.kron(s, s) @ d) / n))
    results = {}
    for N, reps, base in ((100, 12, 1430), (50, 24, 2440)):
        t1 = time.time()
        res = np.array([run_sample(N, base + k) for k in range(reps)])
        results[N] = res
        print("N=%d, %d samples (%.1fs)" % (N, reps, time.time() - t1))
        for i, t in enumerate(TS):
            line = []
            for a, T in enumerate(TEMPS):
                m, h = rk.mean_ci(res[:, i, a])
                line.append("T=%.1f: %.4f +- %.4f" % (T, m, h))
            print("  t=%5.1f  %s" % (t, " | ".join(line)))
    rows = []
    for i, t in enumerate(TS):
        row = [t]
        for N in (100, 50):
            for a in range(len(TEMPS)):
                m, h = rk.mean_ci(results[N][:, i, a])
                row += [m, h]
        rows.append(row)
    rk.write_dat(os.path.join(DATA, "14-dmft-pspin-sim.dat"),
                 ["t", "n100e12", "n100c12", "n100e07", "n100c07", "n100e00", "n100c00",
                  "n50e12", "n50c12", "n50e07", "n50c07", "n50e00", "n50c00"], rows)
    print("runtime %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
