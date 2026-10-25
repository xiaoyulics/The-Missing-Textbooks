"""Chapter 14 (dmft): zero-temperature gradient flow on the spherical 2-spin model (Theorem 14.3).

H(s) = -(1/sqrt 2) <s, W s> on the sphere |s|^2 = N, W ~ GOE(N), so E H(s)H(s') = N (s.s'/N)^2 (the book's
normalization, xi(q) = q^2). The spherical gradient flow from s_0 uniform is s_t = sqrt(N) e^{sqrt2 t W} s_0 / |...|.
Limits (Theorem 14.3), with Phi(u) = int e^{u x} sigma_sc(dx) = I_1(2u)/u:
   C(t, s) = Phi(sqrt2 (t + s)) / sqrt(Phi(2 sqrt2 t) Phi(2 sqrt2 s)),   e(t) = -Phi'(2 sqrt2 t) / (sqrt2 Phi(2 sqrt2 t)),
   aging: C(t_w (1 + x), t_w) -> (4 (1 + x) / (2 + x)^2)^{3/4} as t_w -> infinity.
Simulation: N = 1500, 10 independent GOE matrices and starting points (seeds 1420 + k); every quantity is an
explicit function of the eigendecomposition. Writes figures/data/14-dmft-twospin-*.dat.
"""
import math
import os
import sys
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")   # the machine is shared: two BLAS threads avoid oversubscription
import numpy as np  # noqa: E402
from scipy import integrate, special  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")
S2 = math.sqrt(2.0)
N, REPS = 1500, 10
TWS = (0.5, 2.0, 8.0)


def log_phi(u):
    """log Phi(u) for u >= 0, Phi(u) = I_1(2u)/u (scaled Bessel function to avoid overflow)."""
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    nz = u > 1e-12
    out[nz] = np.log(special.ive(1, 2 * u[nz])) + 2 * u[nz] - np.log(u[nz])
    return out


def mean_lambda(u):
    """Phi'(u)/Phi(u) = 2 I_2(2u) / I_1(2u)."""
    u = np.asarray(u, dtype=float)
    return np.where(u > 1e-12, 2 * special.ive(2, 2 * np.maximum(u, 1e-12)) / special.ive(1, 2 * np.maximum(u, 1e-12)), 0.0)


def C_lim(t, s):
    return np.exp(log_phi(S2 * (t + s)) - 0.5 * (log_phi(2 * S2 * t) + log_phi(2 * S2 * s)))


def e_lim(t):
    return -mean_lambda(2 * S2 * np.asarray(t, dtype=float)) / S2


def aging(x):
    return (4 * (1 + x) / (2 + x) ** 2) ** 0.75


def sim_one(seed, pairs, ts_e):
    gen = rk.rng(seed)
    W = rk.goe(N, gen)
    lam, U = np.linalg.eigh(W)
    g = gen.standard_normal(N)
    c = (U.T @ g) ** 2
    lmax = lam[-1]

    def logQ(u):   # log (1/N) sum_k c_k e^{u lam_k}, stable
        return u * lmax + np.log(np.sum(c * np.exp(u * (lam - lmax))) / N)

    Cs = [math.exp(logQ(S2 * (t + s)) - 0.5 * (logQ(2 * S2 * t) + logQ(2 * S2 * s))) for (t, s) in pairs]
    es = []
    for t in ts_e:
        u = 2 * S2 * t
        wts = c * np.exp(u * (lam - lmax))
        es.append(-float(np.sum(wts * lam) / np.sum(wts)) / S2)
    return np.array(Cs), np.array(es), lam


def main():
    t0 = time.time()
    # checks of the closed forms against quadrature
    for u in (0.3, 2.0, 6.0):
        q0 = integrate.quad(lambda x: math.exp(u * x) * math.sqrt(4 - x * x) / (2 * math.pi), -2, 2)[0]
        q1 = integrate.quad(lambda x: x * math.exp(u * x) * math.sqrt(4 - x * x) / (2 * math.pi), -2, 2)[0]
        print("Phi(%.1f): closed %.10f quad %.10f ; Phi'/Phi closed %.10f quad %.10f"
              % (u, math.exp(float(log_phi(np.array([u]))[0])), q0, float(mean_lambda(np.array([u]))[0]), q1 / q0))
    # asymptotics quoted in the text
    for t in (10.0, 100.0, 1000.0):
        print("t=%g: e(t)+sqrt2 = %.6e, 3/(8t) = %.6e ; C(t,0) t^{3/4} = %.5f (2^{-1/8} pi^{-1/4} = %.5f)"
              % (t, float(e_lim(t)) + S2, 3 / (8 * t), float(C_lim(t, 0.0)) * t ** 0.75, 2 ** (-0.125) * math.pi ** (-0.25)))
    for tw in (10.0, 100.0, 1000.0):
        print("t_w=%g: C(2 t_w, t_w) = %.6f (aging limit %.6f); C(t_w + 1, t_w) = %.6f"
              % (tw, float(C_lim(2 * tw, tw)), aging(1.0), float(C_lim(tw + 1, tw))))
    # curves against x = tau / t_w
    xs = np.logspace(-2, 1.5, 141)
    rows = []
    for x in xs:
        rows.append([x] + [float(C_lim(tw * (1 + x), tw)) for tw in TWS] + [aging(x)])
    rk.write_dat(os.path.join(DATA, "14-dmft-twospin-curves.dat"), ["x", "tw05", "tw2", "tw8", "aging"], rows)
    tgrid = np.logspace(-2, 2, 121)
    rk.write_dat(os.path.join(DATA, "14-dmft-twospin-energy.dat"), ["t", "e"],
                 [[t, float(e_lim(t))] for t in tgrid])
    # simulation
    xs_sim = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    pairs = [(tw * (1 + x), tw) for tw in TWS for x in xs_sim]
    ts_e = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    allC, allE = [], []
    gaps = []
    for k in range(REPS):
        Cs, es, lam = sim_one(1420 + k, pairs, ts_e)
        allC.append(Cs)
        allE.append(es)
        gaps.append(lam[-1] - lam[-2])
    allC, allE = np.array(allC), np.array(allE)
    print("N=%d: mean top eigenvalue gap %.4f" % (N, float(np.mean(gaps))))
    srows = []
    j = 0
    within = 0
    for tw in TWS:
        for x in xs_sim:
            m, hci = rk.mean_ci(allC[:, j])
            th = float(C_lim(tw * (1 + x), tw))
            within += abs(m - th) <= hci
            srows.append([x, tw, m, hci, th])
            print("t_w=%4.1f x=%5.2f: C sim %.4f +- %.4f, limit %.4f" % (tw, x, m, hci, th))
            j += 1
    print("%d of %d intervals contain the limit" % (within, len(pairs)))
    rk.write_dat(os.path.join(DATA, "14-dmft-twospin-sim.dat"), ["x", "tw", "C", "ci", "lim"], srows)
    erows = []
    for i, t in enumerate(ts_e):
        m, hci = rk.mean_ci(allE[:, i])
        erows.append([t, m, hci, float(e_lim(t))])
        print("t=%5.1f: e sim %.5f +- %.5f, limit %.5f" % (t, m, hci, float(e_lim(t))))
    rk.write_dat(os.path.join(DATA, "14-dmft-twospin-energysim.dat"), ["t", "e", "ci", "lim"], erows)
    print("runtime %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
