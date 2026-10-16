"""Escape times of spherical online SGD for single-index models (Chapter 7, figure fig:singleindex:escape).

Target y = h_k(<w_star, x>), student h_k(<w, x>), x ~ N(0, I_d), correlation loss, fresh sample per
step, warm start r_0 = d^{-1/2}. By rotational invariance the overlap r_t = <w_t, w_star> is itself a
Markov chain: given r_t = r, draw independent v, z ~ N(0,1) and Q ~ chi^2_{d-2}, put u = r v + s z
with s = sqrt(1 - r^2) and F = h_k(u) sqrt(k) h_{k-1}(v); then exactly
    r_{t+1} = (r + eta F s z) / sqrt(1 + eta^2 F^2 (z^2 + Q)).
(Here u = <w_star, x>, v = <w, x>, z^2 + Q = |P_w^perp x|^2.) We simulate this chain, which costs O(1)
per step for every d; build/07-singleindex-scratch/chain_check.py compares it with full d-dimensional
SGD at d = 16, 32.

Step sizes: eta = c/d for k = 1, 2 and eta = c d^{-3/2} for k = 3, with c = 0.02.
Escape time: first t with r_t >= 1/2 (k odd) or |r_t| >= 1/2 (k even).
Theory: the population flow dr/dt = (1 - r^2) k r^(k-1) needs time t_1/2 = int_{r_0}^{1/2} dr /
((1 - r^2) k r^(k-1)); the predicted number of steps is t_1/2 / eta.
Runs that have not escaped after 3 times the predicted time of the largest d count as +infinity in the
quartiles (none did in our run). Runtime: about one minute on a laptop.
Output: figures/data/07-singleindex-escape-k{1,2,3}.dat (simulation: d, median, q25, q75, prediction)
and figures/data/07-singleindex-escape-theory.dat (prediction on a fine grid of d).
"""
import os
import numpy as np
from scipy.integrate import quad

C = 0.02
SEED = 20261017
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")


def grad_weight(k, u, v):
    """F = h_k(u) * sqrt(k) * h_{k-1}(v) for the matched link g = phi = h_k."""
    if k == 1:
        return u
    if k == 2:
        return (u * u - 1.0) * v
    if k == 3:
        return 0.5 * (u * u * u - 3.0 * u) * (v * v - 1.0)
    raise ValueError(k)


def eta_of(k, d):
    return C / d if k <= 2 else C * d ** -1.5


def ode_steps(k, d):
    r0 = d ** -0.5
    t = quad(lambda r: 1.0 / ((1.0 - r * r) * k * r ** (k - 1)), r0, 0.5)[0]
    return t / eta_of(k, d)


def escape_times(k, ds, runs, rng, tmax, block=1000):
    d = np.repeat(np.asarray(ds, float), runs)
    eta = np.array([eta_of(k, di) for di in d])
    r = d ** -0.5
    esc = np.full(d.size, -1, dtype=np.int64)
    t = 0
    while t < tmax:
        act = np.nonzero(esc < 0)[0]
        if act.size == 0:
            break
        m = act.size
        V = rng.standard_normal((block, m))
        Z = rng.standard_normal((block, m))
        Q = rng.gamma(shape=(d[act] - 2.0) / 2.0, scale=2.0, size=(block, m))
        rr, ee, ee2 = r[act].copy(), eta[act], eta[act] ** 2
        hit = np.full(m, -1, dtype=np.int64)
        for b in range(block):
            z = Z[b]
            s = np.sqrt(np.maximum(1.0 - rr * rr, 0.0))
            u = rr * V[b] + s * z
            F = grad_weight(k, u, V[b])
            rr = (rr + ee * F * s * z) / np.sqrt(1.0 + ee2 * F * F * (z * z + Q[b]))
            done = (rr >= 0.5) if k % 2 else (np.abs(rr) >= 0.5)
            hit[(hit < 0) & done] = t + b + 1
        r[act] = rr
        esc[act] = hit
        t += block
    return d, esc


def main():
    rng = np.random.default_rng(SEED)
    plan = {1: ([16, 64, 256, 1024, 4096], 200),
            2: ([16, 64, 256, 1024, 4096], 200),
            3: ([16, 32, 64, 128, 256], 100)}
    for k, (ds, runs) in plan.items():
        tmax = int(3 * max(ode_steps(k, x) for x in ds))
        d, esc = escape_times(k, ds, runs, rng, tmax)
        rows = []
        for x in ds:
            e = esc[d == x].astype(float)
            e[e < 0] = np.inf  # did not escape before tmax (counted in the median as +infinity)
            rows.append((x, np.median(e), np.percentile(e, 25), np.percentile(e, 75), ode_steps(k, x),
                         np.mean(e < np.inf)))
        with open(os.path.join(OUT, "07-singleindex-escape-k%d.dat" % k), "w") as f:
            f.write("d median q25 q75 ode escaped\n")
            for row in rows:
                f.write("%d %.6g %.6g %.6g %.6g %.4f\n" % row)
        print("k=%d" % k, ["d=%d med/ode=%.3f esc=%.3f" % (x[0], x[1] / x[4], x[5]) for x in rows], flush=True)
    grid = np.unique(np.round(np.logspace(np.log10(16), np.log10(4096), 60)))
    with open(os.path.join(OUT, "07-singleindex-escape-theory.dat"), "w") as f:
        f.write("d k1 k2 k3\n")
        for x in grid:
            f.write("%d %.6g %.6g %.6g\n" % (x, ode_steps(1, x), ode_steps(2, x), ode_steps(3, x)))


if __name__ == "__main__":
    main()
