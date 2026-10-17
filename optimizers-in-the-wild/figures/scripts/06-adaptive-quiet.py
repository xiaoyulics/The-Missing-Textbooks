#!/usr/bin/env python3
"""Chapter 6 (adaptive), figure fig:adaptive:quiet.

Quiet then loud: one coordinate receives gradients g_t ~ N(0, delta^2) for t0 steps, then the
constant gradient G = 1 for K steps.  Adam runs as implemented in common libraries
(beta1 = 0.9, bias correction, step eta = 1, epsilon = 1e-8 unless stated).

Theory (Proposition prop:adaptive:quietloud, limit delta/G -> 0): the k-th loud step has length
    rho_k = (1 - beta1^(k+1)) / (sqrt(1 - beta2^(k+1)) + epsilon/G),     k = 0, 1, 2, ...
and the excess displacement over the loud stretch, compared with sign descent, is
    E(beta2) = sum_{k>=0} (rho_k - 1)   (epsilon = 0),
whose leading term for beta2 -> 1 is 2 ln 2 / (1 - beta2).

Writes figures/data/06-adaptive-quiet-theory.dat (dense k), 06-adaptive-quiet-sim.dat (simulated
steps at sparse k) and 06-adaptive-quiet-excess.dat.  Seed 0; a few seconds on one core.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
BETA1 = 0.9
DELTA = 1e-3
# (beta2, epsilon) for the four series of the left panel
SERIES = ((0.95, 1e-8), (0.99, 1e-8), (0.999, 1e-8), (0.999, 1e-2))
KMAX = 20000


def rho(k, beta2, eps):
    return (1 - BETA1 ** (k + 1)) / (np.sqrt(1 - beta2 ** (k + 1)) + eps)


def adam_steps(beta2, eps, t0, K, rng):
    """Vectorized over the entries of beta2 and eps; returns |step| at the K loud steps."""
    beta2 = np.asarray(beta2, float)
    eps = np.asarray(eps, float)
    m = np.zeros_like(beta2)
    v = np.zeros_like(beta2)
    p1 = 1.0
    p2 = np.ones_like(beta2)
    out = np.empty((K, len(beta2)))
    quiet = DELTA * rng.standard_normal((t0, len(beta2)))
    for t in range(t0 + K):
        g = quiet[t] if t < t0 else 1.0
        m = BETA1 * m + (1 - BETA1) * g
        v = beta2 * v + (1 - beta2) * g * g
        p1 *= BETA1
        p2 = p2 * beta2
        step = (m / (1 - p1)) / (np.sqrt(v / (1 - p2)) + eps)
        if t >= t0:
            out[t - t0] = np.abs(step)
    return out


def main():
    rng = np.random.default_rng(0)
    # left panel: time course of the step after the jump
    t0 = 20000                                   # 20 memories of the slowest series
    steps = adam_steps([s[0] for s in SERIES], [s[1] for s in SERIES], t0, KMAX + 1, rng)
    kd = np.unique(np.round(np.logspace(0, np.log10(KMAX + 1), 200)).astype(int)) - 1
    with open(os.path.join(DATA, "06-adaptive-quiet-theory.dat"), "w") as fh:
        fh.write("k kp1 th95 th99 th999 th999e\n")
        for k in kd:
            fh.write(f"{k} {k + 1} " + " ".join(f"{rho(k, b2, e):.6g}" for b2, e in SERIES) + "\n")
    ks = np.unique(np.round(np.logspace(0, np.log10(KMAX + 1), 26)).astype(int)) - 1
    with open(os.path.join(DATA, "06-adaptive-quiet-sim.dat"), "w") as fh:
        fh.write("k kp1 sim95 sim99 sim999 sim999e\n")
        for k in ks:
            fh.write(f"{k} {k + 1} " + " ".join(f"{steps[k, j]:.6g}" for j in range(len(SERIES)))
                     + "\n")
    for j, (b2, e) in enumerate(SERIES):
        print(f"beta2={b2}, eps={e:g}: first step sim {steps[0, j]:.4f} theory {rho(0, b2, e):.4f};"
              f" peak sim {steps[:, j].max():.4f} theory {rho(np.arange(KMAX), b2, e).max():.4f}")
    # right panel: excess displacement versus memory
    mems = np.array([20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000], float)
    b2s = 1 - 1 / mems
    t0 = int(20 * mems.max())
    K = int(30 * mems.max())
    st = adam_steps(b2s, np.full(len(b2s), 1e-8), t0, K, rng)
    sim_excess = []
    for j, mem in enumerate(mems):
        kk = int(30 * mem)
        sim_excess.append(np.sum(st[:kk, j] - 1.0))
    md = np.unique(np.round(np.logspace(np.log10(20), 4, 60)))
    with open(os.path.join(DATA, "06-adaptive-quiet-excess.dat"), "w") as fh:
        fh.write("mem exact lead\n")
        for mem in md:
            b2 = 1 - 1 / mem
            k = np.arange(int(60 * mem))
            fh.write(f"{mem:.6g} {np.sum(rho(k, b2, 0.0) - 1):.6g} {2 * np.log(2) * mem:.6g}\n")
    with open(os.path.join(DATA, "06-adaptive-quiet-excess-sim.dat"), "w") as fh:
        fh.write("mem sim exact\n")
        for mem, e in zip(mems, sim_excess):
            k = np.arange(int(60 * mem))
            ex = np.sum(rho(k, 1 - 1 / mem, 0.0) - 1)
            fh.write(f"{mem:.6g} {e:.6g} {ex:.6g}\n")
            print(f"memory {mem:7.0f}: excess sim {e:10.3f}  exact series {ex:10.3f}")


if __name__ == "__main__":
    main()
