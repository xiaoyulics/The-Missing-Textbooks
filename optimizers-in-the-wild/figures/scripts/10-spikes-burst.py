"""Figure 10.2: Adam's step on a block of weights that wakes up after a quiet period.

A linear-regression model with d = 32 weights whose input features are absent (identically zero)
for the first T0 = 10^5 steps of training, so that the block's gradient, and hence its Adam moments
m and v, are exactly zero at step T0. From T0 on the features are N(0, I), the labels are
y = <w_star, x> + sigma * noise with w_star in {-1, +1}^d and sigma = 1, and Adam (with bias
correction counted from the start of training, epsilon = 1e-8) is run with minibatches of B = 4096.
The step size is eta = 1e-6, small enough that the gradient (about w - w_star) stays nearly constant
over the window; the script records |Delta w_i|/eta, the per-coordinate step in units of eta.

Theory (Proposition prop:spikes:burst): after k loud steps with constant gradient G the step is
  a_k = (1 - b1^k) / (sqrt(1 - b2^k) + eps/|G|).
To show the effect of epsilon, the same runs are repeated with the loss multiplied by 1e-7, so that
|G| = 1e-7 = 10 epsilon.

Outputs:
  figures/data/10-spikes-burst-k.dat     k, exact a_k and simulated median step for b2 in {0.95, 0.99, 0.999}
  figures/data/10-spikes-burst-peak.dat  1 - b2, a_1, max_k a_k (eps = 0 and eps/|G| = 0.1), the worst case
                                         C(b1, b2), and the simulated peaks
Deterministic seeds; about 30 s on one core.
"""
import os
import numpy as np

B1 = 0.9
EPS = 1e-8
D = 32
BATCH = 4096
SIGMA = 1.0
ETA = 1e-6
T0 = 100_000


def burst_theory(k, b2, eps_over_g=0.0):
    k = np.asarray(k, dtype=float)
    return (1 - B1 ** k) / (np.sqrt(1 - b2 ** k) + eps_over_g)


def worst_case(b2):
    return (1 - B1) / np.sqrt((1 - b2) * (1 - B1 ** 2 / b2))


def simulate(b2, steps, scale=1.0, seed=0):
    """Median over coordinates of |Delta w_i|/eta for k = 1..steps after the block wakes up."""
    rng = np.random.default_rng(seed)
    w_star = rng.choice([-1.0, 1.0], size=D)
    w = np.zeros(D)
    m = np.zeros(D)
    v = np.zeros(D)
    out = np.empty(steps)
    for k in range(1, steps + 1):
        t = T0 + k  # global step count used by bias correction
        X = rng.standard_normal((BATCH, D))
        y = X @ w_star + SIGMA * rng.standard_normal(BATCH)
        g = scale * (X.T @ (X @ w - y)) / BATCH
        m = B1 * m + (1 - B1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - B1 ** t)
        vhat = v / (1 - b2 ** t)
        step = ETA * mhat / (np.sqrt(vhat) + EPS)
        w -= step
        out[k - 1] = np.median(np.abs(step)) / ETA
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, "..", "data")
    # left panel: the burst as a function of k
    b2s = [0.95, 0.99, 0.999]
    K = 3000
    ks = np.unique(np.round(np.logspace(0, np.log10(K), 70)).astype(int))
    cols = {}
    for i, b2 in enumerate(b2s):
        sim = simulate(b2, K, seed=10 + i)
        cols[b2] = (burst_theory(ks, b2), sim[ks - 1])
        print(f"b2={b2}: theory peak {burst_theory(np.arange(1, K + 1), b2).max():.4f}, simulated peak {sim.max():.4f}")
    with open(os.path.join(data, "10-spikes-burst-k.dat"), "w") as fh:
        fh.write("k th95 sim95 th99 sim99 th999 sim999\n")
        for j, k in enumerate(ks):
            fh.write(f"{k} " + " ".join(f"{cols[b2][0][j]:.6g} {cols[b2][1][j]:.6g}" for b2 in b2s) + "\n")
    # right panel: the peak as a function of 1 - b2
    kk = np.arange(1, 20001)
    with open(os.path.join(data, "10-spikes-burst-peak.dat"), "w") as fh:
        fh.write("omb2 a1 peak peakeps worst sim simeps\n")
        for j, omb2 in enumerate(np.logspace(-1, -4, 31)):
            b2 = 1 - omb2
            a1 = (1 - B1) / np.sqrt(omb2)
            peak = burst_theory(kk, b2).max()
            peak_eps = burst_theory(kk, b2, 0.1).max()
            if j % 3 == 0:
                sim = simulate(b2, 80, seed=100 + j).max()
                sim_eps = simulate(b2, 80, scale=1e-7, seed=200 + j).max()
                print(f"1-b2={omb2:.2e}: a1={a1:.3f} peak={peak:.3f} (sim {sim:.3f}) peak_eps={peak_eps:.3f} "
                      f"(sim {sim_eps:.3f}) worst={worst_case(b2):.3f}")
                fh.write(f"{omb2:.6g} {a1:.6g} {peak:.6g} {peak_eps:.6g} {worst_case(b2):.6g} {sim:.6g} {sim_eps:.6g}\n")
            else:
                fh.write(f"{omb2:.6g} {a1:.6g} {peak:.6g} {peak_eps:.6g} {worst_case(b2):.6g} nan nan\n")
    print("wrote", os.path.normpath(data))


if __name__ == "__main__":
    main()
