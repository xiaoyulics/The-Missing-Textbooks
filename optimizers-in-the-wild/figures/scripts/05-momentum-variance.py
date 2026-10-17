"""Chapter 5 (momentum): stationary loss of heavy ball on a noisy quadratic mode.

Heavy ball x+ = x - eta (h x + zeta) + beta (x - x_prev) with zeta ~ N(0, sigma^2) i.i.d., h = 1,
sigma = 1.  Theorem thm:momentum:noise gives the stationary variance exactly; written in terms of the
effective step u = eta h/(1-beta), the stationary loss E[h x^2/2] divided by its first-order value
u sigma^2/4 is
    ratio(u) = 1 / (1 - u (1-beta) / (2 (1+beta))),
which diverges at the stability limit u = 2(1+beta)/(1-beta) (u = 2 for SGD, beta = 0).
The simulation runs 2000 independent chains from x = 0 for a burn-in of 2000 steps and averages
x^2 over the next 6000 steps, for beta in {0, 0.5, 0.9} and several u (seed 0).
Output: figures/data/05-momentum-variance.dat      u r0 r5 r9      (curves; nan beyond the limit)
        figures/data/05-momentum-variance-sim.dat  beta u ratio    (points)
Runs in about ten seconds.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
BETAS = [0.0, 0.5, 0.9]


def ratio(u, beta):
    r = 1.0 / (1.0 - u * (1 - beta) / (2 * (1 + beta)))
    return r if u < 2 * (1 + beta) / (1 - beta) else np.nan


def simulate(u, beta, rng, chains=2000, burn=2000, T=6000):
    h, sigma = 1.0, 1.0
    eta = u * (1 - beta) / h
    x = np.zeros(chains); xp = np.zeros(chains)
    acc = 0.0
    for t in range(burn + T):
        xn = x - eta * (h * x + sigma * rng.standard_normal(chains)) + beta * (x - xp)
        xp, x = x, xn
        if t >= burn:
            acc += np.mean(x * x)
    loss = 0.5 * h * acc / T
    return loss / (u * sigma ** 2 / 4)


def main():
    rng = np.random.default_rng(0)
    # a log grid, refined just below each stability limit so that the curves reach their asymptotes
    us = list(np.geomspace(0.01, 40, 160))
    for b in BETAS:
        lim = 2 * (1 + b) / (1 - b)
        us += [frac * lim for frac in [0.95, 0.97, 0.98, 0.99]]
    with open(os.path.join(DATA, "05-momentum-variance.dat"), "w") as f:
        f.write("u r0 r5 r9\n")
        for u in sorted(us):
            vals = [ratio(u, b) for b in BETAS]
            f.write(f"{u:.5f} " + " ".join("nan" if np.isnan(v) else f"{v:.6f}" for v in vals) + "\n")
    grid = {0.0: [0.03, 0.1, 0.3, 1.0, 1.5, 1.8], 0.5: [0.03, 0.1, 0.3, 1.0, 3.0, 5.0],
            0.9: [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 20.0, 30.0, 34.0]}
    worst = 0.0
    with open(os.path.join(DATA, "05-momentum-variance-sim.dat"), "w") as f:
        f.write("beta u ratio\n")
        for b in BETAS:
            for u in grid[b]:
                r = simulate(u, b, rng)
                worst = max(worst, abs(r / ratio(u, b) - 1))
                f.write(f"{b:.2f} {u:.4f} {r:.6f}\n")
    print(f"largest relative deviation of Monte Carlo from the exact formula: {worst:.3f}")


if __name__ == "__main__":
    main()
