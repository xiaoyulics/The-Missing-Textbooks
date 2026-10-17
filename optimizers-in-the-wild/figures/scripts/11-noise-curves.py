"""Chapter 11 (noise), figure fig:noise:curves.

Expected loss of SGD in the noisy quadratic model: the exact formula of thm:noise:nqm (curves)
against the average of independent SGD runs (points).

Model: d = 1000 modes, curvatures h_i = 1/i, noise Sigma = H (c_i = h_i), theta_0 = (1,...,1),
Gaussian gradient noise of covariance Sigma/B. Left panel: B = 4 and three step sizes; right
panel: step size 0.26 and three batch sizes. Writes
  figures/data/11-noise-curves.dat      exact expected loss on a log grid of steps, and the floors
  figures/data/11-noise-curves-mc.dat   Monte Carlo averages over R runs at fewer steps
Runtime: about 30 s on one CPU core. Deterministic seed.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")

d = 1000
h = 1.0 / np.arange(1, d + 1)
c = h.copy()
theta0 = np.ones(d)
l0 = 0.5 * h * theta0**2
eps = 0.05 * l0.sum()          # target: 5% of the initial loss
R = 64                          # Monte Carlo runs per setting
TMAX = 4000

# (label, eta, B): panel a = fixed B, three step sizes; panel b = fixed step, three batch sizes
settings = [("a1", 0.10, 4), ("a2", 0.26, 4), ("a3", 1.00, 4),
            ("b1", 0.26, 1), ("b2", 0.26, 4), ("b3", 0.26, 16)]


def exact(t, eta, B):
    """E L(theta_t) = sum_i F_i + (l_i(0) - F_i) (1 - eta h_i)^{2t}  (thm:noise:nqm)."""
    x = eta * h
    F = eta * c / (2 * B * (2 - x))
    rho = (1 - x) ** 2
    return np.array([np.sum(F + (l0 - F) * rho**tt) for tt in t])


def floor(eta, B):
    return np.sum(eta * c / (2 * B * (2 - eta * h)))


def monte_carlo(eta, B, tgrid, rng):
    """average over R runs of L(theta_t) for SGD with Gaussian noise of covariance Sigma/B."""
    th = np.tile(theta0, (R, 1))
    sd = np.sqrt(c / B)
    out = np.empty(len(tgrid))
    want = {int(tt): k for k, tt in enumerate(tgrid)}
    for t in range(TMAX + 1):
        if t in want:
            out[want[t]] = np.mean(0.5 * (th**2) @ h)
        g = th * h + rng.standard_normal(th.shape) * sd
        th = th - eta * g
    return out


tgrid = np.unique(np.round(np.logspace(0, np.log10(TMAX), 160)).astype(int))
tmc = np.unique(np.round(np.logspace(0, np.log10(TMAX), 22)).astype(int))

rng = np.random.default_rng(11)
cols, mcols, done = {}, {}, {}
for lab, eta, B in settings:
    cols[lab] = exact(tgrid, eta, B)
    cols["f" + lab] = np.full(len(tgrid), floor(eta, B))
    if (eta, B) not in done:            # the setting (0.26, 4) appears in both panels
        done[(eta, B)] = monte_carlo(eta, B, tmc, rng)
    mcols[lab] = done[(eta, B)]
    k = np.argmax(np.abs(mcols[lab] / exact(tmc, eta, B) - 1))
    print(f"{lab}: eta={eta} B={B} floor={floor(eta, B):.4f} "
          f"max rel. MC deviation {np.max(np.abs(mcols[lab] / exact(tmc, eta, B) - 1)):.3f} at t={tmc[k]}")

names = ["a1", "a2", "a3", "b1", "b2", "b3"]
with open(os.path.join(OUT, "11-noise-curves.dat"), "w") as f:
    f.write("t " + " ".join(names) + " " + " ".join("f" + n for n in names) + " eps\n")
    for k, t in enumerate(tgrid):
        f.write(f"{t} " + " ".join(f"{cols[n][k]:.6e}" for n in names) + " "
                + " ".join(f"{cols['f' + n][k]:.6e}" for n in names) + f" {eps:.6e}\n")
with open(os.path.join(OUT, "11-noise-curves-mc.dat"), "w") as f:
    f.write("t " + " ".join(names) + "\n")
    for k, t in enumerate(tmc):
        f.write(f"{t} " + " ".join(f"{mcols[n][k]:.6e}" for n in names) + "\n")
print("target eps =", eps)
