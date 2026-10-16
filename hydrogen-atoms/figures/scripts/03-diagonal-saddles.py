"""Chapter 3 (diagonal): saddle-to-saddle dynamics for a general design.

Model: w = u*u - v*v in R^d, u(0) = v(0) = alpha*1, Lhat(w) = (1/2n)||Xw - y||^2, with a random
Gaussian design X (n = 3, d = 6, seed 281, chosen among seeds for four well-separated jumps one of
which deactivates a coordinate) and random labels y.

Theory: the alpha -> 0 limit in accelerated time tau = t/log(1/alpha), computed by the algorithm of
Pesme and Flammarion (2023) in the normalization of the chapter: the dual variable
sigma(tau) = -2 int_0^tau grad Lhat moves linearly between jumps; a jump happens when a coordinate
of sigma reaches +1 or -1; the new saddle minimizes Lhat over the vectors that vanish where
|sigma_i| < 1 and have the sign of sigma_i where |sigma_i| = 1 (nonnegative least squares).
Simulation: the gradient flow on (u, v), integrated in the variables (log u, log v) with an adaptive
Runge-Kutta method (RK45, rtol 1e-10), for alpha = 1e-4, 1e-12, 1e-24.

Writes figures/data/03-diagonal-saddles-flow.dat (coordinates 1-4 and loss versus tau, three alphas)
and figures/data/03-diagonal-saddles-limit.dat (the piecewise constant limit, as a step path).
Deterministic; runs in a few seconds.
"""
import os
import numpy as np
from scipy.optimize import nnls
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")

n, d = 3, 6
rng = np.random.default_rng(281)
X = rng.standard_normal((n, d))
y = rng.standard_normal(n)
H = X.T @ X / n
b = X.T @ y / n


def lhat(w):
    r = X @ w - y
    return r @ r / (2 * n)


def saddle_path(tol=1e-12):
    """Jump times and saddles of the alpha -> 0 limit (chapter normalization)."""
    tau, w, sig = 0.0, np.zeros(d), np.zeros(d)
    taus, saddles = [0.0], [w.copy()]
    for _ in range(50):
        g = H @ w - b
        if np.max(np.abs(g)) < tol:
            break
        waits = np.full(d, np.inf)
        for i in range(d):
            if g[i] < -tol:
                waits[i] = (1 - sig[i]) / (-2 * g[i])
            elif g[i] > tol:
                waits[i] = (sig[i] + 1) / (2 * g[i])
        waits[waits <= 1e-12] = np.inf
        dt = waits.min()
        tau += dt
        sig = np.clip(sig - 2 * dt * g, -1, 1)
        A = np.where(np.abs(sig) > 1 - 1e-9)[0]
        s = np.sign(sig[A])
        z, _ = nnls(X[:, A] * s[None, :], y)
        w = np.zeros(d)
        w[A] = s * z
        taus.append(tau)
        saddles.append(w.copy())
    return np.array(taus), np.array(saddles)


taus, saddles = saddle_path()
active = np.where(np.any(saddles != 0, axis=0))[0]
assert len(active) == 4

tau_grid = np.linspace(0.0, 3.6, 361)
flows = {}
for alpha in [1e-4, 1e-12, 1e-24]:
    L = np.log(1 / alpha)

    def rhs(t, z):
        g = H @ (np.exp(2 * z[:d]) - np.exp(2 * z[d:])) - b
        return np.concatenate([-2 * g, 2 * g])

    sol = solve_ivp(rhs, (0, tau_grid[-1] * L), np.log(alpha) * np.ones(2 * d),
                    t_eval=tau_grid * L, method="RK45", rtol=1e-10, atol=1e-12)
    W = np.exp(2 * sol.y[:d]) - np.exp(2 * sol.y[d:])
    flows[alpha] = (W, np.array([lhat(W[:, j]) for j in range(W.shape[1])]))

with open(os.path.join(OUT, "03-diagonal-saddles-flow.dat"), "w") as f:
    cols = []
    for tag in ["a4", "a12", "a24"]:
        cols += [f"{tag}w{j + 1}" for j in range(4)] + [f"{tag}loss"]
    f.write("tau " + " ".join(cols) + "\n")
    for j, x in enumerate(tau_grid):
        vals = []
        for alpha in [1e-4, 1e-12, 1e-24]:
            W, ls = flows[alpha]
            vals += [W[i, j] for i in active] + [ls[j]]
        f.write(f"{x:.4f} " + " ".join(f"{v:.6f}" for v in vals) + "\n")

with open(os.path.join(OUT, "03-diagonal-saddles-limit.dat"), "w") as f:
    f.write("tau " + " ".join(f"w{j + 1}" for j in range(4)) + " loss\n")
    edges = list(taus) + [tau_grid[-1]]
    for k in range(len(taus)):
        for x in (edges[k], edges[k + 1]):
            f.write(f"{x:.4f} " + " ".join(f"{saddles[k][i]:.6f}" for i in active)
                    + f" {lhat(saddles[k]):.6f}\n")

print("active coordinates:", active + 1)
print("jump times:", np.round(taus, 4))
for k, (t, w) in enumerate(zip(taus, saddles)):
    print(f"saddle {k}: tau={t:.4f}  w={np.round(w[active], 4)}  loss={lhat(w):.4f}")
for alpha in [1e-4, 1e-12, 1e-24]:
    W, _ = flows[alpha]
    lim = np.array([saddles[np.searchsorted(taus, x, side='right') - 1] for x in tau_grid]).T
    away = np.array([np.min(np.abs(x - taus[1:])) > 0.15 for x in tau_grid])
    print(f"alpha={alpha:.0e}: max |flow - limit| at distance > 0.15 from the jumps: "
          f"{np.max(np.abs(W[:, away] - lim[:, away])):.3f}")
