"""Chapter 3 (diagonal): incremental learning under an orthogonal design.

Model: w = u*u - v*v in R^d, u(0) = v(0) = alpha*1, Lhat(w) = (1/2n)||Xw - y||^2 with
X^T X = n I_d (X = sqrt(n) Q, Q with orthonormal columns), noiseless labels y = X w_star.

Theory (exact solution of the chapter): u_i(t)^2 = (p_+ + kappa_i(t) p_-)/(1 + kappa_i(t)),
p_pm = (s_i pm rho_i)/2, rho_i = sqrt(s_i^2 + 4 alpha^4), kappa_i(t) = kappa_i(0) exp(-4 rho_i t),
kappa_i(0) = (p_+ - alpha^2)/(alpha^2 - p_-), v_i^2 = alpha^4/u_i^2, w_i = u_i^2 - v_i^2, s_i = w_star_i.
Simulation: plain gradient descent on (u, v) with step size 1e-3 (the design is a random rotation,
so the coordinates are coupled in the simulation and decouple only through X^T X = n I).

Writes figures/data/03-diagonal-orthogonal-theory.dat (normalized coordinates w_i/w_star_i, exact),
figures/data/03-diagonal-orthogonal-sim.dat (the same from gradient descent, every 0.5 time units),
figures/data/03-diagonal-staircase.dat (training loss in accelerated time tau = t/log(1/alpha) for
alpha = 1e-2, 1e-4, 1e-8, exact, and the alpha -> 0 staircase).
Deterministic (seed 1); runs in a few seconds.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")

d, n = 4, 8
w_star = np.array([1.0, -0.6, 0.35, 0.2])
rng = np.random.default_rng(1)
Q, _ = np.linalg.qr(rng.standard_normal((n, d)))
X = np.sqrt(n) * Q
y = X @ w_star
assert np.allclose(X.T @ X / n, np.eye(d))


def exact_w(t, s, alpha):
    """Exact w_i(t) for target value s (any sign), from the Riccati solution for u_i^2."""
    t = np.asarray(t, dtype=float)
    if s == 0:
        return np.zeros_like(t)
    sg, s = np.sign(s), abs(s)  # symmetry (u, v, s) -> (v, u, -s)
    a2 = alpha * alpha
    rho = np.sqrt(s * s + 4 * a2 * a2)
    p_plus = 0.5 * (s + rho)
    p_minus = -a2 * a2 / p_plus  # = (s - rho)/2, computed without cancellation
    kappa = (p_plus - a2) / (a2 - p_minus) * np.exp(-4 * rho * t)
    u2 = (p_plus + kappa * p_minus) / (1 + kappa)
    return sg * (u2 - a2 * a2 / u2)


def loss(W):
    """Training loss for an orthogonal design and noiseless labels: (1/2)||w - w_star||^2."""
    return 0.5 * np.sum((W - w_star[:, None]) ** 2, axis=0)


# panel (a): alpha = 1e-3, exact curves and gradient descent
alpha = 1e-3
T = 20.0
t_grid = np.linspace(0.0, T, 321)
W_exact = np.array([exact_w(t_grid, s, alpha) for s in w_star])
eta = 1e-3
u = alpha * np.ones(d)
v = alpha * np.ones(d)
record_every = 500  # every 0.5 time units
sim_t, sim_W = [0.0], [u * u - v * v]
for step in range(1, int(round(T / eta)) + 1):
    g = X.T @ (X @ (u * u - v * v) - y) / n
    u, v = u - 2 * eta * u * g, v + 2 * eta * v * g
    if step % record_every == 0:
        sim_t.append(step * eta)
        sim_W.append(u * u - v * v)
sim_t = np.array(sim_t)
sim_W = np.array(sim_W).T

with open(os.path.join(OUT, "03-diagonal-orthogonal-theory.dat"), "w") as f:
    f.write("t r1 r2 r3 r4\n")
    for j, t in enumerate(t_grid):
        f.write(f"{t:.4f} " + " ".join(f"{W_exact[i, j] / w_star[i]:.6f}" for i in range(d)) + "\n")
with open(os.path.join(OUT, "03-diagonal-orthogonal-sim.dat"), "w") as f:
    f.write("t r1 r2 r3 r4\n")
    for j, t in enumerate(sim_t):
        f.write(f"{t:.4f} " + " ".join(f"{sim_W[i, j] / w_star[i]:.6f}" for i in range(d)) + "\n")

# panel (b): loss in accelerated time
tau = np.linspace(0.0, 3.0, 301)
cols = []
for a in [1e-2, 1e-4, 1e-8]:
    tt = tau * np.log(1 / a)
    W = np.array([exact_w(tt, s, a) for s in w_star])
    cols.append(loss(W))
tau_i = 1.0 / (2.0 * np.abs(w_star))
stair = np.array([0.5 * np.sum(w_star[tau_i > x] ** 2) for x in tau])
with open(os.path.join(OUT, "03-diagonal-staircase.dat"), "w") as f:
    f.write("tau a2 a4 a8 stair\n")
    for j, x in enumerate(tau):
        f.write(f"{x:.4f} {cols[0][j]:.6f} {cols[1][j]:.6f} {cols[2][j]:.6f} {stair[j]:.6f}\n")

# checks printed for the text
dev = 0.0
for j, t in enumerate(sim_t):
    dev = max(dev, np.max(np.abs(sim_W[:, j] - np.array([exact_w(t, s, alpha) for s in w_star]))))
print(f"max |GD - exact| over the recorded times: {dev:.2e}")
t_pred = np.log(np.abs(w_star) / alpha ** 2) / (4 * np.abs(w_star))
for s, tp in zip(w_star, t_pred):
    # half time from the exact solution, by bisection
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if exact_w(mid, s, alpha) / s < 0.5:
            lo = mid
        else:
            hi = mid
    print(f"w_star={s:+.3f}: exact half time {lo:.6f}, log(|s|/a^2)/(4|s|) = {tp:.6f}, "
          f"difference {lo - tp:.2e}, -a^2/(2 s^2) = {-alpha ** 2 / (2 * s * s):.2e}")
print("staircase steps at tau_i = 1/(2|w_star_i|):", np.round(tau_i, 4))
