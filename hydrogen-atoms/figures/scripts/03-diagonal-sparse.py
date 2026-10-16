"""Chapter 3 (diagonal): from kernel to sparse interpolation as the initialization shrinks.

Model: w = u*u - v*v in R^d, u(0) = v(0) = alpha*1, gradient flow on
Lhat(w) = (1/2n)||Xw - y||^2, Gaussian design X (n x d), k-sparse target, no noise.

Theory: the limit of the flow is argmin{Phi_alpha(w) : Xw = y}; it is computed from the
optimality conditions w = 2 alpha^2 sinh(4 X^T nu), Xw = y, by damped Newton steps on the
(convex) dual function nu -> (alpha^2/2) sum_i cosh(4 (X^T nu)_i) - <y, nu>.
Simulation: the gradient flow on (u, v), integrated numerically in the variables (log u, log v)
(u and v stay positive): an adaptive Runge-Kutta method (RK45, rtol 1e-9) until the training loss
is below 1e-9, then a stiff solver (BDF, with the exact Jacobian) until it is below 1e-20. The last
phase matters: for small alpha the flow reaches small loss quickly but settles on its limit only on
a time scale that grows like a power of 1/alpha.
Also computed: the error of the minimum-l2 interpolator X^+ y (the kernel limit) and the
margin gamma* of the best dual certificate (the exponent of the rich limit, error ~ alpha^(4 gamma*)).

Writes figures/data/03-diagonal-sparse-theory.dat and figures/data/03-diagonal-sparse-sim.dat.
Deterministic (seed 0); runs in well under a minute.
"""
import os
import numpy as np
from scipy.optimize import linprog
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")

rng = np.random.default_rng(0)
n, d, k = 40, 100, 5
X = rng.standard_normal((n, d))
support = rng.choice(d, size=k, replace=False)
w_star = np.zeros(d)
w_star[support] = rng.choice([-1.0, 1.0], size=k) / np.sqrt(k)  # unit norm target
y = X @ w_star


def phi_interpolator(alpha, nu=None):
    """argmin Phi_alpha(w) subject to Xw = y, by Newton's method on the dual."""
    nu = np.zeros(n) if nu is None else nu.copy()
    a2 = alpha * alpha
    for _ in range(500):
        z = 4.0 * X.T @ nu
        w = 2.0 * a2 * np.sinh(z)
        r = X @ w - y
        if np.linalg.norm(r) < 1e-14:
            break
        J = X @ ((8.0 * a2 * np.cosh(z))[:, None] * X.T)
        step = np.linalg.solve(J, r)
        h0 = 0.5 * a2 * np.sum(np.cosh(z)) - y @ nu
        t = 1.0
        while t > 1e-14:
            nu_new = nu - t * step
            with np.errstate(over="ignore"):
                h1 = 0.5 * a2 * np.sum(np.cosh(4.0 * X.T @ nu_new)) - y @ nu_new
            if np.isfinite(h1) and h1 <= h0 - 0.25 * t * (r @ step):
                break
            t /= 2.0
        nu = nu_new
    return 2.0 * a2 * np.sinh(4.0 * X.T @ nu), nu


H = X.T @ X / n
b = X.T @ y / n


def _rhs(t, z):
    p, q = z[:d], z[d:]
    g = H @ (np.exp(2 * p) - np.exp(2 * q)) - b
    return np.concatenate([-2.0 * g, 2.0 * g])


def _jac(t, z):
    p, q = z[:d], z[d:]
    A = H * (2.0 * np.exp(2 * p))[None, :]
    B = H * (2.0 * np.exp(2 * q))[None, :]
    return np.block([[-2.0 * A, 2.0 * B], [2.0 * A, -2.0 * B]])


def _loss_event(level):
    def ev(t, z):
        p, q = z[:d], z[d:]
        r = X @ (np.exp(2 * p) - np.exp(2 * q)) - y
        return np.log(r @ r / (2 * n) + 1e-300) - np.log(level)
    ev.terminal = True
    return ev


def gradient_flow(alpha):
    """Integrate du/dt = -2 u g, dv/dt = 2 v g (g = grad Lhat(w)) from u = v = alpha*1."""
    z0 = np.log(alpha) * np.ones(2 * d)
    s1 = solve_ivp(_rhs, (0.0, 1e12), z0, method="RK45", rtol=1e-9, atol=1e-12,
                   events=_loss_event(1e-9))
    s2 = solve_ivp(_rhs, (s1.t[-1], 1e14), s1.y[:, -1], method="BDF", jac=_jac,
                   rtol=1e-10, atol=1e-12, events=_loss_event(1e-20))
    p, q = s2.y[:d, -1], s2.y[d:, -1]
    return np.exp(2 * p) - np.exp(2 * q), s1.t[-1], s2.t[-1]


# kernel limit and rich-limit exponent
w_l2 = np.linalg.pinv(X) @ y
err_l2 = np.sum((w_l2 - w_star) ** 2)
S = np.sort(support)
Sc = np.setdiff1d(np.arange(d), S)
sgn = np.sign(w_star[S])
# best certificate: minimize t subject to X_S^T nu = sgn, |X_Sc^T nu| <= t
cvec = np.zeros(n + 1)
cvec[-1] = 1.0
A_ub = np.vstack([np.hstack([X[:, Sc].T, -np.ones((len(Sc), 1))]),
                  np.hstack([-X[:, Sc].T, -np.ones((len(Sc), 1))])])
A_eq = np.hstack([X[:, S].T, np.zeros((k, 1))])
lp = linprog(cvec, A_ub=A_ub, b_ub=np.zeros(2 * len(Sc)), A_eq=A_eq, b_eq=sgn,
             bounds=[(None, None)] * n + [(0, None)], method="highs")
gamma_star = 1.0 - lp.x[-1]

# theory curve, from large to small alpha (warm starts)
alphas = np.logspace(1.0, -4.0, 101)
theory, nu = [], None
for a in alphas:
    w, nu = phi_interpolator(a, nu)
    theory.append(np.sum((w - w_star) ** 2))
theory = np.array(theory)
# power-law guide through the smallest-alpha point, slope 4 gamma*
guide = theory[-1] * (alphas / alphas[-1]) ** (4.0 * gamma_star)

with open(os.path.join(OUT, "03-diagonal-sparse-theory.dat"), "w") as f:
    f.write("alpha err errl2 guide\n")
    for a, e, gd in zip(alphas[::-1], theory[::-1], guide[::-1]):
        f.write(f"{a:.6e} {e:.6e} {err_l2:.6e} {gd:.6e}\n")

# simulation points
sim_alphas = np.logspace(1.0, -4.0, 11)
rows, nu = [], None
for a in sim_alphas:
    w_gf, t_small, t_end = gradient_flow(a)
    w_th, nu = phi_interpolator(a, nu)
    rows.append((a, np.sum((w_gf - w_star) ** 2), np.linalg.norm(w_gf - w_th), t_small, t_end))
with open(os.path.join(OUT, "03-diagonal-sparse-sim.dat"), "w") as f:
    f.write("alpha err dist_to_theory t_loss_1e-9 t_loss_1e-20\n")
    for a, e, dist, t1, t2 in rows[::-1]:
        f.write(f"{a:.6e} {e:.6e} {dist:.3e} {t1:.3e} {t2:.3e}\n")

print(f"n={n} d={d} k={k}  err_l2={err_l2:.4f}  (1-n/d)|w*|^2={(1 - n / d):.4f}")
print(f"gamma*={gamma_star:.4f}  slope 4 gamma*={4 * gamma_star:.4f}")
for a, e, dist, t1, t2 in rows:
    print(f"alpha={a:.1e}  flow err={e:.4e}  |w_flow - w_theory|={dist:.1e}  "
          f"t(loss<1e-9)={t1:.2e}  t(loss<1e-20)={t2:.2e}")
