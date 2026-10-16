"""Chapter 10 (grokking): a diagonal linear network with weight decay groks.

Model (Chapter 3 plus weight decay): w = u*u - v*v in R^d, u(0) = v(0) = alpha*1,
gradient flow on (1/2n)||Xw - y||^2 + (lambda/2)(||u||^2 + ||v||^2), sparse target.
Simulation: the (u, v) gradient flow integrated with a stiff solver (BDF).
Theory: w(t) is the Phi_{a}-minimal point of its fibre with a = alpha*exp(-lambda*t);
the quasi-static prediction is the test error of argmin{Phi_a(w) : Xw = y}.

Writes figures/data/10-grokking-diagonal.dat (one run: train/test error versus t)
and figures/data/10-grokking-delay.dat (half-time of the test error versus alpha, lambda).
Runs in about a minute on a laptop CPU.
"""
import os
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
rng = np.random.default_rng(0)
d, n, k = 100, 30, 3
X = rng.standard_normal((n, d))
w_star = np.zeros(d)
w_star[:k] = 1.0
y = X @ w_star
H = X.T @ X / n


def phi_interpolator(a, nu=None):
    """argmin Phi_a(w) s.t. Xw = y, via its dual: w = 2a^2 sinh(4 X^T nu)."""
    nu = np.zeros(n) if nu is None else nu.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(400):
            z = 4 * X.T @ nu
            w = 2 * a * a * np.sinh(z)
            r = X @ w - y
            rn = np.linalg.norm(r)
            if rn < 1e-12:
                break
            J = X @ ((8 * a * a * np.cosh(z))[:, None] * X.T)
            step = np.linalg.solve(J, r)
            t = 1.0
            while t > 1e-10:
                nn = nu - t * step
                w2 = 2 * a * a * np.sinh(4 * X.T @ nn)
                r2 = np.linalg.norm(X @ w2 - y)
                if np.isfinite(r2) and r2 < rn:
                    break
                t /= 2
            nu = nn
    return w, nu


def rel_test(w):
    return np.sum((w - w_star) ** 2) / np.sum(w_star ** 2)


# static curve: test error of the Phi_a interpolator as a function of a
a_grid = np.logspace(2, -4, 400)
static, nu = [], None
for a in a_grid:
    w, nu = phi_interpolator(a, nu)
    static.append(rel_test(w))
static = np.array(static)
plateau = static[0]                      # min-l2 interpolator (a -> infinity)
half = plateau / 2
# a_half: where the static curve crosses half the plateau (log-linear interpolation)
i = np.argmax(static < half)
la = np.interp(half, [static[i], static[i - 1]], [np.log(a_grid[i]), np.log(a_grid[i - 1])])
a_half = np.exp(la)
# heuristic estimate (Section 6): a^2 log(1/a^2) = ||w_2||^2 / (4 ||w_star||_1)
w2 = np.linalg.pinv(X) @ y
target = np.sum(w2 ** 2) / (4 * np.sum(np.abs(w_star)))
from scipy.optimize import brentq
a_heur = np.sqrt(brentq(lambda s: s * np.log(1 / s) - target, 1e-8, np.exp(-1)))
print(f"plateau {plateau:.4f}, a_half {a_half:.4f}, heuristic a_c {a_heur:.4f}")


def theory_at(a):
    return np.interp(np.log(a), np.log(a_grid[::-1]), static[::-1])


def simulate(alpha, lam, T, ts):
    def rhs(t, th):
        u, v = th[:d], th[d:]
        g = X.T @ (X @ (u * u - v * v) - y) / n
        return np.concatenate([-2 * u * g - lam * u, 2 * v * g - lam * v])

    def jac(t, th):
        u, v = th[:d], th[d:]
        g = X.T @ (X @ (u * u - v * v) - y) / n
        I = np.eye(d)
        Juu = -2 * np.diag(g) - lam * I - 4 * u[:, None] * H * u[None, :]
        Juv = 4 * u[:, None] * H * v[None, :]
        Jvu = 4 * v[:, None] * H * u[None, :]
        Jvv = 2 * np.diag(g) - lam * I - 4 * v[:, None] * H * v[None, :]
        return np.block([[Juu, Juv], [Jvu, Jvv]])

    th0 = alpha * np.ones(2 * d)
    sol = solve_ivp(rhs, (0, T), th0, method="BDF", jac=jac, t_eval=ts, rtol=1e-9, atol=1e-12)
    assert sol.success, sol.message
    W = sol.y[:d] ** 2 - sol.y[d:] ** 2
    train = np.sum((X @ W - y[:, None]) ** 2, axis=0) / np.sum(y ** 2)   # relative to w = 0
    test = np.sum((W - w_star[:, None]) ** 2, axis=0) / np.sum(w_star ** 2)
    return train, test


# one run, the figure of the chapter
alpha, lam = 10.0, 0.01
ts = np.logspace(-5, np.log10(3000), 160)
ts[-1] = 3000.0
train, test = simulate(alpha, lam, ts[-1], ts)
theory = np.array([theory_at(alpha * np.exp(-lam * t)) for t in ts])
with open(os.path.join(OUT, "10-grokking-diagonal.dat"), "w") as f:
    f.write("t trainrel test theory\n")
    for row in zip(ts, train, test, theory):
        f.write("%.6e %.6e %.6e %.6e\n" % row)
t_star = np.log(alpha / a_half) / lam
print(f"alpha={alpha}, lambda={lam}: predicted t* = {t_star:.1f}; heuristic {np.log(alpha / a_heur) / lam:.1f}")


def half_time(ts, test):
    j = np.argmax(test < half)
    return np.exp(np.interp(half, [test[j], test[j - 1]], [np.log(ts[j]), np.log(ts[j - 1])]))


# the delay: half-time of the test error for several alpha and lambda
alphas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
lams = [0.003, 0.01, 0.03]
lt = {}
for lam_ in lams:
    for alpha_ in alphas:
        T = 3 * np.log(alpha_ / a_half) / lam_ + 10 / lam_
        ts_ = np.logspace(-6, np.log10(T), 400)
        ts_[-1] = T
        tr_, te_ = simulate(alpha_, lam_, T, ts_)
        lt[(alpha_, lam_)] = lam_ * half_time(ts_, te_)
        print(f"alpha={alpha_:6.1f} lambda={lam_:.3f}: lambda*t_half sim {lt[(alpha_, lam_)]:.3f}, "
              f"theory {np.log(alpha_ / a_half):.3f}")
with open(os.path.join(OUT, "10-grokking-delay.dat"), "w") as f:
    f.write("alpha sim0003 sim001 sim003 theory heur\n")
    for alpha_ in alphas:
        f.write("%.4f %.6f %.6f %.6f %.6f %.6f\n" % (alpha_, lt[(alpha_, 0.003)], lt[(alpha_, 0.01)],
                                                   lt[(alpha_, 0.03)], np.log(alpha_ / a_half), np.log(alpha_ / a_heur)))
with open(os.path.join(OUT, "10-grokking-static.dat"), "w") as f:
    f.write("a test\n")
    for a, e in zip(a_grid[::4], static[::4]):
        f.write("%.6e %.6e\n" % (a, e))
