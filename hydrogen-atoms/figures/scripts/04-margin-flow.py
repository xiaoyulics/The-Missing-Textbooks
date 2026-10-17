"""Chapter 4, figure fig:margin:flow.

Gradient flow of the exponential loss on a separable two-dimensional data set
(n = 40, y_i x_i drawn from N(mu, 0.2^2 I) and rescaled to max norm 1, seed 31).
The flow is integrated exactly (LSODA, rtol 1e-10) in the time variable
s = log(1+t), up to t = 1e9.  Writes figures/data/04-margin-flow.dat with
  t        time
  lossx    gamma*^2 t L(t)                (simulation)
  lossbd   gamma*^2 t /(1+gamma*^2 t)     (theorem: upper bound on lossx)
  norm     ||w(t)||                        (simulation)
  normlo   log((1+g^2 t)/n)/g  (theorem lower bound, clipped at 0)
  normas   log(t)/g + <u*, w~>             (heuristic asymptote)
  marg     min_i y_i<w,x_i> / (g ||w||)    (simulation, normalized margin / gamma*)
  marglo   1 - log n / log((1+g^2 t)/n)    (theorem lower bound, clipped at 0)
  margas   same ratio for w = w^ log t + w~ (heuristic asymptote)
  angle    ||w/||w|| - u*||                (simulation)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

rng = np.random.default_rng(31)
n = 40
mu = np.array([0.55, 0.25])
Z = mu + 0.2 * rng.standard_normal((n, 2))
Z /= np.max(np.linalg.norm(Z, axis=1))        # rows are y_i x_i, max norm 1

# Max-margin direction: closest point of the convex hull of the y_i x_i to 0.
f = lambda q: 0.5 * np.sum((q @ Z) ** 2)
g = lambda q: Z @ (q @ Z)
res = minimize(f, np.ones(n) / n, jac=g, method='SLSQP', bounds=[(0, 1)] * n,
               constraints=[{'type': 'eq', 'fun': lambda q: q.sum() - 1}],
               options={'ftol': 1e-16, 'maxiter': 2000})
q = res.x
zs = q @ Z
gam = np.linalg.norm(zs)
us = zs / gam
what = us / gam
assert (Z @ us).min() > gam - 1e-8
sv = np.where(q > 1e-6)[0]
alpha = q / gam ** 2                            # w^ = sum alpha_i y_i x_i
wt = np.linalg.solve(Z[sv], -np.log(n * alpha[sv]))   # two support vectors in R^2
print(f"gamma* = {gam:.6f}, u* = {us}, support vectors {sv}, alpha = {alpha[sv]}, w~ = {wt}")
print("second smallest ratio y<w^,x>:", np.sort(Z @ what)[len(sv)])

def rhs(s, w):
    e = np.exp(-Z @ w)
    return np.exp(s) * (e @ Z) / n               # dt/ds = 1+t = e^s

s_end = np.log1p(1e9)
ss = np.linspace(np.log1p(1e-1), s_end, 90)
sol = solve_ivp(rhs, (0, s_end), np.zeros(2), t_eval=ss, rtol=1e-10, atol=1e-12, method='LSODA')
W = sol.y.T
t = np.expm1(ss)
L = np.mean(np.exp(-W @ Z.T), axis=1)
nw = np.linalg.norm(W, axis=1)
mm = (W @ Z.T).min(axis=1)
G = gam ** 2 * t
rows = []
for k in range(len(t)):
    wp = what * np.log(t[k]) + wt
    margas = (Z @ wp).min() / np.linalg.norm(wp) / gam if t[k] > 50 else np.nan
    rows.append([t[k], G[k] * L[k], G[k] / (1 + G[k]), nw[k],
                 max(np.log((1 + G[k]) / n) / gam, 0), np.log(t[k]) / gam + us @ wt,
                 mm[k] / nw[k] / gam, max(1 - np.log(n) / np.log((1 + G[k]) / n), 0) if 1 + G[k] > n else 0,
                 margas, np.linalg.norm(W[k] / nw[k] - us)])
rows = np.array(rows)
# sanity checks of the theorems
assert np.all(rows[:, 1] <= rows[:, 2] + 1e-9)
assert np.all(rows[:, 3] >= rows[:, 4] - 1e-9)
assert np.all(rows[:, 3] <= np.log(1 + G) / gam ** 2 + 1e-9)
assert np.all(rows[:, 6] >= rows[:, 7] - 1e-9)
assert np.all(L >= 1 / (1 + t) - 1e-12)
print("final: gamma^2 t L =", rows[-1, 1], " margin ratio =", rows[-1, 6], " heuristic =", rows[-1, 8],
      " norm =", rows[-1, 3], " heuristic norm =", rows[-1, 5], " angle =", rows[-1, 9])
np.savetxt('figures/data/04-margin-flow.dat', rows,
           header='t lossx lossbd norm normlo normas marg marglo margas angle', comments='', fmt='%.6g')
