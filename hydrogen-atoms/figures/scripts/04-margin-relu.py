"""Chapter 4, figure fig:margin:relu.

A two-layer ReLU network f(x) = sum_r a_r relu(<w_r, x>) without biases (homogeneous
of degree 2), width m = 50, on n = 30 points of the unit circle labeled by
sign(x_1 x_2), which no linear classifier separates.  Initialization N(0, 0.3^2)
for all weights, seed 0, exponential loss.  Gradient flow is integrated by the
explicit Euler method in the clock tau with d tau = L dt (steps of size
dtau / L in t, dtau = 1e-3), which follows the flow while the loss decays.
Writes figures/data/04-margin-relu.dat with columns
  t      time of the flow
  loss   training loss L
  nm     normalized margin  min_i y_i f(x_i) / ||theta||^2
  sm     smoothed margin   -log(n L) / ||theta||^2
  up     sm + log(n) / ||theta||^2   (upper bound for nm)
  sq     ||theta||^2
"""
import numpy as np

rng = np.random.default_rng(0)
n, m, init, dtau, tmax = 30, 50, 0.3, 1e-3, 1e10
phi = rng.uniform(0, 2 * np.pi, n)
X = np.c_[np.cos(phi), np.sin(phi)]
y = np.sign(np.cos(phi) * np.sin(phi))
W = init * rng.standard_normal((m, 2))
a = init * rng.standard_normal(m)
t = 0.0
grid = 10 ** np.arange(-1, 10.01, 0.1)
k = 0
rows = []
prev, drop = -np.inf, 0.0
while t < tmax:
    Z = X @ W.T
    H = np.maximum(Z, 0)
    f = H @ a
    q = y * f
    ex = np.exp(-q)
    L = ex.mean()
    sq = (W * W).sum() + a @ a
    sm = -np.log(n * L) / sq
    if L < 1 / n:
        drop = max(drop, prev - sm)
        prev = sm
    while k < len(grid) and t >= grid[k]:
        rows.append([t, L, q.min() / sq, sm, sm + np.log(n) / sq, sq])
        k += 1
    e = ex * y / n
    ga = H.T @ e
    gW = ((e[:, None] * (Z > 0)) * a).T @ X
    h = dtau / L
    W += h * gW
    a += h * ga
    t += h
rows = np.array(rows)
print("largest decrease of the smoothed margin after L < 1/n:", drop)
print("first time with L < 1/n:", rows[np.argmax(rows[:, 1] < 1 / n), 0])
print("final normalized margin", rows[-1, 2], "smoothed", rows[-1, 3], "||theta||^2", rows[-1, 5])
np.savetxt('figures/data/04-margin-relu.dat', rows, header='t loss nm sm up sq', comments='', fmt='%.6g')
