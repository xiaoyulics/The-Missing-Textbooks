"""Numbers drawn in the volume's ornaments (laboratory.sty).

1. The foot of each chapter opening: sixteen draws from a standard normal (NumPy
   default_rng, seed 3407), their running mean, and a 95% t-interval for the mean.
2. The title page: a scaling law L(u) = E + A * 10^(-alpha * u), u = log10(compute),
   measured at six scales with five runs each (multiplicative noise), fitted by least
   squares in log L, with a 95% band for the fitted curve from a bootstrap over runs.

Prints TikZ-ready coordinates in millimetres. Run: python3 figures/scripts/00-emblem.py
"""
import numpy as np
from scipy import optimize, stats

# --- 1. running mean of sixteen runs -------------------------------------------
rng = np.random.default_rng(3407)
y = rng.standard_normal(16)
print("% runs: j / value / running mean / 95% t half-width (units of sigma)")
for j in range(1, 17):
    m = y[:j].mean()
    h = stats.t.ppf(0.975, j - 1) * y[:j].std(ddof=1) / np.sqrt(j) if j > 1 else 0.0
    print(f"{j}/{y[j-1]:.3f}/{m:.3f}/{h:.3f},")

# --- 2. scaling law, fit and bootstrap band ----------------------------------------
E, A, alpha = 0.80, 7.2, 0.30            # true law (never drawn)
U = np.arange(0.25, 3.0, 0.5)            # six scales, in decades of compute
R = 5                                    # runs per scale
noise = 0.05                             # multiplicative run-to-run noise
rng = np.random.default_rng(3407)
L = (E + A * 10 ** (-alpha * U))[:, None] * np.exp(noise * rng.standard_normal((len(U), R)))

def law(u, p):
    e, a, al = p
    return e + a * 10 ** (-al * u)

def fit(Lruns):
    u = np.repeat(U, Lruns.shape[1]); l = Lruns.ravel()
    res = optimize.least_squares(lambda p: np.log(law(u, p)) - np.log(l), x0=[0.5, 6.0, 0.35],
                                 bounds=([0, 0, 0.01], [5, 50, 2]))
    return res.x

p_hat = fit(L)
print("% fitted E, A, alpha:", np.round(p_hat, 4))
B = 2000
boot = np.empty((B, 3))
for b in range(B):
    idx = rng.integers(0, R, size=(len(U), R))
    boot[b] = fit(np.take_along_axis(L, idx, axis=1))
ugrid = np.linspace(-0.7, 5.0, 58)
curves = np.array([law(ugrid, p) for p in boot])
lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
mid = law(ugrid, p_hat)

def mm(u, l):       # page coordinates: 35 mm per decade of compute, 60 mm per decade of loss
    return 24 + 35 * u, 22 + 60 * np.log10(l)

print("% fitted curve (x mm, y mm)")
print(" ".join(f"({x:.2f},{yy:.2f})" for x, yy in zip(*mm(ugrid, mid))))
print("% band, lower then upper")
print(" ".join(f"({x:.2f},{yy:.2f})" for x, yy in zip(*mm(ugrid, lo))))
print(" ".join(f"({x:.2f},{yy:.2f})" for x, yy in zip(*mm(ugrid, hi))))
print("% runs (x mm, y mm) and means with 95% t-interval (x, y, ylo, yhi)")
for i, u in enumerate(U):
    xs, ys = mm(np.full(R, u), L[i])
    print("runs", " ".join(f"({x + 0.9 * (k - 2):.2f},{yy:.2f})" for k, (x, yy) in enumerate(zip(xs, ys))))
    lm = np.log(L[i]); m = lm.mean(); h = stats.t.ppf(0.975, R - 1) * lm.std(ddof=1) / np.sqrt(R)
    x0, y0 = mm(u, np.exp(m)); _, ylo = mm(u, np.exp(m - h)); _, yhi = mm(u, np.exp(m + h))
    print(f"mean {x0:.2f}/{y0:.2f}/{ylo:.2f}/{yhi:.2f}")
print("% band half-width in loss decades at u = 2.75 and u = 5:",
      np.round(np.interp([2.75, 5.0], ugrid, (np.log10(hi) - np.log10(lo)) / 2), 4))
