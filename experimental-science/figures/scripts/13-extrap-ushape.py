"""Lab 13.2: a non-monotone scaling curve, and the trends that a short range of it suggests.

Digits with 20% of the training labels replaced by random labels (labkit's label_noise).
A one-hidden-layer ReLU network of width w (N = 75 w + 10 parameters) is trained by Adam at
learning rate 0.001, constant schedule, batch 64, for 200 epochs, long enough for the wider
networks to fit every training label, noisy ones included. Eighteen widths from 2 to 2048,
six runs per width; run s uses seed s for initialization, data order, split and label noise.

The test error first falls with width, then rises toward the width at which the network
can just fit the noisy training set, then falls again (double descent). Two windows of the
curve are fitted by a straight line in (ln N, ln error) on all runs in the window, and the
fit is extrapolated with the t prediction interval for one new run: window A
on the first descent and window B on the rise, the miniature of an inverse-scaling trend.
The local slope of the curve over sliding windows of three widths is also reported.

Writes figures/data/13-extrapolation-ushape-*.dat and prints the numbers quoted in the text.
About two to three minutes with two threads.
Run from the volume root: python3 figures/scripts/13-extrap-ushape.py
"""
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk  # noqa: E402

T0 = time.time()
WIDTHS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048]
SEEDS = range(6)
NOISE = 0.2
OUT = "figures/data/13-extrapolation-ushape"


def train():
    err = np.zeros((len(WIDTHS), len(SEEDS)))
    tr = np.zeros_like(err)
    N = np.zeros(len(WIDTHS), dtype=int)
    for i, w in enumerate(WIDTHS):
        t = time.time()
        for j, s in enumerate(SEEDS):
            cfg = lk.Config(data="digits", label_noise=NOISE, width=w, depth=1, optimizer="adam",
                            lr=1e-3, epochs=200, batch_size=64, schedule="constant")
            r = lk.run(cfg, init=s, order=s, split=s, noise=s)
            err[i, j] = r["test_err"]
            tr[i, j] = 1.0 - r["train_acc"]
            N[i] = r["params"]
        print(f"  w={w:5d} N={N[i]:7d} test {100 * err[i].mean():5.2f}%  train {100 * tr[i].mean():5.2f}%"
              f"  ({(time.time() - t) / len(SEEDS):.2f}s per run)", flush=True)
    return err, tr, N


cache = os.environ.get("EXTRAP_CACHE")
if cache and os.path.exists(cache):
    z = np.load(cache)
    err, tr, N = z["err"], z["tr"], z["N"]
    print(f"loaded cached runs from {cache}")
else:
    err, tr, N = train()
    if cache:
        np.savez(cache, err=err, tr=tr, N=N)

n_s = err.shape[1]
tq = stats.t.ppf(0.975, n_s - 1)
m = 100 * err.mean(1)
hw = tq * 100 * err.std(1, ddof=1) / np.sqrt(n_s)
mtr = 100 * tr.mean(1)
print("\nmean test error (%) with 95% t-interval, and training error on the noisy labels:")
for i, w in enumerate(WIDTHS):
    print(f"  w={w:5d} N={N[i]:7d}  test {m[i]:5.2f} +- {hw[i]:4.2f}   train {mtr[i]:5.2f}")
imin = int(np.argmin(m[:10]))
ipk = imin + int(np.argmax(m[imin:]))
print(f"first minimum at w={WIDTHS[imin]} ({m[imin]:.2f}%), peak at w={WIDTHS[ipk]} ({m[ipk]:.2f}%), "
      f"largest width {WIDTHS[-1]}: {m[-1]:.2f}%")
thr = [WIDTHS[i] for i in range(len(WIDTHS)) if mtr[i] < 1.0]
print(f"smallest width with mean training error below 1%: {thr[0] if thr else None}; "
      f"20% of 1258 training labels are noisy, of which 9/10 are wrong (about 226 examples)")

lk.write_dat(OUT + "-runs.dat", {"N": np.repeat(N, n_s), "err": 100 * err.ravel(),
                                 "train": 100 * tr.ravel()})
lk.write_dat(OUT + "-means.dat", {"N": N, "mean": m, "hw": hw, "train": mtr})

x = np.log(N.astype(float))
Yl = np.log(err)


def window_fit(idx):
    """OLS of ln(error) on ln N over all runs at the widths idx; returns a function giving the
    prediction and the half-width of the 95% interval for one new run (or for the mean)."""
    xf = np.repeat(x[idx], n_s)
    yf = Yl[idx].ravel()
    n = len(xf)
    xb = xf.mean()
    sxx = ((xf - xb) ** 2).sum()
    b = ((xf - xb) * yf).sum() / sxx
    a = yf.mean() - b * xb
    s2 = ((yf - a - b * xf) ** 2).sum() / (n - 2)
    tt = stats.t.ppf(0.975, n - 2)

    def pred(x0, new_run=True):
        return a + b * x0, tt * np.sqrt(s2 * ((1.0 if new_run else 0.0) + 1 / n + (x0 - xb) ** 2 / sxx))
    return pred, (b, tt * np.sqrt(s2 / sxx)), np.sqrt(s2), n


grid = np.log(np.geomspace(N[0] * 0.85, N[-1] * 1.2, 90))
for name, ws in [("A", [4, 6, 8, 12]), ("B", [16, 24, 32, 48, 64])]:
    idx = np.array([WIDTHS.index(w) for w in ws])
    pred, (b, hb), s, n = window_fit(idx)
    print(f"\nwindow {name}: widths {ws}, N {N[idx[0]]}-{N[idx[-1]]}, {n} runs; slope {b:.3f} per ln N "
          f"(95% interval {b - hb:.3f} to {b + hb:.3f}; {b * np.log(10):.3f} per decade), residual sd {s:.3f}")
    g = np.sort(np.r_[grid[grid > x[idx[0]]], x[idx[0]], x[idx[-1]]])
    p, h = pred(g)
    inside = g <= x[idx[-1]] + 1e-9
    outside = g >= x[idx[-1]] - 1e-9
    nan = np.full(len(g), np.nan)
    lk.write_dat(OUT + f"-fit{name}.dat", {"N": np.exp(g), "predin": np.where(inside, 100 * np.exp(p), nan),
                                          "predout": np.where(outside, 100 * np.exp(p), nan),
                                          "lo": 100 * np.exp(p - h), "hi": 100 * np.exp(np.minimum(p + h, 0))})
    tot = cov = 0
    for i in range(idx[-1] + 1, len(WIDTHS)):
        p0, h0 = pred(x[i])
        inside = np.abs(Yl[i] - p0) <= h0
        tot += n_s
        cov += inside.sum()
        if WIDTHS[i] in (64, 128, 256, 512, 1024, 2048):
            print(f"  w={WIDTHS[i]:5d}: predicted {100 * np.exp(p0):6.2f}% [{100 * np.exp(p0 - h0):.2f}, "
                  f"{100 * np.exp(min(p0 + h0, 0)):.2f}]  observed mean {m[i]:.2f}%  runs inside {inside.sum()}/{n_s}")
    print(f"  runs beyond the window inside their 95% prediction interval: {cov}/{tot}")

# the first descent fitted by a law with a floor, err = E + A N^-alpha (least squares in logs)
from scipy import optimize  # noqa: E402
idx = np.array([WIDTHS.index(w) for w in [4, 6, 8, 12]])
xf = np.repeat(x[idx], n_s)
yf = Yl[idx].ravel()
best = None
for lnE in [np.log(0.05), np.log(0.09), np.log(0.1)]:
    for al in [0.5, 1.0, 2.0, 4.0]:
        sol = optimize.least_squares(lambda p: np.logaddexp(p[0], p[1] - p[2] * xf) - yf,
                                     [lnE, yf[0] + al * xf[0], al], method="lm", max_nfev=4000)
        if best is None or sol.cost < best.cost:
            best = sol
print(f"\nwindow A fitted with a floor: E = {100 * np.exp(best.x[0]):.2f}%, alpha = {best.x[2]:.2f}; "
      f"predicted at w=64: {100 * np.exp(np.logaddexp(best.x[0], best.x[1] - best.x[2] * x[WIDTHS.index(64)])):.2f}%, "
      f"at w=2048: {100 * np.exp(np.logaddexp(best.x[0], best.x[1] - best.x[2] * x[-1])):.2f}%")

# local slopes over sliding windows of three consecutive widths
cen, sl, lo, hi = [], [], [], []
for i in range(len(WIDTHS) - 2):
    idx = np.arange(i, i + 3)
    xf = np.repeat(x[idx], n_s)
    yf = Yl[idx].ravel()
    res = stats.linregress(xf, yf)
    tt = stats.t.ppf(0.975, len(xf) - 2)
    cen.append(np.exp(x[idx].mean()))
    sl.append(res.slope)
    lo.append(res.slope - tt * res.stderr)
    hi.append(res.slope + tt * res.stderr)
cen, sl, lo, hi = map(np.array, (cen, sl, lo, hi))
lk.write_dat(OUT + "-slopes.dat", {"N": cen, "slope": sl, "lo": lo, "hi": hi})
print("\nlocal slope d ln(error) / d ln N over three consecutive widths (95% interval):")
for c_, s_, l_, h_ in zip(cen, sl, lo, hi):
    sign = "+" if l_ > 0 else ("-" if h_ < 0 else "0")
    print(f"  N~{c_:8.0f}: {s_:+.3f} [{l_:+.3f}, {h_:+.3f}] {sign}")

print(f"\ntotal time {time.time() - T0:.0f}s")
