"""Lab 13.1: backtesting extrapolations of a scaling curve on the teacher task.

A ladder of 13 networks with two hidden layers of width 4 to 256 (61 to 68,353 parameters)
is trained under the online protocol: one pass over D = 65,536 fresh examples, Adam at
learning rate 0.01 (the best of 0.001, 0.003 and 0.01 at every width in a pilot), cosine
schedule, batch 64. Thirty ladders are trained, every run with its own seeds (init, order
and split all equal to 1000 r + j for ladder r and rung j), so that the errors of different
runs are independent.

Each backtest fits the k smallest rungs of one ladder (k = 5, 6, 7, 8) and predicts the log
test loss of every larger rung of the same ladder, with a 95% prediction interval, by two
forms:
  line  ln L = a + b ln N, ordinary least squares, the t prediction interval for one new run
        (exact if the form is right and the errors are normal);
  sat   ln L = ln(E + A N^-alpha), nonlinear least squares in log space with a grid of
        starting points, the delta-method prediction interval with n - 3 degrees of freedom.
Coverage is the fraction of the 30 ladders whose observed run falls in the interval, with a
Clopper-Pearson interval. A positive control repeats both procedures on
simulated ladders drawn from a saturating law with normal errors of the observed size. The
misspecification proposition is checked against the bias of the line.

Writes figures/data/13-extrapolation-backtest-*.dat and prints the numbers quoted in the
text. About two to three minutes with two threads.
Run from the volume root: python3 figures/scripts/13-extrap-backtest.py
"""
import os
import sys
import time

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk  # noqa: E402

T0 = time.time()
D = 2 ** 16
LR = 0.01
WIDTHS = [4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256]
R = 30
WINDOWS = [5, 6, 7, 8]
SHOW_K = 6          # the window shown in the figure
OUT = "figures/data/13-extrapolation-backtest"


def train_ladders():
    L = np.zeros((R, len(WIDTHS)))
    N = np.zeros(len(WIDTHS), dtype=int)
    for r in range(R):
        for j, w in enumerate(WIDTHS):
            cfg = lk.Config(data="teacher", train_size=D, epochs=1, schedule="cosine", width=w,
                            optimizer="adam", lr=LR, batch_size=64)
            s = 1000 * r + j
            res = lk.run(cfg, init=s, order=s, split=s)
            L[r, j] = res["test_loss"]
            N[j] = res["params"]
        if r % 10 == 9:
            print(f"  trained {r + 1} ladders ({time.time() - T0:.0f}s)", flush=True)
    return L, N


cache = os.environ.get("EXTRAP_CACHE")
if cache and os.path.exists(cache):
    z = np.load(cache)
    L, N = z["L"][:R], z["N"]
    print(f"loaded cached runs from {cache}")
else:
    L, N = train_ladders()
    if cache:
        np.savez(cache, L=L, N=N)
x = np.log(N.astype(float))
Y = np.log(L)
nr = len(WIDTHS)


# ------------------------------------------------------------------------------------------
# the two forms and their prediction intervals
# ------------------------------------------------------------------------------------------
def line_fit(xf, yf):
    n = len(xf)
    xb = xf.mean()
    sxx = ((xf - xb) ** 2).sum()
    b = ((xf - xb) * yf).sum() / sxx
    a = yf.mean() - b * xb
    res = yf - a - b * xf
    s2 = (res ** 2).sum() / (n - 2)
    return a, b, s2, xb, sxx


def line_pred(fit, n, x0, new_run=True):
    a, b, s2, xb, sxx = fit
    pred = a + b * x0
    var = s2 * ((1.0 if new_run else 0.0) + 1.0 / n + (x0 - xb) ** 2 / sxx)
    return pred, stats.t.ppf(0.975, n - 2) * np.sqrt(var)


def sat_model(p, xx):
    with np.errstate(over="ignore"):
        return np.logaddexp(p[0], p[1] - p[2] * xx)


def sat_fit(xf, yf):
    best = None
    for lnE in [np.log(1e-4), yf.min() - 0.3, yf.min() - 0.05]:
        for al in [0.1, 0.3, 0.6, 1.0]:
            p0 = [lnE, yf[0] + al * xf[0], al]
            try:
                sol = optimize.least_squares(lambda p: sat_model(p, xf) - yf, p0, method="lm",
                                             max_nfev=4000)
            except (ValueError, FloatingPointError):
                continue
            if np.isfinite(sol.cost) and (best is None or sol.cost < best.cost):
                best = sol
    n = len(xf)
    s2 = 2 * best.cost / (n - 3)
    cov = s2 * np.linalg.pinv(best.jac.T @ best.jac)
    return best.x, s2, cov


def sat_pred(fit, n, x0, new_run=True):
    p, s2, cov = fit
    lnf = sat_model(p, np.atleast_1d(x0))
    wE = np.exp(p[0] - lnf)                      # d ln f / d ln E
    wT = 1.0 - wE                                # d ln f / d ln A
    g = np.stack([wE, wT, -np.atleast_1d(x0) * wT])
    var = np.einsum("in,ij,jn->n", g, cov, g) + (s2 if new_run else 0.0)
    return lnf, stats.t.ppf(0.975, n - 3) * np.sqrt(np.maximum(var, 0))


def clopper(k, n):
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return lo, hi


# ------------------------------------------------------------------------------------------
# description of the runs
# ------------------------------------------------------------------------------------------
print(f"\nladder: widths {WIDTHS}")
print(f"parameters N: {[int(v) for v in N]}  ({np.log10(N[-1] / N[0]):.2f} decades)")
mL = L.mean(0)
hwL = stats.t.ppf(0.975, R - 1) * L.std(0, ddof=1) / np.sqrt(R)
sdlog = Y.std(0, ddof=1)
for j in range(nr):
    print(f"  w={WIDTHS[j]:4d} N={N[j]:6d} mean loss {mL[j]:.5f} +- {hwL[j]:.5f}  sd(ln L) {sdlog[j]:.3f}")
print(f"teacher noise floor 0.01; mean loss at the largest rung {mL[-1]:.4f}")
loc = np.diff(Y.mean(0)) / np.diff(x)
print("local slopes d ln L / d ln N between rungs:", np.round(loc, 3))
lk.write_dat(OUT + "-runs.dat", {"N": np.tile(N, R), "loss": L.ravel(),
                                 "ladder": np.repeat(np.arange(R), nr)})
lk.write_dat(OUT + "-means.dat", {"N": N, "mean": mL, "hw": hwL})

# ------------------------------------------------------------------------------------------
# backtests
# ------------------------------------------------------------------------------------------
results = {}
for k in WINDOWS:
    tgt = np.arange(k, nr)
    cov_l = np.zeros(len(tgt), dtype=int)
    cov_s = np.zeros(len(tgt), dtype=int)
    in_l = np.zeros((R, len(tgt)), dtype=bool)
    in_s = np.zeros((R, len(tgt)), dtype=bool)
    err_l = np.zeros((R, len(tgt)))
    err_s = np.zeros((R, len(tgt)))
    hw_l = np.zeros((R, len(tgt)))
    hw_s = np.zeros((R, len(tgt)))
    for r in range(R):
        fl = line_fit(x[:k], Y[r, :k])
        fs = sat_fit(x[:k], Y[r, :k])
        pl, hl = line_pred(fl, k, x[tgt])
        ps, hs = sat_pred(fs, k, x[tgt])
        err_l[r] = pl - Y[r, tgt]
        err_s[r] = ps - Y[r, tgt]
        hw_l[r], hw_s[r] = hl, hs
        in_l[r] = np.abs(err_l[r]) <= hl
        in_s[r] = np.abs(err_s[r]) <= hs
        cov_l += in_l[r]
        cov_s += in_s[r]
    dist = np.log10(N[tgt] / N[k - 1])
    results[k] = dict(tgt=tgt, dist=dist, cov_l=cov_l, cov_s=cov_s, err_l=err_l, err_s=err_s,
                      hw_l=hw_l, hw_s=hw_s, in_l=in_l, in_s=in_s)
    print(f"\nwindow k={k}: fit N in [{N[0]}, {N[k - 1]}] ({np.log10(N[k - 1] / N[0]):.2f} decades)")
    print("  decades beyond      :", np.round(dist, 2))
    print("  line coverage       :", np.round(cov_l / R, 3))
    print("  line mean error ln  :", np.round(err_l.mean(0), 3))
    print("  line median halfwid :", np.round(np.median(hw_l, 0), 3))
    print("  sat  coverage       :", np.round(cov_s / R, 3))
    print("  sat  mean error ln  :", np.round(err_s.mean(0), 3))
    print("  sat  median halfwid :", np.round(np.median(hw_s, 0), 3))

tot_bt = sum(R * len(results[k]["tgt"]) for k in WINDOWS)
cl = sum(results[k]["cov_l"].sum() for k in WINDOWS)
cs = sum(results[k]["cov_s"].sum() for k in WINDOWS)
print(f"\n{R * len(WINDOWS)} backtests and {tot_bt} forecasts per form; "
      f"overall coverage line {cl / tot_bt:.3f}, sat {cs / tot_bt:.3f}")
# pooled coverage by distance; the forecasts of one ladder share its noise, so the interval
# is a percentile bootstrap that resamples whole ladders
rng_b = np.random.default_rng(7)
boot = rng_b.integers(0, R, size=(4000, R))
for lab, lo_d, hi_d in [("within 0.55 decades", 0, 0.55), ("0.55-1.2 decades", 0.55, 1.2),
                        ("beyond 1.2 decades", 1.2, 9)]:
    per_l = np.zeros(R)
    per_s = np.zeros(R)
    cells = 0
    for k in WINDOWS:
        m = (results[k]["dist"] > lo_d) & (results[k]["dist"] <= hi_d)
        per_l += results[k]["in_l"][:, m].sum(1)
        per_s += results[k]["in_s"][:, m].sum(1)
        cells += m.sum()
    bl = per_l[boot].sum(1) / (cells * R)
    bs = per_s[boot].sum(1) / (cells * R)
    print(f"  {lab:22s}: n={cells * R:4d}  line {per_l.sum() / (cells * R):.3f} "
          f"[{np.quantile(bl, 0.025):.3f}, {np.quantile(bl, 0.975):.3f}]  "
          f"sat {per_s.sum() / (cells * R):.3f} [{np.quantile(bs, 0.025):.3f}, {np.quantile(bs, 0.975):.3f}]"
          f"  (ladder bootstrap)")

# ------------------------------------------------------------------------------------------
# positive control: the same procedures on simulated ladders that follow a saturating law
# exactly, with normal errors of the observed size at each rung
# ------------------------------------------------------------------------------------------
print("\npositive control (simulated ladders from a saturating law, normal errors):")
ybar = Y.mean(0)
pfull = sat_fit(x, ybar)[0]
mu = sat_model(pfull, x)
rng = np.random.default_rng(13)
S = 600
control = {}
for k in [SHOW_K, 8]:
    tgt = np.arange(k, nr)
    cnt_s = np.zeros(len(tgt))
    cnt_l = np.zeros(len(tgt))
    for _ in range(S):
        ysim = mu + sdlog * rng.standard_normal(nr)
        ps, hs = sat_pred(sat_fit(x[:k], ysim[:k]), k, x[tgt])
        cnt_s += np.abs(ps - ysim[tgt]) <= hs
        pl, hl = line_pred(line_fit(x[:k], ysim[:k]), k, x[tgt])
        cnt_l += np.abs(pl - ysim[tgt]) <= hl
    control[k] = (cnt_l / S, cnt_s / S)
    print(f"  k={k}: sat coverage {np.round(cnt_s / S, 3)}")
    print(f"        line coverage {np.round(cnt_l / S, 3)}")
print(f"  (law used: E={np.exp(pfull[0]):.4f}, A={np.exp(pfull[1]):.3f}, alpha={pfull[2]:.3f})")

# coverage table for the figure (window SHOW_K and window 8)
for k in [SHOW_K, 8]:
    res = results[k]
    cols = {"dist": res["dist"], "N": N[res["tgt"]]}
    for form in ["l", "s"]:
        c = res["cov_" + form]
        lo_hi = np.array([clopper(ci, R) for ci in c])
        cols["cov" + form] = c / R
        cols["lo" + form] = lo_hi[:, 0]
        cols["hi" + form] = lo_hi[:, 1]
        cols["hw" + form] = np.median(res["hw_" + form], 0)
        cols["err" + form] = res["err_" + form].mean(0)
    cols["ctll"], cols["ctls"] = control[k]
    lk.write_dat(OUT + f"-coverage-k{k}.dat", cols)

# ------------------------------------------------------------------------------------------
# one backtest drawn in full: ladder 0, window SHOW_K
# ------------------------------------------------------------------------------------------
r0 = 0
k = SHOW_K
fl = line_fit(x[:k], Y[r0, :k])
fs = sat_fit(x[:k], Y[r0, :k])
grid = np.sort(np.r_[np.log(np.geomspace(N[0] * 0.9, N[-1] * 1.25, 80)), x[k - 1]])
pl, hl = line_pred(fl, k, grid)
ps, hs = sat_pred(fs, k, grid)
inside = grid <= x[k - 1] + 1e-9
outside = grid >= x[k - 1] - 1e-9
nan = np.full(len(grid), np.nan)
lk.write_dat(OUT + "-fit.dat", {
    "N": np.exp(grid), "linein": np.where(inside, np.exp(pl), nan), "lineout": np.where(outside, np.exp(pl), nan),
    "linelo": np.exp(pl - hl), "linehi": np.exp(pl + hl),
    "satin": np.where(inside, np.exp(ps), nan), "satout": np.where(outside, np.exp(ps), nan),
    "satlo": np.exp(ps - hs), "sathi": np.exp(ps + hs)})
lk.write_dat(OUT + "-ladder0.dat", {"N": N, "fitted": np.where(np.arange(nr) < k, L[r0], np.nan),
                                    "heldout": np.where(np.arange(nr) >= k, L[r0], np.nan)})
p, s2, cov = fs
print(f"\nladder 0, window k={k}: line slope {fl[1]:.3f}, residual sd {np.sqrt(fl[2]):.3f}")
print(f"  saturating law: E={np.exp(p[0]):.4f}, A={np.exp(p[1]):.3f}, alpha={p[2]:.3f}, "
      f"residual sd {np.sqrt(s2):.3f}, sd(ln E)={np.sqrt(cov[0, 0]):.2f}")
pl_end, hl_end = line_pred(fl, k, x[-1])
ps_end, hs_end = sat_pred(fs, k, x[-1])
print(f"  at N={N[-1]}: observed {L[r0, -1]:.4f}; line {np.exp(pl_end):.4f} "
      f"[{np.exp(pl_end - hl_end):.4f}, {np.exp(pl_end + hl_end):.4f}]; "
      f"sat {np.exp(ps_end[0]):.4f} [{np.exp(ps_end[0] - hs_end[0]):.4f}, {np.exp(ps_end[0] + hs_end[0]):.4f}]")
Es = []
for r in range(R):
    Es.append(np.exp(sat_fit(x[:k], Y[r, :k])[0][0]))
Es = np.array(Es)
print(f"  fitted E over the {R} ladders (k={k}): median {np.median(Es):.4f}, "
      f"quartiles {np.quantile(Es, 0.25):.4f}-{np.quantile(Es, 0.75):.4f}, "
      f"min {Es.min():.2e}, max {Es.max():.4f}")

# ------------------------------------------------------------------------------------------
# the misspecification proposition: bias of the line from the curvature in the window
# ------------------------------------------------------------------------------------------
print("\nmisspecification check (bias of the line, natural-log units):")
for k in WINDOWS:
    xf = x[:k]
    xb = xf.mean()
    u = xf - xb
    m2, m3, m4 = (u ** 2).mean(), (u ** 3).mean(), (u ** 4).mean()
    c = np.polyfit(xf, ybar[:k], 2)[0]          # curvature of the mean curve in the window
    tgt = results[k]["tgt"]
    d = x[tgt] - xb
    r_x0 = d ** 2 - (m3 / m2) * d - m2
    pred_bias = -c * r_x0                       # E[prediction] - truth
    kappa = m4 - m2 ** 2 - m3 ** 2 / m2
    s2bar = np.mean([line_fit(xf, Y[r, :k])[2] for r in range(R)])
    noise2 = np.mean(sdlog[:k] ** 2)
    print(f"  k={k}: curvature c={c:.4f} per (ln N)^2; c*sqrt(kappa)={c * np.sqrt(kappa):.3f}; "
          f"mean S^2 {s2bar:.4f} vs noise {noise2:.4f} + c^2 n kappa/(n-2) {c ** 2 * k * kappa / (k - 2):.4f}")
    # can one ladder see the curvature? t-test of the quadratic term in each ladder's window
    rej = 0
    for r in range(R):
        Xq = np.c_[np.ones(k), u, u ** 2]
        beta, rss, *_ = np.linalg.lstsq(Xq, Y[r, :k], rcond=None)
        s2q = float(rss[0]) / (k - 3)
        se_c = np.sqrt(s2q * np.linalg.inv(Xq.T @ Xq)[2, 2])
        rej += abs(beta[2] / se_c) > stats.t.ppf(0.975, k - 3)
    se_one = np.sqrt(noise2 / (k * kappa))
    print(f"     one ladder: standard error of the curvature about {se_one:.3f}; "
          f"quadratic term significant at 5% in {rej}/{R} ladders")
    # the curvature of the mean curve is the mean of the ladders' curvatures; its 95% interval
    cs_r = np.array([np.polyfit(xf, Y[r, :k], 2)[0] for r in range(R)])
    hc = stats.t.ppf(0.975, R - 1) * cs_r.std(ddof=1) / np.sqrt(R)
    print(f"     curvature over ladders: {cs_r.mean():.4f}, 95% interval {cs_r.mean() - hc:.4f} to {cs_r.mean() + hc:.4f}")
    print("     predicted bias :", np.round(pred_bias, 4))
    print("       range over the curvature's interval:", np.round(-(c - hc) * r_x0, 3), "to", np.round(-(c + hc) * r_x0, 3))
    obshw = stats.t.ppf(0.975, R - 1) * results[k]["err_l"].std(0, ddof=1) / np.sqrt(R)
    print("     observed bias  :", np.round(results[k]["err_l"].mean(0), 3), "+-", np.round(obshw, 3))
    print("     predicted / observed:", np.round(pred_bias / results[k]["err_l"].mean(0), 2))
    if k == SHOW_K:
        lk.write_dat(OUT + "-bias.dat", {"dist": results[k]["dist"], "pred": pred_bias,
                                         "obs": results[k]["err_l"].mean(0),
                                         "obshw": obshw})

print(f"\ntotal time {time.time() - T0:.0f}s")
