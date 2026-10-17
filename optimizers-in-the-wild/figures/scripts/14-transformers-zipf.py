"""Chapter 14 (transformers): gradient descent, sign descent and Adam on the Zipf table model.

Model. K rows w_j in R^m (m = 16), frequencies q_j = 1/(j Z_K) with Z_K = 1 + 1/2 + ... + 1/K, loss
L(W) = sum_j q_j |w_j - y_j|^2 / 2, initialization W = 0.  Every row has the same target y, the
m quantiles of a Gaussian rescaled to |y| = 1, so every row starts with error norm 1 and L_0 = 1/2.

Theory (curves).
  gd<K>    exact GD loss with step 1/q_1:  (1/Z_K) sum_{j>=2} (1/j) (1 - 1/j)^(2t)   (Theorem zipf (ii))
  asy<K>   asymptotic line (ln(K/(2t)) - gamma)/Z_K, written only where it is positive
  sign     exact sign-descent loss of one row (Corollary): coordinate errors |y_i| - eta t, then a
           two-cycle; by the scale invariance it is the normalized loss for every K and every q
  adam     Adam (beta1 = 0.9, beta2 = 0.999, epsilon = 0) run on a single row of unit frequency; by the
           scale invariance (Proposition clock) it is the normalized loss for every K and every q
  b1..b4   exact GD loss of the frequency buckets j in [1,10), [10,100), [100,1000), [1000,10^4]
           for K = 10^4, each normalized by its own initial loss
Simulation (points): the optimizers actually run on the K x m model in floating point.
  gd<K>    full-batch GD with step 1/q_1 for K = 10^2, 10^3, 10^4 (m = 1 suffices for GD, see below)
  sign4, adam4   sign descent (eta = 0.005) and Adam (eta = 0.005) on the Zipf model with K = 10^4
  sign2, adam2   the same with K = 10^2
  gdb1..gdb4, adamb1..adamb4   bucket losses of GD and Adam for K = 10^4
GD acts on each coordinate separately with a factor that depends only on q_j, so its normalized loss
depends on the targets only through |y_j| = 1; the GD runs therefore use m = 1 (y_j = 1), which keeps
the 10^5-step runs cheap.  Deterministic (no randomness).  Runtime: about 30 seconds on one core.
Outputs: figures/data/14-transformers-zipf-curves.dat, 14-transformers-zipf-points.dat,
         14-transformers-zipf-buckets.dat, 14-transformers-zipf-bucketpts.dat
"""
import os
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
GAMMA = 0.5772156649015329
M = 16
ETA_SIGN = 0.005
ETA_ADAM = 0.005
B1, B2 = 0.9, 0.999


def target():
    y = norm.ppf((np.arange(M) + 0.5) / M)
    return y / np.linalg.norm(y)


def zipf(K):
    j = np.arange(1, K + 1, dtype=float)
    Z = np.sum(1.0 / j)
    return j, Z, 1.0 / (j * Z)


def gd_exact(K, t):
    j, Z, q = zipf(K)
    jj = j[1:]
    return np.array([np.sum((1 - 1 / jj) ** (2 * s) / jj) for s in t]) / Z


def bucket_exact(K, t, lo, hi):
    j = np.arange(lo, hi, dtype=float)
    w = 1.0 / j
    f = np.clip(1 - 1 / j, 0, None)
    return np.array([np.sum(w * f ** (2 * s)) for s in t]) / np.sum(w)


def sign_exact(t, y, eta):
    """Sign descent on one row from w = 0: coordinate error a = |y_i| shrinks by eta per step,
    then alternates between eta - r and r, r = a - n eta, n = floor(a/eta)."""
    a = np.abs(y)
    n = np.floor(a / eta)
    r = a - n * eta
    out = []
    for s in t:
        e = np.where(s <= n, a - s * eta, np.where(((s - n) % 2) == 1, eta - r, r))
        e = np.where((r == 0) & (s >= n), 0.0, e)
        out.append(np.sum(e ** 2) / np.sum(a ** 2))
    return np.array(out)


def run_rows(q, Y, opt, tmax, record, eta):
    """Run opt on L(W) = sum_j q_j |w_j - y_j|^2/2 from W = 0; return normalized losses at the
    record times, overall and per bucket (buckets given by a list of index arrays)."""
    W = np.zeros_like(Y)
    Mo = np.zeros_like(Y)
    V = np.zeros_like(Y)
    w = q[:, None]
    L0 = np.sum(w * 0.5 * Y ** 2)
    rec = {}
    rs = set(record)
    for s in range(tmax + 1):
        if s in rs:
            rec[s] = W.copy()
        if s == tmax:
            break
        G = w * (W - Y)
        if opt == "gd":
            W = W - eta * G
        elif opt == "sign":
            W = W - eta * np.sign(G)
        else:
            Mo = B1 * Mo + (1 - B1) * G
            V = B2 * V + (1 - B2) * G * G
            mh = Mo / (1 - B1 ** (s + 1))
            vh = V / (1 - B2 ** (s + 1))
            W = W - eta * mh / np.maximum(np.sqrt(vh), 1e-300)
    return rec, L0


def norm_loss(rec, q, Y, L0, idx=None):
    out = []
    for s in sorted(rec):
        E = rec[s] - Y
        if idx is None:
            out.append(np.sum(q[:, None] * 0.5 * E ** 2) / L0)
        else:
            qq = q[idx][:, None]
            out.append(np.sum(qq * 0.5 * E[idx] ** 2) / np.sum(qq * 0.5 * Y[idx] ** 2))
    return np.array(out)


def main():
    y = target()
    # theory curves on a log grid of t
    tc = np.unique(np.round(np.logspace(0, 5, 160)).astype(int))
    curves = {"t": tc}
    for K, name in [(100, "2"), (1000, "3"), (10000, "4")]:
        j, Z, q = zipf(K)
        curves["gd" + name] = gd_exact(K, tc)
        asy = (np.log(K / (2 * tc)) - GAMMA) / Z
        curves["asy" + name] = np.where(asy > 0, asy, np.nan)
    curves["sign"] = sign_exact(tc, y, ETA_SIGN)
    # Adam on one row of unit frequency (the scale-invariant prediction)
    one = np.ones(1)
    rec, L0 = run_rows(one, y[None, :], "adam", int(tc.max()), list(tc), ETA_ADAM)
    curves["adam"] = norm_loss(rec, one, y[None, :], L0)
    buckets = [(1, 10), (10, 100), (100, 1000), (1000, 10001)]
    bc = {"t": tc}
    for k, (lo, hi) in enumerate(buckets):
        bc["b%d" % (k + 1)] = bucket_exact(10000, tc, lo, hi)
    bc["adam"] = curves["adam"]

    # simulations
    tp = np.array([1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000,
                   10000, 20000, 30000, 50000, 100000])
    pts = {"t": tp}
    for K, name in [(100, "2"), (1000, "3"), (10000, "4")]:
        j, Z, q = zipf(K)
        Y1 = np.ones((K, 1))
        rec, L0 = run_rows(q, Y1, "gd", int(tp.max()), list(tp), 1.0 / q[0])
        pts["gd" + name] = norm_loss(rec, q, Y1, L0)
    bp = {"t": tp}
    for K, name in [(100, "2"), (10000, "4")]:
        j, Z, q = zipf(K)
        Y = np.tile(y, (K, 1))
        tmax = 5000
        tps = [s for s in tp if s <= tmax]
        for opt, eta in [("sign", ETA_SIGN), ("adam", ETA_ADAM)]:
            rec, L0 = run_rows(q, Y, opt, tmax, tps, eta)
            v = norm_loss(rec, q, Y, L0)
            pts[opt + name] = np.concatenate([v, np.full(len(tp) - len(v), np.nan)])
            if K == 10000 and opt == "adam":
                for k, (lo, hi) in enumerate(buckets):
                    idx = np.arange(lo - 1, min(hi, K + 1) - 1)
                    vb = norm_loss(rec, q, Y, L0, idx)
                    bp["adamb%d" % (k + 1)] = np.concatenate([vb, np.full(len(tp) - len(vb), np.nan)])
    j, Z, q = zipf(10000)
    Y1 = np.ones((10000, 1))
    rec, L0 = run_rows(q, Y1, "gd", int(tp.max()), list(tp), 1.0 / q[0])
    for k, (lo, hi) in enumerate(buckets):
        idx = np.arange(lo - 1, hi - 1)
        bp["gdb%d" % (k + 1)] = norm_loss(rec, q, Y1, L0, idx)

    def write(name, cols):
        keys = list(cols.keys())
        path = os.path.join(DATA, name)
        with open(path, "w") as f:
            f.write(" ".join(keys) + "\n")
            for r in range(len(cols["t"])):
                f.write(" ".join(("nan" if not np.isfinite(cols[k][r]) else "%.6g" % cols[k][r])
                                 for k in keys) + "\n")
        print("wrote", path)

    write("14-transformers-zipf-curves.dat", curves)
    write("14-transformers-zipf-points.dat", pts)
    write("14-transformers-zipf-buckets.dat", bc)
    write("14-transformers-zipf-bucketpts.dat", bp)
    # checks quoted in the text
    for K in (100, 1000, 10000):
        j, Z, q = zipf(K)
        print("K", K, "Z_K", round(Z, 4), "per-decade drop ln10/Z_K", round(np.log(10) / Z, 4))
    print("max |sim - exact| GD K=1e4:", np.nanmax(np.abs(pts["gd4"] - gd_exact(10000, tp))))
    tps = [s for s in tp if s <= 5000]
    rec, L0 = run_rows(np.ones(1), y[None, :], "adam", 5000, tps, ETA_ADAM)
    one_row = norm_loss(rec, np.ones(1), y[None, :], L0)
    print("max |Adam Zipf K=1e4 - Adam one row|:", np.max(np.abs(pts["adam4"][:len(tps)] - one_row)))
    print("max |sign Zipf K=1e4 - exact formula|:",
          np.max(np.abs(pts["sign4"][:len(tps)] - sign_exact(np.array(tps), y, ETA_SIGN))))
    print("max |sign/Adam K=1e2 - K=1e4|:", np.nanmax(np.abs(pts["sign2"] - pts["sign4"])),
          np.nanmax(np.abs(pts["adam2"] - pts["adam4"])))
    for s in (10, 100, 1000):
        print("t", s, "GD K=1e4 exact", gd_exact(10000, [s])[0],
              "asymptotic", (np.log(10000 / (2 * s)) - GAMMA) / zipf(10000)[1])


if __name__ == "__main__":
    main()
