"""Figure 1 of Chapter 1 (Double Descent): exact finite-sample test error of
minimum-norm least squares versus the number p of features, for n = 40 samples,
compared with Monte Carlo simulation.

Panel (a), well specified: x ~ N(0, I_p), y = <w_star, x> + noise,
    ||w_star||^2 = 1, noise variance sigma^2 = 0.25.
Panel (b), weak features: z ~ N(0, I_d) with d = 100, y = <beta, z> (no label
    noise), ||beta|| = 1; the learner uses the first p coordinates of z.
    Two coefficient profiles: evenly spread (beta_j^2 = 1/d) and decaying
    (beta_j^2 proportional to j^(-0.8)).

Writes figures/data/01-doubledescent-risk-theory.dat (every p) and
figures/data/01-doubledescent-risk-mc.dat (a grid of p, with standard errors).
Deterministic (seeded); runs in well under a minute on a laptop CPU.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # small matrices: threads only add overhead
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N_SAMPLES = 40          # n
R2, S2 = 1.0, 0.25      # ||w_star||^2 and sigma^2 in panel (a)
D = 100                 # number of features of the truth in panel (b)
TRIALS = 4000

def exact_excess(n, p, head2, noise2):
    """E||w_hat - w_star||^2 for min-norm least squares (the exact-risk theorem of the chapter)."""
    if p <= n - 2:
        return noise2 * p / (n - p - 1)
    if p >= n + 2:
        return head2 * (1 - n / p) + noise2 * n / (p - n - 1)
    return np.inf if noise2 > 0 else (0.0 if p <= n else head2 * (1 - n / p))

def betas():
    even = np.full(D, 1.0 / np.sqrt(D))
    dec = np.arange(1, D + 1) ** (-0.4)
    dec /= np.linalg.norm(dec)
    return even, dec

def test_error_weak(n, p, beta):
    head2, tail2 = np.sum(beta[:p] ** 2), np.sum(beta[p:] ** 2)
    return tail2 + exact_excess(n, p, head2, tail2)   # sigma^2 = 0 here

def minnorm_batch(X, y):
    """Minimum-norm least-squares solutions for a batch X (T,n,p), y (T,n)."""
    T, n, p = X.shape
    if p <= n:
        G = np.einsum("tij,tik->tjk", X, X)
        b = np.einsum("tij,ti->tj", X, y)
        return np.linalg.solve(G, b[..., None])[..., 0]
    G = np.einsum("tij,tkj->tik", X, X)
    a = np.linalg.solve(G, y[..., None])[..., 0]
    return np.einsum("tij,ti->tj", X, a)

def mc_wellspec(rng, n, p, trials, batch=500):
    errs = []
    w = np.zeros(p); w[0] = np.sqrt(R2)          # the risk depends on w only through ||w||
    for start in range(0, trials, batch):
        T = min(batch, trials - start)
        X = rng.standard_normal((T, n, p))
        y = X @ w + np.sqrt(S2) * rng.standard_normal((T, n))
        wh = minnorm_batch(X, y)
        errs.append(S2 + np.sum((wh - w) ** 2, axis=1))   # exact test error of each fit
    e = np.concatenate(errs)
    return e.mean(), e.std(ddof=1) / np.sqrt(len(e))

def mc_weak(rng, n, p, beta, trials, batch=500):
    errs = []
    for start in range(0, trials, batch):
        T = min(batch, trials - start)
        Z = rng.standard_normal((T, n, D))
        y = Z @ beta
        wh = minnorm_batch(Z[:, :, :p], y)
        errs.append(np.sum(beta[p:] ** 2) + np.sum((wh - beta[:p]) ** 2, axis=1))
    e = np.concatenate(errs)
    return e.mean(), e.std(ddof=1) / np.sqrt(len(e))

def fmt(v):
    return "nan" if not np.isfinite(v) else f"{v:.6g}"

def main():
    rng = np.random.default_rng(20261017)
    n = N_SAMPLES
    even, dec = betas()
    with open(os.path.join(DATA, "01-doubledescent-risk-theory.dat"), "w") as f:
        f.write("p wellspec weakeven weakdecay\n")
        for p in range(1, 201):
            ws = S2 + exact_excess(n, p, R2, S2)
            we = test_error_weak(n, p, even) if p <= D else np.nan
            wd = test_error_weak(n, p, dec) if p <= D else np.nan
            f.write(f"{p} {fmt(ws)} {fmt(we)} {fmt(wd)}\n")
    grid_a = list(range(2, 36, 3)) + [36] + [44, 46] + list(range(50, 201, 10))
    grid_b = list(range(2, 36, 3)) + [36] + [44, 46] + list(range(50, 101, 5))
    rows = {}
    for p in grid_a:
        rows.setdefault(p, {})["ws"] = mc_wellspec(rng, n, p, TRIALS)
    for p in grid_b:
        rows.setdefault(p, {})["we"] = mc_weak(rng, n, p, even, TRIALS)
        rows[p]["wd"] = mc_weak(rng, n, p, dec, TRIALS)
    with open(os.path.join(DATA, "01-doubledescent-risk-mc.dat"), "w") as f:
        f.write("p wellspec wellspecse weakeven weakevense weakdecay weakdecayse\n")
        for p in sorted(rows):
            r = rows[p]
            ws = r.get("ws", (np.nan, np.nan)); we = r.get("we", (np.nan, np.nan)); wd = r.get("wd", (np.nan, np.nan))
            f.write(f"{p} {fmt(ws[0])} {fmt(ws[1])} {fmt(we[0])} {fmt(we[1])} {fmt(wd[0])} {fmt(wd[1])}\n")
    # agreement report (z-scores) on stdout
    for p in sorted(rows):
        r = rows[p]
        out = [f"p={p:3d}"]
        if "ws" in r:
            ex = S2 + exact_excess(n, p, R2, S2); m, se = r["ws"]; out.append(f"(a) {ex:.4f} vs {m:.4f}+-{se:.4f} z={(m-ex)/se:+.1f}")
        if "we" in r:
            for key, beta in (("we", even), ("wd", dec)):
                ex = test_error_weak(n, p, beta); m, se = r[key]; out.append(f"({key}) {ex:.4f} vs {m:.4f} z={(m-ex)/se:+.1f}")
        print("  ".join(out))

if __name__ == "__main__":
    main()
