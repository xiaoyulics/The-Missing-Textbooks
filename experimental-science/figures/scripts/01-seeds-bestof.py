"""Lab 1.3 (Chapter 1, Seeds): what the best of k runs reports.

The recipe of Lab 1.1 is run N = 400 times on split 0 with fresh initialization and
data-order seeds, once on clean labels and once with 20% of the training labels replaced
by random labels (noise seed fixed, so the noisy training set is part of the task). The
539 test images are divided at random into a development half (270) and a test half (269).

For k = 1..50 we compute, exactly over all k-subsets of the N runs:
  best   E[lowest test-half error among k runs]   (reporting the best run on the set that scores it)
  honest E[test-half error of the run with the lowest development-half error among k]
           (ties broken uniformly at random)
and compare them with the normal predictions  mean - SD * e_k  and  mean - rho * SD * e_k,
where rho is the correlation between a run's development and test errors. Bands: 95%
percentile bootstrap over runs (400 replicates).

Writes figures/data/01-seeds-bestof-clean.dat and figures/data/01-seeds-bestof-noisy.dat
Prints every number quoted in the text.
Run from the volume root:  python3 figures/scripts/01-seeds-bestof.py   (about 70 s)
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from scipy import stats, integrate
from scipy.special import gammaln
import labkit as lk

T0 = time.time()
N = 400
KS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50])
B = 400
rng = np.random.default_rng(3407)


def e_k(k):
    """Expected maximum of k independent standard normals."""
    if k == 1:
        return 0.0
    f = lambda z: z * k * stats.norm.pdf(z) * stats.norm.cdf(z) ** (k - 1)
    return integrate.quad(f, -12, 12, limit=200)[0]


def logcomb(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)
    return np.where((b < 0) | (b > a), -np.inf, out)


def expected_min(x, k):
    """Unbiased E[min of k] from all k-subsets of x (same as labkit.expected_best on -x)."""
    s = np.sort(x)
    n = len(s)
    w = np.exp(logcomb(n - np.arange(1, n + 1), k - 1) - logcomb(n, k))
    return float(np.dot(w, s))


def expected_selected(dev, test, k):
    """E[test value of the run with the smallest dev value in a uniform random k-subset],
    ties among the smallest dev values broken uniformly at random; exact."""
    n = len(dev)
    vals, inv = np.unique(dev, return_inverse=True)
    sum_test = np.bincount(inv, weights=test)
    cnt = np.bincount(inv)
    greater = n - np.cumsum(cnt)                 # runs with a strictly larger dev value
    t = np.arange(k)[:, None]                    # number of tied companions in the subset
    lc = (logcomb(cnt[None, :] - 1, t) + logcomb(greater[None, :], k - 1 - t) - logcomb(n, k))
    p_run = (np.exp(lc) / (t + 1)).sum(0)        # selection probability of each run in a group
    assert abs(np.dot(p_run, cnt) - 1) < 1e-9
    return float(np.dot(p_run, sum_test))


def collect(label_noise):
    cfg = lk.Config(width=64, depth=2, optimizer="momentum", lr=0.05, epochs=20, batch_size=64,
                    label_noise=label_noise)
    data = lk.load(cfg, split=0, noise=0)
    Xte, yte = data[2], data[3]
    m = len(yte)
    perm = np.random.default_rng(101).permutation(m)
    dev_idx, test_idx = perm[: (m + 1) // 2], perm[(m + 1) // 2:]
    D = np.empty(N); T = np.empty(N); W = np.zeros((N, m))
    for r in range(N):
        res = lk.run(cfg, init=r, order=r, noise=0, data=data, return_model=True)
        with torch.no_grad():
            wrong = (res["model"](Xte).argmax(1) != yte).numpy().astype(float)
        W[r] = wrong
        D[r] = 100 * wrong[dev_idx].mean()
        T[r] = 100 * wrong[test_idx].mean()
    return D, T, len(dev_idx), len(test_idx), W


def churn_report(W):
    """Split the seed variance of the error on all m items into item churn (the diagonal
    of the item covariance) and co-movement (the off-diagonal part, an estimate of the
    variance of a run's population error), with a bootstrap over runs."""
    n, m = W.shape
    def parts(Wb):
        q = Wb.mean(0)
        sv = (100 * Wb.mean(1)).var(ddof=1)
        ch = 1e4 * (q * (1 - q)).sum() * n / (n - 1) / m ** 2
        return sv, ch
    sv, ch = parts(W)
    shares = []
    for b in range(400):
        sv_b, ch_b = parts(W[rng.integers(0, n, size=n)])
        shares.append(ch_b / sv_b)
    lo_, hi_ = np.percentile(shares, [2.5, 97.5])
    q = W.mean(0)
    print(f"  all {m} items: seed variance {sv:.4f} %^2 (SD {math.sqrt(sv):.3f}); item churn {ch:.4f} "
          f"= {ch/sv:.0%} (95% [{lo_:.0%}, {hi_:.0%}]); co-movement {sv-ch:+.4f}; items ever wrong "
          f"{(q>0).sum()}, always wrong {(q==1).sum()}")


def analyse(name, D, T, m_dev, m_test):
    mu, sd = T.mean(), T.std(ddof=1)
    rho = np.corrcoef(D, T)[0, 1]
    z = np.arctanh(rho); h = 1.96 / math.sqrt(N - 3)
    ek = np.array([e_k(k) for k in KS])
    best = np.array([expected_min(T, k) for k in KS])
    hon = np.array([expected_selected(D, T, k) for k in KS])
    bb = np.empty((B, len(KS))); hb = np.empty((B, len(KS)))
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        bb[b] = [expected_min(T[idx], k) for k in KS]
        hb[b] = [expected_selected(D[idx], T[idx], k) for k in KS]
    blo, bhi = np.percentile(bb, [2.5, 97.5], axis=0)
    hlo, hhi = np.percentile(hb, [2.5, 97.5], axis=0)
    pred = mu - sd * ek
    hpred = mu - rho * sd * ek
    lk.write_dat(f"figures/data/01-seeds-bestof-{name}.dat",
                 {"k": KS, "ek": ek, "best": best, "blo": blo, "bhi": bhi, "pred": pred,
                  "honest": hon, "hlo": hlo, "hhi": hhi, "hpred": hpred,
                  "mean": np.full(len(KS), mu)})
    mu_all = (D * m_dev + T * m_test) / (m_dev + m_test)
    print(f"== {name}: {N} runs, development half {m_dev} items, test half {m_test} items")
    print(f"  all 539 items: mean {mu_all.mean():.3f}%, SD {mu_all.std(ddof=1):.3f}")
    print(f"  test half: mean {mu:.3f}% (95% t +- {lk.t_interval(T)[1]:.3f}), SD {sd:.3f}; "
          f"dev half: mean {D.mean():.3f}%, SD {D.std(ddof=1):.3f}")
    print(f"  corr(dev, test) rho = {rho:.3f} (95% [{np.tanh(z-h):.3f}, {np.tanh(z+h):.3f}]); "
          f"Spearman {stats.spearmanr(D, T)[0]:.3f}; Cov(dev, test) = {np.cov(D, T)[0,1]:+.4f} %^2 "
          f"(estimates the variance of a run's population error)")
    print(f"  test-half skewness {stats.skew(T):.3f}, excess kurtosis {stats.kurtosis(T):.3f}")
    for k, e, bst, lo_, hi_, pr, hn, hl, hh, hp in zip(KS, ek, best, blo, bhi, pred, hon, hlo, hhi, hpred):
        if k in (1, 2, 5, 10, 20, 50):
            print(f"  k={k:2d} e_k={e:.4f}  best-on-test {bst:.3f} [{lo_:.3f},{hi_:.3f}] (normal {pr:.3f})"
                  f"  best-on-dev->test {hn:.3f} [{hl:.3f},{hh:.3f}] (normal {hp:.3f})"
                  f"  real share of apparent gain {(mu-hn)/(mu-bst) if k>1 else float('nan'):.2f}")
    share = (mu - hon[1:]) / (mu - best[1:])
    print(f"  real share of the apparent gain over k = 2..50: from {share.min():.2f} to {share.max():.2f}")
    return mu, sd, rho


# sanity check of the exact estimators against brute force on a small example
xs = rng.integers(0, 6, size=9).astype(float); ys = rng.normal(size=9)
from itertools import combinations
bf_min = np.mean([min(xs[list(c)]) for c in combinations(range(9), 3)])
bf_sel = np.mean([np.mean([ys[i] for i in c if xs[i] == min(xs[list(c)])]) for c in combinations(range(9), 3)])
assert abs(bf_min - expected_min(xs, 3)) < 1e-12 and abs(bf_sel - expected_selected(xs, ys, 3)) < 1e-12
assert abs(expected_min(xs, 3) + lk.expected_best(-xs, 3)) < 1e-12

for name, ln in (("clean", 0.0), ("noisy", 0.2)):
    t1 = time.time()
    D, T, md, mt, W = collect(ln)
    print(f"(runs: {time.time()-t1:.0f} s)")
    analyse(name, D, T, md, mt)
    churn_report(W)
print("e_k table:", " ".join(f"e_{k}={e_k(k):.4f}" for k in range(2, 11)))
print("sqrt(2 ln k):", " ".join(f"{math.sqrt(2*math.log(k)):.3f}" for k in range(2, 11)))
print("further e_k:", " ".join(f"e_{k}={e_k(k):.4f}" for k in (20, 25, 50, 100, 200, 625, 10_000)))
print(f"[{time.time()-T0:.1f} s]")
