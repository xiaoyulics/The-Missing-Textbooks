"""Lab 5.2 (Chapter 5, Controls): a search against its placebo, its sham and its controls.

Reads the truth table of Lab 5.1 (figures/data/05-controls-space.dat; run
05-controls-space.py first) and trains, for every one of the 192 architectures, four
further runs with fresh seeds (init = order = noise = 1000 + s, s = 0..3) on the 958
training images, recording the validation error (300 images) after every epoch. The
validation error after f epochs of one run is the proxy of fidelity f; with a constant
learning rate it is exactly what a separate f-epoch run with that seed would give.

On this tabular benchmark every search is a sequence of lookups. A search that evaluates an
architecture draws one of its four proxy runs at random, as NAS-Bench-101 does. The arms,
each repeated R = 2000 times with independent search seeds, are

  placebo   one architecture drawn uniformly from the space, no evaluation (random sampling)
  RS        random search: k uniform draws, keep the one with the best proxy (the sham search)
  EV        regularized evolution (aging evolution; population 10, tournament 3, one-attribute
            mutations), k proxy evaluations in all, keep the best proxy seen
  oracle    k uniform draws, keep the one with the best TRUE error (positive control: a perfect
            proxy)
  shuffled  random search whose proxy values belong to other, randomly permuted architectures
            (negative control: a proxy with no information)

The score of a search is the true error (Lab 5.1 table) of the architecture it returns, and its
report is the proxy value of that architecture. Everything is done twice: on the full space
and on the curated subspace (width 64 or 128).

Writes figures/data/05-controls-search-{proxy,curves,runs,fidelity}.dat and prints the
numbers quoted in the chapter. Runtime about three minutes on two threads.
Run from the volume root: python3 figures/scripts/05-controls-search.py
"""
import os
import sys
import time
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats, integrate  # noqa: E402

DEPTHS = (1, 2, 3, 4)
WIDTHS = (16, 32, 64, 128)
ACTS = ("relu", "tanh", "gelu")
DROPS = (0.0, 0.25)
BNS = (False, True)
SPACE = list(itertools.product(DEPTHS, WIDTHS, ACTS, DROPS, BNS))   # same order as Lab 5.1
CHOICES = (DEPTHS, WIDTHS, ACTS, DROPS, BNS)
PROXY_SEEDS = [1000 + s for s in range(4)]
EPOCHS = 30
KS = (1, 2, 4, 8, 16, 32, 64)
R = 2000
FID_MAIN = 2          # epochs of the short-training proxy used in the main comparison
FIDS = (1, 2, 3, 5, 10, 20, 30)


def e_k(k):
    """Expected maximum of k independent standard normals."""
    if k == 1:
        return 0.0
    f = lambda x: x * k * stats.norm.pdf(x) * stats.norm.cdf(x) ** (k - 1)
    return integrate.quad(f, -12, 12, limit=200)[0]


def load_truth():
    path = os.path.join(lk.DATA_DIR, "05-controls-space.dat")
    if not os.path.exists(path):
        sys.exit("run figures/scripts/05-controls-space.py first")
    with open(path) as fh:
        head = fh.readline().split()
        rows = [ln.split() for ln in fh]
    col = {h: i for i, h in enumerate(head)}
    truth = np.zeros(len(SPACE))
    for r in rows:
        truth[int(r[col["id"]])] = float(r[col["mean"]])
    return truth


def proxy_runs():
    """V[a, s, e]: validation error of architecture a, proxy seed s, after e + 1 epochs."""
    Xtr, ytr, _, _ = lk.digits(split=0)
    perm = np.random.default_rng(5).permutation(len(ytr))
    tr, va = perm[:958], perm[958:]
    data = (Xtr[tr], ytr[tr], Xtr[va], ytr[va])
    per_epoch = 958 // 64
    V = np.zeros((len(SPACE), len(PROXY_SEEDS), EPOCHS))
    for a, (depth, width, act, drop, bn) in enumerate(SPACE):
        cfg = lk.Config(depth=depth, width=width, activation=act, dropout=drop, batchnorm=bn,
                        optimizer="adam", lr=1e-3, batch_size=64, epochs=EPOCHS,
                        eval_every=per_epoch)
        for j, s in enumerate(PROXY_SEEDS):
            curve = lk.run(cfg, init=s, order=s, noise=s, data=data)["curve"]
            V[a, j] = [1.0 - c[2] for c in curve]
    return V


class Searcher:
    """Searches on the table restricted to the architectures in `ids`."""

    def __init__(self, ids, truth, V, fid, rng):
        self.ids = np.asarray(ids)
        self.truth, self.P, self.rng = truth, V[:, :, fid - 1], rng
        self.index = {SPACE[i]: i for i in self.ids}
        self.choices = [sorted({SPACE[i][d] for i in self.ids}, key=str) for d in range(5)]

    def draw(self, k):
        return self.rng.choice(self.ids, size=k)

    def proxy(self, a, perm=None):
        b = a if perm is None else perm[a]
        return self.P[b, self.rng.integers(self.P.shape[1])]

    def placebo(self, k):
        a = self.draw(1)[0]
        return a, np.nan

    def random_search(self, k, perm=None):
        cand = self.draw(k)
        scores = np.array([self.proxy(a, perm) for a in cand])
        j = np.argmin(scores)
        return cand[j], scores[j]

    def oracle(self, k):
        cand = self.draw(k)
        return cand[np.argmin(self.truth[cand])], np.nan

    def mutate(self, a):
        arch = list(SPACE[a])
        while True:
            d = self.rng.integers(5)
            opts = [v for v in self.choices[d] if v != arch[d]]
            if opts:
                arch[d] = opts[self.rng.integers(len(opts))]
                return self.index[tuple(arch)]

    def evolution(self, k, pop_size=10, tournament=3):
        pop, best, best_score = [], None, np.inf
        for t in range(k):
            if t < pop_size:
                a = self.draw(1)[0]
            else:
                sample = [pop[i] for i in self.rng.choice(len(pop), tournament, replace=False)]
                parent = min(sample, key=lambda x: x[1])[0]
                a = self.mutate(parent)
            sc = self.proxy(a)
            pop.append((a, sc))
            if len(pop) > pop_size:
                pop.pop(0)           # aging: remove the oldest
            if sc < best_score:
                best, best_score = a, sc
        return best, best_score


def run_arm(searcher, arm, k, perm=None):
    out = np.zeros((R, 2))
    for r in range(R):
        if arm == "shuffled":
            a, s = searcher.random_search(k, perm=perm[r])
        else:
            a, s = getattr(searcher, arm)(k)
        out[r] = searcher.truth[a], s
    return out


def rho_of(V, truth, ids, f):
    """Pearson correlation of one proxy evaluation (fidelity f) with the true error."""
    return np.corrcoef(V[ids, :, f - 1].ravel(), np.repeat(truth[ids], V.shape[1]))[0, 1]


def main(V=None):
    t0 = time.time()
    truth = load_truth()
    if V is None:
        V = proxy_runs()
    print(f"proxy runs: {V.shape[0] * V.shape[1]} runs of {EPOCHS} epochs, {time.time() - t0:.0f} s")
    cur = np.array([a[1] >= 64 for a in SPACE])
    spaces = {"full": np.arange(len(SPACE)), "curated": np.where(cur)[0]}
    tab = np.genfromtxt(os.path.join(lk.DATA_DIR, "05-controls-space.dat"), names=True,
                        dtype=None, encoding=None)
    single = np.zeros((len(SPACE), 6))
    for row in tab:
        single[row["id"]] = [row[f"e{j}"] for j in range(1, 7)]

    # --- 1. proxy quality: Kendall tau and Pearson rho of one proxy evaluation vs the truth
    for name, ids in spaces.items():
        taus = [np.mean([stats.kendalltau(V[ids, s, f - 1], truth[ids])[0]
                         for s in range(V.shape[1])]) for f in FIDS]
        rhos = [rho_of(V, truth, ids, f) for f in FIDS]
        print(f"[{name}] proxy fidelity (epochs): " + ", ".join(
            f"{f}: tau {t:.2f} rho {r:.2f}" for f, t, r in zip(FIDS, taus, rhos)))
        pair = [stats.kendalltau(single[ids, i], single[ids, j])[0]
                for i in range(6) for j in range(i + 1, 6)]
        print(f"[{name}] seed-to-seed Kendall tau of two single test runs: {np.mean(pair):.2f} "
              f"(range {np.min(pair):.2f} to {np.max(pair):.2f})")
    rank_cur = {i: int((truth[cur] < truth[i]).sum()) + 1 for i in np.where(cur)[0]}
    found = int(np.where(cur)[0][np.argmin(V[cur, :, 29].mean(1))])
    fav2 = int(np.where(cur)[0][np.argmin(V[cur, :, FID_MAIN - 1].mean(1))])
    print(f"[curated] best by 30-epoch validation (mean of 4 runs): {SPACE[found]}, true error "
          f"{100 * truth[found]:.2f}%, rank {rank_cur[found]} of {cur.sum()}")
    print(f"[curated] best by {FID_MAIN}-epoch validation (mean of 4 runs): {SPACE[fav2]}, true error "
          f"{100 * truth[fav2]:.2f}%, rank {rank_cur[fav2]} of {cur.sum()}")
    lk.write_dat("figures/data/05-controls-search-proxy.dat", {
        "id": np.arange(len(SPACE)), "curated": cur.astype(int), "truth": truth,
        "p2": V[:, 0, FID_MAIN - 1], "p2mean": V[:, :, FID_MAIN - 1].mean(1),
        "p30": V[:, 0, 29], "p30mean": V[:, :, 29].mean(1),
        "found": (np.arange(len(SPACE)) == found).astype(int)})

    # --- 2. the searches with the short-training proxy
    curves = {"k": list(KS)}
    indiv = {"x": [], "err": [], "space": []}
    rng_plot = np.random.default_rng(11)
    for si, (name, ids) in enumerate(spaces.items()):
        rng = np.random.default_rng(100 + si)
        S = Searcher(ids, truth, V, FID_MAIN, rng)
        mu_bar, sig_T, best = truth[ids].mean(), truth[ids].std(), truth[ids].min()
        rho = rho_of(V, truth, ids, FID_MAIN)
        print(f"[{name}] space mean {100 * mu_bar:.2f}%, SD {100 * sig_T:.2f} pp, best {100 * best:.2f}%, "
              f"proxy ({FID_MAIN} epochs) rho {rho:.2f}")
        for arm in ("placebo", "random_search", "evolution", "oracle", "shuffled"):
            means, hws, sds = [], [], []
            for k in KS:
                perm = (np.array([_perm(ids, len(SPACE), rng) for _ in range(R)])
                        if arm == "shuffled" else None)
                res = run_arm(S, arm, k, perm)
                m, h = lk.t_interval(res[:, 0])
                means.append(m)
                hws.append(h)
                sds.append(res[:, 0].std(ddof=1))
                if arm == "random_search":
                    sel = res[:40, 0]
                    indiv["x"] += list(k * np.exp(rng_plot.uniform(-0.13, 0.13, len(sel))))
                    indiv["err"] += list(sel)
                    indiv["space"] += [si] * len(sel)
            curves[f"{arm}_{name}"] = means
            curves[f"{arm}_{name}_hw"] = hws
            print(f"[{name}] {arm:13s} " + "  ".join(
                f"k={k}: {100 * m:.2f}+-{100 * h:.2f} (sd {100 * s:.2f})"
                for k, m, h, s in zip(KS, means, hws, sds)))
        curves[f"pred_{name}"] = [mu_bar - rho * sig_T * e_k(k) for k in KS]
        curves[f"mean_{name}"] = [mu_bar] * len(KS)
        curves[f"best_{name}"] = [best] * len(KS)
        print(f"[{name}] Gaussian prediction for RS: " + "  ".join(
            f"k={k}: {100 * p:.2f}" for k, p in zip(KS, curves[f"pred_{name}"])))
        for k, a, b, ha, hb in zip(KS, curves[f"random_search_{name}"], curves[f"evolution_{name}"],
                                   curves[f"random_search_{name}_hw"], curves[f"evolution_{name}_hw"]):
            if k >= 16:
                print(f"[{name}] k={k}: evolution - random search = {100 * (b - a):+.2f} pp "
                      f"(+- {100 * np.hypot(ha, hb):.2f})")
    for c in ("full", "curated"):    # percent, for plotting
        for key in list(curves):
            if key.endswith(c) or key.endswith(c + "_hw"):
                curves[key] = list(100 * np.asarray(curves[key]))
    lk.write_dat("figures/data/05-controls-search-curves.dat", curves)
    indiv["err"] = list(100 * np.asarray(indiv["err"]))
    lk.write_dat("figures/data/05-controls-search-runs.dat", indiv)

    # --- 3. evolution against random search with better proxies
    for si, (name, ids) in enumerate(spaces.items()):
        for f in (10, 30):
            for k in (32, 64):
                Sa = Searcher(ids, truth, V, f, np.random.default_rng(700 + 10 * si + f + k))
                Sb = Searcher(ids, truth, V, f, np.random.default_rng(800 + 10 * si + f + k))
                a = run_arm(Sa, "random_search", k)[:, 0]
                b = run_arm(Sb, "evolution", k)[:, 0]
                h = np.hypot(lk.t_interval(a)[1], lk.t_interval(b)[1])
                print(f"[{name}] proxy {f} epochs, k={k}: random search {100 * a.mean():.2f}%, "
                      f"evolution {100 * b.mean():.2f}%, difference {100 * (b.mean() - a.mean()):+.2f} "
                      f"+- {100 * h:.2f} pp")

    # --- 4. what a search reports about itself (30-epoch proxy: same units as the truth)
    for si, (name, ids) in enumerate(spaces.items()):
        mu_bar = truth[ids].mean()
        mp = V[ids, :, 29].mean()
        rho30 = rho_of(V, truth, ids, 30)
        rng = np.random.default_rng(300 + si)
        S30 = Searcher(ids, truth, V, 30, rng)
        for k in (8, 32):
            res = run_arm(S30, "random_search", k)
            tg, rg = mu_bar - res[:, 0].mean(), mp - res[:, 1].mean()
            print(f"[{name}] winner's curse, RS k={k}, 30-epoch proxy (rho {rho30:.2f}): reported gain "
                  f"{100 * rg:.2f} pp, true gain {100 * tg:.2f} pp, ratio {rg / tg:.2f} "
                  f"(1/rho^2 = {1 / rho30 ** 2:.2f})")
        res = run_arm(S30, "shuffled", 16, perm=np.array([_perm(ids, len(SPACE), rng) for _ in range(R)]))
        print(f"[{name}] negative control (shuffled 30-epoch proxy), k=16: true gain "
              f"{100 * (mu_bar - res[:, 0].mean()):+.2f} +- {100 * lk.t_interval(res[:, 0])[1]:.2f} pp, "
              f"reported gain {100 * (mp - res[:, 1].mean()):.2f} pp")

    # --- 5. the proxy theorem across fidelities: gain of RS(k=16) over the placebo
    k = 16
    fid = {}
    for si, (name, ids) in enumerate(spaces.items()):
        mu_bar, sig_T = truth[ids].mean(), truth[ids].std()
        scale = sig_T * e_k(k)
        rhos, ng, nh = [0.0], [], []
        rng = np.random.default_rng(900 + si)
        S = Searcher(ids, truth, V, 1, rng)
        res = run_arm(S, "shuffled", k, perm=np.array([_perm(ids, len(SPACE), rng) for _ in range(R)]))
        m, h = lk.t_interval(mu_bar - res[:, 0])
        ng.append(m / scale)
        nh.append(h / scale)
        line = [f"shuffled: {100 * m:.2f}+-{100 * h:.2f}"]
        for f in FIDS:
            S = Searcher(ids, truth, V, f, np.random.default_rng(500 + 10 * si + f))
            m, h = lk.t_interval(mu_bar - run_arm(S, "random_search", k)[:, 0])
            r = rho_of(V, truth, ids, f)
            rhos.append(r)
            ng.append(m / scale)
            nh.append(h / scale)
            line.append(f"{f} ep (rho {r:.2f}): {100 * m:.2f}+-{100 * h:.2f}, pred {100 * r * scale:.2f}")
        S = Searcher(ids, truth, V, 1, np.random.default_rng(950 + si))
        m, h = lk.t_interval(mu_bar - run_arm(S, "oracle", k)[:, 0])
        rhos.append(1.0)
        ng.append(m / scale)
        nh.append(h / scale)
        line.append(f"oracle: {100 * m:.2f}+-{100 * h:.2f}, pred {100 * scale:.2f}")
        fid[f"rho_{name}"], fid[f"ng_{name}"], fid[f"ng_{name}_hw"] = rhos, ng, nh
        print(f"[{name}] RS k=16 gain over the placebo (pp): " + "; ".join(line))
        print(f"[{name}] sigma_T e_16 = {100 * scale:.2f} pp; headroom mean - best = "
              f"{100 * (mu_bar - truth[ids].min()):.2f} pp")
    lk.write_dat("figures/data/05-controls-search-fidelity.dat", fid)
    print("e_k:", ", ".join(f"e_{k}={e_k(k):.3f}" for k in KS))
    print(f"total {time.time() - t0:.0f} s")


def _perm(ids, n, rng):
    p = np.arange(n)
    p[ids] = rng.permutation(ids)
    return p


if __name__ == "__main__":
    main()
