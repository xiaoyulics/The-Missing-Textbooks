"""Lab 15.1 of *Deep Learning as an Experimental Science*: classifiers under random
test-time formats.

Six classifiers of the scikit-learn digits are trained once each on split 0 (seed 0):
softmax regression ("linear"), the two-layer MLP of Chapter 1 ("MLP"), a one-layer MLP of
width 128 ("MLP-1"), the MLP with tanh units ("MLP-tanh"), a two-layer MLP of width 256
trained with Adam ("MLP-Adam"), and the small convolutional network of Chapter 6 ("CNN").
Inputs are scaled to [0, 1] and standardized with one mean and one standard deviation of
the training set, so that a test-time change of the pixels reaches the network unscaled.

A format is a meaning-preserving change of the 539 test images, built from seven atomic
choices, like the formats of a prompt grammar: a partial one-pixel shift (none or one of
four directions), contrast (x0.8, x1, x1.2), a grey background (+0, +0.05, +0.1), a light
blur (no, yes), pixel noise (sd 0, 0.05, 0.1; a fixed pattern per format), thicker strokes
(no, yes) and a gamma curve (0.8, 1, 1.25): 1,620 formats in all. Every network is scored
on every image under every format. The 1,620 x 539 table of each network is the
population from which the resampling experiments draw formats and items with
replacement, so that the variance formulas of Chapter 15 hold exactly.

Run from the volume root:  python3 figures/scripts/15-instruments-formats.py
Writes figures/data/15-instruments-formats-{scatter,reversal,budget,tau}.dat and prints
the numbers quoted in Chapter 15. About a minute on two threads.
"""
import os, sys, time, math, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import replace
from scipy import stats

R_REV, R_BUDGET, R_TAU = 20_000, 4_000, 5_000     # resamples
FS = [1, 2, 4, 8, 16, 32, 64]                      # numbers of formats averaged
M_SMALL = 100                                      # items per format in the nested variant
BUDGET = 1000                                      # evaluations per model in the budget study
PAIR = ("MLP", "CNN")

# ---- data -------------------------------------------------------------------------------
X, y = lk._digits_raw()
X = X.reshape(-1, 8, 8)
_perm = np.random.default_rng(10_000).permutation(len(y))       # labkit's split 0
_nte = int(round(0.3 * len(y)))
TE, TR = _perm[:_nte], _perm[_nte:]
XTR_RAW, YTR, XTE_RAW, YTE = X[TR], y[TR], X[TE], y[TE]
MU, SD = XTR_RAW.mean(), XTR_RAW.std()


def standardize(A):
    return torch.from_numpy(((A.reshape(len(A), 64) - MU) / SD).astype(np.float32))


DATA = (standardize(XTR_RAW), torch.from_numpy(YTR), standardize(XTE_RAW), torch.from_numpy(YTE))


class CNN(nn.Module):
    """The convolutional network of Chapter 6: two 3x3 convolutions (16 and 32 channels),
    a 2x2 max-pool and a linear classifier, 9,930 parameters."""

    def __init__(self, c1=16, c2=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.fc = nn.Linear(c2 * 16, 10)

    def forward(self, x):
        x = x.view(-1, 1, 8, 8)
        x = F.max_pool2d(F.relu(self.conv2(F.relu(self.conv1(x)))), 2)
        return self.fc(x.flatten(1))


SGD = lk.Config(optimizer="momentum", lr=0.05, momentum=0.9, batch_size=64, epochs=20)
ADAM = lk.Config(optimizer="adam", lr=1e-3, batch_size=64, epochs=30)
MODELS = [  # name, constructor, recipe
    ("linear", lambda: lk.MLP(64, 10, 64, depth=0), SGD),
    ("MLP", lambda: lk.MLP(64, 10, 64, depth=2), SGD),
    ("MLP-1", lambda: lk.MLP(64, 10, 128, depth=1), SGD),
    ("MLP-tanh", lambda: lk.MLP(64, 10, 64, depth=2, activation="tanh"), SGD),
    ("MLP-Adam", lambda: lk.MLP(64, 10, 256, depth=2), ADAM),
    ("CNN", CNN, SGD),
]


def train(ctor, cfg, seed=0):
    """One run with labkit's seeding and batching (initialization, order)."""
    Xtr, ytr, _, _ = DATA
    torch.manual_seed(50_000 + seed)
    model = ctor()
    opt = lk.make_optimizer(cfg, model.parameters())
    n, bs = Xtr.shape[0], cfg.batch_size
    total = int(round(cfg.epochs * (n // bs)))
    g = torch.Generator().manual_seed(60_000 + seed)
    perm, pos = torch.randperm(n, generator=g), 0
    for step in range(total):
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g), 0
        idx = perm[pos:pos + bs]
        pos += bs
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


# ---- the format grammar ------------------------------------------------------------------
SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
CONTRAST = [0.8, 1.0, 1.2]
BACKGROUND = [0.0, 0.05, 0.1]
BLUR = [0, 1]
NOISE = [0.0, 0.05, 0.1]
THICK = [0, 1]
GAMMA = [0.8, 1.0, 1.25]
GRAMMAR = list(itertools.product(range(len(SHIFTS)), CONTRAST, BACKGROUND, BLUR, NOISE, THICK, GAMMA))
CLEAN = GRAMMAR.index((0, 1.0, 0.0, 0, 0.0, 0, 1.0))


def _shift(A, dx, dy):
    out = np.zeros_like(A)
    out[:, max(-dy, 0):8 + min(-dy, 0), max(-dx, 0):8 + min(-dx, 0)] = \
        A[:, max(dy, 0):8 + min(dy, 0), max(dx, 0):8 + min(dx, 0)]
    return out


def _neighbours(A):
    P = np.pad(A, ((0, 0), (1, 1), (1, 1)))
    return np.stack([P[:, i:i + 8, j:j + 8] for i in range(3) for j in range(3)])


def apply_format(A, fid):
    s, c, b, blur, noise, thick, gamma = GRAMMAR[fid]
    Y = A.copy()
    if thick:
        Y = np.maximum(Y, 0.3 * _neighbours(Y).max(0))
    if blur:
        Y = 0.7 * Y + 0.3 * _neighbours(Y).mean(0)
    Y = np.clip(c * Y ** gamma, 0.0, 1.0)
    if s:
        Y = 0.7 * Y + 0.3 * _shift(Y, *SHIFTS[s])
    Y = Y + b
    if noise > 0:
        Y = Y + noise * np.random.default_rng(90_000 + fid).standard_normal(Y.shape)
    return np.clip(Y, 0.0, 1.0)


def main():
    t0 = time.time()
    names = [nm for nm, _, _ in MODELS]
    nets = {}
    for nm, ctor, cfg in MODELS:
        t = time.time()
        nets[nm] = train(ctor, cfg)
        print(f"trained {nm:9s} ({lk.n_params(nets[nm]):,} parameters, {time.time() - t:.1f}s)", flush=True)
    yte = torch.from_numpy(YTE)
    C = np.zeros((len(names), len(GRAMMAR), len(YTE)), dtype=bool)     # correct?
    with torch.no_grad():
        for fid in range(len(GRAMMAR)):
            Xf = standardize(apply_format(XTE_RAW, fid))
            for k, nm in enumerate(names):
                C[k, fid] = (nets[nm](Xf).argmax(1) == yte).numpy()
    print(f"scored {len(names)} networks x {len(GRAMMAR)} formats x {len(YTE)} images "
          f"({time.time() - t0:.0f}s)")
    acc = 100.0 * C.mean(2)                                            # networks x formats

    print("\naccuracy (%) under the clean format and over the 1,620 formats:")
    for k, nm in enumerate(names):
        a = acc[k]
        print(f"  {nm:9s} clean {a[CLEAN]:6.2f}   mean {a.mean():6.2f}   sd {a.std():5.2f}   "
              f"min {a.min():6.2f}   max {a.max():6.2f}   spread {a.max() - a.min():5.2f}")
    order = np.argsort(acc.mean(1))
    print("  ranking by mean over formats (worst to best):", [names[k] for k in order])
    print("  ranking under the clean format (worst to best):", [names[k] for k in np.argsort(acc[:, CLEAN], kind="stable")])

    # ---- the featured pair ----------------------------------------------------------------
    i1, i2 = names.index(PAIR[0]), names.index(PAIR[1])
    a1, a2 = acc[i1], acc[i2]
    d = a2 - a1                                     # CNN minus MLP, per format
    Delta, s = d.mean(), d.std()
    cov = np.mean((a1 - a1.mean()) * (a2 - a2.mean()))
    s_sep = math.sqrt(a1.var() + a2.var())
    rev1 = np.mean(d < 0) + 0.5 * np.mean(d == 0)
    print(f"\n{PAIR[1]} minus {PAIR[0]}, per format: Delta {Delta:.3f}, sd s {s:.3f}, "
          f"correlation {np.corrcoef(a1, a2)[0, 1]:.3f}, sigma_u = sqrt(cov) {math.sqrt(cov):.2f}, "
          f"sd of single scores {a1.std():.2f} / {a2.std():.2f}; skewness {stats.skew(d):.2f}, "
          f"excess kurtosis {stats.kurtosis(d):.2f}")
    print(f"  clean-format difference {d[CLEAN]:.2f}; formats on which {PAIR[0]} wins: {np.mean(d < 0):.3f}, ties {np.mean(d == 0):.3f}")
    print(f"  reversal on one shared format: observed {rev1:.3f}, normal model {stats.norm.cdf(-abs(Delta) / s):.3f}")
    print(f"  reversal on separate formats: normal model {stats.norm.cdf(-abs(Delta) / s_sep):.3f}")
    print(f"  quantiles of the difference (0, 5, 25, 50, 75, 95, 100%): {np.round(np.quantile(d, [0, .05, .25, .5, .75, .95, 1]), 2)}")
    G = np.array(GRAMMAR, dtype=float)
    labels = ["shift", "contrast", "background", "blur", "noise", "thick", "gamma"]
    print("  mean difference by atomic choice:")
    for j, lab in enumerate(labels):
        vals = sorted(set(G[:, j]))
        print(f"    {lab:10s} " + "  ".join(f"{v:g}: {d[G[:, j] == v].mean():5.2f}" for v in vals))
    worst = np.argsort(d)[:3]
    print("  three formats most favourable to the MLP:", [(GRAMMAR[f], round(d[f], 2)) for f in worst])

    # two facets: per-item paired difference D[f, z] in points
    D = 100.0 * (C[i2].astype(np.int8) - C[i1].astype(np.int8))           # formats x items
    row, col, grand = D.mean(1), D.mean(0), D.mean()
    var_s = np.mean((row - grand) ** 2)
    var_i = np.mean((col - grand) ** 2)
    var_si = np.mean((D - row[:, None] - col[None, :] + grand) ** 2)
    print(f"\ncomponents of the per-item difference (points^2; population of 1,620 formats x 539 images):")
    print(f"  setting {var_s:.2f} (sd {math.sqrt(var_s):.2f}), item {var_i:.1f} (sd {math.sqrt(var_i):.2f}), "
          f"setting x item {var_si:.1f} (sd {math.sqrt(var_si):.2f}); total {D.var():.1f}")
    disc = np.mean(D != 0)
    print(f"  share of (format, image) pairs on which the two networks disagree: {disc:.4f}")
    lead = 2.0
    print(f"  Sclar's criterion, d = {lead:g} points: P(CNN ahead by >= d) {np.mean(d >= lead):.3f}, "
          f"P(MLP ahead by >= d) {np.mean(d <= -lead):.3f} (the probability that a lead of the CNN on one "
          f"format is reversed by at least d on an independent format)")
    m_all = len(YTE)
    v_fixed = var_s + (var_i + var_si) / m_all
    v_rand = (var_s + var_i + var_si) / m_all
    print(f"  all {m_all} images under one random format: sd {math.sqrt(v_fixed):.3f}, reversal (normal model) "
          f"{stats.norm.cdf(-abs(Delta) / math.sqrt(v_fixed)):.3f}; a fresh random format for each image: "
          f"sd {math.sqrt(v_rand):.3f}, reversal (normal model) {stats.norm.cdf(-abs(Delta) / math.sqrt(v_rand)):.3f}")
    rr = np.random.default_rng(16)
    fz = rr.integers(0, len(GRAMMAR), size=(R_BUDGET, m_all))
    zz = rr.integers(0, m_all, size=(R_BUDGET, m_all))
    est = D[fz, zz].mean(1)
    print(f"    simulated, a fresh format for each image (images and formats drawn with replacement, {R_BUDGET} times): "
          f"sd {est.std():.3f}, reversal {np.mean(est < 0) + 0.5 * np.mean(est == 0):.3f}")
    r1 = np.random.default_rng(17)      # a generator of its own, so that the other resamples are unchanged
    f1 = r1.integers(0, len(GRAMMAR), size=R_REV)
    est1 = np.empty(R_REV)
    for lo in range(0, R_REV, 2000):    # in blocks, to keep the index arrays small
        z1 = r1.integers(0, m_all, size=(min(2000, R_REV - lo), m_all))
        est1[lo:lo + len(z1)] = D[f1[lo:lo + len(z1), None], z1].mean(1)
    print(f"    simulated, one format for all images (images drawn with replacement, {R_REV} times): "
          f"sd {est1.std():.3f}, reversal {np.mean(est1 < 0) + 0.5 * np.mean(est1 == 0):.3f}")
    Fstar = math.sqrt(BUDGET * var_s / var_i)
    print(f"  crossed design, budget {BUDGET}: best number of formats F* = sqrt(N) sd_s/sd_i = {Fstar:.1f}, "
          f"sd at F* {math.sqrt(2 * math.sqrt(var_s * var_i) / math.sqrt(BUDGET) + var_si / BUDGET):.3f}")
    print(f"  floor of a crossed design on all 539 images: sd {math.sqrt(var_i / 539):.3f}")

    rng = np.random.default_rng(15)
    nF = len(GRAMMAR)
    rows = {"F": FS, "shared": [], "sharedpred": [], "sep": [], "seppred": [], "small": [], "smallpred": [],
            "sharedlo": [], "sharedhi": [], "seplo": [], "sephi": [], "smalllo": [], "smallhi": []}
    sign = np.sign(Delta)

    def rev_rate(diff):
        r = np.mean(sign * diff < 0) + 0.5 * np.mean(diff == 0)
        h = 1.96 * math.sqrt(r * (1 - r) / len(diff))
        return r, max(r - h, 0.0), r + h

    print("\nreversal probability by number of formats F (observed over resamples / normal model):")
    for Fk in FS:
        f1 = rng.integers(0, nF, size=(R_REV, Fk))
        f2 = rng.integers(0, nF, size=(R_REV, Fk))
        shared = (a2[f1] - a1[f1]).mean(1)
        sep = a2[f1].mean(1) - a1[f2].mean(1)
        # nested: fresh M_SMALL images for every format, the same images for both networks
        z = rng.integers(0, len(YTE), size=(R_REV, Fk, M_SMALL))
        small = D[f1[:, :, None], z].mean((1, 2))
        p_sh = stats.norm.cdf(-math.sqrt(Fk) * abs(Delta) / s)
        p_sep = stats.norm.cdf(-math.sqrt(Fk) * abs(Delta) / s_sep)
        p_sm = stats.norm.cdf(-abs(Delta) / math.sqrt(var_s / Fk + (var_i + var_si) / (Fk * M_SMALL)))
        for key, arr, pred in (("shared", shared, p_sh), ("sep", sep, p_sep), ("small", small, p_sm)):
            r, lo, hi = rev_rate(arr)
            rows[key].append(r); rows[key + "lo"].append(lo); rows[key + "hi"].append(hi)
            rows[key + "pred"].append(pred)
        print(f"  F={Fk:3d}: shared {rows['shared'][-1]:.4f} / {p_sh:.4f}   separate {rows['sep'][-1]:.4f} / {p_sep:.4f}"
              f"   shared, {M_SMALL} images each {rows['small'][-1]:.4f} / {p_sm:.4f}")
    lk.write_dat("figures/data/15-instruments-formats-reversal.dat", rows)

    # ---- a fixed budget of evaluations: crossed versus nested designs --------------------------
    designs = [F_ for F_ in (1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000) if BUDGET % F_ == 0]
    bud = {"F": [], "m": [], "crossed": [], "crossedpred": [], "nested": [], "nestedpred": []}
    print(f"\nstandard deviation of the estimated difference (points) with {BUDGET} evaluations per network:")
    for F_ in designs:
        m_ = BUDGET // F_
        fr = rng.integers(0, nF, size=(R_BUDGET, F_))
        zc = rng.integers(0, len(YTE), size=(R_BUDGET, m_))
        crossed = D[fr[:, :, None], zc[:, None, :]].mean((1, 2))
        zn = rng.integers(0, len(YTE), size=(R_BUDGET, F_, m_))
        nested = D[fr[:, :, None], zn].mean((1, 2))
        pc = math.sqrt(var_s / F_ + var_i / m_ + var_si / BUDGET)
        pn = math.sqrt(var_s / F_ + (var_i + var_si) / BUDGET)
        for key, val in (("F", F_), ("m", m_), ("crossed", crossed.std()), ("crossedpred", pc),
                         ("nested", nested.std()), ("nestedpred", pn)):
            bud[key].append(val)
        print(f"  F={F_:4d} m={m_:4d}: crossed {crossed.std():.3f} (formula {pc:.3f})   "
              f"nested {nested.std():.3f} (formula {pn:.3f})")
    lk.write_dat("figures/data/15-instruments-formats-budget.dat", bud)

    # ---- a leaderboard of six networks: Kendall's tau with the ranking by the mean ------------
    mu = acc.mean(1)
    pairs = list(itertools.combinations(range(len(names)), 2))
    sd_pair = {(a, b): (acc[b] - acc[a]).std() for a, b in pairs}
    tau = {"F": FS, "tau": [], "lo": [], "hi": [], "pred": [], "two": [], "twopred": []}
    print("\nsix networks: Kendall's tau between the ranking on F formats and the ranking by the mean")
    for a, b in pairs:
        print(f"    {names[a]:9s} vs {names[b]:9s}: gap {mu[b] - mu[a]:6.2f}, sd of per-format gap {sd_pair[(a, b)]:.2f}")
    for Fk in FS:
        f1 = rng.integers(0, nF, size=(R_TAU, Fk))
        f2 = rng.integers(0, nF, size=(R_TAU, Fk))
        m1 = acc[:, f1].mean(2)                    # networks x resamples
        m2 = acc[:, f2].mean(2)
        t_true, t_two = np.zeros(R_TAU), np.zeros(R_TAU)
        for a, b in pairs:
            s_true = np.sign(mu[b] - mu[a])
            sg1, sg2 = np.sign(m1[b] - m1[a]), np.sign(m2[b] - m2[a])
            t_true += sg1 * s_true
            t_two += sg1 * sg2
        t_true /= len(pairs); t_two /= len(pairs)
        p = [stats.norm.cdf(-math.sqrt(Fk) * abs(mu[b] - mu[a]) / sd_pair[(a, b)]) for a, b in pairs]
        pred = np.mean([1 - 2 * pp for pp in p])
        pred2 = np.mean([(1 - 2 * pp) ** 2 for pp in p])
        m_, h_ = t_true.mean(), 1.96 * t_true.std() / math.sqrt(R_TAU)
        tau["tau"].append(m_); tau["lo"].append(m_ - h_); tau["hi"].append(m_ + h_); tau["pred"].append(pred)
        tau["two"].append(t_two.mean()); tau["twopred"].append(pred2)
        print(f"  F={Fk:3d}: tau with the mean ranking {m_:.3f} (normal model {pred:.3f}); "
              f"tau between two independent sets of F formats {t_two.mean():.3f} (model {pred2:.3f}); "
              f"P(same winner as the mean) {np.mean(m1.argmax(0) == mu.argmax()):.3f}")
    lk.write_dat("figures/data/15-instruments-formats-tau.dat", tau)

    # ---- scatter of the pair, every format -------------------------------------------------
    lk.write_dat("figures/data/15-instruments-formats-scatter.dat",
                 {"mlp": a1, "cnn": a2, "clean": [int(f == CLEAN) for f in range(nF)]}, fmt="{:.3f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
