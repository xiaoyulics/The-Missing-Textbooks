"""Lab 2.3 of Chapter 2 (Instability): is a twin a new draw from the run distribution?

Run from the volume root:

    python3 figures/scripts/02-instability-draws.py

Same network, optimizer and regimes as Lab 2.2 (02-instability-twins.py): the digits MLP,
SGD with momentum 0.9, learning rate 0.1, batch size 64, 50 epochs; clean labels, or 20% of
the training labels replaced by random labels.

Part A. In each regime, 100 base runs (init = order = i), their one-ulp twins (one randomly
chosen initial weight moved by one unit in the last place), and 100 runs with new seeds
(init = order = 1000 + i). In the noisy regime also 100 "early" twins, whose initial
weights are moved by 1% of their norm in a random direction (they diverge early, Lab 2.2).
For each group: the test-error distribution, two-sample tests of twins against new seeds,
the correlation between each base run and its twin, the ratio of mean squared differences
(Proposition "what a twin measures"), and the test-set disagreement (churn).

Part B. The regime map: for learning rates 0.0125 ... 0.15, clean and noisy labels, 20
pairs per cell: the disagreement between a run and its one-ulp twin, divided by the mean
disagreement between independent runs.

Writes figures/data/02-instability-draws-*.dat and prints the numbers quoted in the
chapter. Runtime: about seven minutes on a shared laptop (431 s at a load average
of about 6).
"""
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk  # noqa: E402  (two threads, deterministic algorithms)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy import stats  # noqa: E402

T0 = time.time()
CFG = lk.Config(width=64, depth=2, optimizer="momentum", lr=0.1, momentum=0.9,
                batch_size=64, epochs=50)
N_RUNS = 100
N_MAP = 20
LRS = [0.0125, 0.025, 0.05, 0.1, 0.15]
B_BOOT = 10_000


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def one_ulp(model, rng):
    """Move one uniformly chosen parameter by one unit in the last place, up or down."""
    params = list(model.parameters())
    sizes = np.array([p.numel() for p in params])
    j = int(rng.integers(0, sizes.sum()))
    t = int(np.searchsorted(np.cumsum(sizes), j, side="right"))
    k = j - int(np.cumsum(sizes)[t - 1]) if t > 0 else j
    direction = float("inf") if rng.random() < 0.5 else float("-inf")
    with torch.no_grad():
        v = params[t].view(-1)
        v[k] = torch.nextafter(v[k], torch.tensor(direction))


def random_direction(model, rel, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        norm = flat(model).norm()
        d = [torch.randn(p.shape, generator=g) for p in model.parameters()]
        dn = torch.sqrt(sum((x ** 2).sum() for x in d))
        for p, x in zip(model.parameters(), d):
            p.add_(rel * norm * x / dn)


def train(cfg, data, init, order, perturb=None):
    """One run, exactly as labkit.run, with an optional perturbation after initialization.
    Returns the test-set predictions and the final parameter vector."""
    Xtr, ytr, Xte, yte = data
    torch.manual_seed(50_000 + init)
    model = lk.MLP(Xtr.shape[1], 10, cfg.width, cfg.depth, cfg.activation, cfg.batchnorm, cfg.dropout)
    if perturb is not None:
        perturb(model)
    opt = lk.make_optimizer(cfg, model.parameters())
    n = Xtr.shape[0]
    bs = min(cfg.batch_size, n)
    total = int(round(cfg.epochs * max(1, n // bs)))
    g_order = torch.Generator().manual_seed(60_000 + order)
    torch.manual_seed(80_000)
    perm, pos = torch.randperm(n, generator=g_order), 0
    for step in range(total):
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g_order), 0
        idx = perm[pos:pos + bs]
        pos += bs
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(Xte).argmax(1)
    return pred, flat(model)


def runs(cfg, data, i, early=False):
    """Base run i, its one-ulp twin, a run with new seeds, and optionally an early twin."""
    out = {"base": train(cfg, data, i, i),
           "ulp": train(cfg, data, i, i, perturb=lambda m: one_ulp(m, np.random.default_rng(3000 + i))),
           "new": train(cfg, data, 1000 + i, 1000 + i)}
    if early:
        out["early"] = train(cfg, data, i, i, perturb=lambda m: random_direction(m, 1e-2, 7000 + i))
    return out


def boot_ratio(num, den, seed=0):
    """Percentile bootstrap interval for mean(num) / mean(den), resampling pairs."""
    rng = np.random.default_rng(seed)
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), size=(B_BOOT, len(num)))
    reps = num[idx].mean(1) / den[idx].mean(1)
    return float(num.mean() / den.mean()), float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def fisher_ci(x, y):
    r = float(np.corrcoef(x, y)[0, 1])
    z, se = np.arctanh(r), 1 / np.sqrt(len(x) - 3)
    return r, float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def perm_test(a, b, seed=0, B=10_000):
    """Two-sided permutation test for a difference in means."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    cnt = 0
    for _ in range(B):
        rng.shuffle(pool)
        cnt += abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= obs - 1e-15
    return (cnt + 1) / (B + 1)


regimes = {"c": lk.digits(label_noise=0.0), "n": lk.digits(label_noise=0.2, noise=0)}
names = {"c": "clean labels", "n": "20% noisy labels"}

# ---------------------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------------------
wide = {"i": list(range(N_RUNS)), "jx": list(np.random.default_rng(5).uniform(-0.28, 0.28, N_RUNS))}
store = {}
for r, data in regimes.items():
    yte = data[3]
    groups = ["base", "ulp", "early", "new"] if r == "n" else ["base", "ulp", "new"]
    res = [runs(CFG, data, i, early=(r == "n")) for i in range(N_RUNS)]
    store[r] = res
    err = {g: np.array([(x[g][0] != yte).float().mean().item() for x in res]) for g in groups}
    dis = {g: np.array([(x["base"][0] != x[g][0]).float().mean().item() for x in res]) for g in groups[1:]}
    same = sum(torch.equal(x["base"][1], x["ulp"][1]) for x in res)
    for g in groups:
        wide[f"{r}_{g}"] = list(err[g])
    pos = {g: k + 1 for k, g in enumerate(groups)}
    summ = {"pos": [], "mean": [], "hw": []}
    print(f"\n== Part A, {names[r]}: {N_RUNS} runs per group, {time.time() - T0:.0f} s so far")
    for g in groups:
        m, hw = lk.t_interval(err[g])
        summ["pos"].append(pos[g])
        summ["mean"].append(m)
        summ["hw"].append(hw)
        print(f"   {g:6s} test error {100 * m:6.2f} +- {100 * hw:4.2f} %  (SD {100 * err[g].std(ddof=1):4.2f})")
    lk.write_dat(f"figures/data/02-instability-draws-{'clean' if r == 'c' else 'noisy'}.dat", summ)
    for g in groups[1:-1]:
        p_perm = perm_test(err[g], err["new"], seed=1)
        p_ks = stats.ks_2samp(err[g], err["new"]).pvalue
        rho, lo, hi = fisher_ci(err["base"], err[g])
        R, Rlo, Rhi = boot_ratio((err["base"] - err[g]) ** 2, (err["base"] - err["new"]) ** 2, seed=2)
        C, Clo, Chi = boot_ratio(dis[g], dis["new"], seed=3)
        print(f"   {g} twins vs new seeds: permutation p = {p_perm:.2f}, KS p = {p_ks:.2f}")
        print(f"      corr(base, {g}) = {rho:.2f} [{lo:.2f}, {hi:.2f}];"
              f" mean squared difference ratio = {R:.2f} [{Rlo:.2f}, {Rhi:.2f}]")
        print(f"      disagreement with base: {g} {100 * dis[g].mean():.2f}%, new seeds {100 * dis['new'].mean():.2f}%;"
              f" ratio {C:.2f} [{Clo:.2f}, {Chi:.2f}]")
    rho, lo, hi = fisher_ci(err["base"], err["new"])
    print(f"   control: corr(base, new seeds) = {rho:.2f} [{lo:.2f}, {hi:.2f}]")
    print(f"   one-ulp twins whose final weights equal their base bit for bit: {same} of {N_RUNS}")
lk.write_dat("figures/data/02-instability-draws-runs.dat", wide)

# ---------------------------------------------------------------------------------------
# Part B: the regime map
# ---------------------------------------------------------------------------------------
mp = {"lr": LRS}
pairs = {"lr": [], "c": [], "n": []}
for r, data in regimes.items():
    yte = data[3]
    ratio, lo_, hi_, errs = [], [], [], []
    per_pair = []
    for lr in LRS:
        if lr == CFG.lr:
            res = store[r][:N_MAP]                 # the same seeds as Part A
        else:
            cfg = replace(CFG, lr=lr)
            res = [runs(cfg, data, i) for i in range(N_MAP)]
        d_t = np.array([(x["base"][0] != x["ulp"][0]).float().mean().item() for x in res])
        d_n = np.array([(x["base"][0] != x["new"][0]).float().mean().item() for x in res])
        C, Clo, Chi = boot_ratio(d_t, d_n, seed=4)
        ratio.append(C)
        lo_.append(Clo)
        hi_.append(Chi)
        errs.append(np.mean([(x["base"][0] != yte).float().mean().item() for x in res]))
        per_pair.append(d_t / d_n.mean())
        print(f"   map {names[r]:17s} lr {lr:<7g}: twin/independent disagreement {C:.2f} [{Clo:.2f}, {Chi:.2f}]"
              f"  (independent {100 * d_n.mean():.2f}%, test error {100 * errs[-1]:.2f}%)  {time.time() - T0:.0f} s")
    mp[f"{r}_ratio"], mp[f"{r}_lo"], mp[f"{r}_hi"] = ratio, lo_, hi_
    mp[f"{r}_elo"] = [c - l for c, l in zip(ratio, lo_)]
    mp[f"{r}_ehi"] = [h - c for c, h in zip(ratio, hi_)]
    mp[f"{r}_err"] = errs
    if r == "c":
        pairs["lr"] = [lr for lr in LRS for _ in range(N_MAP)]
    pairs[r] = [v for arr in per_pair for v in arr]
lk.write_dat("figures/data/02-instability-draws-map.dat", mp)
lk.write_dat("figures/data/02-instability-draws-mappairs.dat", pairs)
print(f"\nruntime {time.time() - T0:.0f} s")
