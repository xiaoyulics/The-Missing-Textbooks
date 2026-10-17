"""Lab 8.2 (Chapter 8, Ablations): the home-field advantage of an ablation.

The network and pruning procedure of Lab 8.1 (two-layer perceptron of width 128 on the
digits; iterative magnitude pruning to 2.8% of each hidden layer and 21% of the output
layer, resetting to the initial values after each of seven rounds) are run with learning
rate 0.1 for the dense training and the pruning rounds. With the final mask of each seed
fixed, three arms are retrained over a grid of ten learning rates from 0.01 to 0.5:

  ticket   the mask with the original initial values,
  reinit   the mask with a fresh draw from the initial distribution,
  sham     the mask shuffled within each layer, with the original initial values,

and the dense network is trained at the same learning rates for reference. All runs use
SGD with momentum 0.9, batch size 64, a cosine schedule and 30 epochs; 20 seeds, each with
its own split, initialization and data order.

For each ablated arm the script computes the effect at a fixed learning rate h (ablated
error minus ticket error, paired by seed), the re-tuned effect (best ablated mean minus
best ticket mean over the grid), and the regrets of both arms at h, which satisfy the
identity of the chapter's home-field proposition exactly. Intervals: 95% t-intervals over
seeds for paired differences; percentile bootstrap over seeds (10,000 resamples) for
quantities that involve a minimum over the grid.

A resampling study then imitates small studies: each draws 6 of the 20 seeds, chooses the
ticket's learning rate as the one with the lowest mean error over 3 of them, reports that
error, and runs the ablated arm at the chosen rate on the other 3 seeds (fresh runs). The
expected reported effect is compared with the re-tuned effect, and its excess is split into
the selection bonus and the home-field term of the chapter's corollary (10,000 studies).

Writes
  figures/data/08-ablations-homefield-runs.dat    one row per seed and learning rate (errors in %)
  figures/data/08-ablations-homefield-means.dat   one row per learning rate: means and half-widths
  figures/data/08-ablations-homefield-effects.dat effects of the two ablations up to rate 0.3
Prints every number quoted in the text.
Run from the volume root:  python3 figures/scripts/08-ablations-homefield.py   (about 5 min)
"""
import os, sys, time, math, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import labkit as lk

T0 = time.time()
WIDTH, DEPTH, EPOCHS = 128, 2, 30
RATE, ROUNDS = 0.4, 7
LR_MASK = 0.1
GRID = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
SEEDS = list(range(20))
ARMS = ["ticket", "reinit", "sham", "dense"]


def weights(model):
    return [m.weight for m in model.net if isinstance(m, torch.nn.Linear)]


def build(seed):
    torch.manual_seed(seed)
    return lk.MLP(64, 10, WIDTH, DEPTH)


def train(cfg, model, masks, data, order):
    """Train with the pruned weights held at zero; return the test error in %."""
    Xtr, ytr, Xte, yte = data
    opt = lk.make_optimizer(cfg, model.parameters())
    n, bs = Xtr.shape[0], cfg.batch_size
    total = int(round(cfg.epochs * (n // bs)))
    g = torch.Generator().manual_seed(60_000 + order)
    perm, pos = torch.randperm(n, generator=g), 0
    W = weights(model)
    with torch.no_grad():
        for w, m in zip(W, masks):
            w.mul_(m)
    for step in range(total):
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g), 0
        idx = perm[pos:pos + bs]
        pos += bs
        for gr in opt.param_groups:
            gr["lr"] = cfg.lr * lk.lr_factor(cfg, step, total)
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for w, m in zip(W, masks):
            w.grad.mul_(m)
        opt.step()
        with torch.no_grad():
            for w, m in zip(W, masks):
                w.mul_(m)
    with torch.no_grad():
        out = model(Xte)
        if not torch.isfinite(out).all():
            return 90.0
        return (out.argmax(1) != yte).float().mean().item() * 100


def prune_round(model, masks):
    new, W = [], weights(model)
    for li, (w, m) in enumerate(zip(W, masks)):
        r = RATE / 2 if li == len(W) - 1 else RATE
        k = int(round(r * m.sum().item()))
        mags = w.detach().abs().clone()
        mags[~m.bool()] = float("inf")
        mm = m.clone().flatten()
        mm[torch.argsort(mags.flatten())[:k]] = 0
        new.append(mm.view_as(m))
    return new


def shuffled(masks, seed):
    g = torch.Generator().manual_seed(seed)
    return [m.flatten()[torch.randperm(m.numel(), generator=g)].view_as(m) for m in masks]


def fresh_copy(m0, theta0):
    m = copy.deepcopy(m0)
    m.load_state_dict(theta0)
    return m


# --------------------------------------------------------------------------------------
# Runs: Y[arm] has shape (seeds, grid)
# --------------------------------------------------------------------------------------
Y = {a: np.zeros((len(SEEDS), len(GRID))) for a in ARMS}
for i, s in enumerate(SEEDS):
    cfg = lk.Config(width=WIDTH, depth=DEPTH, optimizer="momentum", lr=LR_MASK, momentum=0.9,
                    epochs=EPOCHS, batch_size=64, schedule="cosine")
    data = lk.load(cfg, split=s)
    m0 = build(50_000 + s)
    theta0 = copy.deepcopy(m0.state_dict())
    masks = [torch.ones_like(w) for w in weights(m0)]
    model = fresh_copy(m0, theta0)
    train(cfg, model, masks, data, s)
    for r in range(ROUNDS):
        masks = prune_round(model, masks)
        if r < ROUNDS - 1:
            model = fresh_copy(m0, theta0)
            train(cfg, model, masks, data, s)
    sham = shuffled(masks, 95_000 + s)
    ones = [torch.ones_like(w) for w in weights(m0)]
    for j, lr in enumerate(GRID):
        c = lk.Config(**{**cfg.__dict__, "lr": lr})
        Y["ticket"][i, j] = train(c, fresh_copy(m0, theta0), masks, data, s)
        Y["reinit"][i, j] = train(c, build(90_000 + s), masks, data, s)
        Y["sham"][i, j] = train(c, fresh_copy(m0, theta0), sham, data, s)
        Y["dense"][i, j] = train(c, fresh_copy(m0, theta0), ones, data, s)
    if (i + 1) % 5 == 0:
        print(f"  {i + 1} seeds done [{time.time() - T0:.0f}s]", flush=True)

n, K = len(SEEDS), len(GRID)
tq = lambda x: lk.t_interval(x)
print(f"\nMask from learning rate {LR_MASK}; final densities as in Lab 8.1")
print("\nTest error (%) by retraining learning rate: mean +- 95% t half-width over 20 seeds")
for j, lr in enumerate(GRID):
    print(f"  lr {lr:5.2f}: " + "  ".join(f"{a} {tq(Y[a][:, j])[0]:6.2f} +- {tq(Y[a][:, j])[1]:5.2f}" for a in ARMS)
          + f"  | diverged (error 90): " + " ".join(f"{a} {(Y[a][:, j] >= 89.9).sum()}" for a in ARMS))

means = {a: Y[a].mean(0) for a in ARMS}
best = {a: int(np.argmin(means[a])) for a in ARMS}
for a in ARMS:
    print(f"  best learning rate of {a}: {GRID[best[a]]} (mean error {means[a][best[a]]:.2f})")

# --------------------------------------------------------------------------------------
# The home-field identity: fixed-h effect = re-tuned effect + regret_ablated(h) - regret_full(h)
# (errors: lower is better; the effect of an ablation is ablated error minus ticket error)
# --------------------------------------------------------------------------------------
B = 10_000
rng = np.random.default_rng(20261018)
boot_idx = rng.integers(0, n, size=(B, n))


def boot(stat):
    vals = np.array([stat(ix) for ix in boot_idx])
    return np.percentile(vals, [2.5, 97.5])


for a in ("reinit", "sham"):
    f_star, g_star = means["ticket"].min(), means[a].min()
    retuned = g_star - f_star
    lo, hi = boot(lambda ix: Y[a][ix].mean(0).min() - Y["ticket"][ix].mean(0).min())
    print(f"\n{a}: re-tuned effect {retuned:.2f} (bootstrap 95% {lo:.2f} to {hi:.2f}); "
          f"ticket best at {GRID[best['ticket']]}, {a} best at {GRID[best[a]]}")
    print(f"  {'h':>5s} {'fixed-h effect':>22s} {'regret ' + a:>16s} {'regret ticket':>14s} {'check':>8s}")
    for j, lr in enumerate(GRID):
        d = Y[a][:, j] - Y["ticket"][:, j]
        m, h = tq(d)
        reg_g = means[a][j] - g_star
        reg_f = means["ticket"][j] - f_star
        print(f"  {lr:5.2f} {m:+10.2f} +- {h:5.2f} {reg_g:16.2f} {reg_f:14.2f} {retuned + reg_g - reg_f - m:+8.1e}")

# paired regret intervals at the ticket's plateau, for the text
print("\nRegret of each ablated arm at h, against its own best grid value (paired t over seeds):")
for a in ("reinit", "sham"):
    for j, lr in enumerate(GRID):
        if lr in (0.05, 0.1, 0.2, 0.3):
            m, h = tq(Y[a][:, j] - Y[a][:, best[a]])
            print(f"  {a} at {lr}: {m:+.2f} +- {h:.2f}")
print("Dense network at h against its best grid value:")
for j, lr in enumerate(GRID):
    m, h = tq(Y["dense"][:, j] - Y["dense"][:, best["dense"]])
    print(f"  dense at {lr}: {m:+.2f} +- {h:.2f}")
print("Ticket at h against its best grid value:")
for j, lr in enumerate(GRID):
    m, h = tq(Y["ticket"][:, j] - Y["ticket"][:, best["ticket"]])
    print(f"  ticket at {lr}: {m:+.2f} +- {h:.2f}")

# --------------------------------------------------------------------------------------
# Small studies that tune the ticket on their own runs (the corollary)
# --------------------------------------------------------------------------------------
S = 10_000
srng = np.random.default_rng(3407)
print("\nSmall studies: 3 seeds choose and report the ticket's learning rate, 3 fresh seeds run the ablation")
for a in ("reinit", "sham"):
    f_star, g_star = means["ticket"].min(), means[a].min()
    D, V, G, H = np.zeros(S), np.zeros(S), np.zeros(S), np.zeros(S, dtype=int)
    for b in range(S):
        sel = srng.choice(n, size=6, replace=False)
        tune, abl = sel[:3], sel[3:]
        v = Y["ticket"][tune].mean(0)
        jh = int(np.argmin(v))
        H[b] = jh
        V[b] = v[jh]                           # reported ticket error at the chosen rate
        D[b] = Y[a][abl, jh].mean() - v[jh]    # reported ablation effect
        G[b] = means[a][jh]                    # population error of the ablated arm at the chosen rate
    sel_bonus = f_star - V.mean()
    home = G.mean() - g_star
    print(f"  {a}: mean reported effect {D.mean():.2f} (sd {D.std():.2f}); re-tuned effect {g_star - f_star:.2f}; "
          f"excess {D.mean() - (g_star - f_star):.2f} = selection bonus {sel_bonus:.2f} + home-field {home:.2f}")
    freq = np.bincount(H, minlength=K) / S
    print("    chosen learning rate frequencies: " + ", ".join(f"{GRID[k]}: {freq[k]:.2f}" for k in range(K) if freq[k] > 0))
    print(f"    reported effect quartiles {np.percentile(D, [25, 50, 75]).round(2)}; "
          f"fraction of studies reporting more than twice the re-tuned effect {(D > 2 * (g_star - f_star)).mean():.2f}")

# --------------------------------------------------------------------------------------
# Data for the figure
# --------------------------------------------------------------------------------------
jit = np.random.default_rng(7).uniform(-0.025, 0.025, size=(n, K))
rows = {"seed": [], "lr": [], "x": []}
for a in ARMS:
    rows[a] = []
for i in range(n):
    for j, lr in enumerate(GRID):
        rows["seed"].append(SEEDS[i]); rows["lr"].append(lr)
        rows["x"].append(10 ** (math.log10(lr) + jit[i, j]))
        for a in ARMS:
            rows[a].append(Y[a][i, j])
lk.write_dat("figures/data/08-ablations-homefield-runs.dat", rows)
cols = {"lr": GRID}
for a in ARMS:
    ms, hs = zip(*[tq(Y[a][:, j]) for j in range(K)])
    cols[a], cols["h" + a] = list(ms), list(hs)
for a in ("reinit", "sham"):
    ms, hs = zip(*[tq(Y[a][:, j] - Y["ticket"][:, j]) for j in range(K)])
    cols["d" + a], cols["hd" + a] = list(ms), list(hs)
lk.write_dat("figures/data/08-ablations-homefield-means.dat", cols)
# effects at each rate up to 0.3 (at higher rates the ticket itself begins to fail), with the
# re-tuned effects as constant columns for the reference lines
keep = [j for j, lr in enumerate(GRID) if lr <= 0.3]
eff = {"lr": [GRID[j] for j in keep]}
for a in ("reinit", "sham"):
    eff["d" + a] = [cols["d" + a][j] for j in keep]
    eff["hd" + a] = [cols["hd" + a][j] for j in keep]
    eff["r" + a] = [means[a].min() - means["ticket"].min()] * len(keep)
lk.write_dat("figures/data/08-ablations-homefield-effects.dat", eff)
print(f"\nTotal time {time.time() - T0:.0f}s")
