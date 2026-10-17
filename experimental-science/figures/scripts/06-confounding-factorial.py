"""Lab 6.1 of *Deep Learning as an Experimental Science*: the architecture and the recipe.

A 2x2 factorial on the scikit-learn digits. Architecture: the two-layer MLP of Chapter 1
(width 64, 8,970 parameters) or a small convolutional network defined below (9,930
parameters). Recipe: the old recipe of Chapter 1 (SGD with momentum 0.9, learning rate
0.05, constant, batch 64, 20 epochs, plain cross-entropy) or a new one (AdamW, learning rate
2e-3, weight decay 0.05, 50 warm-up steps and cosine decay, label smoothing 0.1, 60 epochs).

Twenty blocks: in block b all four cells use split b and the seeds init = order = noise = b,
so that contrasts are computed within a block (same training and test images) and their
intervals are t-intervals over the 20 blocks. Both architectures are trained by the same
loop, which reproduces labkit.run (checked at start-up for the MLP).

Run from the volume root:  python3 figures/scripts/06-confounding-factorial.py
Writes figures/data/06-confounding-factorial-{runs,means,credit}.dat and prints the numbers
quoted in Chapter 6. Two to five minutes on two threads, depending on the load of the
machine (about five CPU-minutes).
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import replace

BLOCKS = list(range(20))
OLD = lk.Config(optimizer="momentum", lr=0.05, momentum=0.9, batch_size=64, epochs=20)
NEW = lk.Config(optimizer="adamw", lr=2e-3, weight_decay=0.05, batch_size=64, epochs=60,
                schedule="cosine", warmup=50, label_smoothing=0.1)
RECIPES = [("old", OLD), ("new", NEW)]
ARCHS = ["mlp", "cnn"]


class CNN(nn.Module):
    """Two 3x3 convolutions (16 and 32 channels, padding 1, ReLU), a 2x2 max-pool and a
    linear classifier: 9,930 parameters, about the MLP's 8,970."""

    def __init__(self, c1=16, c2=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.fc = nn.Linear(c2 * 16, 10)

    def forward(self, x):
        x = x.view(-1, 1, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        return self.fc(x.flatten(1))


def make_model(arch):
    return CNN() if arch == "cnn" else lk.MLP(64, 10, width=64, depth=2)


def macs(arch):
    """Multiply-accumulates per image in a forward pass."""
    if arch == "mlp":
        return 64 * 64 + 64 * 64 + 64 * 10
    return 64 * 16 * 9 + 64 * 32 * 16 * 9 + 512 * 10


def train(arch, cfg, data, init, order, noise):
    """One run; the same seeding, batching and evaluation as labkit.run."""
    t0 = time.time()
    Xtr, ytr, Xte, yte = data
    torch.manual_seed(50_000 + init)
    model = make_model(arch)
    opt = lk.make_optimizer(cfg, model.parameters())
    n = Xtr.shape[0]
    bs = min(cfg.batch_size, n)
    per_epoch = max(1, n // bs)
    total = int(round(cfg.epochs * per_epoch))
    g_order = torch.Generator().manual_seed(60_000 + order)
    g_noise = torch.Generator().manual_seed(70_000 + noise)
    torch.manual_seed(80_000 + noise)
    perm, pos = torch.randperm(n, generator=g_order), 0
    for step in range(total):
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g_order), 0
        idx = perm[pos:pos + bs]
        pos += bs
        xb, yb = Xtr[idx], ytr[idx]
        if cfg.augment:
            xb = lk._augment(xb, g_noise)
        for group in opt.param_groups:
            group["lr"] = cfg.lr * lk.lr_factor(cfg, step, total)
        loss = F.cross_entropy(model(xb), yb, label_smoothing=cfg.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    ev = lk.evaluate(model, Xte, yte, False)
    err = 1.0 - ev["acc"] if math.isfinite(ev["loss"]) else 0.9
    return err, time.time() - t0, lk.n_params(model)


def tint(x):
    m, h = lk.t_interval(np.asarray(x, dtype=float))
    return m, h


def main():
    t_start = time.time()
    # the shared loop reproduces labkit.run exactly for the MLP
    d0 = lk.load(OLD, split=0)
    a = lk.run(replace(OLD, epochs=2), init=0, order=0, data=d0)["test_err"]
    b = train("mlp", replace(OLD, epochs=2), d0, 0, 0, 0)[0]
    assert abs(a - b) < 1e-12, (a, b)
    print(f"check: training loop matches labkit.run ({a:.6f})")
    print("parameters:", {ar: lk.n_params(make_model(ar)) for ar in ARCHS},
          "MACs per image:", {ar: macs(ar) for ar in ARCHS},
          f"ratio {macs('cnn') / macs('mlp'):.1f}")

    E = np.zeros((2, 2, len(BLOCKS)))      # test error (%), [arch, recipe, block]
    S = np.zeros((2, 2, len(BLOCKS)))      # seconds per run
    for bi, blk in enumerate(BLOCKS):
        data = lk.load(OLD, split=blk)
        for ai, ar in enumerate(ARCHS):
            for ri, (rn, cfg) in enumerate(RECIPES):
                err, sec, _ = train(ar, cfg, data, init=blk, order=blk, noise=blk)
                E[ai, ri, bi], S[ai, ri, bi] = 100 * err, sec
        print(f"block {blk:2d}: " + "  ".join(f"{ARCHS[ai]}-{RECIPES[ri][0]} {E[ai, ri, bi]:.2f}"
              for ai in range(2) for ri in range(2)) + f"   ({time.time() - t_start:.0f}s)", flush=True)

    m01 = 539  # test items per split
    print(f"\none test item is {100 / m01:.3f} points of test error")
    print("cell means (test error %, mean and 95% t-half-width over 20 blocks), sd, seconds per run:")
    for ai, ar in enumerate(ARCHS):
        for ri, (rn, _) in enumerate(RECIPES):
            m, h = tint(E[ai, ri])
            print(f"  {ar:3s} {rn}: {m:.2f} +- {h:.2f}   sd {E[ai, ri].std(ddof=1):.2f}   "
                  f"min {E[ai, ri].min():.2f} max {E[ai, ri].max():.2f}   {S[ai, ri].mean():.2f}s")

    # gains are reductions of test error, in points; arch 1 = CNN, recipe 1 = new
    mo, mn, co, cn = E[0, 0], E[0, 1], E[1, 0], E[1, 1]
    naive = mo - cn                       # MLP-old minus CNN-new: what a paper introducing the CNN reports
    naive2 = co - mn                      # CNN-old minus MLP-new: what a paper introducing the MLP reports
    alpha_old = mo - co                   # architecture effect (CNN over MLP) at the old recipe
    alpha_new = mn - cn                   # ... at the new recipe
    rho_mlp = mo - mn                     # recipe effect on the MLP
    rho_cnn = co - cn                     # recipe effect on the CNN
    inter = alpha_new - alpha_old         # interaction contrast I
    theta_a = (alpha_old + alpha_new) / 2  # main effect of architecture = Shapley credit
    theta_r = (rho_mlp + rho_cnn) / 2
    rows = {
        "naive gain, CNN-new over MLP-old": naive,
        "naive gain, MLP-new over CNN-old": naive2,
        "architecture effect at old recipe (alpha_old)": alpha_old,
        "architecture effect at new recipe (alpha_new)": alpha_new,
        "recipe effect on MLP (rho_mlp)": rho_mlp,
        "recipe effect on CNN (rho_cnn)": rho_cnn,
        "interaction contrast I = alpha_new - alpha_old": inter,
        "main effect architecture (Shapley)": theta_a,
        "main effect recipe (Shapley)": theta_r,
    }
    print("\ncontrasts (gain in points = reduction of test error; mean and 95% t-interval over blocks):")
    for k, v in rows.items():
        m, h = tint(v)
        print(f"  {k:48s} {m:6.2f} +- {h:.2f}   [{m - h:.2f}, {m + h:.2f}]")
    # unpaired versions, for comparison with the blocked analysis
    def welch_hw(x, y):
        vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
        nu = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
        from scipy import stats
        return stats.t.ppf(0.975, nu) * math.sqrt(vx + vy)
    print("  unpaired (Welch) half-widths: naive %.2f, alpha_old %.2f, alpha_new %.2f"
          % (welch_hw(mo, cn), welch_hw(mo, co), welch_hw(mn, cn)))
    se_i_unpaired = math.sqrt(sum(E[a_, r_].var(ddof=1) for a_ in range(2) for r_ in range(2)) / len(BLOCKS))
    print(f"  interaction: blocked sd/sqrt(n) {inter.std(ddof=1) / math.sqrt(len(BLOCKS)):.3f}, "
          f"unpaired se {se_i_unpaired:.3f}")
    share = lambda part: 100 * part.mean() / naive.mean()
    print(f"  architecture's share of the naive gain: architecture first {share(alpha_old):.0f}%, "
          f"recipe first {share(alpha_new):.0f}%, Shapley {share(theta_a):.0f}%")
    print(f"  blocks in which the CNN wins at old / new recipe: {(alpha_old > 0).sum()} / {(alpha_new > 0).sum()} of {len(BLOCKS)};"
          f" blocks in which MLP-new beats CNN-old: {(naive2 > 0).sum()}, ties {(naive2 == 0).sum()}")
    # tuned comparison: each architecture with its better recipe (both prefer the new one here)
    best = [min(range(2), key=lambda ri: E[ai, ri].mean()) for ai in range(2)]
    tuned = E[0, best[0]] - E[1, best[1]]
    m, h = tint(tuned)
    print(f"  tuned comparison (each with its better recipe: MLP {RECIPES[best[0]][0]}, CNN {RECIPES[best[1]][0]}): {m:.2f} +- {h:.2f}")
    print(f"  seconds per run: CNN/MLP ratio old {S[1, 0].mean() / S[0, 0].mean():.1f}, new {S[1, 1].mean() / S[0, 1].mean():.1f}")

    # ---- data for the figures --------------------------------------------------------------
    rng = np.random.default_rng(6)
    jit = rng.uniform(-0.07, 0.07, size=len(BLOCKS))
    cols = {"arch": [], "recipe": [], "block": [], "x": [], "err": []}
    for ai in range(2):
        for ri in range(2):
            for bi, blk in enumerate(BLOCKS):
                cols["arch"].append(ai); cols["recipe"].append(ri); cols["block"].append(blk)
                cols["x"].append(ri + (-0.13 if ai == 0 else 0.13) + jit[bi])
                cols["err"].append(E[ai, ri, bi])
    lk.write_dat("figures/data/06-confounding-factorial-runs.dat", cols)
    for ai, ar in enumerate(ARCHS):
        mm = [tint(E[ai, ri]) for ri in range(2)]
        lk.write_dat(f"figures/data/06-confounding-factorial-means-{ar}.dat",
                     {"x": [ri + (-0.13 if ai == 0 else 0.13) for ri in range(2)],
                      "mean": [m_ for m_, _ in mm], "hw": [h_ for _, h_ in mm]})
    # credit: x = 1 architecture first, 2 recipe first, 3 Shapley
    arch_credit = [alpha_old, alpha_new, theta_a]
    rec_credit = [rho_cnn, rho_mlp, theta_r]
    ca = [tint(v) for v in arch_credit]
    cr = [tint(v) for v in rec_credit]
    lk.write_dat("figures/data/06-confounding-factorial-credit.dat",
                 {"x": [1, 2, 3], "xa": [0.9, 1.9, 2.9], "xr": [1.1, 2.1, 3.1],
                  "arch": [m_ for m_, _ in ca], "archhw": [h_ for _, h_ in ca],
                  "recipe": [m_ for m_, _ in cr], "recipehw": [h_ for _, h_ in cr],
                  "naive": [naive.mean()] * 3})
    print(f"\nnaive gain {naive.mean():.2f}; credits written. total time {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
