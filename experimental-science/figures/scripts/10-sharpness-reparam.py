"""Lab 10.2 of *Deep Learning as an Experimental Science*: rescaling a trained network.

Five networks of one configuration of the sweep of Lab 10.1 (width 128, learning rate 0.1, batch
size 32, no weight decay, seeds 0-4; two hidden layers, plain SGD until the training
cross-entropy reaches 0.02 on the 539 training images of split 0) are rescaled without changing the function they
compute: for c > 0 the first layer's weights and biases and the second layer's biases are
multiplied by c and the output weights divided by c (the rescaling theorem of Chapter 10 with
alpha = (c, 1, 1/c)). For 17 values of c from 0.01 to 100 the script measures the top Hessian
eigenvalue of the training loss (power iteration on Hessian-vector products), the perturbation
sharpness P_0.05 and the adaptive perturbation sharpness P^ad_0.5 of Lab 10.1 (common random
numbers), the training loss and the test error, and it checks that the logits are unchanged.
It also computes, at c = 1, the block eigenvalues mu (output weights) and nu (the coordinates
that the rescaling multiplies by c), whose bounds c^2 mu and nu/c^2 the top eigenvalue must
exceed. About 20 seconds.

Run from the volume root:  python3 figures/scripts/10-sharpness-reparam.py
"""
import os, sys, math, time, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import labkit as lk

_spec = importlib.util.spec_from_file_location(
    "sweep", os.path.join(os.path.dirname(os.path.abspath(__file__)), "10-sharpness-sweep.py"))
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)          # reuse training and sharpness code of Lab 10.1 (its main does not run)

WIDTH, LR, BATCH, WD = 128, 0.1, 32, 0.0
SEEDS = (0, 1, 2, 3, 4)
CS = [10 ** (k / 4) for k in range(-8, 9)]


def rescale(model, c):
    """Apply T_c in place: (W1, b1) * c, b2 * c, W3 / c; returns the model."""
    lin = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        lin[0].weight.mul_(c); lin[0].bias.mul_(c)
        lin[1].bias.mul_(c)
        lin[2].weight.div_(c)
    return model


def block_top(model, X, y, names):
    """Top eigenvalue of the Hessian block of the parameters whose names are in `names`."""
    params = list(model.named_parameters())
    n, hvp = sw.hessian_operator(model, X, y)
    mask = torch.cat([torch.full((p.numel(),), float(nm in names)) for nm, p in params])

    def block_hvp(v):
        return hvp(v * mask) * mask
    gen = torch.Generator().manual_seed(7)
    v = torch.randn(n, generator=gen) * mask
    v /= v.norm()
    lam = 0.0
    for it in range(1, 2001):
        hv = block_hvp(v)
        new = torch.dot(v, hv).item()
        v = hv / hv.norm()
        if it > 1 and abs(new - lam) <= 1e-7 * abs(new):
            return new
        lam = new
    return lam


if __name__ == "__main__":
    t_start = time.time()
    data = lk.digits(split=0, test_frac=0.7, train_size=sw.TRAIN_SIZE)
    Xtr, ytr, Xte, yte = data
    rows = {k: [] for k in ("seed", "c", "lam", "mu_bound", "nu_bound", "s_abs", "s_ad", "train_loss", "test_err", "maxdiff")}
    summary = []
    for seed in SEEDS:
        model, ce, steps, ok = sw.train_to_target(data, WIDTH, LR, BATCH, WD, seed)
        base_state = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            logits0 = torch.cat([model(Xtr), model(Xte)])
        mu = block_top(model, Xtr, ytr, {"net.4.weight"})
        nu = block_top(model, Xtr, ytr, {"net.0.weight", "net.0.bias", "net.2.bias"})
        for c in CS:
            model.load_state_dict(base_state)
            rescale(model, c)
            n, hvp = sw.hessian_operator(model, Xtr, ytr)
            lam, its = sw.power_iteration(n, hvp, iters=3000, tol=1e-7, seed=seed)
            with torch.no_grad():
                logits = torch.cat([model(Xtr), model(Xte)])
                maxdiff = (logits - logits0).abs().max().item()
                train_loss = F.cross_entropy(model(Xtr), ytr).item()
            rows["seed"].append(seed); rows["c"].append(c); rows["lam"].append(lam)
            rows["mu_bound"].append(c * c * mu); rows["nu_bound"].append(nu / (c * c))
            rows["s_abs"].append(sw.perturbation_sharpness(model, Xtr, ytr, sw.SIG_ABS, False, seed=seed))
            rows["s_ad"].append(sw.perturbation_sharpness(model, Xtr, ytr, sw.SIG_AD, True, seed=seed))
            rows["train_loss"].append(train_loss); rows["test_err"].append(sw.errors(model, Xte, yte))
            rows["maxdiff"].append(maxdiff)
        model.load_state_dict(base_state)
        print(f"seed {seed}: {steps} steps, ce {ce:.4f}, mu {mu:.4f}, nu {nu:.4f}, sqrt(mu nu) {math.sqrt(mu*nu):.4f}, "
              f"lam(1) {rows['lam'][-9]:.4f}", flush=True)
        summary.append((mu, nu))

    r = {k: np.array(v) for k, v in rows.items()}
    print(f"largest change of any logit under rescaling: {r['maxdiff'].max():.2e}")
    print(f"test error: {np.unique(np.round(r['test_err'], 6))} (per seed constant: "
          f"{all(np.ptp(r['test_err'][r['seed']==s]) == 0 for s in SEEDS)})")
    print(f"training loss: max relative change {np.max(np.abs(r['train_loss'] / np.repeat(r['train_loss'][8::17], 17) - 1)):.2e}")
    print(" c        lam (mean of 5)   min/max ratio to lam(1)   bound max(c^2 mu, nu/c^2)   s_abs   s_ad   test err")
    means = {k: [] for k in ("c", "lam", "lam_lo", "lam_hi", "bound_mu", "bound_nu", "s_abs", "s_ad", "test_err",
                             "rel_lam", "rel_sabs", "rel_sad", "rel_err")}
    lam1 = r["lam"][r["c"] == 1.0]
    sabs1 = r["s_abs"][r["c"] == 1.0]; sad1 = r["s_ad"][r["c"] == 1.0]; err1 = r["test_err"][r["c"] == 1.0]
    for c in CS:
        sel = np.isclose(r["c"], c)
        lam, bmu, bnu = r["lam"][sel], r["mu_bound"][sel], r["nu_bound"][sel]
        print(f"{c:9.4f}  {lam.mean():12.4f}   {np.min(lam/lam1):9.3f} {np.max(lam/lam1):9.3f}   "
              f"{np.mean(np.maximum(bmu, bnu)):12.4f}  {r['s_abs'][sel].mean():8.4f} {r['s_ad'][sel].mean():.4f} "
              f"{r['test_err'][sel].mean():.3f}")
        means["c"].append(c); means["lam"].append(lam.mean()); means["lam_lo"].append(lam.min()); means["lam_hi"].append(lam.max())
        means["bound_mu"].append(bmu.mean()); means["bound_nu"].append(bnu.mean())
        means["s_abs"].append(r["s_abs"][sel].mean()); means["s_ad"].append(r["s_ad"][sel].mean())
        means["test_err"].append(r["test_err"][sel].mean())
        means["rel_lam"].append(np.mean(lam / lam1)); means["rel_sabs"].append(np.mean(r["s_abs"][sel] / sabs1))
        means["rel_sad"].append(np.mean(r["s_ad"][sel] / sad1)); means["rel_err"].append(np.mean(r["test_err"][sel] / err1))
    dev_ad = max(np.max(np.abs(r["s_ad"][r["seed"] == s] / r["s_ad"][(r["seed"] == s) & (r["c"] == 1.0)] - 1)) for s in SEEDS)
    sabs_ratio = [np.max(r["s_abs"][r["seed"] == s]) / r["s_abs"][(r["seed"] == s) & (r["c"] == 1.0)][0] for s in SEEDS]
    print(f"adaptive sharpness: largest relative change over c {dev_ad:.1e}; "
          f"fixed-sigma sharpness: largest over smallest ratio per network {min(sabs_ratio):.0f}-{max(sabs_ratio):.0f}")
    ratio = np.array(means["lam"]).max() / np.array(means["lam"]).min()
    print(f"range of mean lam over c: {min(means['lam']):.4f} to {max(means['lam']):.1f} (ratio {ratio:.3g})")
    print(f"lam >= max(c^2 mu, nu/c^2) for every run: {bool(np.all(r['lam'] >= np.maximum(r['mu_bound'], r['nu_bound']) * (1 - 1e-6)))}")
    ends = np.isclose(r["c"], CS[0]) | np.isclose(r["c"], CS[-1])
    excess = r["lam"][ends] / np.maximum(r["mu_bound"], r["nu_bound"])[ends] - 1
    print(f"at c = {CS[0]:g} and {CS[-1]:g}: lam exceeds the bound by a relative {excess.min():.1e} to {excess.max():.1e}")
    mus, nus = np.array(summary).T
    print(f"mu {mus.mean():.4f} (range {mus.min():.4f}-{mus.max():.4f}), nu {nus.mean():.4f} (range {nus.min():.4f}-{nus.max():.4f})")
    lk.write_dat("figures/data/10-sharpness-reparam-runs.dat", {k: rows[k] for k in ("seed", "c", "lam", "s_abs", "s_ad", "test_err")})
    lk.write_dat("figures/data/10-sharpness-reparam-mean.dat", means)
    print(f"total {time.time()-t_start:.0f} s", flush=True)
