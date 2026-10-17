"""Lab 10.1 of *Deep Learning as an Experimental Science*: three sharpnesses in one sweep.

A full factorial sweep of the digits MLP (two hidden layers of ReLU units, plain SGD without
momentum) over learning rate {0.03, 0.1, 0.3}, batch size {8, 32, 128}, width {32, 128, 512} and
weight decay {0, 1e-3, 3e-3}: 81 configurations, two runs each, every run with its own seeds
(init = order = run seed), 162 runs. Every run is trained until its cross-entropy on the whole
training set falls to 0.02, the stopping rule of Jiang et al. (2020), so that sharpness is
compared at equal training loss. For every run the script measures
    lam      the top eigenvalue of the Hessian of the training loss (power iteration on
             Hessian-vector products; cross-checked against Lanczos on every 27th run),
    s_abs    perturbation sharpness P_0.05: mean increase of the training loss under Gaussian
             perturbations of standard deviation 0.05 in every coordinate (32 draws),
    s_ad     adaptive perturbation sharpness P^ad_0.5: the same with standard deviation
             0.5 |theta_i| in coordinate i (invariant under function-preserving rescalings),
    norm     the Euclidean norm of the parameters,
and the generalization gap, test error minus training error in percentage points. It prints
the reliability of the gap (one-way analysis of variance over configurations), and for each
measure Kendall's tau with the gap: pooled, given width, given the configuration (the two runs
of a configuration), the granulated coefficients psi_i and Psi of Jiang et al., and the split
of the pooled tau into pairs within and between widths. Controls: the gap of the replicate run
(positive control), a random measure (negative control), and 1/width and lr/B (trivial
measures). The rows of Table 10.1 are printed in LaTeX form.

Data: scikit-learn digits, labkit split 0 with test_frac=0.7: 539 training and 1258 test images.
One to five minutes with two threads, depending on the load of the machine.
Run from the volume root:  python3 figures/scripts/10-sharpness-sweep.py
Options: an integer N trains on the first N training images only; --shared-seeds reproduces the
first version reported in the text, in which every configuration reused the seeds 0 and 1.
"""
import os, sys, math, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import labkit as lk

LRS = (0.03, 0.1, 0.3)
BATCHES = (8, 32, 128)
WIDTHS = (32, 128, 512)
WDS = (0.0, 1e-3, 3e-3)
SEEDS = (0, 1)
TRAIN_SIZE = None      # all 539 training images of split 0 with test_frac=0.7; an integer subsamples
SHARED_SEEDS = False   # True reproduces the first version of the text: run seed = replicate index
if __name__ == "__main__":
    for a in sys.argv[1:]:
        if a == "--shared-seeds":
            SHARED_SEEDS = True
        else:
            TRAIN_SIZE = int(a)
SUFFIX = ("" if TRAIN_SIZE is None else f"-n{TRAIN_SIZE}") + ("-shared" if SHARED_SEEDS else "")
TARGET = 0.02          # stop when the training cross-entropy reaches this value
MAX_STEPS = 8000
SIG_ABS, SIG_AD, DRAWS = 0.05, 0.5, 32
FACTORS = ("lr", "batch", "width", "wd")


# ---------------------------------------------------------------------------------------
# training to a target loss
# ---------------------------------------------------------------------------------------
def full_ce(model, X, y):
    with torch.no_grad():
        return F.cross_entropy(model(X), y).item()


def train_to_target(data, width, lr, bs, wd, seed):
    Xtr, ytr = data[0], data[1]
    torch.manual_seed(50_000 + seed)                 # labkit's convention for init seeds
    model = lk.MLP(Xtr.shape[1], 10, width, 2)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    n = Xtr.shape[0]
    g = torch.Generator().manual_seed(60_000 + seed)  # labkit's convention for order seeds
    every_far = max(1, 64 // bs)     # check the loss about every 64 examples, and at every step near the target
    steps, ce = 0, full_ce(model, Xtr, ytr)
    perm, pos = torch.randperm(n, generator=g), 0
    while steps < MAX_STEPS:
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g), 0
        idx = perm[pos:pos + bs]
        pos += bs
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps += 1
        if ce < 2 * TARGET or steps % every_far == 0:
            ce = full_ce(model, Xtr, ytr)
            if not math.isfinite(ce):
                break
            if ce <= TARGET:
                return model, ce, steps, True
    return model, full_ce(model, Xtr, ytr), steps, False


# ---------------------------------------------------------------------------------------
# sharpness
# ---------------------------------------------------------------------------------------
def flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def hessian_operator(model, X, y):
    """Return (n_params, hvp) for the Hessian of the mean cross-entropy on (X, y)."""
    params = list(model.parameters())
    loss = F.cross_entropy(model(X), y)
    grads = flat(torch.autograd.grad(loss, params, create_graph=True))

    def hvp(v):
        return flat(torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)).detach()
    return grads.numel(), hvp


def power_iteration(n, hvp, iters=300, tol=1e-6, seed=0):
    """Top eigenvalue by power iteration; if the dominant eigenvalue is negative, shift."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=gen)
    v /= v.norm()
    lam, it = 0.0, 0
    for it in range(1, iters + 1):
        hv = hvp(v)
        new = torch.dot(v, hv).item()
        v = hv / hv.norm()
        if it > 1 and abs(new - lam) <= tol * abs(new):
            lam = new
            break
        lam = new
    if lam < 0:                                       # dominant eigenvalue negative: shift
        shift = -lam
        mu, it2 = power_iteration(n, lambda u: hvp(u) + shift * u, iters, tol, seed + 1)
        return mu - shift, it + it2
    return lam, it


def lanczos_top(n, hvp):
    from scipy.sparse.linalg import LinearOperator, eigsh
    op = LinearOperator((n, n), matvec=lambda x: hvp(torch.from_numpy(np.asarray(x, dtype=np.float32).ravel())).double().numpy(),
                        dtype=np.float64)
    return float(eigsh(op, k=1, which="LA", tol=1e-6, return_eigenvectors=False)[0])


def perturbation_sharpness(model, X, y, sigma, adaptive, draws=DRAWS, seed=0):
    """Mean increase of the training loss under Gaussian perturbations (common random numbers)."""
    params = list(model.parameters())
    base = [p.detach().clone() for p in params]
    gen = torch.Generator().manual_seed(90_000 + seed)
    out = []
    with torch.no_grad():
        L0 = F.cross_entropy(model(X), y).item()
        for _ in range(draws):
            for p, b in zip(params, base):
                z = torch.randn(b.shape, generator=gen)
                p.copy_(b + sigma * (b.abs() if adaptive else 1.0) * z)
            out.append(F.cross_entropy(model(X), y).item() - L0)
        for p, b in zip(params, base):
            p.copy_(b)
    return float(np.mean(out))


def errors(model, X, y):
    with torch.no_grad():
        return 100.0 * (model(X).argmax(1) != y).float().mean().item()


# ---------------------------------------------------------------------------------------
# rank statistics
# ---------------------------------------------------------------------------------------
def tau_a(x, y):
    """Kendall's tau as in Jiang et al. (2020): mean of sign products over pairs; ties count 0."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 2:
        return float("nan")
    sx = np.sign(x[:, None] - x[None, :])
    sy = np.sign(y[:, None] - y[None, :])
    return float((sx * sy)[np.triu_indices(n, 1)].mean())


def mean_ci(values):
    m, h = lk.t_interval(values)
    return m, h


def stratified_tau(rows, measure, outcome, keys):
    """Within-stratum tau for strata defined by equal values of `keys`; returns the list."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    return [tau_a([r[measure] for r in g], [r[outcome] for r in g]) for g in groups.values() if len(g) >= 2]


def conditional_tau(rows, measure, outcome, keys):
    """Mean over strata (equal values of `keys`) of the within-stratum tau; keys=() is pooled."""
    return float(np.mean(stratified_tau(rows, measure, outcome, list(keys)))) if keys else \
        tau_a([r[measure] for r in rows], [r[outcome] for r in rows])


def jackknife_hw(rows, measure, outcome, keys):
    """Half-width of a 95% jackknife interval, deleting one configuration (both seeds) at a time."""
    cfgs = sorted({r["cfg"] for r in rows})
    reps = np.array([conditional_tau([r for r in rows if r["cfg"] != c], measure, outcome, keys) for c in cfgs])
    n = len(cfgs)
    se = math.sqrt((n - 1) / n * np.sum((reps - reps.mean()) ** 2))
    return 1.96 * se


def analyse(rows, measure, outcome="gap", label=None):
    """Pooled tau, tau given width, tau given the configuration (replicate pairs), and the
    granulated coefficients psi_i and Psi of Jiang et al. (2020)."""
    label = label or measure
    out = {}
    for name, keys in (("pooled", ()), ("width", ("width",))):
        out[name] = conditional_tau(rows, measure, outcome, keys)
        out[name + "_hw"] = jackknife_hw(rows, measure, outcome, keys)
    out["config"], out["config_hw"] = mean_ci(stratified_tau(rows, measure, outcome, ["cfg"]))
    psis = []
    for f in FACTORS:
        others = [k for k in FACTORS if k != f] + ["seed"]
        m, h = mean_ci(stratified_tau(rows, measure, outcome, others))
        out["psi_" + f], out["psi_" + f + "_hw"] = m, h
        psis.append(m)
    out["Psi"] = float(np.mean(psis))
    # pair decomposition of the pooled tau (Simpson's decomposition, part (ii)): pairs within and between width strata
    w_arr = np.array([r["width"] for r in rows])
    x = np.array([r[measure] for r in rows], float)
    y = np.array([r[outcome] for r in rows], float)
    prod = np.sign(x[:, None] - x[None, :]) * np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(len(rows), 1)
    same = (w_arr[:, None] == w_arr[None, :])[iu]
    out["w_within"] = float(same.mean())
    out["tau_within_pairs"] = float(prod[iu][same].mean())
    out["tau_between_pairs"] = float(prod[iu][~same].mean())
    print(f"{label:>10s}: pooled {out['pooled']:+.3f}±{out['pooled_hw']:.3f}  "
          f"given width {out['width']:+.3f}±{out['width_hw']:.3f}  "
          f"given configuration {out['config']:+.3f}±{out['config_hw']:.3f}\n            "
          + "  ".join(f"psi_{f} {out['psi_'+f]:+.3f}±{out['psi_'+f+'_hw']:.3f}" for f in FACTORS)
          + f"  Psi {out['Psi']:+.3f}\n            pairs within width strata {out['w_within']:.3f}: "
          f"tau within {out['tau_within_pairs']:+.3f}, between {out['tau_between_pairs']:+.3f}", flush=True)
    return out


# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    t_start = time.time()
    data = lk.digits(split=0, test_frac=0.7, train_size=TRAIN_SIZE)
    Xtr, ytr, Xte, yte = data
    print(f"digits split 0: {len(ytr)} training and {len(yte)} test images", flush=True)
    rows, checks = [], []
    configs = list(itertools.product(LRS, BATCHES, WIDTHS, WDS))
    for ci, (lr, bs, width, wd) in enumerate(configs):
        for seed in SEEDS:
            # every run has its own seeds (init = order = run seed), so that runs of different
            # configurations are independent and the configurations are independent units
            rs = seed if SHARED_SEEDS else len(SEEDS) * ci + seed
            t0 = time.time()
            model, ce, steps, ok = train_to_target(data, width, lr, bs, wd, rs)
            t1 = time.time()
            n, hvp = hessian_operator(model, Xtr, ytr)
            lam, its = power_iteration(n, hvp, seed=rs)
            if (len(rows) % 27) == 0:
                checks.append((lam, lanczos_top(n, hvp)))
            tr, te = errors(model, Xtr, ytr), errors(model, Xte, yte)
            row = dict(cfg=ci, lr=lr, batch=bs, width=width, wd=wd, seed=seed, run_seed=rs, ok=int(ok), steps=steps,
                       ce=ce, train_err=tr, test_err=te, gap=te - tr, lam=lam, its=its,
                       s_abs=perturbation_sharpness(model, Xtr, ytr, SIG_ABS, False, seed=rs),
                       s_ad=perturbation_sharpness(model, Xtr, ytr, SIG_AD, True, seed=rs),
                       norm=math.sqrt(sum((p.detach() ** 2).sum().item() for p in model.parameters())),
                       params=lk.n_params(model), t_train=t1 - t0, t_all=time.time() - t0)
            rows.append(row)
    print(f"{len(rows)} runs in {time.time()-t_start:.0f} s; {sum(r['ok'] for r in rows)} reached the target; "
          f"steps {min(r['steps'] for r in rows)}-{max(r['steps'] for r in rows)}; training losses "
          f"{min(r['ce'] for r in rows):.4f}-{max(r['ce'] for r in rows):.4f}; power iteration used at most "
          f"{max(r['its'] for r in rows)} products", flush=True)
    rel = [abs(a - b) / abs(b) for a, b in checks]
    print(f"power iteration vs Lanczos on {len(checks)} runs: max relative difference {max(rel):.2e}", flush=True)
    for k in ("ce", "train_err", "test_err", "gap", "lam", "s_abs", "s_ad"):
        v = np.array([r[k] for r in rows])
        print(f"  {k:9s} min {v.min():.4g} median {np.median(v):.4g} max {v.max():.4g}")

    # replicate (positive control) and random (negative control) measures
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["cfg"], {})[r["seed"]] = r
    rng = np.random.default_rng(12345)
    for r in rows:
        r["replicate"] = by_cfg[r["cfg"]][1 - r["seed"]]["gap"]
        r["random"] = rng.normal()
        r["noise"] = r["lr"] / r["batch"]
        r["narrowness"] = 1.0 / r["width"]

    # how reproducible is the gap? one-way random effects over the 81 configurations
    g = np.array([[by_cfg[c][s]["gap"] for s in SEEDS] for c in sorted(by_cfg)])
    ms_a = len(SEEDS) * np.var(g.mean(1), ddof=1)
    ms_e = np.mean(np.var(g, axis=1, ddof=1))
    sb2 = (ms_a - ms_e) / len(SEEDS)
    print(f"gap: between-configuration sd {math.sqrt(max(sb2, 0)):.3f}, seed sd {math.sqrt(ms_e):.3f}, "
          f"intraclass correlation {sb2 / (sb2 + ms_e):.3f}", flush=True)

    print("Kendall tau with the generalization gap (95% jackknife intervals over configurations for pooled and "
          "given width; 95% t-intervals over strata for given configuration and psi):", flush=True)
    results = {m: analyse(rows, m) for m in ("lam", "s_abs", "s_ad", "norm", "narrowness", "noise", "replicate", "random")}
    print("Table rows (pooled, psi for lr, batch, width, weight decay, given width; then Psi and given configuration):")
    fmt = lambda m, h: f"\\pmval{{{m:.2f}}}{{{h:.2f}}}" if h > 0 else f"${m:.2f}$"
    for m in ("lam", "s_abs", "s_ad", "norm", "narrowness", "replicate", "random"):
        o = results[m]
        cells = [fmt(o["pooled"], o["pooled_hw"])] + [fmt(o["psi_" + f], o["psi_" + f + "_hw"]) for f in FACTORS] \
            + [fmt(o["width"], o["width_hw"])]
        print(f"  {m:10s} & " + " & ".join(cells) + f"   | Psi {o['Psi']:+.2f}  given configuration {o['config']:+.2f}±{o['config_hw']:.2f}")

    # per-factor means of the measures and of the gap
    for f, levels in zip(FACTORS, (LRS, BATCHES, WIDTHS, WDS)):
        for lev in levels:
            sub = [r for r in rows if r[f] == lev]
            print(f"  {f}={lev:<6g}: gap {np.mean([r['gap'] for r in sub]):5.2f}  lam {np.mean([r['lam'] for r in sub]):6.3f}"
                  f"  s_abs {np.mean([r['s_abs'] for r in sub]):.4f}  s_ad {np.mean([r['s_ad'] for r in sub]):.4f}"
                  f"  norm {np.mean([r['norm'] for r in sub]):6.2f}  steps {np.mean([r['steps'] for r in sub]):7.1f}")

    cols = ["cfg", "lr", "batch", "width", "wd", "seed", "run_seed", "ok", "steps", "ce", "train_err", "test_err", "gap",
            "lam", "s_abs", "s_ad", "norm"]
    lk.write_dat(f"figures/data/10-sharpness-sweep-runs{SUFFIX}.dat", {k: [r[k] for r in rows] for k in cols})
    # width strata: means with 95% t-intervals over the 54 runs of each width
    strat = {"width": [], "gap": [], "gap_hw": []}
    for k in ("lam", "s_ad"):
        strat[k], strat[k + "_hw"] = [], []
    for w in WIDTHS:
        sub = [r for r in rows if r["width"] == w]
        strat["width"].append(w)
        for k in ("gap", "lam", "s_ad"):
            m, h = lk.t_interval([r[k] for r in sub])
            strat[k].append(m)
            strat[k + "_hw"].append(h)
        print(f"  width {w}: gap {strat['gap'][-1]:.2f}±{strat['gap_hw'][-1]:.2f}  lam {strat['lam'][-1]:.3f}±{strat['lam_hw'][-1]:.3f}"
              f"  s_ad {strat['s_ad'][-1]:.4f}±{strat['s_ad_hw'][-1]:.4f}")
    lk.write_dat(f"figures/data/10-sharpness-sweep-width{SUFFIX}.dat", strat)
    print(f"total {time.time()-t_start:.0f} s", flush=True)
