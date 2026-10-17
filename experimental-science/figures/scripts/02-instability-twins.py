"""Lab 2.2 of Chapter 2 (Instability): twin runs.

Run from the volume root:

    python3 figures/scripts/02-instability-twins.py

A twin of a run repeats it with the same seeds and one tiny change:

  repeat   nothing changed (checks bitwise repeatability),
  ulp      one randomly chosen initial weight moved by one unit in the last place
           (torch.nextafter, the smallest change binary32 allows),
  reorder  the examples inside every minibatch presented in another order (the mean
           gradient is mathematically the same; only the floating-point sums change),
  order    a different data-order seed (a new draw of the SGD noise, same initialization),

and an independent run (new initialization and order seeds) for reference. The network is
the digits MLP of labkit (two hidden layers of 64 ReLU units, 8970 parameters), trained by
SGD with momentum 0.9, learning rate 0.1, batch size 64, for 50 epochs (950 steps), in two
regimes that differ only in the labels: clean, and with 20% of the training labels replaced
by random labels (labkit's label_noise=0.2, noise seed 0). Ten pairs per regime.

Part A records the parameter distance between each run and its twins every 5 steps.
Part B (noisy regime) perturbs the initial weights in a random direction by relative sizes
1e-7 ... 1e-2 and records the first step at which the distance exceeds 1.
Part C (noisy regime) records when each base run has fitted half of its mislabeled
training examples.

Writes figures/data/02-instability-twins-dist.dat, -sizes.dat, -memo.dat and prints the
numbers quoted in the chapter. Runtime: about one minute.
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk  # noqa: E402  (two threads, deterministic algorithms)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

T0 = time.time()
CFG = lk.Config(width=64, depth=2, optimizer="momentum", lr=0.1, momentum=0.9,
                batch_size=64, epochs=50)
EVERY = 5            # snapshot interval in steps
PAIRS = 10
THRESH = 1.0         # "diverged": parameter distance above this
SIZES = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


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
    return t, k


def random_direction(model, rel, seed):
    """Add a perturbation of norm rel * ||theta_0|| in a uniformly random direction."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        norm = flat(model).norm()
        d = [torch.randn(p.shape, generator=g) for p in model.parameters()]
        dn = torch.sqrt(sum((x ** 2).sum() for x in d))
        for p, x in zip(model.parameters(), d):
            p.add_(rel * norm * x / dn)


def train(cfg, data, init, order, perturb=None, reorder=None, every=None):
    """One run, exactly as labkit.run (same seeds, same arithmetic), with three hooks:
    perturb(model) after initialization, a seed for permuting the examples inside every
    minibatch, and a snapshot of the parameter vector every `every` steps (and at step 0)."""
    Xtr, ytr, Xte, yte = data
    torch.manual_seed(50_000 + init)
    model = lk.MLP(Xtr.shape[1], 10, cfg.width, cfg.depth, cfg.activation, cfg.batchnorm, cfg.dropout)
    if perturb is not None:
        perturb(model)
    opt = lk.make_optimizer(cfg, model.parameters())
    n = Xtr.shape[0]
    bs = min(cfg.batch_size, n)
    total = cfg.steps if cfg.steps is not None else int(round(cfg.epochs * max(1, n // bs)))
    g_order = torch.Generator().manual_seed(60_000 + order)
    g_in = torch.Generator().manual_seed(90_000 + reorder) if reorder is not None else None
    torch.manual_seed(80_000)
    snaps = [flat(model).clone()] if every else None
    perm, pos = torch.randperm(n, generator=g_order), 0
    for step in range(total):
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g_order), 0
        idx = perm[pos:pos + bs]
        pos += bs
        if g_in is not None:
            idx = idx[torch.randperm(len(idx), generator=g_in)]
        for group in opt.param_groups:
            group["lr"] = cfg.lr * lk.lr_factor(cfg, step, total)
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if every and (step + 1) % every == 0:
            snaps.append(flat(model).clone())
    return model, (torch.stack(snaps) if every else None), total


def load_snapshot(model, vec):
    with torch.no_grad():
        off = 0
        for p in model.parameters():
            m = p.numel()
            p.copy_(vec[off:off + m].view_as(p))
            off += m


def first_above(dist, thresh):
    hit = np.nonzero(dist > thresh)[0]
    return int(hit[0]) * EVERY if len(hit) else np.nan


# the training function reproduces labkit.run bit for bit
_d = lk.load(CFG)
_ref = lk.run(CFG, init=3, order=4, data=_d, return_model=True)
_mine, _, _ = train(CFG, _d, 3, 4)
assert torch.equal(flat(_ref["model"]), flat(_mine)), "train() must reproduce labkit.run"
print("train() reproduces labkit.run bit for bit: True")

regimes = {"c": lk.digits(label_noise=0.0), "n": lk.digits(label_noise=0.2, noise=0)}
names = {"c": "clean labels", "n": "20% noisy labels"}
cols, steps = {}, None
summary = {}
memo_rows = {"pair": [], "t_memo50": [], "t_div_ulp": [], "t_div_reorder": [], "memo_at_div": [],
             "memo_final": []}
size_rows = {"pair": [], "rel": [], "d0": [], "t_div": []}
base_snaps_noisy = {}

for r, data in regimes.items():
    Xtr, ytr, Xte, yte = data
    dis = lambda a, b: (a(Xte).argmax(1) != b(Xte).argmax(1)).float().mean().item()
    err = lambda m: (m(Xte).argmax(1) != yte).float().mean().item()
    rng = np.random.default_rng(2024)          # the same weights in both regimes
    stats = {k: [] for k in ("repeat_max", "ulp_final", "reo_final", "ord_final", "dis_ulp", "dis_reo",
                             "dis_ord", "dis_ind", "err", "ulp_identical", "ulp_where", "div_ulp", "div_reo")}
    curves = {"ulp": [], "reo": [], "ord": []}
    for i in range(PAIRS):
        base, A, total = train(CFG, data, i, i, every=EVERY)
        rep, R_, _ = train(CFG, data, i, i, every=EVERY)
        where = []
        ulp, B, _ = train(CFG, data, i, i, perturb=lambda m: where.append(one_ulp(m, rng)), every=EVERY)
        reo, C, _ = train(CFG, data, i, i, reorder=i, every=EVERY)
        ordt, D, _ = train(CFG, data, i, 500 + i, every=EVERY)
        ind, _, _ = train(CFG, data, 500 + i, 500 + i)
        steps = np.arange(A.shape[0]) * EVERY
        dB, dC, dD = [(S - A).norm(dim=1).numpy() for S in (B, C, D)]
        curves["ulp"].append(dB)
        curves["reo"].append(dC)
        curves["ord"].append(dD)
        with torch.no_grad():
            stats["repeat_max"].append((R_ - A).abs().max().item())
            stats["dis_ulp"].append(dis(base, ulp))
            stats["dis_reo"].append(dis(base, reo))
            stats["dis_ord"].append(dis(base, ordt))
            stats["dis_ind"].append(dis(base, ind))
            stats["err"].append(err(base))
        stats["ulp_final"].append(dB[-1])
        stats["reo_final"].append(dC[-1])
        stats["ord_final"].append(dD[-1])
        stats["ulp_identical"].append(bool(dB[-1] == 0.0))
        stats["ulp_where"].append(where[0])
        stats["div_ulp"].append(first_above(dB, THRESH))
        stats["div_reo"].append(first_above(dC, THRESH))
        if r == "n":
            base_snaps_noisy[i] = A
    for t in ("ulp", "reo", "ord"):
        M = np.array(curves[t])
        if t == "ulp":
            for i in range(PAIRS):
                cols[f"{r}_ulp{i}"] = [v if v > 0 else float("nan") for v in M[i]]
        with np.errstate(divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-zero columns at step 0
            med = np.exp(np.nanmedian(np.log(np.where(M > 0, M, np.nan)), axis=0))
        cols[f"{r}_{t}_med"] = [v if np.isfinite(v) and v > 0 else float("nan") for v in med]
    summary[r] = stats

cols = {"step": list(steps), **cols}
lk.write_dat("figures/data/02-instability-twins-dist.dat", cols)

for r in regimes:
    s = summary[r]
    print(f"\n== Part A, {names[r]}: {PAIRS} pairs, {total} steps")
    print(f"   repeat with identical seeds: largest |difference| in any weight at any step = {max(s['repeat_max'])}")
    print(f"   test error of the base runs: mean {np.mean(s['err']):.4f}")
    q = lambda v: np.quantile(np.asarray(v, dtype=float), [0.0, 0.5, 1.0])
    print("   final parameter distance (min / median / max):")
    print(f"      one-ulp twin   {q(s['ulp_final'])}   (twins bitwise identical to their base at the end:"
          f" {sum(s['ulp_identical'])})")
    print(f"      reorder twin   {q(s['reo_final'])}")
    print(f"      order twin     {q(s['ord_final'])}")
    print("   test-set disagreement with the base run (mean over pairs):")
    print(f"      one-ulp twin {np.mean(s['dis_ulp']):.4f}  reorder twin {np.mean(s['dis_reo']):.4f}"
          f"  order twin {np.mean(s['dis_ord']):.4f}  independent run {np.mean(s['dis_ind']):.4f}")
    print(f"   step at which the distance first exceeds {THRESH}: one-ulp {s['div_ulp']}")
    print(f"                                               reorder {s['div_reo']}")
    print(f"   perturbed parameter (tensor, index): {s['ulp_where']}")

# distance at a few steps: the rounding floor
dfile = cols
for r in regimes:
    for t in (50, 200):
        k = t // EVERY
        vals = [dfile[f"{r}_ulp{i}"][k] for i in range(PAIRS)]
        print(f"   {names[r]}: one-ulp twins at step {t}: median distance {np.nanmedian(vals):.2e};"
              f" reorder twins (median) {dfile[f'{r}_reo_med'][k]:.2e}")

# growth rates in the noisy regime: quiet phase and burst
quiet, burst = [], []
for i in range(PAIRS):
    d = np.array(dfile[f"n_ulp{i}"], dtype=float)
    if not np.isfinite(d[-1]) or d[-1] < THRESH:
        continue
    k4 = np.nonzero(d > 1e-4)[0][0]
    k1 = np.nonzero(d > 1.0)[0][0]
    k50 = 50 // EVERY
    if k4 > k50:
        quiet.append(np.log(d[k4] / d[k50]) / ((k4 - k50) * EVERY))
    burst.append(np.log(d[k1] / d[k4]) / max(1, (k1 - k4) * EVERY))
print(f"\n   noisy regime, one-ulp twins: growth rate of ln(distance) per step, median over pairs:"
      f" quiet phase (step 50 to distance 1e-4) {np.median(quiet):.4f},"
      f" burst (1e-4 to 1) {np.median(burst):.3f}  [{len(burst)} pairs]")

# ---------------------------------------------------------------------------------------
# Part B: divergence step against the size of the initial perturbation (noisy regime)
# ---------------------------------------------------------------------------------------
data = regimes["n"]
for i in range(PAIRS):
    A = base_snaps_noisy[i]
    for rel in SIZES:
        _, B, _ = train(CFG, data, i, i, perturb=lambda m: random_direction(m, rel, 7000 + i), every=EVERY)
        d = (B - A).norm(dim=1).numpy()
        size_rows["pair"].append(i)
        size_rows["rel"].append(rel)
        size_rows["d0"].append(float(d[0]))
        size_rows["t_div"].append(first_above(d, THRESH))
    d = np.array(dfile[f"n_ulp{i}"], dtype=float)
    size_rows["pair"].append(i)
    size_rows["rel"].append(0.0)                     # 0 marks the one-ulp twin
    size_rows["d0"].append(float(d[0]) if np.isfinite(d[0]) else float("nan"))
    size_rows["t_div"].append(summary["n"]["div_ulp"][i])
lk.write_dat("figures/data/02-instability-twins-sizes.dat", size_rows)
print("\n== Part B, noisy regime: first step with distance > 1, by relative size of the initial perturbation")
rel_arr, t_arr, d0_arr = (np.array(size_rows[k], dtype=float) for k in ("rel", "t_div", "d0"))
for rel in [0.0] + SIZES:
    m = rel_arr == rel
    label = "one ulp" if rel == 0 else f"{rel:.0e}"
    print(f"   {label:8s} initial distance median {np.nanmedian(d0_arr[m]):.1e}:"
          f" divergence step median {np.nanmedian(t_arr[m]):.0f}, range {np.nanmin(t_arr[m]):.0f}-{np.nanmax(t_arr[m]):.0f}")
big = (rel_arr >= 1e-5)
slope, icept = np.polyfit(np.log(d0_arr[big]), t_arr[big], 1)
print(f"   least-squares slope of the divergence step on ln(initial distance), sizes 1e-5..1e-2:"
      f" {slope:.1f} steps per unit of ln, i.e. an exponent of {-1 / slope:.3f} per step")
# medians with percentile-bootstrap 95% intervals, and the fitted line for the figure
brng = np.random.default_rng(11)
med = {"d0": [], "t": [], "elo": [], "ehi": []}
for rel in [0.0] + SIZES:
    m = (rel_arr == rel) & np.isfinite(t_arr)
    t = t_arr[m]
    reps = np.median(t[brng.integers(0, len(t), size=(10_000, len(t)))], axis=1)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    med["d0"].append(float(np.median(d0_arr[m])))
    med["t"].append(float(np.median(t)))
    med["elo"].append(float(np.median(t) - lo))
    med["ehi"].append(float(hi - np.median(t)))
lk.write_dat("figures/data/02-instability-twins-sizemed.dat", med)
xs = np.array([np.min(d0_arr[big]) / 3, np.max(d0_arr[big]) * 3])
lk.write_dat("figures/data/02-instability-twins-sizefit.dat", {"d0": list(xs), "t": list(icept + slope * np.log(xs))})

# ---------------------------------------------------------------------------------------
# Part C: when does the base run fit its mislabeled examples? (noisy regime)
# ---------------------------------------------------------------------------------------
Xc, yc, _, _ = lk.digits(label_noise=0.0)
Xn, yn, _, _ = regimes["n"]
flipped = yc != yn
probe = lk.MLP(64, 10, CFG.width, CFG.depth)
for i in range(PAIRS):
    A = base_snaps_noisy[i]
    frac = []
    for k in range(A.shape[0]):
        load_snapshot(probe, A[k])
        with torch.no_grad():
            frac.append(((probe(Xn).argmax(1) == yn) & flipped).float().sum().item() / flipped.sum().item())
    frac = np.array(frac)
    memo_rows["pair"].append(i)
    memo_rows["t_memo50"].append(first_above(frac, 0.5))
    memo_rows["t_div_ulp"].append(summary["n"]["div_ulp"][i])
    memo_rows["t_div_reorder"].append(summary["n"]["div_reo"][i])
    tdiv = summary["n"]["div_ulp"][i]
    memo_rows["memo_at_div"].append(float(frac[int(tdiv) // EVERY]) if np.isfinite(tdiv) else float("nan"))
    memo_rows["memo_final"].append(float(frac[-1]))
lk.write_dat("figures/data/02-instability-twins-memo.dat", memo_rows)
print(f"\n== Part C, noisy regime: {flipped.sum().item()} of {len(yn)} training labels changed by the noise")
print(f"   step at which the base run fits half of them: {memo_rows['t_memo50']}"
      f" (median {np.nanmedian(memo_rows['t_memo50']):.0f})")
print(f"   divergence step of the one-ulp twins: {memo_rows['t_div_ulp']}"
      f" (median {np.nanmedian(memo_rows['t_div_ulp']):.0f})")
print(f"   divergence step of the reorder twins: {memo_rows['t_div_reorder']}"
      f" (median {np.nanmedian(memo_rows['t_div_reorder']):.0f})")
mad = np.array(memo_rows["memo_at_div"], dtype=float)
print(f"   fraction of the mislabeled examples fitted by the base run at the one-ulp divergence step:"
      f" median {np.nanmedian(mad):.2f}, range {np.nanmin(mad):.2f}-{np.nanmax(mad):.2f};"
      f" at the end of training: median {np.median(memo_rows['memo_final']):.2f}")
print(f"\nruntime {time.time() - T0:.0f} s")
