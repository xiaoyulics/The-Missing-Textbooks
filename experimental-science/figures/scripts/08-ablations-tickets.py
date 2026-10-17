"""Lab 8.1 (Chapter 8, Ablations): lottery tickets on the digits at two learning rates.

A two-layer perceptron (width 128, 26,122 parameters) is trained on the scikit-learn digits
with SGD and momentum 0.9, batch size 64, a cosine schedule and 30 epochs. Its weight
matrices are pruned by magnitude, layer by layer, to 2.8% of the weights of each hidden
layer and 21% of the output layer (3.7% of all weights), either in one shot from the trained
dense network or iteratively, removing 40% of the surviving weights of each hidden layer
and 20% of the output layer in each of seven rounds and resetting the survivors to their
initial values after each round (Frankle and Carbin's procedure).

For each of 20 seeds, at each of two learning rates (0.02 and 0.2, used for the whole
pipeline: the dense training, every pruning round and the final training), and for each
pruning schedule, three arms are trained with the same split, data order and epochs:

  ticket   the pruned mask with the original initial values of its weights,
  reinit   the same mask with a fresh draw from the same initial distribution,
  sham     the mask shuffled at random within each layer (the same number of weights per
           layer, at random positions) with the original initial values.

The design is a 2 x 2 x 2 factorial in initialization (ticket, reinit), learning rate and
schedule, blocked by seed, plus the sham as a control arm. The reinit and sham arms are also
trained at two intermediate densities of the iterative schedule, after rounds 3 and 5 (23% and
9% of all weights). Effects are estimated from
per-seed contrasts with 95% t-intervals over the 20 seeds.

Writes
  figures/data/08-ablations-tickets-iter.dat      20 seeds: errors (%) of every arm, iterative
  figures/data/08-ablations-tickets-oneshot.dat   20 seeds: errors (%) of every arm, one shot
  figures/data/08-ablations-tickets-iter-means.dat     means and 95% t half-widths, iterative
  figures/data/08-ablations-tickets-oneshot-means.dat  the same, one shot
Prints every number quoted in the text.
Run from the volume root:  python3 figures/scripts/08-ablations-tickets.py   (about 3 min)
"""
import os, sys, time, math, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import labkit as lk
from scipy import stats

T0 = time.time()
WIDTH, DEPTH, EPOCHS = 128, 2, 30
RATE, ROUNDS = 0.4, 7          # fraction of surviving hidden weights pruned per round (output: half)
LRS = [0.02, 0.2]              # small and large learning rate
SEEDS = list(range(20))
ARMS = ["ticket", "reinit", "sham"]


# --------------------------------------------------------------------------------------
# Masked training, magnitude pruning, shuffled masks
# --------------------------------------------------------------------------------------
def weights(model):
    return [m.weight for m in model.net if isinstance(m, torch.nn.Linear)]


def build(seed):
    torch.manual_seed(seed)
    return lk.MLP(64, 10, WIDTH, DEPTH)


def train(cfg, model, masks, data, order):
    """Train with the pruned weights held at zero; return (test error, train error) in %."""
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
        out_te, out_tr = model(Xte), model(Xtr)
        if not (torch.isfinite(out_te).all() and torch.isfinite(out_tr).all()):
            return 90.0, 90.0
        te = (out_te.argmax(1) != yte).float().mean().item() * 100
        tr = (out_tr.argmax(1) != ytr).float().mean().item() * 100
    return te, tr


def prune_round(model, masks):
    """Remove RATE of the surviving weights of each hidden layer, RATE/2 of the output layer,
    those of smallest magnitude in the trained model."""
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


def prune_oneshot(model, counts):
    """Keep the `counts[l]` weights of largest magnitude in each layer of the trained model."""
    out = []
    for w, c in zip(weights(model), counts):
        mm = torch.zeros(w.numel())
        mm[torch.argsort(w.detach().abs().flatten(), descending=True)[:c]] = 1
        out.append(mm.view_as(w))
    return out


def shuffled(masks, seed):
    g = torch.Generator().manual_seed(seed)
    return [m.flatten()[torch.randperm(m.numel(), generator=g)].view_as(m) for m in masks]


def ticket_net(m0, theta0):
    m = copy.deepcopy(m0)
    m.load_state_dict(theta0)
    return m


# --------------------------------------------------------------------------------------
# The experiment
# --------------------------------------------------------------------------------------
# err[sched][arm] has shape (seeds, lrs); dense errors are shared by both schedules
err = {s: {a: np.zeros((len(SEEDS), len(LRS))) for a in ARMS} for s in ("iter", "oneshot")}
trn = {s: {a: np.zeros((len(SEEDS), len(LRS))) for a in ARMS} for s in ("iter", "oneshot")}
dense = np.zeros((len(SEEDS), len(LRS)))
counts_final = None
MID = (3, 5)                   # rounds after which the arms are also compared (23% and 9% of the weights)
mid = {r: {a: np.zeros((len(SEEDS), len(LRS))) for a in ARMS} for r in MID}
mid_density = {}
# diagnostics: rank correlation of |final| and |initial| weights of the dense run (first layer),
# RMS of the ticket's initial weights relative to the layer, sign agreement, active hidden units
diag = {k: np.zeros((len(SEEDS), len(LRS))) for k in ("spearman", "rms", "sign", "units_ticket", "units_sham")}


def active_units(masks):
    """Hidden units with at least one incoming and one outgoing weight."""
    tot = 0
    for l in range(len(masks) - 1):
        inc, out = masks[l].sum(1) > 0, masks[l + 1].sum(0) > 0
        tot += int((inc & out).sum().item())
    return tot
for i, s in enumerate(SEEDS):
    for j, lr in enumerate(LRS):
        cfg = lk.Config(width=WIDTH, depth=DEPTH, optimizer="momentum", lr=lr, momentum=0.9,
                        epochs=EPOCHS, batch_size=64, schedule="cosine")
        data = lk.load(cfg, split=s)
        m0 = build(50_000 + s)
        theta0 = copy.deepcopy(m0.state_dict())
        ones = [torch.ones_like(w) for w in weights(m0)]
        model = ticket_net(m0, theta0)
        dense[i, j], _ = train(cfg, model, ones, data, s)
        dense_model = model
        w_init, w_fin = weights(m0)[0].detach().abs().flatten(), weights(dense_model)[0].detach().abs().flatten()
        diag["spearman"][i, j] = stats.spearmanr(w_init.numpy(), w_fin.numpy())[0]
        # iterative magnitude pruning with resetting to the initial values
        masks = ones
        for r in range(ROUNDS):
            masks = prune_round(model, masks)
            model = ticket_net(m0, theta0)
            te, tr = train(cfg, model, masks, data, s)
            if r + 1 in MID:   # intermediate densities of the iterative schedule
                mid[r + 1]["ticket"][i, j] = te
                mid[r + 1]["reinit"][i, j], _ = train(cfg, build(90_000 + s), masks, data, s)
                mid[r + 1]["sham"][i, j], _ = train(cfg, ticket_net(m0, theta0), shuffled(masks, 95_000 + s), data, s)
                mid_density[r + 1] = sum(int(m.sum().item()) for m in masks) / sum(m.numel() for m in masks)
        err["iter"]["ticket"][i, j], trn["iter"]["ticket"][i, j] = te, tr
        rms, agree, kept = [], 0, 0
        for w0, wt, mk in zip(weights(m0), weights(model), masks):
            b = mk.bool()
            rms.append((w0[b].pow(2).mean().sqrt() / w0.pow(2).mean().sqrt()).item())
            agree += int((torch.sign(w0[b]) == torch.sign(wt[b])).sum().item())
            kept += int(b.sum().item())
        diag["rms"][i, j], diag["sign"][i, j] = float(np.mean(rms)), agree / kept
        diag["units_ticket"][i, j] = active_units(masks)
        diag["units_sham"][i, j] = active_units(shuffled(masks, 95_000 + s))
        counts = [int(m.sum().item()) for m in masks]
        counts_final = counts
        os_masks = prune_oneshot(dense_model, counts)
        for sched, mk in (("iter", masks), ("oneshot", os_masks)):
            if sched == "oneshot":
                err[sched]["ticket"][i, j], trn[sched]["ticket"][i, j] = train(cfg, ticket_net(m0, theta0), mk, data, s)
            err[sched]["reinit"][i, j], trn[sched]["reinit"][i, j] = train(cfg, build(90_000 + s), mk, data, s)
            err[sched]["sham"][i, j], trn[sched]["sham"][i, j] = train(cfg, ticket_net(m0, theta0),
                                                                     shuffled(mk, 95_000 + s), data, s)
    if (i + 1) % 5 == 0:
        print(f"  {i + 1} seeds done [{time.time() - T0:.0f}s]", flush=True)

n = len(SEEDS)
sizes = [w.numel() for w in weights(build(0))]
print(f"\nNetwork: 64-{WIDTH}-{WIDTH}-10, {lk.n_params(build(0))} parameters, {sum(sizes)} weights")
print("Final mask: weights kept per layer", counts_final, "of", sizes,
      "densities", [f"{c / z:.4f}" for c, z in zip(counts_final, sizes)],
      f"overall {sum(counts_final) / sum(sizes):.4f}")
tq = lambda x: lk.t_interval(x)

# --------------------------------------------------------------------------------------
# Cell means
# --------------------------------------------------------------------------------------
print("\nTest error (%), mean and 95% t-interval half-width over 20 seeds; training error in brackets")
for j, lr in enumerate(LRS):
    m, h = tq(dense[:, j])
    print(f"lr {lr}: dense {m:.2f} +- {h:.2f}")
    for sched in ("iter", "oneshot"):
        line = f"   {sched:8s}"
        for a in ARMS:
            m, h = tq(err[sched][a][:, j])
            line += f" {a} {m:6.2f} +- {h:5.2f} [{trn[sched][a][:, j].mean():5.2f}]"
        print(line)

# ticket against dense: does the ticket match?
print("\nTicket minus dense (paired), mean and 95% half-width:")
for j, lr in enumerate(LRS):
    for sched in ("iter", "oneshot"):
        m, h = tq(err[sched]["ticket"][:, j] - dense[:, j])
        print(f"  lr {lr} {sched}: {m:+.2f} +- {h:.2f}")

# intermediate densities (iterative pruning)
print("\nIntermediate densities of the iterative schedule (paired effects, ablated minus ticket):")
for r in MID:
    for j, lr in enumerate(LRS):
        line = f"  round {r} (density {mid_density[r]:.3f}), lr {lr}: ticket {tq(mid[r]['ticket'][:, j])[0]:.2f}"
        for a in ("reinit", "sham"):
            m, h = tq(mid[r][a][:, j] - mid[r]["ticket"][:, j])
            line += f"; {a} {m:+.2f} +- {h:.2f}"
        print(line)

# paired ablation effects
print("\nAblation effects (paired, ablated minus ticket), mean, 95% half-width, sd of differences:")
for j, lr in enumerate(LRS):
    for sched in ("iter", "oneshot"):
        for a in ("reinit", "sham"):
            d = err[sched][a][:, j] - err[sched]["ticket"][:, j]
            m, h = tq(d)
            print(f"  lr {lr} {sched:8s} {a:6s}: {m:+7.2f} +- {h:5.2f} (sd {d.std(ddof=1):.2f}; "
                  f"{(d > 0).sum()} of {n} seeds positive)")
        d = err[sched]["sham"][:, j] - err[sched]["reinit"][:, j]
        m, h = tq(d)
        print(f"  lr {lr} {sched:8s} sham-reinit (mask positions at a fresh init): {m:+.2f} +- {h:.2f}")

# --------------------------------------------------------------------------------------
# The 2 x 2 x 2 factorial in initialization (I), learning rate (L), schedule (S)
# coded -1/+1: I = -1 ticket, +1 reinit; L = -1 small, +1 large; S = -1 iterative, +1 one shot
# --------------------------------------------------------------------------------------
cells = {}
for (iI, a) in ((-1, "ticket"), (1, "reinit")):
    for (iL, j) in ((-1, 0), (1, 1)):
        for (iS, sched) in ((-1, "iter"), (1, "oneshot")):
            cells[(iI, iL, iS)] = err[sched][a][:, j]
names = {"I": (0,), "L": (1,), "S": (2,), "IxL": (0, 1), "IxS": (0, 2), "LxS": (1, 2), "IxLxS": (0, 1, 2)}
print("\nFactorial effects (mean of c=+1 cells minus mean of c=-1 cells, per seed; 95% t over seeds):")
fact = {}
for nm, idx in names.items():
    per_seed = np.zeros(n)
    for key, y in cells.items():
        c = np.prod([key[k] for k in idx])
        per_seed += c * y / 4.0
    m, h = tq(per_seed)
    fact[nm] = (m, h)
    print(f"  {nm:6s} {m:+8.2f} +- {h:5.2f}")
# the leave-one-out ablation of initialization at each of the four (L, S) settings: simple effects
print("\nSimple effects of initialization (reinit minus ticket) at each setting; check delta = sum of theta_S:")
for (iL, iS) in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
    simple = (cells[(1, iL, iS)] - cells[(-1, iL, iS)]).mean()
    pred = fact["I"][0] + iL * fact["IxL"][0] + iS * fact["IxS"][0] + iL * iS * fact["IxLxS"][0]
    print(f"  L={iL:+d} S={iS:+d}: simple effect {simple:+8.2f}; main effect plus signed interactions {pred:+8.2f}")

print("\nDiagnostics (means over seeds) at lr", LRS)
for k, v in diag.items():
    print(f"  {k:13s}", "  ".join(f"{x:.3f}" for x in v.mean(0)))

# effects on the log scale (ratio of errors), for the text
print("\nRatios of mean errors (reinit / ticket):")
for j, lr in enumerate(LRS):
    for sched in ("iter", "oneshot"):
        print(f"  lr {lr} {sched}: {err[sched]['reinit'][:, j].mean() / err[sched]['ticket'][:, j].mean():.2f}")

# --------------------------------------------------------------------------------------
# Data for the figure: jittered x positions around the two learning rates (log axis)
# --------------------------------------------------------------------------------------
rng = np.random.default_rng(20261018)
offset = {"ticket": -0.13, "reinit": 0.0, "sham": 0.13}   # in units of log10(lr)
for sched in ("iter", "oneshot"):
    cols = {"seed": SEEDS}
    for j, lr in enumerate(LRS):
        for a in ARMS:
            jit = rng.uniform(-0.035, 0.035, size=n)
            cols[f"x{a}{j + 1}"] = 10 ** (math.log10(lr) + offset[a] + jit)
            cols[f"{a}{j + 1}"] = err[sched][a][:, j]
        cols[f"dense{j + 1}"] = dense[:, j]
    lk.write_dat(f"figures/data/08-ablations-tickets-{sched}.dat", cols)
for sched in ("iter", "oneshot"):
    cols = {"lr": LRS}
    for a in ARMS + ["dense"]:
        ms, hs = [], []
        for j, lr in enumerate(LRS):
            m, h = tq(dense[:, j] if a == "dense" else err[sched][a][:, j])
            ms.append(m); hs.append(h)
        cols[f"x{a}"] = [10 ** (math.log10(lr) + offset.get(a, 0.0)) for lr in LRS]
        cols[a] = ms
        cols[f"h{a}"] = hs
    lk.write_dat(f"figures/data/08-ablations-tickets-{sched}-means.dat", cols)
print(f"\nTotal time {time.time() - T0:.0f}s")
