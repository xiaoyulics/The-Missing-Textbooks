"""Lab 6.2 of *Deep Learning as an Experimental Science*: attribution along a path.

A 2^4 factorial on the scikit-learn digits that opens the recipe of Lab 6.1 into three
components. Factors, each switched from its old to its new level:
  A  architecture      the two-layer MLP of width 64        -> the small CNN of Lab 6.1
  O  optimizer         SGD, momentum 0.9, lr 0.05, constant -> AdamW, lr 2e-3, weight decay
                       0.05, 50 warm-up steps, cosine decay
  L  label smoothing   0                                    -> 0.1
  D  duration          20 epochs                            -> 60 epochs
The corner (old, old, old, old) is the MLP with the old recipe of Lab 6.1 and the corner
(new, new, new, new) the CNN with its new recipe. Six blocks (splits 20-25); in block b all
16 cells use split b and the seeds init = order = noise = b. For every factor the script
computes its eight simple effects (one for each setting of the other three), its attribution
along each of the 24 orders in which the four changes can be made, its Shapley attribution,
and the factorial effects of the appendix (Factorial contrasts), all with t-intervals over
blocks. The script 06-confounding-noisy.py runs the same analysis with 20% of the training
labels replaced by random labels (Lab 6.3).

Run from the volume root:  python3 figures/scripts/06-confounding-paths.py
Writes figures/data/06-confounding-paths-{cells,simple,shapley,recipes}.dat and prints the
numbers quoted in Chapter 6. Five to thirteen minutes on two threads, depending on the load
of the machine (five to eight CPU-minutes).
"""
import os, sys, time, math, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import replace

NAMES = ["architecture", "optimizer", "label smoothing", "duration"]
SHORT = ["A", "O", "L", "D"]
K = 4


class CNN(nn.Module):
    """The network of Lab 6.1: two 3x3 convolutions (16 and 32 channels), max-pool, linear."""

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


def config(o, l, d):
    if o:
        c = lk.Config(optimizer="adamw", lr=2e-3, weight_decay=0.05, schedule="cosine", warmup=50)
    else:
        c = lk.Config(optimizer="momentum", lr=0.05, momentum=0.9, schedule="constant")
    return replace(c, batch_size=64, label_smoothing=0.1 if l else 0.0, epochs=60 if d else 20)


def train(a, cfg, data, init, order, noise):
    """One run; the same loop as Lab 6.1 (and as labkit.run for the MLP)."""
    Xtr, ytr, Xte, yte = data
    torch.manual_seed(50_000 + init)
    model = CNN() if a else lk.MLP(64, 10, width=64, depth=2)
    opt = lk.make_optimizer(cfg, model.parameters())
    n = Xtr.shape[0]
    bs = min(cfg.batch_size, n)
    total = int(round(cfg.epochs * max(1, n // bs)))
    g_order = torch.Generator().manual_seed(60_000 + order)
    torch.manual_seed(80_000 + noise)
    perm, pos = torch.randperm(n, generator=g_order), 0
    for step in range(total):
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g_order), 0
        idx = perm[pos:pos + bs]
        pos += bs
        for group in opt.param_groups:
            group["lr"] = cfg.lr * lk.lr_factor(cfg, step, total)
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx], label_smoothing=cfg.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    ev = lk.evaluate(model, Xte, yte, False)
    return 1.0 - ev["acc"] if math.isfinite(ev["loss"]) else 0.9


def tint(x):
    return lk.t_interval(np.asarray(x, dtype=float))


def main(label_noise=0.0, blocks=tuple(range(20, 26)), prefix="06-confounding-paths"):
    t0 = time.time()
    blocks = list(blocks)
    print(f"{prefix}: label noise {label_noise}, blocks {blocks}")
    cells = list(itertools.product((0, 1), repeat=K))      # (a, o, l, d)
    E = {c: np.zeros(len(blocks)) for c in cells}           # test error in %
    for bi, blk in enumerate(blocks):
        data = lk.load(lk.Config(label_noise=label_noise), split=blk, noise=blk)
        for c in cells:
            E[c][bi] = 100 * train(c[0], config(*c[1:]), data, blk, blk, blk)
        print(f"block {blk}: done ({time.time() - t0:.0f}s)", flush=True)
    G = {c: -E[c] for c in cells}                           # gain scale: larger is better
    nb = len(blocks)

    # ---- cells and the eight recipes ---------------------------------------------------------
    print(f"\ncell means, test error % (mean +- 95% t-half-width over {nb} blocks):")
    rec_rows = {"o": [], "l": [], "d": [], "mlp": [], "mlphw": [], "cnn": [], "cnnhw": [],
                "gain": [], "gainhw": []}
    for (o, l, d) in itertools.product((0, 1), repeat=3):
        mm, mh = tint(E[(0, o, l, d)])
        cm, ch = tint(E[(1, o, l, d)])
        gm, gh = tint(E[(0, o, l, d)] - E[(1, o, l, d)])
        print(f"  recipe O{o} L{l} D{d}:  MLP {mm:5.2f} +- {mh:.2f}   CNN {cm:5.2f} +- {ch:.2f}   "
              f"CNN gain {gm:5.2f} +- {gh:.2f}")
        for k_, v_ in zip(rec_rows, [o, l, d, mm, mh, cm, ch, gm, gh]):
            rec_rows[k_].append(v_)
    lk.write_dat(f"figures/data/{prefix}-recipes.dat", rec_rows)
    lk.write_dat(f"figures/data/{prefix}-cells.dat",
                 {"a": [c[0] for c in cells], "o": [c[1] for c in cells], "l": [c[2] for c in cells],
                  "d": [c[3] for c in cells], "mean": [E[c].mean() for c in cells],
                  "hw": [tint(E[c])[1] for c in cells]})
    total = G[(1, 1, 1, 1)] - G[(0, 0, 0, 0)]
    tm, th = tint(total)
    print(f"total gain from (MLP, old recipe) to (CNN, new recipe): {tm:.2f} +- {th:.2f}")
    print(f"MLP-new versus CNN-old (gain of the MLP): {tint(G[(0, 1, 1, 1)] - G[(1, 0, 0, 0)])}")

    # ---- factorial effects (toolkit convention: mean over c_S=+1 minus mean over c_S=-1) -----
    def sign(c, S):
        return np.prod([2 * c[j] - 1 for j in S])
    effects = {}
    print("\nfactorial effects on the gain scale (points), mean +- 95% t-half-width over blocks:")
    for r in range(1, K + 1):
        for S in itertools.combinations(range(K), r):
            per_block = sum(sign(c, S) * G[c] for c in cells) * 2 / len(cells)
            effects[S] = per_block
            m, h = tint(per_block)
            flag = "  *" if abs(m) > h else ""
            print(f"  {''.join(SHORT[j] for j in S):5s} {m:6.2f} +- {h:.2f}{flag}")

    # ---- simple effects, path attributions, Shapley values ----------------------------------
    def cell_with(j, val, z):
        c = [0] * K
        c[j] = val
        for i, v in z.items():
            c[i] = v
        return tuple(c)

    simple = {}                           # (j, frozenset of factors already new) -> per-block gain
    for j in range(K):
        others = [i for i in range(K) if i != j]
        for r in range(len(others) + 1):
            for P in itertools.combinations(others, r):
                z = {i: (1 if i in P else 0) for i in others}
                simple[(j, frozenset(P))] = G[cell_with(j, 1, z)] - G[cell_with(j, 0, z)]

    orders = list(itertools.permutations(range(K)))
    path = {}
    for pi in orders:
        path[pi] = {j: simple[(j, frozenset(pi[:t]))] for t, j in enumerate(pi)}
        assert np.allclose(sum(path[pi][j] for j in range(K)), total)
    shap = {j: sum(path[pi][j] for pi in orders) / len(orders) for j in range(K)}
    assert np.allclose(sum(shap.values()), total)
    # Shapley from the factorial effects (odd sets containing j, divided by their size)
    for j in range(K):
        via = sum(effects[S] / len(S) for S in effects if j in S and len(S) % 2 == 1)
        assert np.allclose(via, shap[j])

    print("\nattributions (gain in points); simple effects by the set of factors already switched:")
    srows = {"x": [], "factor": [], "s": [], "gain": [], "hw": [], "mult": []}
    shrows = {"x": [], "shap": [], "shaphw": [], "main": [], "mainhw": [], "first": [], "last": [],
              "lo": [], "hi": []}
    offs = {0: -0.30, 1: -0.10, 2: 0.10, 3: 0.30}
    for j in range(K):
        others = [i for i in range(K) if i != j]
        print(f"  {NAMES[j]}:")
        vals = []
        for r in range(len(others) + 1):
            for q, P in enumerate(itertools.combinations(others, r)):
                v = simple[(j, frozenset(P))]
                m, h = tint(v)
                mult = math.factorial(r) * math.factorial(K - 1 - r)
                vals.append(m)
                print(f"     after {{{','.join(SHORT[i] for i in P)}}}: {m:6.2f} +- {h:.2f}  ({mult} orders)")
                spread = 0.06 * (q - (math.comb(K - 1, r) - 1) / 2)
                srows["x"].append(j + 1 + offs[r] + spread)
                srows["factor"].append(j + 1); srows["s"].append(r)
                srows["gain"].append(m); srows["hw"].append(h); srows["mult"].append(mult)
        sm, sh = tint(shap[j])
        mm, mh = tint(effects[(j,)])
        first = simple[(j, frozenset())].mean()
        last = simple[(j, frozenset(others))].mean()
        npos = sum(1 for pi in orders if path[pi][j].mean() > 0)
        print(f"     Shapley {sm:.2f} +- {sh:.2f}; main effect {mm:.2f} +- {mh:.2f}; first {first:.2f}; "
              f"last {last:.2f}; range over orders [{min(vals):.2f}, {max(vals):.2f}]; "
              f"positive (helps) in {npos} of 24 orders")
        for k_, v_ in zip(shrows, [j + 1, sm, sh, mm, mh, first, last, min(vals), max(vals)]):
            shrows[k_].append(v_)
    lk.write_dat(f"figures/data/{prefix}-simple.dat", srows)
    lk.write_dat(f"figures/data/{prefix}-shapley.dat", shrows)
    print(f"  Shapley values sum to {sum(shap[j].mean() for j in range(K)):.3f} = total {tm:.3f}")

    def show(pi, label):
        parts = ", ".join(f"{SHORT[j]} {path[pi][j].mean():+.2f}" for j in pi)
        print(f"  {label:38s} {parts}")
    print("\nnamed paths (gain credited to each step, in the order of the path):")
    show((1, 2, 3, 0), "recipe first (O, L, D), then A:")
    show((0, 1, 2, 3), "architecture first, then O, L, D:")
    show((3, 1, 2, 0), "D, O, L, then A:")
    show((2, 3, 1, 0), "L, D, O, then A:")
    show((1, 3, 2, 0), "O, D, L, then A:")
    a_vals = sorted(path[pi][0].mean() for pi in orders)
    print(f"  architecture credit over the 24 orders: min {a_vals[0]:.2f}, max {a_vals[-1]:.2f}, "
          f"share of total {100 * a_vals[0] / tm:.0f}% to {100 * a_vals[-1] / tm:.0f}%; "
          f"Shapley share {100 * shap[0].mean() / tm:.0f}%")

    # ---- the tuned comparison: each architecture with its own best of the 8 recipes ----------
    recs = list(itertools.product((0, 1), repeat=3))
    half = nb // 2
    halves = [list(range(0, half)), list(range(half, nb))]
    print("\ntuned comparison (best of the 8 recipes per architecture):")
    for a, nm in ((0, "MLP"), (1, "CNN")):
        full = min(recs, key=lambda r_: E[(a,) + r_].mean())
        print(f"  {nm}: best recipe on all blocks O{full[0]} L{full[1]} D{full[2]}, "
              f"error {E[(a,) + full].mean():.2f}")
    cross = []
    for hsel, heval in (halves, halves[::-1]):
        tuned = []
        for a in (0, 1):
            r_ = min(recs, key=lambda r_: E[(a,) + r_][hsel].mean())
            tuned.append(E[(a,) + r_][heval])
            print(f"  chosen on blocks {[blocks[i] for i in hsel]}: arch {a} recipe {r_}, "
                  f"error on the other blocks {E[(a,) + r_][heval].mean():.2f}")
        cross.append(tuned[0] - tuned[1])
    m, h = tint(np.concatenate(cross))
    print(f"  cross-fitted CNN gain with each architecture's own best recipe: {m:.2f} +- {h:.2f}")
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
