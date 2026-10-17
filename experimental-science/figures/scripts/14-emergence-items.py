"""Lab 14.2: the same curves measured with m = 50, 500 and 5,000 items.

The networks of Lab 14.1 (Chapter 1's recipe, widths 1 to 512, runs s = 0..19 with seed s
for initialization and data order, split 0) are scored on a fixed benchmark of 5,000 items.
An item is a sequence of 10 test digits drawn uniformly with replacement from the 539 test
digits (generator seed 1414); the benchmarks of 50 and 500 items are the first 50 and 500.
Given a network, the items are independent, and each is an exact match with probability
p^10, with p the network's accuracy on the 539 digits; the per-digit log-likelihood is
averaged over the 10 m digit slots of the m items.

A model family is one network per width; run s gives family s. The figure shows family 0;
the printed statistics cover all 20 families.

Run from the volume root: python3 figures/scripts/14-emergence-items.py  (about 60 s)
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
import labkit as lk

WIDTHS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512]
SEEDS = range(20)
MS = (50, 500, 5000)
L = 10
OUT = "figures/data/14-emergence-items"


def per_item(model, X, y):
    model.eval()
    with torch.no_grad():
        lp = F.log_softmax(model(X), 1)
    model.train()
    return (lp.argmax(1) == y).numpy(), lp[torch.arange(len(y)), y].double().numpy()


def clopper_pearson(x, m, level=0.95):
    a = 1 - level
    lo = 0.0 if x == 0 else stats.beta.ppf(a / 2, x, m - x + 1)
    hi = 1.0 if x == m else stats.beta.ppf(1 - a / 2, x + 1, m - x)
    return lo, hi


def main():
    t0 = time.time()
    data = lk.load(lk.Config())
    Xte, yte = data[2], data[3]
    n_test = len(yte)
    items = np.random.default_rng(1414).integers(0, n_test, size=(max(MS), L))

    res = {}   # (w, s) -> dict
    for w in WIDTHS:
        for s in SEEDS:
            cfg = lk.Config(width=w, optimizer="momentum", lr=0.05, batch_size=64, epochs=20)
            r = lk.run(cfg, init=s, order=s, data=data, return_model=True)
            corr, ll = per_item(r["model"], Xte, yte)
            exact = corr[items].all(axis=1)                    # exact match of each item
            d = dict(N=r["params"], p=corr.mean(), ce_pool=-ll.mean())
            for m in MS:
                slots = ll[items[:m]].ravel()
                d[f"x{m}"] = int(exact[:m].sum())
                d[f"ce{m}"], d[f"ce{m}_hw"] = lk.t_interval(-slots)
                d[f"acc{m}"] = corr[items[:m]].mean()
            res[(w, s)] = d
    print(f"trained and scored {len(res)} networks in {time.time() - t0:.1f} s")

    # ---- family 0: the figure -------------------------------------------------------
    fam = [res[(w, 0)] for w in WIDTHS]
    Ns = np.array([d["N"] for d in fam], float)
    true_em = np.array([d["p"] ** L for d in fam])
    shift = {50: 1 / 1.12, 500: 1.0, 5000: 1.12}             # horizontal offsets for legibility
    for m in MS:
        x = np.array([d[f"x{m}"] for d in fam])
        em = x / m
        lo, hi = zip(*[clopper_pearson(k, m) for k in x])
        lo, hi = np.array(lo), np.array(hi)
        pos = x > 0
        lk.write_dat(f"{OUT}-m{m}.dat", {
            "N": Ns[pos], "Nplot": Ns[pos] * shift[m], "em": em[pos],
            "minus": (em - lo)[pos], "plus": (hi - em)[pos]})
        bound = 1 - 0.05 ** (1 / m)                            # one-sided 95% bound for zero counts
        lk.write_dat(f"{OUT}-m{m}-zero.dat", {
            "N": Ns[~pos], "Nplot": Ns[~pos] * shift[m], "bound": np.full((~pos).sum(), bound)})
        lk.write_dat(f"{OUT}-m{m}-ce.dat", {
            "N": Ns, "Nplot": Ns * shift[m], "ce": [d[f"ce{m}"] for d in fam],
            "hw": [d[f"ce{m}_hw"] for d in fam]})
    lk.write_dat(f"{OUT}-pool.dat", {"N": Ns, "em": true_em, "ce": [d["ce_pool"] for d in fam],
                                     "p": [d["p"] for d in fam]})

    print("\nfamily 0: exact-match counts on m items, and the exact value p^10 over all sequences")
    print("width      N        p^10     x/50   x/500  x/5000   CE on 50 items (95% t)   CE pool  eff")
    for w, d, te in zip(WIDTHS, fam, true_em):
        p = d["p"]
        eff = L * p ** (L - 1) * (1 - p) / (1 - p ** L)
        print(f"{w:5d} {d['N']:6d}  {te:10.2e}  {d['x50']:5d}  {d['x500']:5d}  {d['x5000']:6d}   "
              f"{d['ce50']:.3f} +- {d['ce50_hw']:.3f} ({100 * d['ce50_hw'] / d['ce50']:.1f}%)   "
              f"{d['ce_pool']:.3f}  {eff:.2e}")
    for m in MS:
        print(f"  m={m}: zero counts at widths {[w for w, d in zip(WIDTHS, fam) if d[f'x{m}'] == 0]}; "
              f"rule-of-three bound {1 - 0.05 ** (1 / m):.4f} (3/m = {3 / m:.4f})")
    print("family 0: per-digit accuracy over the 500 digit slots of the 50 items (95% Clopper-Pearson)")
    for w, d in zip(WIDTHS[:6], fam[:6]):
        k = int(round(d["acc50"] * 500))
        lo, hi = clopper_pearson(k, 500)
        print(f"  w={w}: {d['acc50']:.3f} ({lo:.3f}-{hi:.3f}); on all 539 digits {d['p']:.3f}")

    # ---- all 20 families -----------------------------------------------------------
    print("\nall 20 families: share of zero counts by width, observed / predicted mean of (1-p^10)^m")
    for w in WIDTHS[:5]:
        line = f"  w={w}:"
        for m in MS:
            obs = np.mean([res[(w, s)][f"x{m}"] == 0 for s in SEEDS])
            pred = np.mean([(1 - res[(w, s)]["p"] ** L) ** m for s in SEEDS])
            line += f"  m={m}: {obs:.2f}/{pred:.2f}"
        print(line)
    print("apparent onset (smallest width with a nonzero count) across the 20 families:")
    for m in MS:
        onset = []
        for s in SEEDS:
            ws = [w for w in WIDTHS if res[(w, s)][f"x{m}"] > 0]
            onset.append(min(ws))
        vals, cnt = np.unique(onset, return_counts=True)
        print(f"  m={m}: " + ", ".join(f"width {v}: {c}" for v, c in zip(vals, cnt)))
    p1 = np.array([res[(1, s)]["p"] for s in SEEDS])
    print(f"width 1: p^10 over families from {np.min(p1 ** L):.1e} to {np.max(p1 ** L):.1e} "
          f"(median {np.median(p1 ** L):.1e}); items for a nonzero count with probability 1/2 "
          f"at the median: {math.log(2) / np.median(p1 ** L):.1e}")
    rel = [100 * res[(w, s)]["ce50_hw"] / res[(w, s)]["ce50"] for w in WIDTHS for s in SEEDS]
    print(f"CE on 50 items: relative half-width of the 95% interval {min(rel):.1f}% to {max(rel):.1f}% "
          f"over all 300 networks")
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
