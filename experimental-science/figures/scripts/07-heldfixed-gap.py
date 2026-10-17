"""Chapter 7 (What Was Held Fixed), Lab 7.1: the large-batch gap under four protocols.

A two-layer MLP of width 64 on the scikit-learn digits (split 0: 1258 training images, 539 test
images), plain SGD, batch sizes B = 8 ... 1258 (the whole training set). Every run uses the same
schedule shape: a linear warm-up over the first 5% of its steps, then cosine decay to zero.

  protocol (i)   fixed epochs, fixed learning rate: 10 epochs (T = 10 * floor(1258/B) steps), and
                 the learning rate that protocol (ii) selects at B = 8, for every B;
  protocol (ii)  fixed epochs, learning rate tuned for each B on a grid 0.025 * 2^k, k = 0..8;
  protocol (iii) fixed steps, learning rate tuned for each B: every B gets the 1570 steps that
                 B = 8 gets in 10 epochs (grid 0.05 * 2^k, k = 0..5);
  protocol (iv)  fixed steps, fixed learning rate: 1570 steps and the learning rate of (i).
The four protocols are the cells of a 2 x 2 design: budget (epochs or steps) by learning rate
(held at the small-batch value or tuned for each B).

Selection is honest by two-fold cross-fitting: the 539 test images are split at random into halves
H1 (270) and H2 (269). The learning rate is chosen on H1 with two selection seeds (lowest mean
error, ties broken by mean cross-entropy) and scored on H2 with ten fresh seeds, and vice versa.
A reported error counts, for each fresh seed, the mistakes on H2 under the H1 choice and on H1 under
the H2 choice, out of 539. Seeds: selection init = order = 0, 1; reporting init = order = 100..109.

Prints the numbers quoted in the text and writes figures/data/07-heldfixed-gap-*.dat.
About two minutes on a laptop CPU with two threads (longer on a busy machine).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk
import numpy as np
import torch
import torch.nn.functional as F

BATCHES = [8, 16, 32, 64, 128, 256, 512, 1024, 1258]
EPOCHS = 10
GRID_EPOCHS = [0.025 * 2 ** k for k in range(9)]
GRID_STEPS = [0.05 * 2 ** k for k in range(6)]
SEL_SEEDS = [0, 1]
REP_SEEDS = list(range(100, 110))


def steps_for_epochs(B, n):
    return EPOCHS * max(1, n // min(B, n))


def one_run(B, lr, T, seed, data, halves):
    cfg = lk.Config(optimizer="sgd", lr=lr, steps=T, batch_size=B, schedule="cosine",
                    warmup=max(1, int(round(0.05 * T))))
    r = lk.run(cfg, init=seed, order=seed, data=data, return_model=True)
    model = r["model"]
    model.eval()
    out = {"train_loss": r["train_loss"], "train_err": 1 - r["train_acc"]}
    with torch.no_grad():
        for name, (X, y) in halves.items():
            o = model(X)
            wrong = int((o.argmax(1) != y).sum())
            loss = F.cross_entropy(o, y).item()
            if not np.isfinite(loss):
                wrong, loss = len(y), float("inf")
            out[name + "_wrong"] = wrong
            out[name + "_loss"] = loss
    return out


def select(B, T, grid, data, halves, cache):
    """Selected learning rate on each half: lowest mean error over the selection seeds."""
    chosen = {}
    for sel in ("H1", "H2"):
        best = None
        for lr in grid:
            runs = [cache_run(B, lr, T, s, data, halves, cache) for s in SEL_SEEDS]
            key = (np.mean([u[sel + "_wrong"] for u in runs]), np.mean([u[sel + "_loss"] for u in runs]))
            if best is None or key < best[0]:
                best = (key, lr)
        chosen[sel] = best[1]
    return chosen


def cache_run(B, lr, T, seed, data, halves, cache):
    k = (B, round(lr, 10), T, seed)
    if k not in cache:
        cache[k] = one_run(B, lr, T, seed, data, halves)
    return cache[k]


def report(B, T, chosen, data, halves, cache):
    """Per fresh seed: mistakes on H2 under the H1 choice plus mistakes on H1 under the H2 choice."""
    m = sum(len(y) for _, y in halves.values())
    errs, tl = [], []
    for s in REP_SEEDS:
        a = cache_run(B, chosen["H1"], T, s, data, halves, cache)
        b = cache_run(B, chosen["H2"], T, s, data, halves, cache)
        errs.append(100.0 * (a["H2_wrong"] + b["H1_wrong"]) / m)
        tl.append(0.5 * (a["train_loss"] + b["train_loss"]))
    return np.array(errs), np.array(tl)


def main():
    t0 = time.time()
    data = lk.load(lk.Config())
    Xtr, ytr, Xte, yte = data
    n = Xtr.shape[0]
    perm = np.random.default_rng(7007).permutation(len(yte))
    i1, i2 = perm[:270], perm[270:]
    halves = {"H1": (Xte[i1], yte[i1]), "H2": (Xte[i2], yte[i2])}
    print(f"training images {n}, test images {len(yte)} (halves {len(i1)} and {len(i2)})")
    T_fixed = steps_for_epochs(BATCHES[0], n)
    cache = {}
    results = {}
    # protocol (ii): fixed epochs, tuned learning rate
    chosen_ii = {}
    for B in BATCHES:
        T = steps_for_epochs(B, n)
        chosen_ii[B] = select(B, T, GRID_EPOCHS, data, halves, cache)
        results[("ii", B)] = report(B, T, chosen_ii[B], data, halves, cache) + (chosen_ii[B], T)
    print(f"protocol (ii) done ({time.time() - t0:.0f}s)")
    # protocol (i): fixed epochs, the learning rate selected at the smallest batch
    chosen_i = chosen_ii[BATCHES[0]]
    for B in BATCHES:
        T = steps_for_epochs(B, n)
        results[("i", B)] = report(B, T, chosen_i, data, halves, cache) + (chosen_i, T)
    print(f"protocol (i) done ({time.time() - t0:.0f}s)")
    # protocol (iii): fixed steps, tuned learning rate
    for B in BATCHES:
        ch = select(B, T_fixed, GRID_STEPS, data, halves, cache)
        results[("iii", B)] = report(B, T_fixed, ch, data, halves, cache) + (ch, T_fixed)
        print(f"  (iii) B={B} done ({time.time() - t0:.0f}s)", flush=True)
    # protocol (iv): fixed steps, the learning rate of (i)
    for B in BATCHES:
        results[("iv", B)] = report(B, T_fixed, chosen_i, data, halves, cache) + (chosen_i, T_fixed)
    print(f"protocol (iv) done ({time.time() - t0:.0f}s)")

    # summaries
    print("\nprotocol  B     steps  epochs   lr(H1,H2)        test error % (mean, 95% t half-width)   train loss")
    rows = {p: {"B": [], "mean": [], "hw": [], "lo": [], "hi": [], "lr1": [], "lr2": [], "steps": [], "trainloss": []}
            for p in ("i", "ii", "iii", "iv")}
    runs = {p: {"B": [], "x": [], "err": []} for p in ("i", "ii", "iii", "iv")}
    jit = {"i": 0.92, "ii": 0.97, "iii": 1.03, "iv": 1.08}
    for p in ("i", "ii", "iii", "iv"):
        for B in BATCHES:
            errs, tl, ch, T = results[(p, B)]
            mean, hw = lk.t_interval(errs)
            rows[p]["B"].append(B); rows[p]["mean"].append(mean); rows[p]["hw"].append(hw)
            rows[p]["lo"].append(mean - hw); rows[p]["hi"].append(mean + hw)
            rows[p]["lr1"].append(ch["H1"]); rows[p]["lr2"].append(ch["H2"])
            rows[p]["steps"].append(T); rows[p]["trainloss"].append(float(np.mean(tl)))
            for e in errs:
                runs[p]["B"].append(B); runs[p]["x"].append(B * jit[p]); runs[p]["err"].append(e)
            print(f"  ({p:>3}) {B:5d} {T:6d} {T * min(B, n) / n:7.1f}   ({ch['H1']:.3g}, {ch['H2']:.3g})"
                  f"   {mean:6.2f} +- {hw:4.2f}   min {errs.min():5.2f} max {errs.max():5.2f}   {np.mean(tl):.4f}")
        lk.write_dat(f"figures/data/07-heldfixed-gap-{p}.dat", rows[p])
        lk.write_dat(f"figures/data/07-heldfixed-gap-runs-{p}.dat", runs[p])

    # paired differences against the smallest batch, per protocol
    print("\npaired differences in test error against B = 8 (percentage points, mean and 95% t half-width):")
    for p in ("i", "ii", "iii", "iv"):
        base = results[(p, BATCHES[0])][0]
        txt = []
        for B in BATCHES[1:]:
            dmean, dhw = lk.t_interval(results[(p, B)][0] - base)
            txt.append(f"{B}: {dmean:+.2f}+-{dhw:.2f}")
        print(f"  ({p}) " + "; ".join(txt))
    # protocol (iii) against (ii) at the largest batches
    for B in (512, 1024, 1258):
        dmean, dhw = lk.t_interval(results[("ii", B)][0] - results[("iii", B)][0])
        print(f"  (ii) minus (iii) at B={B}: {dmean:+.2f} +- {dhw:.2f}")
    # the 2 x 2 decomposition of the gap at a large batch, relative to protocol (i)
    for B in (128, 1258):
        e = {p: results[(p, B)][0].mean() for p in ("i", "ii", "iii", "iv")}
        print(f"  B={B}: (i) {e['i']:.2f}, (ii) {e['ii']:.2f}, (iii) {e['iii']:.2f}, (iv) {e['iv']:.2f}; "
              f"steps alone remove {e['i'] - e['iv']:.2f}, learning rate alone {e['i'] - e['ii']:.2f}, "
              f"interaction {e['i'] - e['ii'] - e['iv'] + e['iii']:.2f} points")
    pooled = np.concatenate([results[("iii", B)][0] for B in BATCHES])
    print(f"protocol (iii), all batch sizes pooled: {pooled.mean():.2f}% (range of batch-size means "
          f"{min(rows['iii']['mean']):.2f} to {max(rows['iii']['mean']):.2f})")
    print(f"distinct training runs: {len(cache)}; total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
