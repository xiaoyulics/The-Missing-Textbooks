"""Lab 4.1: the same comparison under two tuning protocols.

Momentum SGD against Adam on the digits with 256 distractor features (pure noise, drawn
independently for every image and appended to the 64 pixels), 300 training images, 300
validation images and the remaining 1197 as test set. A two-layer MLP (width 64) is trained
for 100 epochs with batch size 32 and a constant learning rate.

  protocol 2017  each optimizer gets a log grid of 16 learning rates; everything else at its
                 default (momentum 0.9; Adam beta1=0.9, beta2=0.999, eps=1e-8; no weight decay)
  protocol 2019  each optimizer gets 16 random-search trials over all its hyperparameters:
                 momentum: lr, 1-momentum, weight decay;
                 Adam: lr, eps, 1-beta1, 1-beta2, weight decay

In both protocols the trial with the lowest validation error is selected (ties broken by the
validation loss) and retrained with five fresh seeds; the mean test error of those five runs
is the protocol's result. The whole procedure is replicated on R = 10 data splits (each with
its own distractors, search draws and seeds), and differences are paired by split.

Run from the volume root:  python3 figures/scripts/04-budgets-protocols.py
"""
import os, sys, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk
import numpy as np
import torch

D_NOISE, N_TRAIN, N_VAL = 256, 300, 300
EPOCHS, BATCH, WIDTH = 100, 32, 64
BUDGET = 16     # trials per optimizer and protocol
R = 10          # replicates (data splits)
N_FRESH = 5     # fresh seeds for the selected configuration
OPTS = ("momentum", "adam")


def make_data(split, d_noise=D_NOISE):
    """Digits split `split`, with d_noise distractor features; returns train/val and test."""
    Xa, ya, Xb, yb = lk.digits(split=split, test_frac=0.3)
    X, y = torch.cat([Xa, Xb]), torch.cat([ya, yb])          # standardized on the first part
    if d_noise:
        g = torch.Generator().manual_seed(90_000 + split)
        X = torch.cat([X, torch.randn(X.shape[0], d_noise, generator=g)], 1)
    va, tr, te = slice(0, N_VAL), slice(N_VAL, N_VAL + N_TRAIN), slice(N_VAL + N_TRAIN, None)
    return (X[tr], y[tr], X[va], y[va]), (X[te], y[te])


def trial(h, data, test, seed):
    """One run of configuration h: (validation error, validation loss, test error)."""
    cfg = lk.Config(epochs=EPOCHS, batch_size=BATCH, width=WIDTH, **h)
    r = lk.run(cfg, init=seed, order=seed, data=data, return_model=True)
    if not math.isfinite(r["test_loss"]):
        return 0.9, float("inf"), 0.9
    ev = lk.evaluate(r["model"], test[0], test[1], False)
    if not math.isfinite(ev["loss"]):
        return 0.9, float("inf"), 0.9
    return r["test_err"], r["test_loss"], 1.0 - ev["acc"]


def grid_2017(opt):
    if opt == "momentum":
        return [dict(optimizer="momentum", lr=float(x)) for x in np.logspace(-3, 0, BUDGET)]
    return [dict(optimizer="adam", lr=float(x)) for x in np.logspace(-4.5, -1.5, BUDGET)]


def sample_2019(opt, rng):
    """One draw from the full search space (log-uniform in every coordinate).
    Adam's step is about lr when eps is small and about lr*|m|/eps when eps is large, so its
    learning rate is drawn relative to max(eps, 1e-2), as Choi et al. drew lr/eps."""
    if opt == "momentum":
        return dict(optimizer="momentum", lr=10 ** rng.uniform(-3, 0),
                    momentum=1 - 10 ** rng.uniform(-2, 0), weight_decay=10 ** rng.uniform(-5, -1))
    eps = 10 ** rng.uniform(-8, 2)
    return dict(optimizer="adam", lr=10 ** rng.uniform(-4, -1) * max(1.0, eps / 1e-2), eps=eps,
                beta1=1 - 10 ** rng.uniform(-2, 0), beta2=1 - 10 ** rng.uniform(-4, -1),
                weight_decay=10 ** rng.uniform(-5, -1))


def main():
    t0 = time.time()
    out = {}      # (protocol, opt) -> list over replicates of dict
    for r in range(R):
        data, test = make_data(split=r)
        for p, protocol in enumerate(("2017", "2019")):
            for o, opt in enumerate(OPTS):
                if protocol == "2017":
                    hs = grid_2017(opt)
                else:
                    rng = np.random.default_rng(4_000 + 10 * r + o)
                    hs = [sample_2019(opt, rng) for _ in range(BUDGET)]
                res = [trial(h, data, test, 100_000 * r + 10_000 * p + 1_000 * o + j) for j, h in enumerate(hs)]
                j = min(range(BUDGET), key=lambda i: (res[i][0], res[i][1]))
                fresh = [trial(hs[j], data, test, 900_000 + 1_000 * r + 100 * p + 10 * o + s)[2]
                         for s in range(N_FRESH)]
                out.setdefault((protocol, opt), []).append(
                    dict(h=hs[j], val=res[j][0], own=res[j][2], fresh=fresh, mean=float(np.mean(fresh))))
        print(f"replicate {r} done ({time.time() - t0:.0f}s)", flush=True)

    # ---- data for the figure: every fresh run (hollow), and replicate means with t-intervals
    xpos = {("2017", "momentum"): 0.85, ("2017", "adam"): 1.15, ("2019", "momentum"): 1.85, ("2019", "adam"): 2.15}
    rng = np.random.default_rng(0)
    cols = {"x": [], "err": [], "opt": [], "rep": []}
    mcols = {"x": [], "mean": [], "hw": [], "opt": []}
    for key, reps in out.items():
        for r, d in enumerate(reps):
            for e in d["fresh"]:
                cols["x"].append(xpos[key] + rng.uniform(-0.07, 0.07))
                cols["err"].append(100 * e)
                cols["opt"].append(OPTS.index(key[1]))
                cols["rep"].append(r)
        m, hw = lk.t_interval([100 * d["mean"] for d in reps])
        mcols["x"].append(xpos[key]); mcols["mean"].append(m); mcols["hw"].append(hw)
        mcols["opt"].append(OPTS.index(key[1]))
    for name, c in (("momentum", 0), ("adam", 1)):
        sel = [i for i, v in enumerate(cols["opt"]) if v == c]
        lk.write_dat(f"figures/data/04-budgets-protocols-runs-{name}.dat",
                     {k: [cols[k][i] for i in sel] for k in ("x", "err", "rep")})
        sel = [i for i, v in enumerate(mcols["opt"]) if v == c]
        lk.write_dat(f"figures/data/04-budgets-protocols-means-{name}.dat",
                     {k: [mcols[k][i] for i in sel] for k in ("x", "mean", "hw")})

    # ---- numbers quoted in the text
    print("\nTest error (%) of the selected configuration, mean over 5 fresh seeds;")
    print(f"mean over R={R} replicates with 95% t-interval half-width")
    for key, reps in out.items():
        m, hw = lk.t_interval([100 * d["mean"] for d in reps])
        own = np.mean([100 * d["own"] for d in reps])
        print(f"  protocol {key[0]} {key[1]:9s} {m:6.2f} +- {hw:4.2f}   (selected trial's own run: {own:6.2f})")
    from scipy import stats
    for protocol in ("2017", "2019"):
        d = [100 * (a["mean"] - b["mean"]) for a, b in zip(out[(protocol, "adam")], out[(protocol, "momentum")])]
        m, hw = lk.t_interval(d)
        wins = sum(x < 0 for x in d)
        p_sign = stats.binomtest(wins, R, 0.5).pvalue
        print(f"  protocol {protocol}: Adam - momentum = {m:+.2f} +- {hw:.2f} points (paired by split; "
              f"median {np.median(d):+.2f}; Adam lower in {wins} of {R}, two-sided sign test p = {p_sign:.3f})")
    for key, reps in out.items():
        bad = [(r, 100 * e) for r, d in enumerate(reps) for e in d["fresh"] if e > 0.5]
        if bad:
            r0 = bad[0][0]
            print(f"  failed fresh runs (test error > 50%) under protocol {key[0]} {key[1]}: {bad} "
                  f"(selected {reps[r0]['h']}; all five reruns on that split: "
                  f"{[round(100 * e, 2) for e in reps[r0]['fresh']]})")
    for opt in OPTS:
        d = [100 * (a["mean"] - b["mean"]) for a, b in zip(out[("2019", opt)], out[("2017", opt)])]
        m, hw = lk.t_interval(d)
        print(f"  {opt:9s}: protocol 2019 - protocol 2017 = {m:+.2f} +- {hw:.2f} points")
    print("\nSelected hyperparameters:")
    for key, reps in out.items():
        print(f"  protocol {key[0]} {key[1]}:")
        for d in reps:
            print("     " + ", ".join(f"{k}={v:.3g}" for k, v in d["h"].items() if k != "optimizer")
                  + f"  (val {100 * d['val']:.1f}, test {100 * d['mean']:.2f})")
    eps = [d["h"]["eps"] for d in out[("2019", "adam")]]
    wd_a = [d["h"]["weight_decay"] for d in out[("2019", "adam")]]
    wd_m = [d["h"]["weight_decay"] for d in out[("2019", "momentum")]]
    print(f"  protocol 2019 Adam: eps >= 1e-3 in {sum(e >= 1e-3 for e in eps)} of {R}; median eps {np.median(eps):.2g}; "
          f"median weight decay {np.median(wd_a):.2g} (momentum: {np.median(wd_m):.2g})")
    n_runs = R * 2 * 2 * (BUDGET + N_FRESH)
    print(f"\n{n_runs} runs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
