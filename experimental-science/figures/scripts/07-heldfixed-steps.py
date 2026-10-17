"""Chapter 7 (What Was Held Fixed), Lab 7.2: steps to a target test error against the batch size.

The network and data of Lab 7.1 (two-layer MLP of width 64, digits split 0, plain SGD), now with a
constant learning rate and no warm-up, the setting of the noisy quadratic model. For each batch size
B = 8 ... 1258 and each learning rate on the grid 0.1 * 2^(k/2), k = 0..9, a run is evaluated on
both halves H1, H2 of the test set (the halves of Lab 7.1) after every step up to 64 and then at
steps spaced by 3%, and the first step at which its error reaches each target (10%, 5%, 3.5%) is
recorded; a run stops when every target is reached on both halves, at 1500 steps, or on divergence.

Two-fold cross-fitting as in Lab 7.1: for each target, the learning rate with the lowest median
first-hitting step on H1 over four selection seeds (0..3) is scored on H2 with eight fresh seeds
(100..107), and vice versa; each (B, target) thus has 16 reported hitting times. The script fits
T(B) = T_min (1 + B_crit / B_eff) to their medians by least squares in log steps, where
1/B_eff = (1/B)(n - B)/(n - 1) is the variance factor of a batch drawn without replacement from the
n = 1258 training images, with bootstrap intervals over seeds. Finally it measures McCandlish et
al.'s simple noise scale B_simple = tr(Sigma)/|G|^2 and noise scale B_noise = tr(H Sigma)/(G'HG) from
per-example gradients on the whole training set, at the points where a reference run (B = 64, its
selected learning rate for the 5% target, seed 0) first reaches each target on all 539 test images.

Prints the numbers quoted in the text and writes figures/data/07-heldfixed-steps-*.dat.
About two minutes on a laptop CPU with two threads.
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares

BATCHES = [8, 16, 32, 64, 128, 256, 512, 1024, 1258]
GRID = [0.1 * 2 ** (k / 2) for k in range(10)]
TARGETS = [0.10, 0.05, 0.035]
SEL_SEEDS = [0, 1, 2, 3]
REP_SEEDS = list(range(100, 108))
CAP = 1500
EVAL = sorted(set(list(range(1, 65)) + [int(round(64 * 1.03 ** k)) for k in range(300) if 64 * 1.03 ** k <= CAP] + [CAP]))
EVALSET = set(EVAL)


def hitting_run(B, lr, seed, data, halves, return_model_at=None):
    """Train with a constant learning rate; first step at which each half reaches each target.
    Mirrors labkit.run step by step (same seeds, batches and update)."""
    Xtr, ytr, _, _ = data
    cfg = lk.Config(optimizer="sgd", lr=lr, steps=CAP, batch_size=B, schedule="constant")
    torch.manual_seed(50_000 + seed)
    model = lk.MLP(Xtr.shape[1], 10, cfg.width, cfg.depth, cfg.activation)
    opt = lk.make_optimizer(cfg, model.parameters())
    n = Xtr.shape[0]
    bs = min(B, n)
    g_order = torch.Generator().manual_seed(60_000 + seed)
    perm, pos = torch.randperm(n, generator=g_order), 0
    hits = {(h, t): math.inf for h in halves for t in TARGETS}
    snaps = {}
    for step in range(CAP):
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g_order), 0
        idx = perm[pos:pos + bs]
        pos += bs
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        s = step + 1
        if s in EVALSET:
            if not math.isfinite(loss.item()):
                break
            with torch.no_grad():
                model.eval()
                for h, (X, y) in halves.items():
                    err = (model(X).argmax(1) != y).float().mean().item()
                    for t in TARGETS:
                        if hits[(h, t)] == math.inf and err <= t + 1e-12:
                            hits[(h, t)] = s
                if return_model_at is not None:
                    Xa, ya = return_model_at
                    err = (model(Xa).argmax(1) != ya).float().mean().item()
                    for t in TARGETS:
                        if t not in snaps and err <= t + 1e-12:
                            snaps[t] = (s, {k: v.detach().clone() for k, v in model.state_dict().items()})
                model.train()
            if all(v < math.inf for v in hits.values()) and (return_model_at is None or len(snaps) == len(TARGETS)):
                break
    return hits, snaps


def median_inf(v):
    v = np.sort(np.asarray(v, dtype=float))
    return float(np.median(v))


def fit_hyperbola(Bs, T, n):
    """Least squares in log steps for T = Tmin (1 + Bcrit / Beff)."""
    Bs = np.asarray(Bs, float)
    inv_eff = (1.0 / Bs) * (n - Bs) / (n - 1)
    ok = np.isfinite(T)
    def resid(p):
        lt, lb = p
        return np.log(np.exp(lt) * (1 + np.exp(lb) * inv_eff[ok])) - np.log(T[ok])
    p0 = [np.log(np.min(T[ok])), np.log(64.0)]
    r = least_squares(resid, p0)
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


def noise_scales(model, X, Y):
    """B_simple = tr(Sigma)/|G|^2 and B_noise = tr(H Sigma)/(G' H G) on the full training set."""
    from torch.func import functional_call, vmap, grad, jvp
    params = {k: v.detach() for k, v in model.named_parameters()}
    def loss_one(p, x, y):
        return F.cross_entropy(functional_call(model, p, (x.unsqueeze(0),)), y.unsqueeze(0))
    def loss_all(p):
        return F.cross_entropy(functional_call(model, p, (X,)), Y)
    g = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)
    keys = list(params)
    flat = torch.cat([g[k].reshape(len(Y), -1) for k in keys], 1).double()
    G = flat.mean(0)
    D = flat - G
    trS = (D ** 2).sum(1).mean().item()
    G2 = (G ** 2).sum().item()
    shapes = [params[k].shape for k in keys]
    sizes = [int(np.prod(s)) for s in shapes]
    def unflat(v):
        out, i = {}, 0
        for k, s, z in zip(keys, shapes, sizes):
            out[k] = v[i:i + z].reshape(s).float()
            i += z
        return out
    def hvp(v):
        return jvp(grad(loss_all), (params,), (unflat(v),))[1]
    def hv_flat(v):
        h = hvp(v)
        return torch.cat([h[k].reshape(-1) for k in keys]).double()
    GHG = torch.dot(G, hv_flat(G)).item()
    trHS = 0.0
    for chunk in torch.split(D, 64):
        hv = vmap(hvp)(chunk.float())
        hflat = torch.cat([hv[k].reshape(chunk.shape[0], -1) for k in keys], 1).double()
        trHS += (hflat * chunk).sum().item()
    trHS /= len(Y)
    return trS / G2, (trHS / GHG if GHG > 0 else float("nan")), trS, G2


def main():
    t0 = time.time()
    data = lk.load(lk.Config())
    Xtr, ytr, Xte, yte = data
    n = Xtr.shape[0]
    perm = np.random.default_rng(7007).permutation(len(yte))
    i1, i2 = perm[:270], perm[270:]
    halves = {"H1": (Xte[i1], yte[i1]), "H2": (Xte[i2], yte[i2])}
    cache = {}
    def get(B, lr, s):
        k = (B, round(lr, 10), s)
        if k not in cache:
            cache[k] = hitting_run(B, lr, s, data, halves)[0]
        return cache[k]

    other = {"H1": "H2", "H2": "H1"}
    rep = {}      # (B, target) -> array of 16 hitting times
    chosen = {}   # (B, target, half) -> lr
    for B in BATCHES:
        for t in TARGETS:
            for h in ("H1", "H2"):
                scores = [(median_inf([get(B, lr, s)[(h, t)] for s in SEL_SEEDS]), lr) for lr in GRID]
                chosen[(B, t, h)] = min(scores)[1]
        for t in TARGETS:
            vals = []
            for s in REP_SEEDS:
                vals.append([get(B, chosen[(B, t, h)], s)[(other[h], t)] for h in ("H1", "H2")])
            rep[(B, t)] = np.array(vals)  # seeds x folds
        print(f"B={B:5d} done ({time.time() - t0:.0f}s); selected learning rates (H1, H2): " +
              "; ".join(f"{100 * t:g}%: {chosen[(B, t, 'H1')]:.3g}, {chosen[(B, t, 'H2')]:.3g}" for t in TARGETS),
              flush=True)

    rng = np.random.default_rng(1)
    summary = {}
    for t in TARGETS:
        med = np.array([median_inf(rep[(B, t)].ravel()) for B in BATCHES])
        Tmin, Bcrit = fit_hyperbola(BATCHES, med, n)
        boots, asym = [], []
        for _ in range(1000):
            idx = rng.integers(0, len(REP_SEEDS), len(REP_SEEDS))
            mb = np.array([median_inf(rep[(B, t)][idx].ravel()) for B in BATCHES])
            if np.all(np.isfinite(mb)):
                boots.append(fit_hyperbola(BATCHES, mb, n))
                asym.append(BATCHES[0] * mb[0] / mb.min())
        boots = np.array(boots)
        a_lo, a_hi = np.quantile(asym, [0.025, 0.975])
        a_est = BATCHES[0] * med[0] / med.min()
        lo_b, hi_b = np.quantile(boots[:, 1], [0.025, 0.975])
        lo_t, hi_t = np.quantile(boots[:, 0], [0.025, 0.975])
        # bootstrap intervals for each median
        mlo, mhi = [], []
        for B in BATCHES:
            bm = [median_inf(rep[(B, t)][rng.integers(0, len(REP_SEEDS), len(REP_SEEDS))].ravel()) for _ in range(1000)]
            mlo.append(np.quantile(bm, 0.025)); mhi.append(np.quantile(bm, 0.975))
        summary[t] = (med, Tmin, Bcrit, (lo_b, hi_b), (lo_t, hi_t))
        print(f"\ntarget {100 * t:g}%: fitted T_min = {Tmin:.1f} ({lo_t:.1f} to {hi_t:.1f}), B_crit = {Bcrit:.0f} "
              f"({lo_b:.0f} to {hi_b:.0f}), D_min = T_min*B_crit = {Tmin * Bcrit:.0f} examples; "
              f"crossing of the asymptotes, (examples at B={BATCHES[0]})/(fewest steps) = {a_est:.0f} ({a_lo:.0f} to {a_hi:.0f})")
        Bs = np.array(BATCHES, float)
        inv_eff = (1 / Bs) * (n - Bs) / (n - 1)
        fitted = Tmin * (1 + Bcrit * inv_eff)
        for B, m_, a, b, f_ in zip(BATCHES, med, mlo, mhi, fitted):
            print(f"   B={B:5d}  median steps {m_:7.1f}  (95% bootstrap {a:.1f} to {b:.1f})  fit {f_:7.1f}  "
                  f"examples {B * m_:8.0f}  lr (H1,H2) = ({chosen[(B, t, 'H1')]:.3g}, {chosen[(B, t, 'H2')]:.3g})")
        tag = f"{int(round(1000 * t)):03d}"
        lk.write_dat(f"figures/data/07-heldfixed-steps-t{tag}.dat",
                     {"B": BATCHES, "med": med, "lo": mlo, "hi": mhi,
                      "lr1": [chosen[(B, t, "H1")] for B in BATCHES], "lr2": [chosen[(B, t, "H2")] for B in BATCHES],
                      "lrgeo": [math.sqrt(chosen[(B, t, "H1")] * chosen[(B, t, "H2")]) for B in BATCHES],
                      "examples": [B * m_ for B, m_ in zip(BATCHES, med)]})
        runsB, runsT = [], []
        for B in BATCHES:
            for v in rep[(B, t)].ravel():
                if np.isfinite(v):
                    runsB.append(B); runsT.append(v)
        lk.write_dat(f"figures/data/07-heldfixed-steps-runs-t{tag}.dat", {"B": runsB, "steps": runsT})
        grid = np.geomspace(6, 1400, 120)
        inv = (1 / grid) * np.clip(n - grid, 0, None) / (n - 1)
        lk.write_dat(f"figures/data/07-heldfixed-steps-fit-t{tag}.dat",
                     {"B": grid, "fit": Tmin * (1 + Bcrit * inv), "tmin": np.full_like(grid, Tmin),
                      "perfect": Tmin * Bcrit / grid})
    nonhit = sum(int(np.sum(~np.isfinite(v))) for v in rep.values())
    print(f"\nreported hitting times that never reached their target within {CAP} steps: {nonhit} of {16 * len(rep)}")

    # noise scales along a reference run
    Bref, tref = 64, 0.05
    lr_ref = math.sqrt(chosen[(Bref, tref, "H1")] * chosen[(Bref, tref, "H2")])
    lr_ref = min(GRID, key=lambda g: abs(math.log(g / lr_ref)))
    _, snaps = hitting_run(Bref, lr_ref, 0, data, halves, return_model_at=(Xte, yte))
    model = lk.MLP(Xtr.shape[1], 10)
    torch.manual_seed(50_000)
    init_model = lk.MLP(Xtr.shape[1], 10)
    bs, bn, _, _ = noise_scales(init_model, Xtr, ytr)
    print(f"\nnoise scales on the training set, reference run B={Bref}, lr={lr_ref:.3g}, seed 0:")
    print(f"   at initialization: B_simple = {bs:.1f}, B_noise = {bn:.1f}")
    nsrows = {"target": [], "step": [], "Bsimple": [], "Bnoise": [], "Bcrit": []}
    for t in TARGETS:
        if t in snaps:
            s, sd = snaps[t]
            model.load_state_dict(sd)
            bs, bn, trS, G2 = noise_scales(model, Xtr, ytr)
            print(f"   first reaching {100 * t:g}% (step {s}): B_simple = {bs:.1f}, B_noise = {bn:.1f} "
                  f"(tr Sigma = {trS:.3g}, |G|^2 = {G2:.3g}); fitted B_crit = {summary[t][2]:.0f}")
            nsrows["target"].append(100 * t); nsrows["step"].append(s); nsrows["Bsimple"].append(bs)
            nsrows["Bnoise"].append(bn); nsrows["Bcrit"].append(summary[t][2])
    lk.write_dat("figures/data/07-heldfixed-steps-noise.dat", nsrows)
    print(f"distinct training runs: {len(cache)}; total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
