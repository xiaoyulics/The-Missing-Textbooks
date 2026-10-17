"""Lab 12.2 (12-protocol-counting): what is counted as a parameter.

Reanalyzes the matched-schedule runs (protocol C) of Lab 12.1, which must be run first
(python3 figures/scripts/12-protocol-schedules.py writes figures/data/12-protocol-schedules-runs.dat).
No new training runs.

The two-layer MLP of width w has N = w^2 + 11w + 1 parameters: an input layer of 9w (the
analogue of an embedding), a hidden layer of w^2 + w, and an output layer of w + 1 (the analogue of
the head). Three conventions:
  full      size N, compute 6 N D (every layer costs operations, so this is the true count);
  relabel   size N~ = w^2 + w (input and output layers left out), compute still 6 N D;
  kaplan    size N~ and compute 6 N~ D, as when embedding parameters and operations are left out.
For each convention the script locates the compute-optimal size on isoFLOP curves exactly as in
Lab 12.1 and fits N_opt ~ C^a, with 95% intervals from a bootstrap over seeds. It then compares
the measured exponents with the counting proposition of Section 12.5, using alpha_N, alpha_D estimated
from the full-count frontier and the known irreducible loss E = 0.01, and tabulates the
proposition for a Transformer family with Kaplan et al.'s vocabulary (no runs).

Run from the volume root:  python3 figures/scripts/12-protocol-counting.py
"""
import os
import sys
import math
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labkit as lk
import numpy as np

spec = importlib.util.spec_from_file_location("schedules", os.path.join(HERE, "12-protocol-schedules.py"))
sch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sch)

OUT = "figures/data/12-protocol-counting"
E = 0.01                     # irreducible loss of the teacher task (noise variance)
NBOOT = 2000
WIDTHS = sch.WIDTHS


def n_tilde(w):
    return w * w + w         # the hidden layer only


def read_runs():
    path = os.path.normpath(os.path.join(HERE, "..", "data", "12-protocol-schedules-runs.dat"))
    with open(path) as f:
        head = f.readline().split()
        rows = [dict(zip(head, map(float, line.split()))) for line in f if line.strip()]
    runs = {}
    for r in rows:
        if int(r["proto"]) == 0:
            runs[("C", int(r["seed"]), int(r["width"]), int(r["steps"]))] = r["loss"]
    return runs


def epsilon(w):
    """d ln N / d ln N~ for the MLP family, with w a continuous width."""
    return (2 * w + 11) / (2 * w + 1) * n_tilde(w) / sch.n_full(w)


def dlog_eps(w, h=1e-4):
    """d ln(epsilon) / d ln N~, numerically."""
    w1, w2 = w * (1 - h), w * (1 + h)
    return (math.log(epsilon(w2)) - math.log(epsilon(w1))) / (math.log(n_tilde(w2)) - math.log(n_tilde(w1)))


def width_of_tilde(nt):
    return (-1 + math.sqrt(1 + 4 * nt)) / 2


def width_of_full(n):
    return (-11 + math.sqrt(121 - 4 * (1 - n))) / 2


if __name__ == "__main__":
    runs = read_runs()
    seeds = sorted({k[1] for k in runs})
    full = {w: sch.n_full(w) for w in WIDTHS}
    tilde = {w: n_tilde(w) for w in WIDTHS}
    levels = sch.LEVELS
    # Kaplan-style levels: the same 13 levels moved down by the median of N~/N on the frontier
    klevels = levels * 0.55
    conv = {"full": (full, full, levels), "kaplan": (tilde, tilde, klevels)}

    rng = np.random.default_rng(12)
    est, xo_all, boot, boot_xo = {}, {}, {c: [] for c in conv}, {c: [] for c in conv}
    for c, (sz, cs, lv) in conv.items():
        est[c], xo_all[c] = sch.exponent(sch.mean_table(runs, "C", seeds), lv, sz, cs)
    for b in range(NBOOT):
        ss = list(rng.choice(seeds, size=len(seeds), replace=True))
        tab = sch.mean_table(runs, "C", ss)
        for c, (sz, cs, lv) in conv.items():
            a, xo = sch.exponent(tab, lv, sz, cs)
            boot[c].append(a)
            boot_xo[c].append([np.nan if v is None else v for v in xo])
    # relabel only: the full-count optima, expressed in N~
    to_t = lambda x: math.log(n_tilde(width_of_full(math.exp(x))))
    xo_rel = [None if v is None else to_t(v) for v in xo_all["full"]]
    ok = [i for i, v in enumerate(xo_rel) if v is not None]
    est["relabel"] = np.polyfit(np.log(levels[ok]), [xo_rel[i] for i in ok], 1)[0]
    boot["relabel"], boot_xo["relabel"] = [], []
    for row in boot_xo["full"]:
        xr = [np.nan if not np.isfinite(v) else to_t(v) for v in row]
        okr = [i for i, v in enumerate(xr) if np.isfinite(v)]
        boot_xo["relabel"].append(xr)
        boot["relabel"].append(np.polyfit(np.log(levels[okr]), [xr[i] for i in okr], 1)[0] if len(okr) >= 4 else np.nan)
    xo_all["relabel"] = xo_rel
    q = lambda v: (np.nanquantile(v, 0.025), np.nanquantile(v, 0.975))
    print("allocation exponent under three counting conventions (95% bootstrap intervals over seeds):")
    for c in ("full", "relabel", "kaplan"):
        lo, hi = q(boot[c])
        print(f"  {c:8s}: a = {est[c]:.3f}  [{lo:.3f}, {hi:.3f}]")
    for c in ("relabel", "kaplan"):
        d = np.array(boot[c]) - np.array(boot["full"])
        lo, hi = q(d)
        print(f"  {c} - full: {est[c] - est['full']:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    # optima files
    for c in ("full", "relabel", "kaplan"):
        lv = klevels if c == "kaplan" else levels
        xo, bx = xo_all[c], np.array(boot_xo[c], dtype=float)
        ok = [i for i, v in enumerate(xo) if v is not None]
        a, b0 = np.polyfit(np.log(lv[ok]), [xo[i] for i in ok], 1)
        lk.write_dat(OUT + f"-opt-{c}.dat", {
            "C": lv[ok], "Nstar": [math.exp(xo[i]) for i in ok],
            "lo": [math.exp(np.nanquantile(bx[:, i], 0.025)) for i in ok],
            "hi": [math.exp(np.nanquantile(bx[:, i], 0.975)) for i in ok],
            "fit": np.exp(b0 + a * np.log(lv[ok]))})
        print(f"  {c:8s} optima: " + " ".join(f"{cc:.1e}:{math.exp(xo[i]):.0f}" for cc, i in zip(lv[ok], ok)))

    # the counting proposition, with exponents estimated from the full-count frontier
    sch_opt = os.path.normpath(os.path.join(HERE, "..", "data", "12-protocol-schedules-opt-C.dat"))
    with open(sch_opt) as f:
        head = f.readline().split()
        rows = [dict(zip(head, map(float, line.split()))) for line in f if line.strip()]
    lc = np.log([r["C"] for r in rows]); ll = np.log([r["Lstar"] - E for r in rows])
    gamma = -np.polyfit(lc, ll, 1)[0]
    a = est["full"]
    aN, aD = gamma / a, gamma / (1 - a)
    print(f"frontier: L* - E ~ C^-{gamma:.3f}; with a = {a:.3f}: alpha_N = {aN:.2f}, alpha_D = {aD:.2f}")
    print("counting proposition, part (iii): predicted local exponents along the Kaplan-count frontier:")
    pred_k, pred_r = [], []
    for x in [v for v in xo_all["kaplan"] if v is not None]:
        w = width_of_tilde(math.exp(x))
        eps, de = epsilon(w), dlog_eps(w)
        pred_k.append(aD / (aD + aN * eps - de))
        print(f"   N~* = {math.exp(x):7.0f} (w = {w:5.1f}): epsilon = {eps:.3f}, dln eps = {de:.3f}, exponent {pred_k[-1]:.3f}")
    print("counting proposition, part (ii): predicted local exponents a/epsilon along the relabeled frontier:")
    for x in [v for v in xo_all["relabel"] if v is not None]:
        w = width_of_tilde(math.exp(x))
        pred_r.append(a / epsilon(w))
        print(f"   N~* = {math.exp(x):7.0f} (w = {w:5.1f}): epsilon = {epsilon(w):.3f}, exponent {pred_r[-1]:.3f}")
    print(f"  mean over the frontier: part (iii) {np.mean(pred_k):.3f}, part (ii) {np.mean(pred_r):.3f}")

    # the counting proposition for a Transformer family: N~ = 12 l d^2 with d/l = 40; embeddings
    # (50257 + 1024) d, so N = N~ + omega N~^(1/3); exponents of Hoffmann et al. (0.34, 0.28)
    omega = (50257 + 1024) * (40 / 12) ** (1 / 3)
    aN_h, aD_h = 0.34, 0.28
    fam = {k: [] for k in ("Ntilde", "ratio", "eps", "relabel", "kaplan")}
    a_h = aD_h / (aN_h + aD_h)
    print(f"Transformer family (omega = {omega:.0f}), a = {a_h:.3f} in full counts:")
    for e10 in [3, 4, 5, 6, 7, 8, 9, 10]:
        nt = 10.0 ** e10
        u = omega * nt ** (-2 / 3)
        eps = (1 + u / 3) / (1 + u)
        de = (2 / 3) ** 2 * u / ((1 + u) ** 2 * eps)
        fam["Ntilde"].append(nt); fam["ratio"].append(1 + u); fam["eps"].append(eps)
        fam["relabel"].append(a_h / eps); fam["kaplan"].append(aD_h / (aD_h + aN_h * eps - de))
        print(f"   N~ = 1e{e10}: N/N~ = {1 + u:8.2f}, epsilon = {eps:.3f}, relabel {a_h/eps:.3f},"
              f" kaplan {fam['kaplan'][-1]:.3f}")
    print(f"   limit as N~ -> 0 (kaplan): {aD_h / (aD_h + aN_h / 3):.3f}")
    # the part (iii) exponent over Kaplan et al.'s range of sizes (768 to 1.5e9), on a fine grid
    ng = np.logspace(math.log10(768), math.log10(1.5e9), 20001)
    ug = omega * ng ** (-2 / 3)
    eg = (1 + ug / 3) / (1 + ug)
    kg = aD_h / (aD_h + aN_h * eg - (2 / 3) ** 2 * ug / ((1 + ug) ** 2 * eg))
    print(f"   over 768..1.5e9 (kaplan): {kg[0]:.3f} at 768, maximum {kg.max():.3f} at N~ = {ng[kg.argmax()]:.2g},"
          f" {kg[-1]:.3f} at 1.5e9")
    lk.write_dat(OUT + "-family.dat", fam)
