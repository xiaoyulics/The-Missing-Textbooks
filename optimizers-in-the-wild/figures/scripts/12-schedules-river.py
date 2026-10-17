#!/usr/bin/env python3
"""Figure 12.1 (Optimizers in the Wild, Chapter 12): loss curves of four learning-rate schedules in
the river valley, exact expected loss (the recursion of the river-valley theorem) against SGD runs.

Model. A quadratic river with 200 directions, curvatures h_j = 0.02 j^-2 and initial losses
(h_j/2) y_j^2 proportional to j^-2 (total 0.5), so that the river loss decays like (area)^(-1/2);
no gradient noise along the river. Forty hills with curvatures log-spaced in [0.2, 2], started at
zero, with independent gradient noise of variance 10/40 each. SGD theta <- theta - eta_t (grad + noise),
peak step size 0.04, T = 4000 steps, schedules: constant, cosine to zero, linear decay to zero,
warmup-free WSD with a linear cooldown over the last 20% of the steps.

Output: figures/data/12-schedules-river.dat (exact, every 20 steps) and
        figures/data/12-schedules-river-sgd.dat (mean of 100 SGD runs with Gaussian noise, every 100 steps).
Printed: the final values quoted in the text, and the tuned peaks of the four schedules (exact, no
randomness; computed after the data files are written).
Runtime: about 10 seconds (numpy only, seed 12).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def instance(q=2.0, n_river=200, h_river=0.02, river_loss=0.5, n_hill=40, h_lo=0.2, h_hi=2.0, noise=10.0):
    j = np.arange(1, n_river + 1)
    h_r = h_river * j ** -2.0
    c_r = j ** -q
    c_r = river_loss * c_r / c_r.sum()            # (h/2) y^2 at the start
    h_h = np.geomspace(h_lo, h_hi, n_hill)
    h = np.concatenate([h_r, h_h])
    y0sq = np.concatenate([2 * c_r / h_r, np.zeros(n_hill)])
    s2 = np.concatenate([np.zeros(n_river), np.full(n_hill, noise / n_hill)])
    return h, s2, y0sq, n_river


def schedule(name, T, peak, c=0.2):
    """eta_1..eta_{T}: the step sizes used for the steps t = 1..T (theta_1 is the start)."""
    t = np.arange(1, T + 1)
    if name == "const":
        e = np.ones(T)
    elif name == "cosine":
        e = 0.5 * (1 + np.cos(np.pi * t / T))
    elif name == "linear":
        e = 1 - t / T
    elif name == "wsd":
        K = int(round(c * T))
        e = np.ones(T)
        e[T - K:] = 1 - np.arange(1, K + 1) / K
    else:
        raise ValueError(name)
    return peak * e


def exact(eta, h, s2, y0sq, n_river, every):
    """E y^2 recursion of the theorem; returns (steps, total, river, hill) after every `every` steps."""
    v = y0sq.astype(float).copy()
    rows = []
    for t in range(len(eta) + 1):
        if t % every == 0:
            rr = 0.5 * np.sum(h[:n_river] * v[:n_river])
            hh = 0.5 * np.sum(h[n_river:] * v[n_river:])
            rows.append((t, rr + hh, rr, hh))
        if t < len(eta):
            v = (1 - eta[t] * h) ** 2 * v + eta[t] ** 2 * s2
    return np.array(rows)


def sgd(etas, h, s2, y0sq, n_river, runs, every, rng):
    """Run len(etas) schedules x runs SGD trajectories at once; returns mean (total, river, hill)."""
    M, T = etas.shape
    d = len(h)
    y = np.broadcast_to(np.sqrt(y0sq), (M, runs, d)).copy()
    sd = np.sqrt(s2)
    out = []
    for t in range(T + 1):
        if t % every == 0:
            l_r = 0.5 * np.sum(h[:n_river] * y[:, :, :n_river] ** 2, axis=2).mean(axis=1)
            l_h = 0.5 * np.sum(h[n_river:] * y[:, :, n_river:] ** 2, axis=2).mean(axis=1)
            out.append((t, l_r + l_h, l_r, l_h))
        if t < T:
            noise = rng.standard_normal((M, runs, d)) * sd
            y -= etas[:, t][:, None, None] * (h * y + noise)
    return out


def main():
    rng = np.random.default_rng(12)
    h, s2, y0sq, nr = instance()
    T, peak = 4000, 0.04
    names = ["const", "cosine", "linear", "wsd"]
    etas = np.array([schedule(nm, T, peak) for nm in names])
    ex = [exact(e, h, s2, y0sq, nr, every=20) for e in etas]
    with open(os.path.join(DATA, "12-schedules-river.dat"), "w") as f:
        f.write("t const cosine linear wsd wsdriver wsdhill cosriver coshill\n")
        for i in range(len(ex[0])):
            f.write("%d %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n" % (
                ex[0][i, 0], ex[0][i, 1], ex[1][i, 1], ex[2][i, 1], ex[3][i, 1],
                ex[3][i, 2], ex[3][i, 3], ex[1][i, 2], ex[1][i, 3]))
    sim = sgd(etas, h, s2, y0sq, nr, runs=100, every=100, rng=rng)
    with open(os.path.join(DATA, "12-schedules-river-sgd.dat"), "w") as f:
        f.write("t const cosine linear wsd wsdriver wsdhill cosriver coshill\n")
        for (t, tot, rr, hh) in sim:
            f.write("%d %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n" % (
                t, tot[0], tot[1], tot[2], tot[3], rr[3], hh[3], rr[1], hh[1]))
    # summary for the text
    fin = {nm: ex[k][-1, 1] for k, nm in enumerate(names)}
    print("final expected loss:", {k: round(v, 4) for k, v in fin.items()})
    print("final SGD mean:", {nm: round(sim[-1][1][k], 4) for k, nm in enumerate(names)})
    print("WSD at start of cooldown (t=3200): total %.4f river %.4f hill %.4f" % tuple(ex[3][160, 1:]))
    print("WSD at end: total %.4f river %.4f hill %.4f" % tuple(ex[3][-1, 1:]))
    print("cosine at end: total %.4f river %.4f hill %.4f" % tuple(ex[1][-1, 1:]))
    print("const at end: total %.4f river %.4f hill %.4f" % tuple(ex[0][-1, 1:]))
    floor = np.sum(peak * s2[nr:] / (2 * (2 - peak * h[nr:])))
    print("noise floor at the peak: %.4f (eta*s/4 = %.4f)" % (floor, peak * 10 / 4))
    tuned_peaks(h, s2, y0sq, nr, T, names)


def final_loss_batch(E, h, s2, y0sq):
    """Exact final expected loss for each row of step sizes E (no randomness)."""
    v = np.broadcast_to(y0sq, (E.shape[0], len(h))).astype(float).copy()
    for t in range(E.shape[1]):
        e = E[:, t][:, None]
        v = (1 - e * h) ** 2 * v + e ** 2 * s2
    return 0.5 * (v * h).sum(axis=1)


def final_loss_log(eta, h, s2, y0sq, nr):
    """Exact final expected loss with the hill second moments summed in the log domain, for peaks at
    which some early steps are unstable (eta h > 2) and intermediate values would overflow."""
    v = y0sq[:nr].astype(float).copy()
    for e in eta:
        v = (1 - e * h[:nr]) ** 2 * v
    total = 0.5 * np.sum(h[:nr] * v)
    with np.errstate(divide="ignore"):
        for i in range(nr, len(h)):            # hills start at zero: only the injected noise counts
            lq = 2 * np.log(np.abs(1 - eta * h[i]))
            after = np.concatenate([np.cumsum(lq[::-1])[::-1][1:], [0.0]])   # sum over later steps
            terms = np.log(eta ** 2 * s2[i]) + after
            m = terms.max()
            total += 0.5 * h[i] * np.exp(m + np.log(np.sum(np.exp(terms - m))))
    return total


def tuned_peaks(h, s2, y0sq, nr, T, names):
    """Numbers quoted in the text: each schedule's best peak among the peaks at which every step is
    stable (peak * max h < 2), and cosine at peaks beyond that limit."""
    for nm in names:
        peaks = np.geomspace(0.01, 0.9999, 60)                    # coarse search, then a fine one
        L = final_loss_batch(np.array([schedule(nm, T, p) for p in peaks]), h, s2, y0sq)
        k = np.argmin(L)
        peaks = np.linspace(peaks[max(k - 1, 0)], peaks[min(k + 1, len(peaks) - 1)], 41)
        L = final_loss_batch(np.array([schedule(nm, T, p) for p in peaks]), h, s2, y0sq)
        k = np.argmin(L)
        print("tuned (stable peaks) %-6s: best peak %.3f, final loss %.4f" % (nm, peaks[k], L[k]))
    for p in [1.5, 1.9, 2.0]:
        print("cosine at peak %.1f (beyond the limit): final loss %.4f" % (
            p, final_loss_log(schedule("cosine", T, p), h, s2, y0sq, nr)))


if __name__ == "__main__":
    main()
