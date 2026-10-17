"""Chapter 2 (Sign Descent), figure fig:sign:noise.

Panel (a): one coordinate of signSGD on L(theta) = h theta^2 / 2 with minibatch gradients
g = h theta + (mean of B per-example noises).  Two per-example noise laws with mean 0 and
variance sigma^2: Gaussian, and a rare-event law sigma (X - p) / sqrt(p (1 - p)) with X ~ Bernoulli(p)
(most examples push slightly one way, a rare one strongly the other way).  The iterates live on the
lattice theta_0 + eta Z and form a birth-death chain; its stationary law follows from detailed
balance, pi(x + eta) P_down(x + eta) = pi(x) P_up(x), with P_down = P(g > 0), P_up = P(g < 0)
computed exactly (normal and binomial distribution functions, in logarithms).  Curves: the exact stationary
E|grad L|; points: long simulations (time and ensemble averages).

Panel (b): the rare class of Proposition prop:sign:rareclass.  Labels y ~ Bernoulli(pi), one bias b,
cross-entropy loss, minibatch gradient sigmoid(b) - mean(y).  If (1 - pi)^B > 1/2, signSGD drifts
to -infinity with speed eta (2 (1 - pi)^B - 1) per step (exact once sigmoid(b) < 1/B, which holds
from b = b* when B < 1/pi).  Points: averages of b_t over runs; lines: the exact drift.  Signum
(sign of an EMA of minibatch gradients, beta = 0.9) is shown for comparison.

Writes figures/data/02-sign-noise-eta.dat and figures/data/02-sign-noise-rare.dat.
Runs in well under a minute on one core.
"""
import os
import numpy as np
from scipy.stats import norm, binom

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

H = 1.0
SIGMA = 1.0
B = 16
P = 0.04          # (1 - P)^B = 0.520 > 1/2
SEED = 11


def p_gauss(x, b):
    """log P(g > 0) and log P(g < 0)."""
    z = H * x * np.sqrt(b) / SIGMA
    return norm.logcdf(z), norm.logcdf(-z)


def p_rare(x, b, p):
    # g > 0 iff K > kappa, with K ~ Bin(b, p)
    kappa = b * (p - H * x * np.sqrt(p * (1 - p)) / SIGMA)
    fl = np.floor(kappa)
    isint = np.isclose(kappa, np.round(kappa), rtol=0.0, atol=1e-12)
    with np.errstate(divide="ignore"):
        lpd = binom.logsf(fl, b, p)
        lpu = binom.logcdf(np.where(isint, np.round(kappa) - 1, fl), b, p)
    return lpd, lpu


def stationary_mean_abs_grad(probs, eta, offset, centre, width):
    k = int(np.ceil(width / eta))
    base = offset + eta * np.round((centre - offset) / eta)
    x = base + eta * np.arange(-k, k + 1)
    lpd, lpu = probs(x)
    # the recurrent class: no move down below the last point with P_down = 0, no move up above
    # the first point with P_up = 0
    lo = max([j for j in range(len(x)) if not np.isfinite(lpd[j])], default=0)
    hi = min([j for j in range(len(x)) if not np.isfinite(lpu[j])], default=len(x) - 1)
    logpi = np.full_like(x, -np.inf)
    logpi[lo] = 0.0
    for j in range(lo, hi):
        logpi[j + 1] = logpi[j] + lpu[j] - lpd[j + 1]
    pi = np.exp(logpi - logpi[lo:hi + 1].max())
    pi /= pi.sum()
    assert (lo > 0 or pi[0] < 1e-12) and (hi < len(x) - 1 or pi[-1] < 1e-12), "window too small"
    return float(np.sum(pi * np.abs(H * x)))


def simulate_mean_abs_grad(sampler, eta, offset, centre, rng, chains=1000):
    x0 = offset + eta * np.round((centre - offset) / eta)
    x = np.full(chains, x0)
    burn = max(3000, int(8.0 / eta))
    meas = max(4000, int(8.0 / eta))
    acc = 0.0
    for t in range(burn + meas):
        g = H * x + sampler(chains, rng)
        x = x - eta * np.sign(g)
        if t >= burn:
            acc += np.abs(H * x).mean()
    return acc / meas


def panel_a(rng):
    c = SIGMA * np.sqrt(P / (1 - P))
    etas = np.logspace(-3, np.log10(0.3), 11)
    gs = lambda n, r: r.standard_normal(n) * SIGMA / np.sqrt(B)
    rs = lambda n, r: SIGMA * (r.binomial(B, P, size=n) / B - P) / np.sqrt(P * (1 - P))
    rows = []
    for eta in etas:
        off = 0.123456 * eta
        ge = stationary_mean_abs_grad(lambda x: p_gauss(x, B), eta, off, 0.0, 1.5 + 60 * eta)
        re = stationary_mean_abs_grad(lambda x: p_rare(x, B, P), eta, off, c / H, 1.5 + 60 * eta)
        gsim = simulate_mean_abs_grad(gs, eta, off, 0.0, rng)
        rsim = simulate_mean_abs_grad(rs, eta, off, c / H, rng)
        bound = eta * H / 2 + SIGMA / np.sqrt(B)
        ou = (2 * np.pi) ** (-0.25) * np.sqrt(eta * H * SIGMA) * B ** (-0.25)
        rows.append((eta, ge, gsim, re, rsim, bound, ou, c))
        print(f"eta={eta:.4g}: gauss exact {ge:.5f} sim {gsim:.5f} (heuristic {ou:.5f}); "
              f"rare exact {re:.5f} sim {rsim:.5f}; bound {bound:.4f}; bias c={c:.4f}")
    with open(os.path.join(DATA, "02-sign-noise-eta.dat"), "w") as f:
        f.write("eta gaussexact gausssim rareexact raresim bound heuristic bias\n")
        for r in rows:
            f.write(" ".join(f"{v:.6e}" for v in r) + "\n")


def sigmoid(b):
    return 1.0 / (1.0 + np.exp(-b))


def panel_b(rng):
    pic = 0.01
    eta = 0.01
    steps = 20000
    runs = 200
    bstar = np.log(pic / (1 - pic))
    every = 500
    cols = {}
    for bsz in (16, 32, 64):
        b = np.full(runs, bstar)
        traj = [b.mean()]
        for t in range(1, steps + 1):
            ybar = rng.binomial(bsz, pic, size=runs) / bsz
            b = b - eta * np.sign(sigmoid(b) - ybar)
            if t % every == 0:
                traj.append(b.mean())
        cols[f"sim{bsz}"] = traj
        q = (1 - pic) ** bsz
        print(f"B={bsz}: q={q:.4f}, drift {-eta * (2 * q - 1):.6f}/step, b_T={traj[-1]:.2f}, "
              f"predicted {bstar - eta * (2 * q - 1) * steps:.2f}")
    # Signum, beta = 0.9, B = 16
    b = np.full(runs, bstar)
    m = np.zeros(runs)
    traj = [b.mean()]
    for t in range(1, steps + 1):
        ybar = rng.binomial(16, pic, size=runs) / 16
        m = 0.9 * m + 0.1 * (sigmoid(b) - ybar)
        b = b - eta * np.sign(m)
        if t % every == 0:
            traj.append(b.mean())
    cols["signum16"] = traj
    print(f"Signum B=16: b_T={traj[-1]:.3f} (b*={bstar:.3f})")
    ts = np.arange(0, steps + 1, every)
    with open(os.path.join(DATA, "02-sign-noise-rare.dat"), "w") as f:
        f.write("t bstar theory16 theory32 theory64 sim16 sim32 sim64 signum16\n")
        for j, t in enumerate(ts):
            th = [bstar - eta * (2 * (1 - pic) ** bsz - 1) * t for bsz in (16, 32, 64)]
            f.write(f"{t:d} {bstar:.6f} {th[0]:.6f} {th[1]:.6f} {th[2]:.6f} "
                    f"{cols['sim16'][j]:.6f} {cols['sim32'][j]:.6f} {cols['sim64'][j]:.6f} "
                    f"{cols['signum16'][j]:.6f}\n")
    print("threshold batch size ln2/(-ln(1-pi)) =", np.log(2) / -np.log(1 - pic))


def main():
    rng = np.random.default_rng(SEED)
    panel_a(rng)
    panel_b(rng)


if __name__ == "__main__":
    main()
