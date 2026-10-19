"""Chapter 3 (Consistency): coherence checks of a forecaster, against coherent liars and against an
accurate but incoherent forecaster.

The world has three binary events, so eight outcomes; the true distribution P is drawn from a flat
Dirichlet law (seed 0). A forecaster is asked, in separate contexts, for the probability q(E) of
events E (nonempty proper subsets of the outcomes) and for conditional probabilities q(E | F).
A check draws its events at random and accepts if a coherence relation holds within tau = 0.02:
  N  negation      |q(E) + q(not E) - 1| <= tau
  A  additivity    |q(E or F) + q(E and F) - q(E) - q(F)| <= tau   (all four events proper, nonempty)
  C  conditional   |q(E and F) - q(E | F) q(F)| <= tau             (Bayes's rule)
Forecasters:
  coherent liar   Q = (1 - lam) P + lam * (point mass on the least likely outcome); every check
                  accepts it (the coherent-liar proposition), however far Q is from P
  noisy           q = P(E) + sigma Z, with fresh Z ~ N(0,1) for every query (unbiased, incoherent);
                  N accepts with probability erf(tau / (2 sigma)), A with erf(tau / (2 sqrt 2 sigma))
  one-half        q = 1/2 for every query: passes N and A, fails C
Inaccuracy is the root mean square of q(E) - P(E) over all events (the square root of the excess
Brier score).
Writes figures/data/03-consistency-forecast-*.dat
Run from the volume root:  python3 figures/scripts/03-consistency-forecast.py
"""
import os
import sys
import time

import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk

TAU = 0.02
CHECKS = 20_000
FULL = 255                                  # the sure event; events are 8-bit masks 1..254
EVENTS = np.arange(1, FULL)


def prob(dist, mask):
    """Probability of the events (bit masks over the 8 outcomes) under dist."""
    bits = (np.asarray(mask)[..., None] >> np.arange(8)) & 1
    return bits @ dist


def draw_pairs(gen, size, kind):
    """Random event pairs (E, F) for the additivity (kind 'A') or conditional (kind 'C') check."""
    out_e, out_f = [], []
    while len(out_e) < size:
        e = gen.choice(EVENTS, size=size)
        f = gen.choice(EVENTS, size=size)
        inter, union = e & f, e | f
        ok = (inter != 0) & (union != FULL) if kind == "A" else (inter != 0)
        out_e.extend(e[ok]); out_f.extend(f[ok])
    return np.array(out_e[:size]), np.array(out_f[:size])


class Coherent:
    def __init__(self, dist):
        self.dist = dist

    def q(self, mask, gen):
        return prob(self.dist, mask)

    def qcond(self, e, f, gen):
        pf = prob(self.dist, f)
        pef = prob(self.dist, e & f)
        return np.where(pf > 0, pef / np.where(pf > 0, pf, 1.0), 0.5)


class Noisy:
    def __init__(self, dist, sigma):
        self.dist, self.sigma = dist, sigma

    def q(self, mask, gen):
        return prob(self.dist, mask) + self.sigma * gen.standard_normal(np.shape(mask))

    def qcond(self, e, f, gen):
        return Coherent(self.dist).qcond(e, f, gen) + self.sigma * gen.standard_normal(np.shape(e))


class Half:
    def q(self, mask, gen):
        return np.full(np.shape(mask), 0.5)

    def qcond(self, e, f, gen):
        return np.full(np.shape(e), 0.5)


def run_checks(fc, gen, n=CHECKS):
    """Acceptance counts of the three checks (each query answered separately)."""
    e = gen.choice(EVENTS, size=n)
    passN = np.abs(fc.q(e, gen) + fc.q(FULL ^ e, gen) - 1) <= TAU
    e, f = draw_pairs(gen, n, "A")
    passA = np.abs(fc.q(e | f, gen) + fc.q(e & f, gen) - fc.q(e, gen) - fc.q(f, gen)) <= TAU
    e, f = draw_pairs(gen, n, "C")
    passC = np.abs(fc.q(e & f, gen) - fc.qcond(e, f, gen) * fc.q(f, gen)) <= TAU
    return int(passN.sum()), int(passA.sum()), int(passC.sum())


def rmse(fc_q, dist):
    return float(np.sqrt(np.mean((fc_q - prob(dist, EVENTS)) ** 2)))


def main():
    gen = pk.rng(0)
    P = gen.dirichlet(np.ones(8))
    pE = prob(P, EVENTS)
    print("true outcome probabilities:", np.round(P, 3))
    worst = np.zeros(8); worst[int(np.argmin(P))] = 1.0

    # coherent liars
    rows = []
    print("\ncoherent liars Q = (1-lam) P + lam * point mass (acceptance of N, A, C checks)")
    for lam in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        Q = (1 - lam) * P + lam * worst
        kN, kA, kC = run_checks(Coherent(Q), gen)
        k = kN + kA + kC
        lo, hi = pk.clopper_pearson(k, 3 * CHECKS)
        d = rmse(prob(Q, EVENTS), P)
        print(f"  lam = {lam:4.2f}  rms error {d:.4f}  N {kN/CHECKS:.4f}  A {kA/CHECKS:.4f}  C {kC/CHECKS:.4f}")
        rows.append((d, k / (3 * CHECKS), k / (3 * CHECKS) - lo, hi - k / (3 * CHECKS)))
    pk.write_dat("figures/data/03-consistency-forecast-liars.dat", ["rmse", "acc", "em", "ep"], rows)

    # noisy, unbiased forecasters
    rows = []
    print("\nnoisy forecasters q = P + sigma Z (acceptance; theory erf(tau/(2 sigma)), erf(tau/(2 sqrt2 sigma)))")
    for sigma in np.geomspace(0.002, 0.3, 12):
        kN, kA, kC = run_checks(Noisy(P, sigma), gen)
        loN, hiN = pk.clopper_pearson(kN, CHECKS)
        loA, hiA = pk.clopper_pearson(kA, CHECKS)
        rN, rA = kN / CHECKS, kA / CHECKS
        print(f"  sigma = {sigma:.4f}  N {rN:.4f} ({erf(TAU/(2*sigma)):.4f})  "
              f"A {rA:.4f} ({erf(TAU/(2*np.sqrt(2)*sigma)):.4f})  C {kC/CHECKS:.4f}")
        rows.append((sigma, rN, rN - loN, hiN - rN, rA, rA - loA, hiA - rA))
    pk.write_dat("figures/data/03-consistency-forecast-noisy.dat",
                 ["sigma", "accN", "emN", "epN", "accA", "emA", "epA"], rows)
    grid = np.geomspace(0.001, 1.0, 80)
    pk.write_dat("figures/data/03-consistency-forecast-theory.dat", ["sigma", "thN", "thA"],
                 [(s, float(erf(TAU / (2 * s))), float(erf(TAU / (2 * np.sqrt(2) * s)))) for s in grid])

    # the forecaster that always says one half
    kN, kA, kC = run_checks(Half(), gen)
    d = rmse(np.full(EVENTS.size, 0.5), P)
    k = kN + kA + kC
    lo, hi = pk.clopper_pearson(k, 3 * CHECKS)
    print(f"\none-half forecaster: rms error {d:.4f}; N {kN/CHECKS:.3f}, A {kA/CHECKS:.3f}, C {kC/CHECKS:.3f}; "
          f"mixture {k/(3*CHECKS):.4f}")
    pk.write_dat("figures/data/03-consistency-forecast-half.dat", ["rmse", "acc", "em", "ep"],
                 [(d, k / (3 * CHECKS), k / (3 * CHECKS) - lo, hi - k / (3 * CHECKS))])
    print(f"range of true event probabilities: {pE.min():.3f} .. {pE.max():.3f}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nruntime {time.perf_counter() - t0:.1f} s")
